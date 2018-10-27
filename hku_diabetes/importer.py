# -*- coding: utf-8 -*-
"""Importer for importing resources from the data directory.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from os import listdir, makedirs
from os.path import exists, join, isfile
from typing import Dict
from typing import Type

import pandas as pd

from .config import DefaultConfig


def import_all(config: Type[DefaultConfig] = DefaultConfig) -> Dict[str, pd.DataFrame]:
    """Imports all resources and returns a dictionary of resources.

    It searches for sub-directory in config.raw_data_path and 
    checks against config.required_resources. If the resources is
    required, it first tries to import the CSV file with the same name
    in config.processed_Data. If this fails, it searches for all files 
    within the resource directory and checks against the file name ending
    and extension against config.data_file_extension. After loading the data
    files, it then saves all the data as CSV file to be imported directly next
    time. It also calls data cleaning logic to convert the column names of the
    resources to something better.

    Args:
        config: Configuration class, default to DefaultConfig.

    Returns:
        A dictionary containing all required resources as DataFrames.

    Example:
        >>> from hku_diabetes.importer import import_all
        >>> data = import_all()
    """
    data = OrderedDict()
    for resource_name in config.required_resources:
        if resource_name == 'Demographic':
            continue  # need the separate routine below to import demographic data
        tic = time.time()
        resource_key = resource_name
        data[resource_key] = import_resource(resource_name)
        print('Finished importing %s, time passed: %is' % (resource_name,
                                                           time.time() - tic))

    # Special routine for Demographic data
    try:
        data['Demographic'] = pd.read_csv(
            "%s/Demographic.csv" % (config.processed_data_path), index_col=0)
    except IOError:
        data['Demographic'] = pd.read_excel(
            "%s/total 7307_DOB_DOD_SEX.xlsx" % (config.raw_data_path),
            index_col=1,
            header=0)
        data['Demographic'].to_csv(
            "%s/Demographic.csv" % (config.processed_data_path))
    _cleaning(data)
    return data


def import_resource(resource_name: str,
                    config: Type[DefaultConfig] = DefaultConfig) -> pd.DataFrame:
    """Imports one particular resource.

    This function is a sub-routine called by import_all to import one 
    particular resource. It first tries to import the CSV file with the same name
    in config.processed_Data. If this fails, it searches for all files within
    the resource directory and checks against the file name ending and extension against
    config.data_file_extension. After loading the data files, it then saves all the data as
    CSV file to be imported directly next time.

    Args:
        resource_name: The name of the resource to be loaded.
        config: Configuration class, default to DefaultConfig.

    Returns:
        A DataFrame of the resource, with patient_id as index, and column names matching
        the raw data file.

    Example:
        >>> from hku_diabetes.importer import import_resource
        >>> resource = import_resource('Creatinine')
    """
    csv_filename = "%s/%s.csv" % (config.processed_data_path, resource_name)
    try:
        resource = pd.read_csv(csv_filename, index_col=0)
    except IOError:
        path = "%s/%s" % (config.raw_data_path, resource_name)
        files = [file for file in listdir(path) if isfile(join(path, file))]
        data_files = [
            join(path, file) for file in files
            if file.endswith(config.data_file_extensions)
        ]
        with ProcessPoolExecutor() as executor:
            df_lists = executor.map(_read_html_file, data_files)
        dfs = [df_list[0] for df_list in df_lists if df_list]
        resource = pd.concat(dfs)
        if not exists(config.processed_data_path):
            makedirs(config.processed_data_path)
        resource.to_csv(csv_filename)
    return resource


def _read_html_file(filepath: str) -> pd.DataFrame:
    """Helper function to read a single HTML file,
    called by ProcessPoolExecutor.
    """
    with open(filepath, 'r') as file:
        df = pd.read_html(file.read(), index_col=0, header=0)
    return df


def _cleaning(data: dict):
    """Data cleaning logic to convert the column names of the resources to
    something better."""
    for resource_name, resource in data.items():
        if 'LIS Reference Datetime' in resource:
            resource['Datetime'] = pd.to_datetime(
                resource['LIS Reference Datetime'])
        if 'LIS Result: Numeric Result' in resource:
            resource['Value'] = resource['LIS Result: Numeric Result']
        if resource_name in ("Creatinine", "Hba1C"):
            data[resource_name] = resource[['Datetime', 'Value']]
        if resource_name == "Demographic":
            data[resource_name] = resource[['DOB', 'Sex']]
    return
