# -*- coding: utf-8 -*-
"""Web scrapper for interacting with drugoffice.gov.hk.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import requests
import pandas as pd
from os import makedirs
from os.path import exists

from hku_diabetes.config import DefaultConfig
from hku_diabetes.config import TestConfig


def get_all_trade_names(config: type = DefaultConfig) -> pd.DataFrame:
    """Obtain all drug trade name for generic names in drug_trade_names.csv.

    This function expects to find 'drug_trade_names.csv' in config.raw_data_path.
    It then calls get_one_trade_names recurisvely for each generic name, and
    concatenate all the trade names into a single DataFrame.

    Returns:
        A DataFrame of drug trade names.

    Example:
        >>> trade_names = get_all_trade_names()
    """    
    try:
        trade_names = pd.read_csv(
            "%s/drug_trade_names.csv" %TestConfig.processed_data_path, index_col=0)
    except IOError:  
        if not exists(config.processed_data_path):
            makedirs(config.processed_data_path)          
        generic_names = pd.read_csv(
            "%s/drug_generic_names.csv" %TestConfig.raw_data_path)
        if config is TestConfig:
            generic_names = generic_names.iloc[:,:2]
        trade_names_list = []
        for category_name in generic_names:
            for index, generic_name in enumerate(
                    generic_names[category_name]):
                if config is TestConfig and index>=3:
                    break
                if isinstance(generic_name, str): 
                    names = get_one_trade_names(generic_name)
                    names['generic_name'] = generic_name
                    names['category_name'] = category_name
                    trade_names_list.append(names)              
        trade_names = pd.concat(trade_names_list)
        trade_names.set_index('generic_name', inplace=True)
        trade_names.to_csv(
            "%s/drug_trade_names.csv" %TestConfig.processed_data_path)
        print("Obtained all drug trade names.")
    return trade_names


def get_one_trade_names(generic_name: str) -> pd.DataFrame:
    """Obtain the list of drug trade name from drugoffice.gov.hk.

    Args:
        generic_name: The name of drug to be searched.

    Returns:
        A DataFrame of drug trade names.

    Example:
        >>> trade_names = get_one_trade_names('Benazepril')
    """
    request_url = 'https://www.drugoffice.gov.hk/eps/drug/productSearchOneFieldAction'
    data = {
    	'fromLang': 'en',
    	'fromSection': 'consumer',
    	'keyword':generic_name,
    	'pageNoRequested': 1,
    	'perPage': 1000,
    	'searchType':'O',
    	'userType':'E'
    }
    r = requests.post(request_url, data=data)
    df_list = pd.read_html(r.text, index_col=0, header=0)
    trade_names = df_list[0]
    trade_names = trade_names.iloc[3:, :4]	# Get rid of the rows and columns that are parsed incorrectly.
    trade_names.columns = ['Name of Product', 'Certificate Holder', 'Reg. No', 'Ingredients']
    print("Pulled trade names for : %s" %generic_name)
    return trade_names