# -*- coding: utf-8 -*-
"""Core data analytics logic.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import os
import pickle
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
import pandas as pd
from matplotlib.dates import date2num
from scipy.interpolate import pchip_interpolate
from scipy.stats import linregress

from .config import DefaultConfig
from .config import TestConfig


class Analyser:
    """Execute core analytics logic.

    This class implements the main execution sequence of the HKU diabetes
    regression analysis. It saves the results of the regression and CKD
    thresholds as csv, and all other intermediate steps as pickle.

    Args:
        config: Configuration class, default to DefaultConfig.

    Attributes:
        patient_ids: A list of valid patient IDs analysed.
        intermediate: A dictionary of all objects in intermediate steps.
        results: A dictionary containing regression results and ckd values.
    """

    def __init__(self, *, config: Type[DefaultConfig] = DefaultConfig):
        self.config = config
        self.patient_ids = []
        self.intermediate = {}
        self.results = {'regression': pd.DataFrame(), 'ckd': pd.DataFrame()}

    def _save(self):
        """Save analytics results to file.

        This should only be called by the run method.
        """
        if not os.path.exists(self.config.results_path):
            os.makedirs(self.config.results_path)
        with open('%s/intermediate.pickle' % self.config.results_path,
                  'wb') as file:
            pickle.dump(self.intermediate, file, pickle.HIGHEST_PROTOCOL)
        for key, item in self.results.items():
            item = item.dropna()
            item.index = self.patient_ids
            item.to_csv("%s/%s.csv" % (self.config.results_path, key))
        print("Finished saving analyser data")

    def load(self) -> Dict[str, pd.DataFrame]:
        """Load analytics results from file.

        Call this method to load the previous analytics results. Calling
        script should catch FileNotFoundError and call the run method.

        Raises:
            FileNotFoundError: No results files are found in config.results_path.

        Returns:
            A dictionary containing results for regression and ckd as
            DataFrame.

        Example:
            >>> from hku_diabetes.analytics import Analyser
            >>> from hku_diabetes.importer import import_all
            >>> analyser = Analyser()
            >>> try:
            >>>     results = analyser.load()
            >>> except FileNotFoundError:
            >>>     data = import_all()
            >>>     results = analyser.run(data)
        """
        try:
            with open('%s/intermediate.pickle' % self.config.results_path,
                      'rb') as file:
                self.intermediate = pickle.load(file)
        except FileNotFoundError as e:
            print("No results files are found in config.results_path")
            raise e
        else:
            self.patient_ids = [x['patient_id'] for x in self.intermediate]
            for key in self.results:
                self.results[key] = pd.read_csv(
                    "%s/%s.csv" % (self.config.results_path, key), index_col=0)
            print("Finished loading analyser data")
        return self.results

    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Execute the main date analytics sequence.

        Call this method to execute the actual data analytics.
        All results are saved in path specified by config.results_path.

        Args:
            data: A dictionary at least containing Creatinine, Hb1aC,
                and Demographics as DataFrames.

        Returns:
            A dictionary containing results for regression and ckd as
            DataFrame.

        Example:
            >>> from hku_diabetes.analytics import Analyser
            >>> from hku_diabetes.importer import import_all
            >>> analyser = Analyser()
            >>> data = import_all()
            >>> results = analyser.run(data)
        """
        tic = time.time()
        dropna(data)
        intersect(data)
        evaluate_eGFR(data)
        patient_ids = data['Creatinine'].index.unique().sort_values()
        if self.config is TestConfig:
            patient_ids = patient_ids[:self.config.test_samples]
        with ProcessPoolExecutor() as executor:
            intermediate_results = executor.map(analyse_subject,
                                                itertools.repeat(data),
                                                patient_ids,
                                                itertools.repeat(self.config))
        self.intermediate = [x for x in intermediate_results if x is not None]
        self.patient_ids = [x['patient_id'] for x in self.intermediate]
        self.results['regression'] = pd.DataFrame(
            [x['regression'] for x in self.intermediate])
        self.results['ckd'] = pd.DataFrame(
            [x['ckd'] for x in self.intermediate],
            columns=self.config.ckd_thresholds)
        self._save()
        print('Finished analysis, time passed: %is' % (time.time() - tic))
        return self.results


def analyse_subject(data: Dict[str, pd.DataFrame],
                    patient_id: int,
                    config: Type[DefaultConfig] = DefaultConfig) -> Union[None, dict]:
    """Compute the regression result and ckd values for one subject.

    This function takes the data of one subject and compute its corresponding
    regression results and ckd values. It is called by Analyser.run via a
    ProcessPoolExecutor. It checks if either the Creatinine or Hb1aC has the
    minimum number of rows required by config.min_analysis_samples, and returns
    None if fails.

    Args:
        data: A dictionary at least containing Creatinine, Hb1aC,
            and Demographics as DataFrames, and only contains rows
            for one subject.
        patient_id: ID of the patient as int.
        config: Configuration class, default to DefaultConfig.

    Returns:
        Either None or a dictionary of results including regression and ckd,
        as well as intermediate steps including patient_id, Creatinine,
        Hba1C, regression, ckd, Creatinine_LP, and cumulative_Hba1C.

    Example:
        >>> from hku_diabetes import analytics
        >>> from hku_diabetes.importer import import_all
        >>> data = import_all()
        >>> patient_id = 802
        >>> intermediate = analytics.analyse_subject(data, patient_id)
    """
    Creatinine = data['Creatinine'].loc[[patient_id]].sort_values('Datetime')
    Hba1C = data['Hba1C'].loc[[patient_id]].sort_values('Datetime')
    if len(Creatinine) < config.min_analysis_samples or len(
            Hba1C) < config.min_analysis_samples:
        # Too few data points for proper analysis
        return None
    Creatinine, Hba1C = remove_duplicate(Creatinine, Hba1C)
    # Low pass filtering of the eGFR as there are too many measurements in some days
    Creatinine_LP = Creatinine.resample(
        config.eGFR_low_pass, on='Datetime').mean().dropna()
    # Convert the datetime to matplotlib datetime objects
    Creatinine_time = date2num(Creatinine['Datetime'])
    Hba1C_time = date2num(Hba1C['Datetime'])
    Creatinine_LP_time = date2num(Creatinine_LP.index)
    time_range = find_time_range(Creatinine_time, Hba1C_time, config)
    cumulative_Hba1C = np.cumsum(
        pchip_interpolate(Hba1C_time, Hba1C['Value'], time_range))
    cumulative_Hba1C = pchip_interpolate(time_range, cumulative_Hba1C,
                                         Creatinine_LP_time)
    inverse_regression = np.poly1d(
        np.polyfit(Creatinine_LP['eGFR'], cumulative_Hba1C, 1))
    intermediate = OrderedDict()
    intermediate['patient_id'] = patient_id
    intermediate['Creatinine'] = Creatinine
    intermediate['Hba1C'] = Hba1C
    intermediate['regression'] = linregress(cumulative_Hba1C,
                                            Creatinine_LP['eGFR'])
    intermediate['ckd'] = inverse_regression(config.ckd_thresholds)
    intermediate['Creatinine_LP'] = Creatinine_LP
    intermediate['cumulative_Hba1C'] = cumulative_Hba1C
    return intermediate


def find_time_range(Creatinine_time: np.ndarray,
                    Hba1C_time: np.ndarray,
                    config: Type[DefaultConfig] = DefaultConfig) -> np.ndarray:
    """Finds the longest possible overlapping time range between Creatinine and Hba1C.

        Args:
            Creatinine_time: Array of Creatinine datetime as Matplotlib dates.
            Hba1C_time: Array of Hba!C datetime as Matplotlib dates.
            config: Configuration class, default to DefaultConfig.

        Returns:
            An array of longest possible overlapping datetime as Matplotlib dates.

        Example:
            >>> from matplotlib.dates import date2num
            >>> from hku_diabetes import analytics
            >>> from hku_diabetes.importer import import_all
            >>> data = import_all()
            >>> patient_id = 802
            >>> Creatinine = data['Creatinine'].loc[[patient_id]]
            >>> Hba1C = data['Hba1C'].loc[[patient_id]]
            >>> Creatinine_time = date2num(Creatinine['Datetime'])
            >>> Hba1C_time = date2num(Hba1C['Datetime'])
            >>> time_range = analytics.find_time_range(Creatinine_time, Hba1C_time)
    """
    latest_startime = max(min(Creatinine_time), min(Hba1C_time))
    earliest_endtime = min(max(Creatinine_time), max(Hba1C_time))
    time_range = np.arange(
        latest_startime, earliest_endtime,
        (earliest_endtime - latest_startime) / config.interpolation_samples)
    return time_range


def dropna(data: Dict[str, pd.DataFrame]):
    """Calls dropna of all DataFrames in the data dictionary

    Args:
        data: A dictionary at least containing Creatinine, Hb1aC,
            and Demographics as DataFrames.

    Example:
        >>> from hku_diabetes import analytics
        >>> from hku_diabetes.importer import import_all
        >>> data = import_all()
        >>> analytics.dropna(data)
    """
    for key in data:
        data[key] = data[key].dropna()


def evaluate_eGFR(data: Dict[str, pd.DataFrame]):
    """Evaluates the eGFR value for each row of the Creatinine DataFrame.

    This function takes the Sex and DOB from the Demographic DataFrame
    for each patient, and computes the corresponding Age of the patient at the
    time of each row of the Creatinine measurement.
    It uses the referenced eGFR formula assuming all subjects are not African.
    The computed eGFR values are inserted for all rows of the creatinine DataFrame.

    Reference:
    http://www.sydpath.stvincents.com.au/tests/ChemFrames/MDRDBody.htm

    Args:
        data: A dictionary at least containing Creatinine, Hb1aC,
            and Demographics as DataFrames.

    Example:
        >>> from hku_diabetes import analytics
        >>> from hku_diabetes.importer import import_all
        >>> data = import_all()
        >>> analytics.evaluate_eGFR(data)
        >>> print(data['Creatinine']['eGFR'])
    """
    unique_patient_ids = set(data['Creatinine'].index)
    data['Creatinine'].loc[unique_patient_ids, 'DOB'] = pd.to_datetime(
        data['Demographic'].loc[unique_patient_ids, 'DOB'])
    data['Creatinine'].loc[unique_patient_ids, 'Sex'] = data[
        'Demographic'].loc[unique_patient_ids, 'Sex']
    data['Creatinine']['Age'] = (data['Creatinine']['Datetime'] -
                                 data['Creatinine']['DOB']).dt.days // 365
    Scr = data['Creatinine']['Value']
    Age = data['Creatinine']['Age']
    Sex = data['Creatinine']['Sex']
    data['Creatinine']['eGFR'] = (
        175 * ((0.0113 * Scr)**
               (-1.154)) * (Age**(-0.203)) * (0.742 if Sex is 'F' else 1))


def intersect(data: Dict[str, pd.DataFrame]):
    """Finds the intersects of unique patients from each DataFrame.

    Args:
        data: A dictionary at least containing Creatinine, Hb1aC,
            and Demographics as DataFrames.

    Example:
        >>> from hku_diabetes import analytics
        >>> from hku_diabetes.importer import import_all
        >>> data = import_all()
        >>> analytics.intersect(data)
    """
    for resource_name, resource in data.items():
        try:
            unique_patient_ids = set(resource.index) & unique_patient_ids
        except NameError:
            unique_patient_ids = set(resource.index)
    for resource_name, resource in data.items():
        data[resource_name] = resource.loc[unique_patient_ids]


def remove_duplicate(Creatinine: pd.DataFrame,
                     Hba1C: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Removes duplicate measurements taken at the same datetime.

    For some reasons, more than one entries are recorded at the same time
    and same date, but containing diferent values. This was observed for both
    Creatinine and Hba1c. This function finds the such entries and only keeps
    the first record.

    Args:
        Creatinine: Creatinine DataFrame.
        Hba1C: Hba1C DataFrame.

    Returns:
        A tuple of filtered Creatinine and Hba1C DataFrame

    Example:
        >>> from hku_diabetes import analytics
        >>> from hku_diabetes.importer import import_all
        >>> data = import_all()
        >>> patient_id = 802
        >>> Creatinine = data['Creatinine'].loc[[patient_id]]
        >>> Hba1C = data['Hba1C'].loc[[patient_id]]
        >>> Creatinine, Hba1C = analytics.remove_duplicate(Creatinine, Hba1C)
    """
    Creatinine_time = date2num(Creatinine['Datetime'])
    Hba1C_time = date2num(Hba1C['Datetime'])
    Creatinine = Creatinine.iloc[[True] + list(np.diff(Creatinine_time) > 0)]
    Hba1C = Hba1C.iloc[[True] + list(np.diff(Hba1C_time) > 0)]
    return Creatinine, Hba1C
