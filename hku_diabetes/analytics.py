# -*- coding: utf-8 -*-
"""Core data analytics logic.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import decimal
import itertools
import os
import pickle
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from typing import Dict
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
    run analysis. It saves the results of the run and CKD
    thresholds as csv, and all other subject_data steps as pickle.

    Args:
        config: Configuration class, default to DefaultConfig.

    Attributes:
        patient_ids: A list of valid patient IDs analysed.
        intermediate: A dictionary of all objects in subject_data steps.
        results: A dictionary containing run results and ckd values.
    """

    def __init__(self, *, config: Type[DefaultConfig] = DefaultConfig):
        self.config = config
        self.patient_ids = []
        self.subject_data = {}
        self.results = {'regression': pd.DataFrame(), 'ckd': pd.DataFrame()}

    def _save(self):
        """Save analytics results to file.

        This should only be called by the run method.
        """
        if not os.path.exists(self.config.results_path):
            os.makedirs(self.config.results_path)
        with open('%s/subject_data.pickle' % self.config.results_path,
                  'wb') as file:
            pickle.dump(self.subject_data, file, pickle.HIGHEST_PROTOCOL)
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
            with open('%s/subject_data.pickle' % self.config.results_path,
                      'rb') as file:
                self.subject_data = pickle.load(file)
        except FileNotFoundError as e:
            print("No results files are found in config.results_path")
            raise e
        else:
            self.patient_ids = [x['patient_id'] for x in self.subject_data]
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
            >>> analytics.evaluate_eGFR(data)
            >>> analyser = Analyser()
            >>> data = import_all()
            >>> results = analyser.run(data)
        """
        tic = time.time()
        patient_ids = data['Creatinine'].index.unique().sort_values()
        print("patient_ids before intersecting: %i" %len(patient_ids))
        dropna(data, self.config)
        intersect(data, self.config)
        evaluate_eGFR(data)
        patient_ids = data['Creatinine'].index.unique().sort_values()
        print("patient_ids after intersecting: %i" %len(patient_ids))
        if self.config is TestConfig:
            patient_ids = patient_ids[:self.config.test_samples]
            subject_data = []
            for i, patient_id in enumerate(patient_ids):
                subject_data.append(analyse_subject(data, patient_id, self.config))
                if subject_data[-1]:
                    print("Processing subject %i, prescriptions: %i" % (i, len(subject_data[-1]['prescriptions'])))
        else:
            with ProcessPoolExecutor() as executor:
                subject_data = executor.map(analyse_subject,
                                                    itertools.repeat(data),
                                                    patient_ids,
                                                    itertools.repeat(self.config))
        self.subject_data = [x for x in subject_data if x is not None]
        self.patient_ids = [x['patient_id'] for x in self.subject_data]
        self.results['regression'] = pd.DataFrame(
            [x['regression'] for x in self.subject_data])
        self.results['ckd'] = pd.DataFrame(
            [x['ckd'] for x in self.subject_data],
            columns=self.config.ckd_thresholds)
        self.group_analysis()
        self._save()
        print('Finished analysis, time passed: %is' % (time.time() - tic))
        return self.results


    def group_analysis(self):
        # selected = self.select_group(primary='SGLT2i')
        # selected = self.select_group(primary='DDP4i')
        selected = self.select_group(primary='SGLT2i', exclude='DDP4i')
        # selected = self.select_group(primary='DDP4i', exclude='SGLT2i')
        import pdb; pdb.set_trace()

    def select_group(self, primary, exclude=None, low_init_eGFR=True):
        selected = []
        for subject in self.subject_data:
            if primary in subject['prescriptions']['category'].tolist():
                selected.append(subject)
        return selected

def analyse_subject(data: Dict[str, pd.DataFrame],
                    patient_id: int,
                    config: Type[DefaultConfig] = DefaultConfig) -> Union[None, dict]:
    """Compute the regression result and ckd values for one subject.

    This function takes the data of one subject and compute its corresponding
    regression results and ckd values. It is called by Analyser.regression via a
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
        as well as subject_data steps including patient_id, Creatinine,
        Hba1C, regression, ckd, Creatinine_LP, and cumulative_Hba1C.

    Example:
        >>> from hku_diabetes import analytics
        >>> from hku_diabetes.importer import import_all
        >>> data = import_all()
        >>> analytics.evaluate_eGFR(data)
        >>> patient_id = 802
        >>> subject_data = analytics.analyse_subject(data, patient_id)
    """
    Creatinine = data['Creatinine'].loc[[patient_id]].sort_values('Datetime')
    Hba1C = data['Hba1C'].loc[[patient_id]].sort_values('Datetime')
    LDL = data['LDL'].loc[[patient_id]].sort_values('Datetime')
    medication = data['Medication'].loc[[patient_id]].sort_values('Prescription Start Date')
    demographic = data['Demographic'].loc[[patient_id]]
    diagnosis = data['Diagnosis'].loc[[patient_id]].sort_values('Reference Date')
    procedure = data['Procedure'].loc[[patient_id]].sort_values('Procedure Date (yyyy-mm-dd)')
    Creatinine = remove_duplicate(Creatinine)
    Hba1C = remove_duplicate(Hba1C)

    diagnosis = convert_code(diagnosis, 'All Diagnosis Code (ICD9)', 'Reference Date', config.diagnosis_code)
    procedure = convert_code(procedure, 'All Procedure Code', 'Procedure Date (yyyy-mm-dd)', config.procedure_code)

    if 'dialysis' in diagnosis['name'].tolist() or 'dialysis' in procedure['name'].tolist():
        # Exclude patients on dialysis, as their creatinine does not represent intrinsic eGFR.
        return None

    if len(Creatinine) < config.min_analysis_samples or len(
            Hba1C) < config.min_analysis_samples:
        # Too few data points for proper analysis
        return None

    if (config.filter_by_starting_eGFR and 
        Creatinine.iloc[0]['eGFR']<config.starting_eGFR):
        # Remove the subject from analysis if the starting eGFR is too small.
        return None

    if config.filter_by_starting_eGFR:
        # eGFR tends to drop over time, the first time it drops below the threshold is the 
        # first_invalid_eGFR
        try:
            first_invalid_eGFR = (Creatinine['eGFR'] < config.starting_eGFR).tolist().index(True)
        except ValueError:
            first_valid_eGFR = 0
        else:
            first_valid_eGFR = first_invalid_eGFR - 1
        Creatinine = Creatinine.iloc[first_valid_eGFR:]

    # Low pass filtering of the eGFR as there are too many measurements in some days
    Creatinine_LP = Creatinine.resample(
        config.eGFR_low_pass, on='Datetime').mean().dropna()
    
    if len(Creatinine_LP) < config.min_analysis_samples:
        # Too few data points for proper analysis
        return None

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
    subject_data = OrderedDict()


    prescriptions = get_continuous_prescriptions(medication, Creatinine, config)

    subject_data['patient_id'] = patient_id
    subject_data['date of death'] = demographic['DOD']
    subject_data['prescriptions'] = prescriptions
    subject_data['diagnosis'] = diagnosis
    subject_data['procedure'] = procedure
    subject_data['Creatinine'] = Creatinine
    subject_data['Hba1C'] = Hba1C
    subject_data['LDL'] = LDL
    subject_data['regression'] = linregress(cumulative_Hba1C,
                                            Creatinine_LP['eGFR'])
    subject_data['ckd'] = inverse_regression(config.ckd_thresholds)
    subject_data['Creatinine_LP'] = Creatinine_LP
    subject_data['cumulative_Hba1C'] = cumulative_Hba1C
    return subject_data


def find_time_range(Creatinine_time: np.ndarray,
                    Hba1C_time: np.ndarray,
                    config: Type[DefaultConfig] = DefaultConfig) -> np.ndarray:
    """Finds the appropriate time range between Creatinine and Hba1C.

        Args:
            Creatinine_time: Array of Creatinine datetime as Matplotlib dates.
            Hba1C_time: Array of Hba1C datetime as Matplotlib dates.
            config: Configuration class, default to DefaultConfig.

        Returns:
            An array of datetime as Matplotlib dates.

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


def convert_code(items, code_key, date_key, mapping):
    """Convert all procedure or convert_code code to general categories"""
    if items[code_key].dtype == 'O':
      has_E = items[code_key].str.contains('E', regex=False)   # Some diagnosis or procedure code is not numeric
      items = items[has_E == False]
      has_V = items[code_key].str.contains('V', regex=False)   # Some diagnosis or procedure code is not numeric
      items['code'] = pd.to_numeric(items[code_key].str.strip('V'))
    else:
      items['code'] = items[code_key]
    decoded_items = []
    for name, item_codes in mapping.items():
        for code in item_codes:
            if type(code) is str:
                code = float(code.strip('V'))
                candidates = items[has_V]
            else:
                if items[code_key].dtype == 'O':
                    candidates = items[has_V == False]
                else:
                    candidates = items
            decimal_point = -1 * decimal.Decimal(str(code)).as_tuple().exponent
            match_code = np.floor(candidates['code'] * 10**decimal_point) == code * 10**decimal_point
            matched_items = candidates[match_code]
            for i, row in matched_items.iterrows():
                decoded_items.append({
                    'name': name,
                    'code': row['code'],
                    'date': row[date_key]
                })
    decoded_items = pd.DataFrame(decoded_items, columns=['name', 'code', 'date']).sort_values('date')
    return decoded_items


def get_continuous_prescriptions(medication, Creatinine, config):
    """Reduce medication entries to the number of continuous prescriptions."""
    available_drug_categories = medication.columns[21:]
    continuous_prescriptions = []
    for category in available_drug_categories:
        prescriptions = medication.loc[medication[category]]
        if len(prescriptions) > 0:
            if category == 'DDP4i':
                low_eGFR = Creatinine[Creatinine['eGFR'] < 45]
                if not low_eGFR.empty:
                    cut_off = low_eGFR['Datetime'].iloc[0]
                    prescriptions = prescriptions[pd.to_datetime(prescriptions['Prescription End Date']) < cut_off]
                    if len(prescriptions) == 0:
                        continue
            prescription_start = pd.to_datetime(prescriptions['Prescription Start Date'])
            prescription_end = pd.to_datetime(prescriptions['Prescription End Date'])
            prescription_gap = prescription_start[1:] - prescription_end[:-1]
            is_discontinuous = prescription_gap.dt.days > config.max_continuous_prescription_gap
            continuous_prescription_start = prescription_start[[True] + is_discontinuous.tolist()]
            continuous_prescription_end = prescription_start[is_discontinuous.tolist() + [True]]
            for start, end in zip(continuous_prescription_start, continuous_prescription_end):
                continuous_prescriptions.append({'category': category,
                                                 'name': prescriptions['Drug Name'].iloc[0],
                                                 'start': start,
                                                 'end': end})
    continuous_prescriptions = pd.DataFrame(continuous_prescriptions,
                                            columns=['category', 'name', 'start', 'end']).sort_values('start')
    for category in available_drug_categories:
        continuous_prescriptions['concurrent %s' % category] = False
    for i, prescription in continuous_prescriptions.iterrows():
        for j, match in continuous_prescriptions.iterrows():
            if prescription['category']==match['category']:
                continue
            if prescription['start'] < match['start'] < prescription['end']:
                continuous_prescriptions.loc[i, 'concurrent %s' % match['category']] = True
    return continuous_prescriptions


def dropna(data: Dict[str, pd.DataFrame], config: Type[DefaultConfig] = DefaultConfig):
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
    for key in ["Creatinine", "Hba1C"]:
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


def intersect(data: Dict[str, pd.DataFrame], config: Type[DefaultConfig] = DefaultConfig):
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
    for resource_name in config.must_have_resources:
        resource = data[resource_name]
        try:
            unique_patient_ids = set(resource.index) & unique_patient_ids
        except NameError:
            unique_patient_ids = set(resource.index)
        print("%s: %i" %(resource_name, len(unique_patient_ids)))
    for resource_name, resource in data.items():
        data[resource_name] = resource.loc[unique_patient_ids]


def remove_duplicate(resource: pd.DataFrame) -> pd.DataFrame:
    """Removes duplicate measurements taken at the same datetime.

    For some reasons, more than one entries are recorded at the same time
    and same date, but containing different values. This was observed for both
    Creatinine and Hba1c. This function finds the such entries and only keeps
    the first record.

    Args:
        resource: A DataFrame of the resource to remove duplicate.

    Returns:
        A DataFrame with duplicates removed.

    Example:
        >>> from hku_diabetes import analytics
        >>> from hku_diabetes.importer import import_all
        >>> data = import_all()
        >>> patient_id = 802
        >>> Creatinine = data['Creatinine'].loc[[patient_id]]
        >>> Creatinine = analytics.remove_duplicate(Creatinine)
    """
    resource_time = date2num(resource['Datetime'])
    resource = resource.iloc[[True] + list(np.diff(resource_time) > 0)]
    return resource
