# -*- coding: utf-8 -*-
"""Script to call the package scrapper.
"""
import itertools
import pandas as pd
import re
import sys
import time

from concurrent.futures import ProcessPoolExecutor

from hku_diabetes.scraper import get_all_trade_names
from hku_diabetes.config import RunConfig
from hku_diabetes.config import TestConfig
from hku_diabetes.importer import import_resource

COMBINATION_WORDS = ['plus', 'hct']


def match_trade_name(trade_name_tuple, medication):
    tic = time.time()
    name = trade_name_tuple[1]['first_word']
    name = re.sub('[^A-Za-z0-9\-]+', ' ', str(name))
    name = re.sub('[\-]+', '', str(name))
    name = name.lower()
    print("Annotating medication table with %s" %name)
    category_name = trade_name_tuple[1]['category_name']
    generic_name = trade_name_tuple[0] 
    matched_rows=[]
    need_inspection=[]
    for j, med in enumerate(medication['Drug Name']):
        med = re.sub('[^A-Za-z0-9]+', ' ', str(med))
        med = re.sub('[\-]+', '', str(med))
        med = med.lower()
        if re.search(name, med):
            matched_rows.append(j)
            inspection = False
            for word in COMBINATION_WORDS:
                if word in med:
                    inspection = True
            need_inspection.append(inspection)
    annotated = medication.iloc[matched_rows]
    annotated['generic_name'] = generic_name
    annotated['category_name'] = category_name
    annotated['need_inspection'] = need_inspection
    print('Finished %s, time passed: %is' %(name, (time.time() - tic)))
    return annotated


if __name__ == '__main__':
    if 'run' in sys.argv:
        Config = RunConfig
    else:
        Config = TestConfig
    trade_names = get_all_trade_names(RunConfig)
    medication = import_resource('Medication', config=Config)
    trade_names['first_word'] = [name.split(' ')[0] for name in trade_names['Name of Product']]
    trade_names.drop_duplicates(['first_word', 'category_name'], inplace=True)
    medication["category_name"] = None
    medication["generic_name"] = None    
    medication["need_inspection"] = False

    trade_names = trade_names[trade_names['category_name'] == 'CCB']
    if 'run' not in sys.argv:
        medication = medication.iloc[:100]
        trade_names = trade_names.iloc[: 10]

    with ProcessPoolExecutor() as executor:
        matched_rows_generator = executor.map(match_trade_name,
                                            trade_names.iterrows(),
                                            itertools.repeat(medication))
    annotated_medication = pd.concat(list(matched_rows_generator))
    annotated_medication.drop_duplicates(inplace=True)

    # for trade_name_tuple in trade_names.iterrows():
    #     matched_rows = match_trade_name(trade_name_tuple, medication)

    annotated_medication.to_csv("%s/annotated_medication.csv" %Config.processed_data_path)
