# -*- coding: utf-8 -*-
"""Script to call the package scrapper.
"""
import re
import sys

from hku_diabetes.scraper import get_all_trade_names
from hku_diabetes.config import RunConfig
from hku_diabetes.config import TestConfig
from hku_diabetes.importer import import_resource

COMBINATION_WORDS = ['plus', 'hct']
if __name__ == '__main__':
    if 'run' in sys.argv:
        Config = RunConfig
    else:
        Config = TestConfig
    trade_names = get_all_trade_names(Config)
    medication = import_resource('Medication', config=Config)
    trade_names = trade_names[trade_names['category_name'] == 'CCB']
    trade_names['first_word'] = [name.split(' ')[0] for name in trade_names['Name of Product']]
    trade_names.drop_duplicates(['first_word', 'category_name'], inplace=True)
    medication["drug_category"] = None
    medication["need_inspection"] = False    
    for i, trade_name_row in trade_names.iterrows():
        name = trade_name_row['Name of Product']
        category_name = trade_name_row['category_name']
        for j, med_row in medication.iterrows():
            med = med_row['Drug Name']
            name = re.sub('[^A-Za-z0-9]+', '', str(name))
            med = re.sub('[^A-Za-z0-9]+', '', str(med))
            if name in med.split(" "):
                med_row['drug_category'] = category_name
                for word in COMBINATION_WORDS:
                    if word in med:
                        med_row['need_inspection'] = True
    medication.to_csv("%s/annotated_medication.csv" %config.processed_data_path)
