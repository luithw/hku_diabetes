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
COMBINATION_DRUGS = [
    'BLOPRESS', 
    'ATACAND', 
    'TEVETEN', 
    'HYZAAR', 
    'OSARTIL',
    'OLMESARTAN',
    'OLMETEC',
    'MICARDIS',
    'DIOVAN',
    'EXFORGE'
    'VALSARTAN'
    ]
INSPECTION_CATEGORY = ['Long acting nitrate']


def add_label(row, key, value):
    if row[key] is None:
        row[key] = value
    else:
        row[key] = row[key] + ', ' + value
    return row


def match_trade_name(trade_name_tuple, medication):
    tic = time.time()
    name = trade_name_tuple[1]['search_name']
    name = re.sub('[\/]+', ' ', str(name))
    name = re.sub('[^A-Za-z0-9\-]+', ' ', str(name))
    name = re.sub('[\-]+', '', str(name))
    name = name.lower()
    generic_name = trade_name_tuple[0]    
    trade_name_row = trade_name_tuple[1]
    category_name = trade_name_row['category_name']    
    trade_name = trade_name_row['Name of Product']
    print("Annotating medication table with generic_name: %s and trade_name: %s" %(generic_name, name))
    matched_rows=[]
    need_inspection=[]
    for j, (med, unique_id) in enumerate(zip(medication['Drug Name'], medication['unique_id'])):
        # med = re.sub('[\/]+', ' ', str(med))    
        med = re.sub('[^A-Za-z0-9]+', ' ', str(med))
        med = re.sub('[\-]+', '', str(med))
        med = med.lower()
        # if ((name == 'amlodpine' or name == 'valsartan') 
        #     and unique_id == 6628) : breakpoint()
        matched = False
        if len(name.split(" ")) > 1:
            matched = True
            for word in name.split(" "):
                if word not in med.split(" "):
                    matched = False
        else:
            if name in med.split(" "):
                matched = True

        if matched:            
            matched_rows.append(j)
            inspection = False
            if category_name in INSPECTION_CATEGORY:
                inspection = True
            for combination_drug in COMBINATION_DRUGS:
                if combination_drug.lower() in med:
                    # breakpoint()
                    for word in med:
                        if word not in COMBINATION_WORDS:
                            inspection = True
            need_inspection.append(inspection)
    annotated = medication.iloc[matched_rows]
    for i, (name, row) in enumerate(annotated.iterrows()):  
        row = add_label(row, 'generic_name', generic_name)
        row =add_label(row, 'category_name', category_name)
        row = add_label(row, 'trade_name', trade_name) 
        row['need_inspection'] = row['need_inspection'] or need_inspection
        annotated.iloc[i] = row
    print('Finished %s, time passed: %is' %(name, (time.time() - tic)))
    return annotated


if __name__ == '__main__':
    tic = time.time()    
    if 'run' in sys.argv or 'run' in globals():
        Config = RunConfig
    else:
        Config = TestConfig

    # try:
    #     annotated = pd.read_csv("%s/annotated_medication.csv" 
    #         %Config.processed_data_path, index_col=0)
    #     unannotated = pd.read_csv("%s/unannotated_medication.csv" 
    #         %Config.processed_data_path, index_col=0)
    #     need_inspection = pd.read_csv("%s/need_inspection_medication.csv" 
    #         %Config.processed_data_path, index_col=0)

    # # except IOError:    
    drug_names = get_all_trade_names(RunConfig)
    # Add the generic names to be pretended to be trade name so that it can also be searched.
    drug_names['generic_name'] = drug_names.index
    drug_names['search_name'] = [name.split(' ')[0] for name in drug_names['Name of Product']]    
    generic_names_excel = pd.read_excel(
        "%s/Drug names.xlsx" % config.raw_data_path, sheet_name=None)    
    for sheet_name, generic_names in generic_names_excel.items():
        if sheet_name == "To notes":
            # Ignore the To notes sheet
            continue
        if config is TestConfig:
            generic_names = generic_names.iloc[:, :2]
        for category_name in generic_names:    
            for i, generic_name in enumerate(generic_names[category_name]):
                if isinstance(generic_name, str):
                    drug_names = drug_names.append({
                        'Name of Product':generic_name,
                        'generic_name':generic_name,
                        'category_name': category_name,
                        'search_name': generic_name
                        }, ignore_index=True)
    drug_names.set_index('generic_name', drop=False, inplace = True)
    drug_names.drop_duplicates(['search_name', 'category_name'], inplace=True)

    assert not drug_names.loc['Cilnidipine'].empty

    medication = import_resource('Medication', config=Config)
    unannotated = medication[['Drug Name', 'Route']].drop_duplicates()
    unannotated['unique_id']=range(len(unannotated))
    unannotated['category_name'] = None
    unannotated['generic_name'] = None
    unannotated['trade_name'] = None        
    unannotated['need_inspection'] = False
    google_name = []
    for i, name in enumerate(unannotated['Drug Name']):
        google_name.append('=HYPERLINK("https://www.google.com.hk/search?q=%s","Google")' %name)
    unannotated['Google'] = google_name

    # drug_names = drug_names[drug_names['category_name'] == 'CCB']
    # drug_names = drug_names.loc['Felodipine']

    # if 'run' not in sys.argv and 'run' not in globals():
    #     unannotated = unannotated.iloc[:100]
    #     drug_names = drug_names.iloc[8:100]

    annotated_list=[]
    need_inspection_list = []

    if 'run' in sys.argv or 'run' in globals():
        with ProcessPoolExecutor() as executor:
            matched_rows_gen = executor.map(match_trade_name,
                                                drug_names.iterrows(),
                                                itertools.repeat(unannotated))
        annotated = pd.concat(list(matched_rows_gen))
        annotated.drop_duplicates(inplace=True)
    else:
        matched_rows = []
        for trade_name_tuple in drug_names.iterrows():
            matched_rows.append(match_trade_name(trade_name_tuple, unannotated))
        annotated = pd.concat(matched_rows)            

    need_inspection = annotated[annotated['need_inspection'] == 1]
    unannotated_unique_id = set(unannotated['unique_id']) - set(annotated['unique_id'])
    unannotated = unannotated[unannotated['unique_id'].isin(unannotated_unique_id)]

    annotated.to_csv("%s/annotated_medication.xlsx" 
        %Config.processed_data_path)
    unannotated.to_csv("%s/unannotated_medication.xlsx" 
        %Config.processed_data_path)
    need_inspection.to_csv("%s/need_inspection_medication.xlsx" 
        %Config.processed_data_path)
    annotated.to_excel("%s/annotated_medication.xlsx" 
        %Config.processed_data_path)
    unannotated.to_excel("%s/unannotated_medication.xlsx" 
        %Config.processed_data_path)
    need_inspection.to_excel("%s/need_inspection_medication.xlsx" 
        %Config.processed_data_path)

    print('Finished all annotation, time passed: %is' %(time.time() - tic))
