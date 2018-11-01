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
    generic_name = trade_name_tuple[0]    
    trade_name_row = trade_name_tuple[1]
    category_name = trade_name_row['category_name']    
    trade_name = trade_name_row['Name of Product']
    matched_rows=[]
    need_inspection=[]
    for j, (med, unique_id) in enumerate(zip(medication['Drug Name'], medication['unique_id'])):
        med = re.sub('[^A-Za-z0-9]+', ' ', str(med))
        med = re.sub('[\-]+', '', str(med))
        med = med.lower()
        # if re.search(name, med):
        if name in med.split(" "):            
            matched_rows.append(j)
            inspection = False
            for word in COMBINATION_WORDS:
                if word in med:
                    inspection = True
            need_inspection.append(inspection)
    annotated = medication.iloc[matched_rows]
    annotated['generic_name'] = generic_name
    annotated['category_name'] = category_name
    annotated['trade_name'] = trade_name    
    annotated['need_inspection'] = need_inspection
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
    trade_names = get_all_trade_names(RunConfig)
    # Add the generic names to be pretended to be trade name so that it can also be searched.
    trade_names['generic_name'] = trade_names.index
    generic_names = pd.DataFrame(trade_names.index.drop_duplicates())
    generic_names['Name of Product'] = generic_names['generic_name']
    generic_names.set_index('generic_name')
    trade_names = pd.concat([trade_names, generic_names])
    trade_names['first_word'] = [name.split(' ')[0] for name in trade_names['Name of Product']]
    trade_names.drop_duplicates(['first_word', 'category_name'], inplace=True)

    medication = import_resource('Medication', config=Config)

    trade_names = trade_names[trade_names['category_name'] == 'CCB']
    # trade_names = trade_names.loc['Amlodipine']        

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

    if 'run' not in sys.argv:
        unannotated = unannotated.iloc[:100]
        trade_names = trade_names.iloc[8: 48]

    annotated_list=[]
    need_inspection_list = []

    for batch in range(len(trade_names) // Config.annotation_batch_size):
        # Run match_trade_name in parelle and in batches. So that the already annotated
        # medication entries can be removed after every batch
        start = batch * Config.annotation_batch_size
        end = (batch+1) * Config.annotation_batch_size
        if end>len(trade_names):
            end = len(trade_names)
        trade_names_batch = trade_names.iloc[start:end]
        with ProcessPoolExecutor() as executor:
            matched_rows_generator = executor.map(match_trade_name,
                                                trade_names_batch.iterrows(),
                                                itertools.repeat(unannotated))
        annotated = pd.concat(list(matched_rows_generator))
        annotated.drop_duplicates(inplace=True)

        # for trade_name_tuple in trade_names_batch.iterrows():
        #     matched_rows = match_trade_name(trade_name_tuple, unannotated)

        need_inspection = annotated[annotated['need_inspection'] == 1]
        unannotated_unique_id = set(unannotated['unique_id']) - set(annotated['unique_id'])
        unannotated = unannotated[unannotated['unique_id'].isin(unannotated_unique_id)]
        annotated_list.append(annotated)
        need_inspection_list.append(need_inspection)

    annotated = pd.concat(annotated_list)
    need_inspection = pd.concat(need_inspection_list)
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
