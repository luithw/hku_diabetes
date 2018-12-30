# -*- coding: utf-8 -*-
"""Script to call the package scrapper.
"""
import numpy as np
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

INSPECTION_CATEGORY = ['Long acting nitrate']
COMBINATION_WORDS = ['plus', 'hct']


def add_label(annotated, additional_label, position, key):
    if key == 'need_inspection':
        annotated[key].iat[position] = annotated[key].iat[position] or additional_label[key]
    else:
        annotated[key].iat[position] = annotated[key].iat[position] + ', ' + additional_label[key]

def match_trade_name(trade_name_tuple, medication, combination_drugs):
    tic = time.time()
    name = trade_name_tuple[1]['search_name']
    name = re.sub('[\/]+', ' ', str(name))
    name = re.sub('[^A-Za-z0-9\-]+', ' ', str(name))
    name = re.sub('[\-]+', '', str(name))
    name = name.lower()
    generic_name = trade_name_tuple[0]    
    trade_name_row = trade_name_tuple[1]
    category_name = trade_name_row['category_name']    
    trade_name = trade_name_row['trade_name']
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
                    break
        else:
            if name in med.split(" "):
                matched = True

        if matched:
            matched_rows.append(j)
            inspection = False
            if category_name in INSPECTION_CATEGORY:
                inspection = True
            for combination_drug in combination_drugs:
                if combination_drug.lower() in med:
                    for word in med:
                        if word not in COMBINATION_WORDS:
                            inspection = True
            need_inspection.append(inspection)
    annotated = medication.iloc[matched_rows]
    annotated['generic_name'] = generic_name
    annotated['category_name'] = category_name
    annotated['trade_name'] = trade_name
    annotated['need_inspection'] = need_inspection
    print('Finished %s, time passed: %is' %(name, (time.time() - tic)))
    return annotated


def auto_annotate(config=RunConfig):
    tic = time.time()    
    drug_names = get_all_trade_names(RunConfig)
    # Add the generic names to be pretended to be trade name so that it can also be searched.
    drug_names['generic_name'] = drug_names.index
    drug_names['search_name'] = [name.split(' ')[0] for name in drug_names['trade_name']]    
    generic_names_excel = pd.read_excel(
        "%s/Drug names.xlsx" % config.raw_data_path, sheet_name=None)    
    for sheet_name, generic_names in generic_names_excel.items():
        if sheet_name == "To notes":
            # Ignore the To notes sheet
            continue
        for category_name in generic_names:    
            for i, generic_name in enumerate(generic_names[category_name]):
                if isinstance(generic_name, str):
                    drug_names = drug_names.append({
                        'trade_name':generic_name,
                        'generic_name':generic_name,
                        'category_name': category_name,
                        'search_name': generic_name
                        }, ignore_index=True)
    drug_names.set_index('generic_name', drop=False, inplace = True)
    drug_names.drop_duplicates(['search_name', 'category_name'], inplace=True)

    assert not drug_names.loc['Cilnidipine'].empty

    medication = import_resource('Medication', config=config)
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

    # Generate the COMBINATION_DRUGS list
    combination_drugs = []
    for word in COMBINATION_WORDS:
        for i, name in enumerate(drug_names['trade_name']):
            name = re.sub('[^A-Za-z0-9]+', ' ', str(name))
            name = re.sub('[\-]+', '', str(name))     
            name = name.lower()   
            if word in name.split():
                combination_drugs.append(drug_names.iloc[i]['search_name'])                       

    if config is TestConfig:
        drug_names = drug_names[((drug_names['category_name'] == 'CCB') |
            (drug_names['category_name'] == 'Long acting nitrate'))]
        # drug_names = drug_names.iloc[10:30]

    if config is TestConfig:
        matched_rows = []
        for trade_name_tuple in drug_names.iterrows():
            matched_rows.append(match_trade_name(trade_name_tuple, unannotated, combination_drugs))
        matched_multiple_annotations = pd.concat(matched_rows)            
    else:
        with ProcessPoolExecutor() as executor:
            matched_rows_gen = executor.map(match_trade_name,
                                                drug_names.iterrows(),
                                                itertools.repeat(unannotated),
                                                itertools.repeat(combination_drugs))
        matched_multiple_annotations = pd.concat(list(matched_rows_gen))
        matched_multiple_annotations.drop_duplicates(inplace=True)
        
    # Need to check for each row to combine medication with multiple annotations
    annotated = matched_multiple_annotations.drop_duplicates('unique_id')
    for position, unique_id in enumerate(annotated['unique_id']):
        multiple_annotations = matched_multiple_annotations.loc[matched_multiple_annotations['unique_id']==unique_id]
        if len(multiple_annotations)>1:
            # Add on the multiple annotations to the first row.
            for i, (index, additional_label) in enumerate(multiple_annotations.iterrows()):
                annotated[annotated['unique_id']==unique_id]['generic_name']
                if i == 1:
                    continue    # No need to add the label of the first row
                add_label(annotated, additional_label, position, 'generic_name')
                add_label(annotated, additional_label, position, 'category_name')
                add_label(annotated, additional_label, position, 'trade_name')
                add_label(annotated, additional_label, position, 'need_inspection')

    unannotated_unique_id = set(unannotated['unique_id']) - set(annotated['unique_id'])
    unannotated = unannotated[unannotated['unique_id'].isin(unannotated_unique_id)]
    need_inspection = annotated[annotated['need_inspection']==1]
    annotated = annotated[annotated['need_inspection'] == False]

    annotated.to_excel("%s/annotated_medication.xlsx" 
        %config.processed_data_path)
    unannotated.to_excel("%s/unannotated_medication.xlsx" 
        %config.processed_data_path)
    need_inspection.to_excel("%s/need_inspection_medication.xlsx" 
        %config.processed_data_path)

    print('Finished all annotation, time passed: %is' %(time.time() - tic))
