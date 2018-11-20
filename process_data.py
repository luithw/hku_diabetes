# -*- coding: utf-8 -*-
"""Main Script to use hku_diabetes."""
from hku_diabetes.config import DefaultConfig
from hku_diabetes.importer import import_resource


class MyConfig(DefaultConfig):
    raw_data_path = "../qmh_data"
    processed_data_path = "../qmh_data/processed"
    required_resources = ['BP', 
                        'Creatinine', 
                        'Demographics', 
                        'HDL', 
                        'Hb', 
                        'Hba1c', 
                        'Heart_Rate', 
                        'LDL', 
                        'Platelet', 
                        'TG', 
                        'Total_cholesterol']


if __name__ == '__main__':
    for resource_name in MyConfig.required_resources:
        if resource_name == 'Demographic':
            continue  # need the separate routine below to import demographic data
        tic = time.time()
        resource_key = resource_name
        data[resource_key] = import_resource(resource_name, config=MyConfig)
        print('Finished importing %s, time passed: %is' % (resource_name,
                                                           time.time() - tic))

