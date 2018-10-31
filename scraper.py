# -*- coding: utf-8 -*-
"""Script to call the package scrapper.
"""
import sys

from hku_diabetes.scraper import get_all_trade_names
from hku_diabetes.config import RunConfig
from hku_diabetes.config import TestConfig

if __name__ == '__main__':
    if 'run' in sys.argv:
        Config = RunConfig
    else:
        Config = TestConfig    
    trade_names = get_all_trade_names(Config)


