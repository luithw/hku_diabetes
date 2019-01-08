# -*- coding: utf-8 -*-
"""Script to call the package scrapper.
"""
import pandas as pd
import sys

from hku_diabetes.scraper import get_all_trade_names
from hku_diabetes.config import RunConfig
from hku_diabetes.config import TestConfig

if __name__ == '__main__':
    if 'regression' in sys.argv:
        config = RunConfig
    else:
        config = TestConfig
    trade_names = get_all_trade_names(config)