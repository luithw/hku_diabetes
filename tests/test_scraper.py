# -*- coding: utf-8 -*-
"""Testing the hku_diabetes.scraper submodule"""
import os

import pytest

from hku_diabetes.config import TestConfig
from hku_diabetes.scraper import get_all_trade_names

def test_get_all_trade_names_from_drugoffice():
    """Test getting the drug names form drugoffice and saving CSV"""
    resource_csv = ("%s/drug_trade_names.csv" 
        % TestConfig.processed_data_path)
    if os.path.exists(resource_csv):
        os.remove(resource_csv)
    trade_names = get_all_trade_names(TestConfig)
    assert len(trade_names.columns) == 5
    assert os.path.exists(resource_csv)


def test_get_all_trade_names_from_file():
    """Test reading from processed file"""
    trade_names = get_all_trade_names(TestConfig)
    assert len(trade_names.columns) == 5


if __name__ == '__main__':
    pytest.main([__file__])
