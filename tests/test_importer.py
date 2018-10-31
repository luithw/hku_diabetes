# -*- coding: utf-8 -*-
"""Testing the hku_diabetes.importer submodule"""
import os

import pytest

from hku_diabetes.config import TestConfig
from hku_diabetes.importer import import_resource

RESOURCE_SHAPES = {
    'Creatinine': (314898, 10),
    'Hba1C': (142018, 10),
    'Medication': (4434498, 21),
    'Diagnosis': (269655, 3),
    'Procedure': (104702, 4),
    'HDL': (104021, 10),
    'LDL': (103547, 10),
    'Demographic': (7307, 20)
}
TEST_RESOURCE = "HDL"


def test_import_raw():
    """Test reading from raw file and saving CSV"""
    resource_csv = "%s/%s.csv" % \
        (TestConfig.processed_data_path, TEST_RESOURCE)
    if os.path.exists(resource_csv):
        os.remove(resource_csv)
    resource = import_resource(TEST_RESOURCE, config=TestConfig)
    assert resource.shape == RESOURCE_SHAPES[TEST_RESOURCE]
    assert os.path.exists(
        "%s/%s.csv" % (TestConfig.processed_data_path, TEST_RESOURCE))


if __name__ == '__main__':
    pytest.main([__file__])
