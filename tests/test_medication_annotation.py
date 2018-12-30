# -*- coding: utf-8 -*-
"""Testing the hku_diabetes.medication_annotation submodule"""
import os
import pytest

from hku_diabetes.config import TestConfig
from hku_diabetes.medication_annotation import auto_annotate


def test_auto_annotation():
    """Test reading from raw file and saving CSV"""
    config = TestConfig
    annotated_file = os.path.join(config.processed_data_path, "annotated_medication.xlsx")
    if os.path.exists(annotated_file):
        os.remove(annotated_file)
    auto_annotate(config=config)
    assert os.path.exists(annotated_file)


if __name__ == '__main__':
    pytest.main([__file__])
