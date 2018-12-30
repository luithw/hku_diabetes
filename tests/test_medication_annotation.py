# -*- coding: utf-8 -*-
"""Testing the hku_diabetes.medication_annotation submodule"""
import os
import pytest

import numpy as np

from hku_diabetes.config import TestConfig
from hku_diabetes.medication_annotation import make_annotation_table
from hku_diabetes.medication_annotation import annotate_records


def test_make_annotation_table():
    config = TestConfig
    annotated_file = os.path.join(config.processed_data_path, "annotated_medication.xlsx")
    if os.path.exists(annotated_file):
        os.remove(annotated_file)
    make_annotation_table(config=config)
    assert os.path.exists(annotated_file)


def test_annotate_records():
    config = TestConfig
    medications = annotate_records(config=config)
    assert np.sum(medications['Metformin']) > 0


if __name__ == '__main__':
    pytest.main([__file__])
