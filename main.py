# -*- coding: utf-8 -*-
"""Main Script to use hku_diabetes."""
import sys

from hku_diabetes import analytics
from hku_diabetes.config import TestConfig
from hku_diabetes.config import RunConfig
from hku_diabetes import importer
from hku_diabetes.medication_annotation import annotate_records
from hku_diabetes.plot import plot_all


if __name__ == '__main__':
    if "regression" in sys.argv:
        config = RunConfig
    else:
        config = TestConfig
    annotate_records(config=config)
    analyser = analytics.Analyser(config=config)
    try:
        analyser.load()
    except FileNotFoundError:
        data = importer.import_all(config=config)
        analyser.regression(data)
    plot_all(analyser)
