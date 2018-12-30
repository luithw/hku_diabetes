# -*- coding: utf-8 -*-
"""Main Script to use hku_diabetes."""
import sys

from hku_diabetes.analytics import Analyser
from hku_diabetes.config import TestConfig
from hku_diabetes.config import RunConfig
from hku_diabetes.importer import import_all
from hku_diabetes.medication_annotation import annotate_records
from hku_diabetes.plot import plot_all


def main():
    """Main sequence to analyse data."""
    if "run" in sys.argv:
        Config = RunConfig
    else:
        Config = TestConfig
    annotate_records(config=Config)        
    # analyser = Analyser(config=Config)
    # try:
    #     analyser.load()
    # except FileNotFoundError:
    #     data = import_all(config=Config)
    #     analyser.run(data)
    # plot_all(analyser)


if __name__ == '__main__':
    main()
