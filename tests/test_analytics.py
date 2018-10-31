# -*- coding: utf-8 -*-
"""Testing the hku_diabetes.analytics submodule"""
import os
import shutil

import pytest

from hku_diabetes.analytics import Analyser
from hku_diabetes.config import TestConfig
from hku_diabetes.importer import import_all
from hku_diabetes.plot import plot_all


def test_file_not_found():
    """Test if FileNotFoundError is raised."""
    if os.path.exists(TestConfig.results_path):
        shutil.rmtree(TestConfig.results_path)
    analyser = Analyser(config=TestConfig)
    with pytest.raises(FileNotFoundError):
        results = analyser.load()
        assert not results['regression'].empty
        assert not results['ckd'].empty


def test_analytics():
    """Test all the results and plots are generated."""
    if os.path.exists(TestConfig.results_path):
        shutil.rmtree(TestConfig.results_path)
    data = import_all(config=TestConfig)
    analyser = Analyser(config=TestConfig)
    results = analyser.run(data)
    assert not results['regression'].empty
    assert not results['ckd'].empty
    assert os.path.exists("%s/regression.csv" % TestConfig.results_path)
    assert os.path.exists("%s/ckd.csv" % TestConfig.results_path)


def test_reload_and_plot():
    """Test the intermediates can be loaded."""
    if os.path.exists(TestConfig.plot_path):
        shutil.rmtree(TestConfig.plot_path)
    analyser = Analyser(config=TestConfig)
    results = analyser.load()
    assert not results['regression'].empty
    assert not results['ckd'].empty
    plot_all(analyser)
    for mode in TestConfig.plot_modes:
        assert os.path.exists("%s/%s.pdf" % (TestConfig.plot_path, mode))
    assert os.path.exists("%s/ckd_distributions.pdf" % TestConfig.plot_path)


if __name__ == '__main__':
    pytest.main([__file__])
