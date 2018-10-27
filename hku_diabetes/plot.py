# -*- coding: utf-8 -*-
"""Data and results visualisation
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import warnings
from decimal import Decimal
from typing import Text

import matplotlib
import numpy as np

matplotlib.use('Agg')  # Need to execute this before importing plt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.dates import date2num
from scipy.stats import ttest_1samp
from scipy.interpolate import pchip_interpolate

from .analytics import Analyser
from .analytics import find_time_range

warnings.filterwarnings("ignore")
fig, ax1 = plt.subplots()


def plot_all(analyser: Analyser):
    """Plots all required PDFs.

    This calls the plot_one function and plot all the required PDFs
    specified by analyser.config.plot_modes.

    Args:
        analyser: An instance of the Analyser class with intermediate
            data available.

    Example:
        >>> from hku_diabetes.analytics import Analyser
        >>> from hku_diabetes.plot import plot_all
        >>> analyser = Analyser()
        >>> results = analyser.load()
        >>> plot_all(analyser)
    """
    for mode in analyser.config.plot_modes:
        plot_one(analyser, mode)


def plot_one(analyser: Analyser, mode: Text):
    """Plot one PDF according to required mode.

    This calls the corresponding private plot functions and plot the
    required PDF. A dot is printed to terminal every 10 figures to indicate
    something is happening.

    Args:
        analyser: An instance of the Analyser class with intermediate
            data available.
        mode: The plot mode required

    Example:
        >>> from hku_diabetes.analytics import Analyser
        >>> from hku_diabetes.plot import plot_one
        >>> analyser = Analyser()
        >>> results = analyser.load()
        >>> plot_one(analyser, 'raw')
    """
    tic = time.time()
    if not os.path.exists(analyser.config.plot_path):
        os.makedirs(analyser.config.plot_path)
    if mode == "regression_distributions":
        __plot_regression_distributions(analyser)
        _plot_ckd_distributions(analyser)
    else:
        plot_func = globals()["_plot_%s" % mode]
        pdf = PdfPages("%s/%s.pdf" % (analyser.config.plot_path, mode))
        for index in range(analyser.config.plot_samples):
            plot_func(analyser, index)
            pdf.savefig(fig)
            plt.clf()
            plt.cla()
            if index % 10 == 0:
                print(".", end="")  # Indicate that something is plotting
        pdf.close()
        print('\nFinished plotting all %s, time passed: %is' %
              (mode, time.time() - tic))


def __plot_regression_distributions(analyser: Analyser):
    pdf = PdfPages(
        "%s/regression_distributions.pdf" % (analyser.config.plot_path))
    for column in analyser.results['regression']:
        x = analyser.results['regression'][column]
        n = len(x)
        t, prob = ttest_1samp(x, analyser.config.t_test_mean[column])
        mean = x.mean()
        std = x.std()
        x = x[np.abs(x - mean) < 3 * std]
        plt.hist(x, bins=30)
        plt.title(column)
        plt.figtext(
            0.89,
            0.68, ('Mean:%0.2f\n'
                   'Std:%0.2f\n'
                   't statistics:%0.2f\n'
                   'p value:%.2E\n'
                   '1-sample t test mean:%0.2f\n'
                   'N:%i') % (mean, std, t, Decimal(prob),
                              analyser.config.t_test_mean[column], n),
            horizontalalignment='right')
        pdf.savefig(fig)
        plt.clf()
        plt.cla()
    pdf.close()
    print("Finished plotting regression_distributions")


def _plot_ckd_distributions(analyser: Analyser):
    pdf = PdfPages("%s/ckd_distributions.pdf" % analyser.config.plot_path)
    for column in analyser.results['ckd']:
        x = analyser.results['ckd'][column]
        n = len(x)
        mean = x.mean()
        std = x.std()
        x = x[np.abs(x - mean) < 3 * std]
        plt.hist(x, bins=30)
        plt.title("eGFR: %i" % int(column))
        plt.figtext(
            0.89,
            0.68, ('Mean:%0.2f\n'
                   'Std:%0.2f\n'
                   'N:%i') % (mean, std, n),
            horizontalalignment='right')
        pdf.savefig(fig)
        plt.clf()
        plt.cla()
    pdf.close()
    print("Finished plotting ckd_distributions")


def _plot_raw(analyser: Analyser, index: int):
    patient_id = analyser.intermediate[index]['patient_id']
    Creatinine = analyser.intermediate[index]['Creatinine']
    Hba1C = analyser.intermediate[index]['Hba1C']
    fig.suptitle(patient_id)
    ax1 = plt.gca()
    ax1.plot(
        Creatinine['Datetime'],
        Creatinine['eGFR'],
        '.-',
        color=analyser.config.eGFR_color)
    ax1.set_ylabel('eGFR', color=analyser.config.eGFR_color)
    ax1.set_xlabel('Time')
    ax1.tick_params(axis='y', labelcolor=analyser.config.eGFR_color)
    ax2 = ax1.twinx()
    ax2.plot(
        Hba1C['Datetime'],
        Hba1C['Value'],
        '.-',
        color=analyser.config.Hba1C_color)
    ax2.set_ylabel('Hba1C', color=analyser.config.Hba1C_color)
    ax2.tick_params(axis='y', labelcolor=analyser.config.Hba1C_color)
    fig.tight_layout()


def _plot_low_pass(analyser: Analyser, index: int):
    patient_id = analyser.intermediate[index]['patient_id']
    Creatinine = analyser.intermediate[index]['Creatinine']
    Hba1C = analyser.intermediate[index]['Hba1C']
    Creatinine_LP = analyser.intermediate[index]['Creatinine_LP']
    fig.suptitle(patient_id)
    ax1 = plt.gca()
    ax1.plot(
        Creatinine_LP.index,
        Creatinine_LP['eGFR'],
        '.-',
        color=analyser.config.eGFR_color)
    ax1.plot(
        Creatinine['Datetime'],
        Creatinine['eGFR'],
        '.--',
        color=analyser.config.eGFR_color)
    ax1.set_ylabel('eGFR', color=analyser.config.eGFR_color)
    ax1.set_xlabel('Time')
    ax1.tick_params(axis='y', labelcolor=analyser.config.eGFR_color)
    ax2 = ax1.twinx()
    ax2.plot(
        Hba1C['Datetime'],
        Hba1C['Value'],
        '.-',
        color=analyser.config.Hba1C_color)
    ax2.set_ylabel('Hba1C', color=analyser.config.Hba1C_color)
    ax2.tick_params(axis='y', labelcolor=analyser.config.Hba1C_color)
    fig.tight_layout()


def _plot_interpolated(analyser: Analyser, index: int):
    patient_id = analyser.intermediate[index]['patient_id']
    Creatinine = analyser.intermediate[index]['Creatinine']
    Hba1C = analyser.intermediate[index]['Hba1C']
    Creatinine_x = date2num(Creatinine['Datetime'])
    Hba1C_x = date2num(Hba1C['Datetime'])
    time_range = find_time_range(Creatinine_x, Hba1C_x, analyser.config)
    fig.suptitle(patient_id)
    ax1 = plt.gca()
    ax1.plot(
        time_range,
        pchip_interpolate(Creatinine_x, Creatinine['eGFR'], time_range),
        '-',
        color=analyser.config.eGFR_color)
    ax1.plot(
        Creatinine['Datetime'],
        Creatinine['eGFR'],
        '.--',
        color=analyser.config.eGFR_color)
    ax1.set_ylabel('eGFR', color=analyser.config.eGFR_color)
    ax1.set_xlabel('Time')
    ax1.tick_params(axis='y', labelcolor=analyser.config.eGFR_color)
    ax2 = ax1.twinx()
    ax2.plot(
        time_range,
        pchip_interpolate(Hba1C_x, Hba1C['Value'], time_range),
        '-',
        color=analyser.config.Hba1C_color)
    ax2.plot(
        Hba1C['Datetime'],
        Hba1C['Value'],
        '.--',
        color=analyser.config.Hba1C_color)
    ax2.set_ylabel('Hba1C', color=analyser.config.Hba1C_color)
    ax2.tick_params(axis='y', labelcolor=analyser.config.Hba1C_color)
    fig.tight_layout()


def _plot_cumulative(analyser: Analyser, index: int):
    patient_id = analyser.intermediate[index]['patient_id']
    Creatinine = analyser.intermediate[index]['Creatinine']
    Hba1C = analyser.intermediate[index]['Hba1C']
    Creatinine_x = date2num(Creatinine['Datetime'])
    Hba1C_x = date2num(Hba1C['Datetime'])
    time_range = find_time_range(Creatinine_x, Hba1C_x, analyser.config)
    fig.suptitle(patient_id)
    ax1 = plt.gca()
    ax1.plot(
        time_range,
        pchip_interpolate(Creatinine_x, Creatinine['eGFR'], time_range),
        '-',
        color=analyser.config.eGFR_color)
    ax1.set_ylabel('eGFR', color=analyser.config.eGFR_color)
    ax1.set_xlabel('Time')
    ax1.tick_params(axis='y', labelcolor=analyser.config.eGFR_color)
    ax2 = ax1.twinx()
    ax2.plot(
        time_range,
        np.cumsum(pchip_interpolate(Hba1C_x, Hba1C['Value'], time_range)),
        '-',
        color=analyser.config.Hba1C_color)
    ax2.set_ylabel('Hba1C', color=analyser.config.Hba1C_color)
    ax2.tick_params(axis='y', labelcolor=analyser.config.Hba1C_color)
    fig.tight_layout()


def _plot_regression(analyser: Analyser, index: int):
    patient_id = analyser.intermediate[index]['patient_id']
    x = analyser.intermediate[index]['Creatinine_LP']['eGFR']
    y = analyser.intermediate[index]['cumulative_Hba1C']
    regression = analyser.results['regression'].iloc[index]
    fig.suptitle(patient_id)
    plt.scatter(x, y)
    plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x))
    plt.figtext(0.15, 0.15, ('Slop:%0.2f\n'
                             'Intercept:%0.2f\n'
                             'R:%0.2f\n'
                             'P-value:%0.5f\n'
                             'Std-err:%0.2f') % tuple(regression))
    plt.ylabel('eGFR')
    plt.xlabel('Cumulative Hba1C')
