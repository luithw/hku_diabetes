
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

from hku_diabetes.analytics import Analyser
from hku_diabetes.analytics import find_time_range
from hku_diabetes.config import TestConfig
from hku_diabetes import analytics
from hku_diabetes.importer import import_all
fig, ax1 = plt.subplots()

data = import_all(config=TestConfig)
analytics.evaluate_eGFR(data)
patient_id = 802
intermediate = analytics.analyse_subject(data, patient_id)
analyser = analytics.Analyser()

patient_id = intermediate['patient_id']
Creatinine = intermediate['Creatinine']
Hba1C = intermediate['Hba1C']
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

pdf = PdfPages("playground.pdf")
pdf.savefig(fig)
plt.clf()
plt.cla()
pdf.close()