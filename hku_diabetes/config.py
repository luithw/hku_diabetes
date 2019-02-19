# -*- coding= utf-8 -*-
"""Configuration classes controlling module behaviours.
"""


class DefaultConfig:
    """Default configuration used by all module classes and functions.

    This is the default configuration class defining all default parameters.
    All classes and functions of the module default to this class whenever they
    accept a config keyword parameter. Extend from this class to create your own
    configuration class.
    """

    # Analytics
    ckd_thresholds = (15, 30, 45, 60, 90)
    """The eGFR threshold values of CKD classifications."""
    min_analysis_samples = 5
    """The minimum number of Creatinine and Hb1aC measurements required for each patient.
        Patient would be skip if the number of measurements is less than this.
    """
    eGFR_low_pass = "90d"
    """The period of eGFR low pass filter. All measurements within the same
        period are averaged to one measurement.
    """
    test_samples = 10
    """The number of samples analysed by the analytics module."""

    filter_by_starting_eGFR = True
    """All patients with starting eGFR less than starting_eGFR are removed"""

    starting_eGFR = 60
    """The starting eGFR of analysis"""

    max_continuous_prescription_gap = 60
    """The maximum number of days between prescriptions for them to be considered continuous."""

    diagnosis_code = {
        'CKD I-V': [585],
        'CKD I': [585.1],
        'CKD II': [585.2],
        'CKD IIII': [585.3],
        'CKD IV': [585.4],
        'CKD V': [585.5],
        'ESRF': [585.6],
        'CKD NOS': [585.9],
        'fracture': [800, 801, 802, 803, 804, 805, 806, 807, 808, 809,
                     810, 811, 812, 813, 814, 815, 816, 817, 818, 819,
                     820, 821, 822, 823, 824, 825, 826, 827, 828, 829,
                     733.1],
        'MI': [410, 411.1, 411.89],
        'stroke': [433, 434, 435, 436, 430, 431, 432, 852, 853],
        'ischemic stroke': [433, 434, 435, 436],
        'hemorrhagic stroke': [430, 431, 432, 852, 853],
        'HF': [398.91, 428],
        'HT': [401, 405],
        'IHD': [410, 411, 412, 414.0, 414.8, 414.9],
        'PVD': [440.2],
        'AF': [427.3, 250.1],
        'pancreatitis': [577.0, 577.1],
        'DM': [249, 250],
        'DKA': [250.1],
        'UTI': [599.0],
        'dialysis': ['V45.1', 'V56.0', 'V56.8', 39.95, 54.98]
    }
    """The ICD9 diagnosis code of different diagnosis. https://en.wikipedia.org/wiki/List_of_ICD-9_codes_390–459"""

    procedure_code = {
        'amputation': [84.0, 84.91],
        'dialysis': [39.95, 54.98]
    }

    # Medication annotation
    annotation_batch_size = 64
    """The number of trade names to be matched per batch of paralle process"""

    # Importer
    data_file_extensions = ("LIS.xls", "DRG.xls", "DX.xls", "PX.xls", "OP.xls", "DOD.xlsx")
    """The file name ending and extension of data files that has actual data."""
    required_resources = [
        "Creatinine", "Hba1C", "Medication", "Diagnosis", "Procedure", "HDL",
        "LDL", "Demographic"
    ]
    must_have_resources = ["Creatinine", "Hba1C", "Demographic"]
    """The resources that each patient must have."""

    # Plots
    plot_modes = [
        "regression_distributions", "run", "cumulative", "low_pass",
        "interpolated", "raw"
    ]
    """The type of plots to be created."""

    interpolation_samples = 100
    """The number of samples to be interpolated in interpolated plots."""
    plot_samples = 1000
    """The number of patients to be plotted for each plot mode."""
    eGFR_color = "tab:red"
    """The colour of eGFR axis and line."""
    Hba1C_color = "tab:blue"
    """The colour of Hb1aC axis and line."""

    # t-test
    t_test_mean = {
        'slope': 0,
        'intercept': 100,
        'rvalue': 0,
        'pvalue': 0.5,
        'stderr': 0
    }
    """The Gaussian mean of the null hypothesis of 1 sample t-test."""

    # Paths
    raw_data_path = "raw_data"
    """The path for importing raw data."""
    processed_data_path = "processed_data"
    """The path for storing processed data."""
    plot_path = "output/plots"
    """The path for exporting plot PDFs."""
    results_path = "output/results"
    """The path for exporting results CSV and subject_data pickles."""


class RunConfig(DefaultConfig):
    """Configuration used for running the full data analytic.
    """

    # plots
    # plot_modes = ["regression_distributions", "run"]
    """The type of plots to be created.

    As it takes a lot of time to generate all the raw plots, only plot the
    run distributions.
    """


class TestConfig(DefaultConfig):
    """Configuration used for development and testing.
    """

    # Analytics
    test_samples = 100
    """The number of samples analysed by the analytics module.

    This allows faster testing time as there is no need to analyse all the data.
    """

    # Medication annotation
    annotation_batch_size = 8
    """The number of trade names to be matched per batch of paralle process"""

    # Plots
    plot_samples = 5
    """The number of patients to be plotted for each plot mode.

    Speed up testing time by plotting less patients.
    """

    # Paths
    processed_data_path = "tmp/processed_data"
    """The path for storing processed data."""
    plot_path = "tmp/output/plots"
    """The path for exporting plot PDFs."""
    results_path = "tmp/output/results"
    """The path for exporting results CSV and subject_data pickles."""
