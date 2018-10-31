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

    # Importer
    data_file_extensions = ("LIS.xls", "DRG.xls", "DX.xls", "PX.xls")
    """The file name ending and extension of data files that has actual data."""
    required_resources = [
        "Creatinine", "Hba1C", "Medication", "Diagnosis", "Procedure", "HDL",
        "LDL"
    ]
    """The resources to be loaded by importer."""

    # Plots
    plot_modes = [
        "regression_distributions", "regression", "cumulative", "low_pass",
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
    """The path for exporting results CSV and intermediate pickles."""


class RunConfig(DefaultConfig):
    """Configuration used for running the full data analytic.
    """

    # Importer
    required_resources = ["Creatinine", "Hba1C"]
    """The resources to be loaded by importer.

    As the current analytics only support Creatinine and Hba1C, there is no
    need to load the other resources.
    """

    # plots
    plot_modes = ["regression_distributions"]
    """The type of plots to be created.

    As it takes a lot of time to generate all the raw plots, only plot the
    regression distributions.
    """


class TestConfig(DefaultConfig):
    """Configuration used for development and testing.
    """

    # Analytics
    test_samples = 10
    """The number of samples analysed by the analytics module.

    This allows faster testing time as there is no need to analyse all the data.
    """

    # Importer
    required_resources = ["Creatinine", "Hba1C"]
    """The resources to be loaded by importer.

    As the current analytics only support Creatinine and Hba1C, there is no
    need to load the other resources.
    """

    # Plots
    plot_samples = 5
    """The number of patients to be plotted for each plot mode.

    Speed up testing time by plotting less patients.
    """

    # Paths
    plot_path = "test_output/plots"
    """The path for exporting plot PDFs.

    Save the plots to an alternate directory so that it doesn't delete
    or overwrite the main output directory.
    """
    results_path = "test_output/results"
    """The path for exporting results CSV and intermediate pickles.

    Save the results to an alternate directory so that it doesn't delete
    or overwrite the main output directory.
    """
