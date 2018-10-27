HKU Diabetes Data Analytics
===========================
`hku_diabetes` is a data analytics module tailored made for analysing diabetes clinical data collected at Queen Mary Hospital of the Hong Kong University.

`hku_diabetes` is compatible with: __Python 3.6-3.7__.

## Installation
If you don't have Python installed already, you could download __Python 3.7__ directly [here](https://www.python.org/downloads/release/python-370/).

You can obtain `hku_diabetes` either by download a zip by clicking the 'Clone or download' button from this page, or simply put the following in your terminal (cmd) if you have git.

```sh
git clone https://github.com/luithw/hku_diabetes.git
```

You then need to install all the required packages by `pip`.

```sh
pip install -r requirements.txt
```

Alternatively, if you just want to use the package, you can install the package itself directly with `pip`.

```sh
pip install hku_diabetes
```

## Perform data analytics
The package expects to find the data under a directory (folder) named `raw_data` in the working directory. Please contact Emmanuel Wong at the Department of Medicine to obtain the data.

Once you have put the data in the working directory. First test everything is OK by executing the following command in your terminal. It execute all the functions with test configuration on a small number of data to make sure there is no error.

```sh
python main.py
```

The first time it is executed, it loads the raw data in inefficient html format and saves them as CSV in the `processed_data` directory, so that it loads faster the next time. 

If the above command is executed without error. Simply add `run` after the previous command. It will then execute on the entire dataset.

```sh
python main.py run
```

## Use the package in your script
First import the diabetes data using the `import_all` function.

```python
from hku_diabetes.importer import import_all

data = import_all()
```

The core analytics logic is available in the `Analyser` class.

```python
from hku_diabetes.analytics import Analyser

analyser = Analyser()
```

The `run` method of `Analyser` takes the data and compute the results. It also saves the results in the `output/plot` directory automatically.

```python
results = analyser.run(data)
```

You can plot all the data and results using the `plot_all` function.

```python
from hku_diabetes.plot import plot_all

plot_all(analyser)
```

To load the previous analysed results without running the whole analytics again, simply call the `load` method of the `Analyser`.

```python
results = analyser.load()
```

You can change the configuration of `inport_all` and `Analyser` by extending the `DefaultConfig` class.

```python
from hku_diabetes.config import DefaultConfig

class MyConfig(DefaultConfig):
    required_resources = ["Creatinine", "Hba1C", "Medication"]	
    ckd_thresholds = (15, 30, 45, 60, 90)
    min_analysis_samples = 10
    eGFR_low_pass = "90d" 

data = import_all(config=MyConfig)
results = analyser.run(data, config=MyConfig)
```

For a full list of available configuration options and functions . Please see the [documentation](https://hku-diabetes.readthedocs.io/en/latest/hku_diabetes.html).

## Support
For any quires, you can send me an email to luithw@hku.hk.

__Enjoy!__
