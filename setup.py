from setuptools import setup
from setuptools import find_packages

long_description = """
hku_diabetes is a data analytics module tailored made for
analysing diabetes clinical data collected at Queen Mary Hospital
of the Hong Kong University.

Read the documentation at: 

hku_diabetes supports python 3.5-3.7
and is distributed under the MIT license.
"""

setup(name='hku_diabetes',
      version='0.0.2',
      description='HKU Diabetes Data Anlytics',
      long_description=long_description,
      author='Tim Lui',
      author_email='lui.thw@cantab.net',
      url='https://github.com/luithw/hku_diabetes',
      license='MIT',
      install_requires=['lxml>=4.2.5',
                        'matplotlib>=3.0.0',
                        'numpy>=1.15.2',
                        'pandas>=0.23.4',                        
                        'scipy>=1.1.0',
                        'six>=1.9.0',
                        'xlrd>=1.1.0'
                        ],
      extras_require={
          'tests': ['pytest'],
      },
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.5',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
      packages=find_packages())
