'''Testing configuration module for Pytest.
This file is read by pytest and the fixtures
defined in it are used in all the tested files.

Note that some variables are extracted from the
configuration YAML config.yaml; to that end,
the configuration dictionary must be loaded in the
first test.

Author: Mikel Sagardia
Date: 2023-01-17
'''

import pytest
import logging
import census_salary as cs

# Fixtures of the census_salary package functions.
# Fixtures are predefined variables passed to test functions;
# in this case, most variables are functions/classes to be tested.

## Parameters

@pytest.fixture
def config_filename():
    '''Configuration filename.'''
    return "config.yaml"

@pytest.fixture
def dataset_path():
    '''Dataset path for training.'''
    #return "./data/census.csv"
    return pytest.config_dict["data_path"]

@pytest.fixture
def target():
    '''Response/Target variable name.'''
    #return "salary"
    return pytest.config_dict["target"]

@pytest.fixture
def model_artifact_path():
    '''Path where model is stored.'''
    #return "./exported_artifacts/model.pickle"
    return pytest.config_dict["model_artifact"]

@pytest.fixture
def processing_artifact_path():
    '''Path where the data processing parameters and pipeline are stored.'''
    #return "./exported_artifacts/processing_parameters.pickle"
    return pytest.config_dict["processing_artifact"]

@pytest.fixture
def evaluation_artifact_path():
    '''Path where evaluation report is stored.'''
    #return "./exported_artifacts/evaluation_report.txt"
    return pytest.config_dict["evaluation_artifact"]

@pytest.fixture
def slicing_artifact_path():
    '''Path where the slicing report is stored.'''
    #return "./exported_artifacts/slice_output.txt"
    return pytest.config_dict["slicing_artifact"]

@pytest.fixture
def categorical_features():
    '''List of categorical features.'''
    return pytest.config_dict["features"]["categorical"]

@pytest.fixture
def numerical_features():
    '''List of numerical features.'''
    return pytest.config_dict["features"]["numerical"]

@pytest.fixture
def num_transformed_features():
    '''Number of final features, after the transformation/processing.'''
    #return 14
    return 108

## Functions

@pytest.fixture
def run_setup():
    '''run_setup() function from census_library.'''
    return cs.run_setup

@pytest.fixture
def run_processing():
    '''run_processing() function from census_library.'''
    return cs.run_processing

@pytest.fixture
def train_pipeline():
    '''train_pipeline() function from census_library.'''
    return cs.train_pipeline

@pytest.fixture
def load_pipeline():
    '''load_pipeline() function from census_library.'''
    return cs.load_pipeline

@pytest.fixture
def predict():
    '''predict() function from census_library.'''
    return cs.predict

## Variable plug-ins

def config_dict_plugin():
    '''Initialize pytest project config container as None:
    pytest.config_dict: dict'''
    return None

def df_plugin():
    '''Initialize pytest dataset container df_train_test as None:
    df_train_test = (df_train, df_test)'''
    return None

def processing_parameters_plugin():
    '''Initialize pytest processing_parameters container as None:
    pytest.processing_parameters: dict'''
    return None

def model_plugin():
    '''Initialize pytest model container as None:
    pytest.model: RandomForestClassifier'''
    return None

def pytest_configure():
    '''Create objects in namespace:
    - `pytest.df_train_test`
    - `pytest.processing_parameters`
    - `pytest.model`
    - `pytest.config_dict`
    '''
    pytest.config_dict = config_dict_plugin()
    pytest.df_train_test = df_plugin() # we can access & modify pytest.df in test functions!
    pytest.processing_parameters = processing_parameters_plugin()
    pytest.model = model_plugin()

def pytest_addoption(parser):
    """Add a command line option to disable logger."""
    parser.addoption(
        "--log-disable", action="append", default=[], help="disable specific loggers"
    )

def pytest_configure(config):
    """Disable the loggers."""
    for name in config.getoption("--log-disable", default=[]):
        logger = logging.getLogger(name)
        logger.propagate = False
