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
from typing import Generator
from fastapi.testclient import TestClient

#import logging
import census_salary as cs
from census_salary import __version__ as model_lib_version
from api import __version__ as api_version
from api.app import API_PROJECT_NAME, INDEX_BODY

# FastAPI app
from api.app import app

# Fixtures of the census_salary package functions.
# Fixtures are predefined variables passed to test functions;
# in this case, most variables are functions/classes to be tested.

## -- Library Parameters

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

## Library Functions

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

## -- Variable plug-ins

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

### -- API Variables and Functions

# This is the test client, which is passed as a fixture.
# An easier alternative would be to define the client in the test_ file:
#   from fastapi.testclient import TestClient
#   from api.app import app
#   client = TestClient(app)
@pytest.fixture()
def client() -> Generator:
    """FastAPI test client."""
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}

@pytest.fixture
def model_lib_version_string():
    """Library version."""
    return model_lib_version

@pytest.fixture
def api_version_string():
    """API version."""
    return api_version

@pytest.fixture
def api_name_string():
    """API version."""
    return API_PROJECT_NAME

@pytest.fixture
def index_string():
    """API version."""
    return INDEX_BODY

@pytest.fixture
def test_data_single() -> dict:
    """Test data JSON/dict. Single row."""
    # This dictionary is the same as in the census library: core.py
    # Note that the field names and values are the already processed ones:
    # no blank spaces, _ instead or -
    d = {
        "inputs": [
            {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        ]
    }
    return d

@pytest.fixture
def test_data_multiple() -> dict:
    """Test data JSON/dict. Multiple rows (two)."""
    # Note that the field names and values are the already processed ones:
    # no blank spaces, _ instead or -
    d = {
        "inputs": [
            {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital_gain": 2174,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            },
            {
                "age": 50,
                "workclass": "Self-emp-not-inc",
                "fnlgt": 83311,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 13,
                "native_country": "United-States"
            }
        ]
    }
    return d
