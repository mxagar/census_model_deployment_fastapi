'''This module tests the functions in the module census_library.py
in the census_salary package. That package builds a model for a
census dataset which is able to predict the salary range of a
person given 14 features.

Altogether, 5 unit tests are defined using pytest:
- test_run_setup()
- test_run_processing()
- test_train_pipeline()
- test_load_pipeline()
- test_predict()

Those tests need to be carried out in the specified order,
because the returned objects are re-used as objects in the pytest namespace.

This test file is run in the CI of Github Actions.
Pylint: 8.70/10.

Note that the testing configuration fixtures
are located in `conftest.py`.
The content from `conftest.py` must be consistent with the
project configuration file `config.yaml`.

To install pytest:

>> pip install -U pytest

The script expects the proper dataset to be located in `./data`
or the folder specified in `config.yaml`.

Author: Mikel Sagardia
Date: 2023-01-17
'''

#import os
#from os import listdir
from os.path import isfile #, join
#import numpy as np
import pytest

# IMPORTANT: the file conftest.py defines the fixtures used in here
# and it contains the necessary imports!

### -- Tests -- ###

def test_run_setup(config_filename, run_setup):
    """Test run_setup() function."""
    df_train, df_test, config = run_setup(config_filename=config_filename, config=None)
    pytest.df_train_test = (df_train, df_test)
    pytest.config_dict = config

    # Data frames
    try:
        assert df_train.shape[0] > 0
        assert df_train.shape[1] > 0
        assert df_test.shape[0] > 0
        assert df_test.shape[1] > 0
    except AssertionError as err:
        print("TESTING run_setup(): ERROR - Data frame has no rows / columns.")
        raise err

    # Configuration dictionary
    try:
        assert isinstance(config, dict)
    except AssertionError as err:
        print("TESTING run_setup(): ERROR - config is not a dictionary.")
        raise err
    try:
        assert len(config.keys()) > 0
    except AssertionError as err:
        print("TESTING run_setup(): ERROR - config is empty.")
        raise err

def test_run_processing(run_processing,
                        categorical_features,
                        numerical_features,
                        num_transformed_features):
    """Test run_processing() function."""
    df = pytest.df_train_test[0] # df_train
    X_transformed, y_transformed, processing_parameters = run_processing(df,
                                                                         pytest.config_dict,
                                                                         training=True,
                                                                         processing_parameters=None)
    pytest.processing_parameters = processing_parameters

    # Processing parameters
    try:
        assert sorted(processing_parameters['numerical_features']) == sorted(numerical_features)
    except AssertionError as err:
        print("TESTING run_processing(): ERROR - numerical_features don't match in config and created processing pipeline.")
        raise err
    try:
        assert sorted(processing_parameters['categorical_features']) == sorted(categorical_features)
    except AssertionError as err:
        print("TESTING run_processing(): ERROR - categorical_features don't match in config and created processing pipeline.")
        raise err

    # Transformed dataset
    try:
        assert X_transformed.shape[0] > 0
        assert y_transformed.shape[0] > 0
    except AssertionError as err:
        print("TESTING run_processing(): ERROR - Transformed arrays are empty.")
        raise err
    try:
        assert X_transformed.shape[0] == y_transformed.shape[0]
    except AssertionError as err:
        print("TESTING run_processing(): ERROR - Transformed arrays have different number of entries.")
        raise err
    try:
        assert X_transformed.shape[1] == num_transformed_features
    except AssertionError as err:
        print("TESTING run_processing(): ERROR - Transformed X has an unexpected number of features.")
        raise err

def test_train_pipeline(config_filename,
                        train_pipeline,
                        model_artifact_path,
                        processing_artifact_path,
                        evaluation_artifact_path,
                        slicing_artifact_path):
    """Test train_pipeline function."""
    model, processing_parameters, _, test_scores = train_pipeline(config_filename=config_filename)
    pytest.model = model
    # Reset pytest plugin: processing_parameters
    # Reason: it might have changed from last test...
    pytest.processing_parameters = processing_parameters

    # Test artifacts are there
    try:
        assert isfile(model_artifact_path)
    except AssertionError as err:
        print("TESTING train_pipeline(): ERROR - Model artifact missing.")
        raise err
    try:
        assert isfile(processing_artifact_path)
    except AssertionError as err:
        print("TESTING train_pipeline(): ERROR - Processing pipeline missing.")
        raise err
    try:
        assert isfile(evaluation_artifact_path)
    except AssertionError as err:
        print("TESTING train_pipeline(): ERROR - Evaluation report missing.")
        raise err
    try:
        assert isfile(slicing_artifact_path)
    except AssertionError as err:
        print("TESTING train_pipeline(): ERROR - Data slicing report missing.")
        raise err

    # Test scores
    try:
        assert test_scores['precision'] >= 0.0
        assert test_scores['recall'] >= 0.0
        assert test_scores['fbeta'] >= 0.0
    except AssertionError as err:
        print("TESTING train_pipeline(): ERROR - Unexpected test scores.")
        raise err

def test_load_pipeline(config_filename,
                       load_pipeline):
    """Test load_pipeline function."""
    model, processing_parameters, config = load_pipeline(config_filename=config_filename)

    # Check types of artifacts
    try:
        assert isinstance(model, type(pytest.model))
    except AssertionError as err:
        print("TESTING load_pipeline(): ERROR - Unexpected type for model.")
        raise err
    try:
        assert isinstance(processing_parameters, type(pytest.processing_parameters))
    except AssertionError as err:
        print("TESTING load_pipeline(): ERROR - Unexpected type for processing_parameters.")
        raise err
    try:
        assert isinstance(config, type(pytest.config_dict))
    except AssertionError as err:
        print("TESTING load_pipeline(): ERROR - Unexpected type for config.")
        raise err

def test_predict(predict, target):
    """Test predict function."""
    X = pytest.df_train_test[1].drop(target, axis=1) # df_test
    pred_decoded = predict(X=X,
                           model=pytest.model,
                           processing_parameters=pytest.processing_parameters)
    # Check that all predicted classes are in df_test: ' <=50K', ' >50K'
    try:
        assert sorted(list(set(list(pred_decoded)))) == sorted(list(pytest.df_train_test[1][target].unique()))
    except AssertionError as err:
        print("TESTING predict(): ERROR - Unexpected classes predicted.")
        raise err
    # Check that array length matches
    try:
        assert len(pred_decoded) == pytest.df_train_test[1].shape[0]
    except AssertionError as err:
        print("TESTING predict(): ERROR - Unexpected length of arrays.")
        raise err
