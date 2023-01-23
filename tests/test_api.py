'''This module tests the API which serves the census-salary model.

Altogether, 5 unit tests are defined using pytest:
- test_load_pipeline()
- test_index()
- test_health()
- test_make_prediction()
- test_make_prediction_multiple()

Note that the testing configuration fixtures
are located in `conftest.py`. Also, the testing client is defined
there and passed as fixture to the tests.

Pylint: 9.33/10.

To install pytest:

>> pip install -U pytest

The script expects
- the configuration file `config.yaml` at the root level
    where we call the tests.
- the inference artifacts located in `./exported_artifacts`
    or the folder specified in `config.yaml`.

Author: Mikel Sagardia
Date: 2023-01-23
'''
import json
import pytest
from fastapi.testclient import TestClient

def test_load_pipeline(config_filename,
                       load_pipeline):
    """Test load_pipeline function.
    This function is checked when the library is tested,
    but we include a test here, since app.py uses this interface.
    Also, we need some of the pipeline objects fro later tests."""

    model, processing_parameters, config = load_pipeline(config_filename=config_filename)
    # Store artifacts in pytest namespace
    pytest.model = model
    pytest.processing_parameters = processing_parameters
    pytest.config_dict = config
    # Check types of artifacts
    try:
        assert model
        assert isinstance(processing_parameters, dict)
        assert isinstance(config, dict)
    except AssertionError as err:
        print("TESTING API load_pipeline(): ERROR - Unexpected type for model pipeline object(s).")
        raise err

def test_index(client: TestClient,
               index_string: str) -> None:
    """Test index() endpoint: status & content."""

    r = client.get("/")
    try:
        assert r.status_code == 200
    except AssertionError as err:
        print("TESTING API index(): ERROR - status code not 200.")
        raise err
    try:
        assert r.text == index_string
    except AssertionError as err:
        print("TESTING API index(): ERROR - website content not the expected.")
        raise err

def test_health(client: TestClient,
                model_lib_version_string: str,
                api_version_string: str,
                api_name_string: str) -> None:
    """Test health() endpoint: status & content."""

    r = client.get("/health")
    try:
        assert r.status_code == 200
    except AssertionError as err:
        print("TESTING API health(): ERROR - status code not 200.")
        raise err
    try:
        assert r.json()['name'] == api_name_string
        assert r.json()['api_version'] == api_version_string
        assert r.json()['model_lib_version'] == model_lib_version_string
    except AssertionError as err:
        print("TESTING API health(): ERROR - health JSON content not the expected.")
        raise err

def test_make_prediction(client: TestClient,
                         model_lib_version_string: str,
                         test_data_single: dict) -> None:
    """Test make_prediction() endpoint: status & content.
    This test takes by default a single data row as JSON/dict."""

    data = json.dumps(test_data_single)
    #r = client.post("/predict", data=data)
    r = client.post("/predict", content=data)
    try:
        assert r.status_code == 200
    except AssertionError as err:
        print("TESTING API predict(): ERROR - status code not 200.")
        raise err
    try:
        assert r.json()["model_lib_version"] == model_lib_version_string
        assert r.json()["timestamp"]
        target_values = list(pytest.processing_parameters['target_processor'].classes_) # <=50K', '>50K'
        for res in r.json()["predictions"]:
            assert res in target_values
    except AssertionError as err:
        print("TESTING API predict(): ERROR - predict return JSON content not the expected.")
        raise err

def test_make_prediction_multiple(client: TestClient,
                                  model_lib_version_string: str,
                                  test_data_multiple: dict) -> None:
    """Test make_prediction() endpoint: status & content.
    This test takes multiple data rows as JSON/dict."""

    # Call test_make_prediction but with multiple inputs
    test_make_prediction(client,
                         model_lib_version_string,
                         test_data_multiple)
