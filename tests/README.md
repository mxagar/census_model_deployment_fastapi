# Census Model Tests

This folder contains two suites of tests implemented with Pytest:

- [`test_census_library.py`](./test_census_library.py): the most important functions in the module `census_library.py` from the `census_salary` package are tested. That package builds a model for a census dataset which is able to predict the salary range of a person given 14 features. Altogether, 5 unit tests are defined:
  - `test_run_setup()`
  - `test_run_processing()`
  - `test_train_pipeline()`
  - `test_load_pipeline()`
  - `test_predict()`
- [`test_api.py`](./test_api.py): this module tests the API which serves the census-salary model. Altogether, 5 unit tests are defined:
  - `test_load_pipeline()`
  - `test_index()`
  - `test_health()`
  - `test_make_prediction()`
  - `test_make_prediction_multiple()`

The file [`conftest.py`](./conftest.py) configures the tests with fixtures. Note that some variables are extracted from the configuration YAML `config.yaml`; to that end, the configuration dictionary must be loaded in the first test.

To run the tests, run in the root folder:

```python
pytest tests
```

Note that the Github Action [python-app.yml](.github/../../.github/workflows/python-app.yml) automatically runs the tests when a new version is pushed to the Github remote repository, i.e., continuous integration (CI) is achieved.
