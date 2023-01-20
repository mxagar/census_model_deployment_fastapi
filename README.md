# Census Model Deployment to Heroku Using FastAPI

In this project, a simple census dataset is used to create a model pipeline, train it and deploy it to Heroku using FastAPI. The dataset consists of 32,561 entries of different people, each with 14 features (age, education, etc.) and the model infers the salary range of an entry.

I forked the starter code for this project from a Udacity [exercise/demo repository](https://github.com/udacity/nd0821-c3-starter-code) and modified it to the present form.

The focus of this project doesn't lie so much on the data processing, but on the techniques and technologies used for model/pipeline deployment. A list of the most important MLOps methods and tools is the following:

- FastAPI
- Heroku
- Type hints and Pydantic
- Pytest
- Python packaging
- DVC

## Table of Contents

- [Census Model Deployment to Heroku Using FastAPI](#census-model-deployment-to-heroku-using-fastapi)
  - [Table of Contents](#table-of-contents)
  - [Dataset](#dataset)
  - [How to Use This Project](#how-to-use-this-project)
    - [Installing Dependencies for Custom Environments](#installing-dependencies-for-custom-environments)
  - [Notes on Theory](#notes-on-theory)
    - [Continuous Integration](#continuous-integration)
  - [Notes on the Implemented Analysis and Modeling](#notes-on-the-implemented-analysis-and-modeling)
    - [Summary of Contents](#summary-of-contents)
  - [Results and Conclusions](#results-and-conclusions)
  - [Next Steps, Improvements](#next-steps-improvements)
  - [References and Links](#references-and-links)
  - [Authorship](#authorship)

## Dataset



## How to Use This Project

The directory of the project consists of the following files:

```
.
├── Instructions.md           # Original challenge instructions
...
```

You can run the notebook at leas in two ways:

1. In a custom environment, e.g., locally or on a container. To that end, you can create a [conda](https://docs.conda.io/en/latest/) environment and install the [dependencies](#installing-dependencies-for-custom-environments) as explained below.
2. In Google Colab. For that, simply click on the following link:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mxagar/census_model_deployment_fastapi/blob/master/census_notebook.ipynb)


### Installing Dependencies for Custom Environments

If you'd like to control where the notebook runs, you need to create a custom environment and install the required dependencies. A quick recipe which sets everything up with [conda](https://docs.conda.io/en/latest/) is the following:

```bash
# Create environment with YAML, incl. packages
conda env create -f conda.yaml
conda activate census
```

List of most important dependencies:

- A
- B

## Notes on Theory

### Continuous Integration

- Github Actions; flake8, pytest, workflow (Python App)

`.github/workflows/python-app.yml`:

```yaml
# This workflow will install Python dependencies, run tests and lint with a single version of Python
# For more information see: https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python

name: Python application

on:
  push:
    branches: [ "master" ]
  pull_request:
    branches: [ "master" ]

permissions:
  contents: read

jobs:
  build:

    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3
    - name: Set up Python 3.8
      uses: actions/setup-python@v3
      with:
        python-version: "3.8.15"
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 pytest
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        # custom library
        pip install .
    - name: Lint with flake8
      run: |
        # stop the build if there are Python syntax errors or undefined names
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    - name: Test with pytest
      run: |
        pytest tests

```

## Notes on the Implemented Analysis and Modeling

### Summary of Contents

- [ ] A
- [ ] B

## Results and Conclusions

## Next Steps, Improvements

- [ ] Implement authentication.
- [ ] Create a simple front-end for the API (e.g., a form).

## References and Links

- A
- B
- C
- Link
- Link

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

If you find this repository useful, you're free to use it, but please link back to the original source.