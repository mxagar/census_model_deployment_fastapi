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
    - [Model Card](#model-card)
    - [Continuous Integration](#continuous-integration)
    - [Data and Model Versioning](#data-and-model-versioning)
    - [FastAPI Application](#fastapi-application)
    - [Deployment to Heroku](#deployment-to-heroku)
    - [Docker Container](#docker-container)
    - [Deployment to AWS EC2](#deployment-to-aws-ec2)
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

### Model Card

The model card, which contains a summary of the model properties, is in the co-located file [`ModelCard.md`](ModelCard.md).

In there, the following topics are covered:

- Dataset
- Model: training and final performance
- Evaluation metrics, data slicing
- Bias and ethical issues (morel fairness)

### Continuous Integration

The project uses Github Actions from Github to implement Continuous Integration. In summary, every time we push a new version to Github (or perform a pull request), the package is checked in a custom container running on Github servers:

- Dependencies are installed, including the package `census_salary`.
- Linting is performed on the code using `flake8`.
- `pytest` is run on the project.

To create the CI Github Action, we simply choose the **Python Application** pre-configured workflow on the Github web interface:

```
Github repository > Actions > New Workflow > Python application: configure
```

The `python-app.yml` workflow YAML file needs to be modified, as well as other files in the repository. Here's a list of things to pay attention to:

- Since we have a package, we need to install it in the container (see the YAML below).
- Check which Python version we are using.
- Add missing/required packages to `requirements.txt`: flake8, pytest, pyyaml, setuptools, etc.
- Add a `.flake8` file to the repository to configure `flake8`, i.e., which folders/files we want to exclude form linting.
- If the tests are in another folder than `test`, modify the `pytest command`.
- If the tests trigger logging, create the logging file in the container (see YAML).
- Make sure all commands defined in the YAML work on our local environment.

Every time we push, the the build job will be run on the Github servers (see YAML); we can check the logs by clicking on the Github web interface:

```
Github repository > Actions > All workflows: Python application >
    Select desired run > Jobs: build
```

The final `python-app.yml` YAML is stored and committed to `.github/workflows`, and its content is the following:

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
    - name: Set up Python 3.10
      uses: actions/setup-python@v3
      with:
        python-version: "3.10"
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install flake8 pytest
        if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
        pip install .
    - name: Lint with flake8
      run: |
        # stop the build if there are Python syntax errors or undefined names
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        # exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    - name: Test with pytest
      run: |
        # Create a logs file
        mkdir logs
        touch logs/census_pipeline.log
        # Test
        pytest tests
```

### Data and Model Versioning

The current version of this project does not use any data or model versioning. The dataset is included in the repository and the inference artifacts are not uploaded anywhere. However, in a professional setting and with increasingly larger artifacts, that is not viable. Two easy options to apply versioning to any kind of artifacts are the following: 

1. **DVC**: We can define remote storage folders (e.g., in GDrive, S3, etc.) and upload large datasets to them while ignoring them in git. DVC requires a buildpack for Heroku, as explained in the [`starter`](starter) folder. Check my [notes on DVC](https://github.com/mxagar/mlops_udacity/blob/main/03_Deployment/MLOpsND_Deployment.md#3-data-and-model-versioning) for more information.

2. **Weights and Biases**: we can create model and dataset artifacts and upload/download them from the W&B servers. An example of mine where W&B functionalities are used extensible is the following: [music_genre_classification](https://github.com/mxagar/music_genre_classification).



### FastAPI Application

### Deployment to Heroku

### Docker Container

### Deployment to AWS EC2

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