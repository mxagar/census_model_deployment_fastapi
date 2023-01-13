# Environment Set up

Working in a command line environment is recommended for ease of use with git and dvc. If on Windows, WSL1 or 2 is recommended.

* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.8" scikit-learn pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.

## Repositories

* Create a directory for the project and initialize git.
    * As you work on the code, continually commit changes. Trained models you want to use in production must be committed to GitHub.
* Connect your local git repo to GitHub.
* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.

# Data

* Download census.csv and commit it to dvc.
* This data is messy, try to open it in pandas and see what you get.
* To clean it, use your favorite text editor to remove all spaces.

# Model

* Using the starter code, write a machine learning model that trains on the clean data and saves the model. Complete any function that has been started.
* Write unit tests for at least 3 functions in the model code.
* Write a function that outputs the performance of the model on slices of the data.
    * Suggestion: for simplicity, the function can just output the performance on slices of just the categorical features.
* Write a model card using the provided template.

# API Creation

*  Create a RESTful API using FastAPI this must implement:
    * GET on the root giving a welcome message.
    * POST that does model inference.
    * Type hinting must be used.
    * Use a Pydantic model to ingest the body from POST. This model should contain an example.
   	 * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

# API Deployment

* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
    * Enable automatic deployments that only deploy if your continuous integration passes.
    * Hint: think about how paths will differ in your local environment vs. on Heroku.
    * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Write a script that uses the requests module to do one POST on your live API.

# Project Requirements

Udacity [rubric](https://review.udacity.com/#!/rubrics/5033/view).

Look at the starter code, many function signatures are provided.

- Git
  - Set up Github Actions: pytest and flake8 on push to main/master
    - At least 6 tests, all pass; flake8 no errors
    - Take screenshot of CI passing: `continuous_integration.png`
- Model building
  - Create a machine learning model
    - train/test split + cross-validation
    - 3 functions implemented: train/save/load model & pipeline, inference, classification metrics
    - script that runs everything: take data, process it, and use the 3 functions above
  - Write unit tests: data validation; at least 3
  - Function with model metrics on slices: for every categorical variable, for each level, compute the metrics and output all to `slice_output.txt`
  - Model card: use the template; include metrics.
- API creation
  - Create a REST API
    - Implement POST and GET; use Pydantic
    - Screenshot of the docs: `example.png`
  - Create tests for API: at least 3 test cases
    - GET: test both status and content
    - One test case for each of the possible inferences
- API deployment
  - Deploy to cloud platform
    - Enable CD: screenshot: `continuous_deloyment.png`
    - Screenshot of GET: `live_get.png`
  - Query live API
    - Screenshot of POST: `live_post.png`


Improvement suggestions:

> - Use codecov or equivalent to measure the test coverage (then write more tests and try to hit 100% coverage!).
> - Implement FastAPI’s advanced features such as authentication.
> - Create a front end for the API.