"""API app of the census library based on FastAPI.

This file defines a FastAPI app with which we can interact and use a ML model
to perform inferences. Altogether, the following methods are implemented:

- GET, index(): Welcome page with links to documentation
- GET, health(): JSON with API and model version is returned 
- POST, make_prediction(): we pass a JSON with features and the inferred target is returned

To use this app LOCALLY, go to the folder where config.yaml is located
and start in the terminal the Uvicorn ASGI web server:

>> uvicorn api.app:app --reload

Then, we open the browser in http://127.0.0.1:8000 and we can start using the API.
We have the following endpoints:

- http://127.0.0.1:8000 : welcome page (index)
- http://127.0.0.1:8000/docs : documentation with testing interfaces
- http://127.0.0.1:8000/health : JSON with API and model version is returned 
- 127.0.0.1:8000/predict ...: we pass a JSON with features and the inferred target is returned;
    see documentation for examples

To use this app HOSTED ON A CLOUD SERVICE, the web server must be run differently.
For instance, if we want to deploy to to HEROKU, we need a Procfile with this line:

    web: uvicorn api.app:app --host=0.0.0.0 --port=${PORT:-5000}

Author: Mikel Sagardia
Date: 2023-01-23
"""

import json
from datetime import datetime

from typing import Any
from fastapi import FastAPI, Request, HTTPException
from pydantic import ValidationError
from fastapi.responses import HTMLResponse
from fastapi.encoders import jsonable_encoder

#import numpy as np
import pandas as pd

from api import __version__ as api_version
from .schemas import Health, PredictionResults

from census_salary import __version__ as model_lib_version
from census_salary import logger
from census_salary import MultipleDataRows
from census_salary import load_pipeline, validate_data, predict

# Constants
# Note that app.py must be called from the folder where config.yaml is,
# i.e., for instance as:
# >> uvicorn api.app:app --reload
API_PROJECT_NAME = "Census Salary Model API"
CONFIG_FILENAME = "./config.yaml"
INDEX_BODY = (
    "<html>"
    "<body style='padding: 10px;'>"
    "<h1>Welcome to the API</h1>"
    "<div>"
    "Check the <a href='/docs'>documentation</a>."
    "</div>"
    "</body>"
    "</html>"
)

# Load model pipeline: model + data processing
model, processing_parameters, config = load_pipeline(config_filename=CONFIG_FILENAME)

# FastAPI app
app = FastAPI(title=API_PROJECT_NAME)

@app.get("/")
def index(request: Request) -> Any:
    """Basic HTML response.
    Default welcome pages of API."""
    body = INDEX_BODY

    return HTMLResponse(content=body)

@app.get("/health")
def health() -> dict:
    """Root get, which returns general API information (version & Co.). 
    """
    health = Health(
        name=API_PROJECT_NAME,
        api_version=api_version,
        model_lib_version=model_lib_version
    )

    # Convert to dict and return
    return health.dict()

# When a the returned object is a Pydantic model/class/object, we define: 
# - as return type hint: Any
# - as response_model parameter of the HTTP method the desired type: response_model=PredictionResults
@app.post("/predict", response_model=PredictionResults, status_code=200)
async def make_prediction(input_data: MultipleDataRows) -> Any:
    """Make a prediction with the trained model
    Make house price predictions with the TID regression model
    """
    # jsonable_encoder() receives an object, like a Pydantic model,
    # and returnns a JSON compatible version; e.g., datetime is converted to str, etc.
    logger.info("Importing data: %s", str(input_data.inputs))
    print(str(input_data.inputs))
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    logger.info("Data imported: %s", str(input_df))

    # Get and validate the data: columns renamed, duplicates dropped, etc.
    #X, errors = validate_data(df=input_df.replace({np.nan: None}))
    X, errors = validate_data(df=input_df) # 
    if "salary" in X.columns:
        X = X.drop("salary", axis=1) # optional removal
    if errors:
        logger.error(f"Input data validation error: %s", str(errors.json()))
        raise HTTPException(status_code=400, detail=json.loads(errors.json()))

    # Make & pack prediction
    results = None
    try:
        # TODO: we could make predict() async and use await here
        logger.info("Making prediction on inputs: %s", str(input_data.inputs))
        pred = predict(X, model, processing_parameters)
        # Pack predictions
        results = PredictionResults(
            model_lib_version = model_lib_version,
            timestamp = str(datetime.now()),
            predictions = list(pred) # pred is np.array, cast it to list
        )
    except ValidationError as error:
        logger.error(f"Prediction validation error.")
        raise HTTPException(status_code=400, detail=json.loads(error.json()))

    if results:
        logger.info("Prediction results: %s", str(results.predictions))
    else:
        logger.info("Prediction unsuccessful.")        

    return results
