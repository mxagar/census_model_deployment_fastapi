"""Census library usage example.
This is not API, but a blueprint for any application.

For the FastAPI application, look in api/

Steps shown in this file:

- Training is performed after loading the config.yaml file
- Inference pipeline (model + processing pipeline) is persisted
- Inference pipeline is loaded
- Data is loaded and scored, i.e., an inference is provided

Author: Mikel Sagardia
Date: 2023-01-20
"""
#import logging
import pandas as pd
import census_salary as cs

# Logging configuration: Defined in core.py from census_salary
from census_salary import logger

if __name__ == '__main__':

    # Train, is not trained yet
    model, processing_parameters, config, test_scores = cs.train_pipeline(config_filename='config.yaml')
    print("Test scores: ")
    print(test_scores)
    logger.info("APP: Test scores: %s", str(test_scores))

    # Load pipeline, if training performed in another execution/session
    logger.info("Loading pipeline...")
    model, processing_parameters, config = cs.load_pipeline(config_filename='config.yaml')

    # Get and check the data
    df = pd.read_csv('./data/census.csv') # original training dataset: features & target
    df, _ = cs.validate_data(df=df) # columns renamed, duplicates dropped, etc.
    X = df.drop("salary", axis=1) # optional
    X = X.iloc[:100, :] # we take a sample

    # Predict salary (values already decoded)
    logger.info("Inference...")
    pred = cs.predict(X, model, processing_parameters)
    print("Prediction: ")
    print(pred)
    logger.info("APP: Prediction: %s", str(pred))
