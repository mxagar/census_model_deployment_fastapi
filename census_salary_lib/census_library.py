"""This module contains creates and persists
a machine learning model for the census dataset.

The following steps are carried out:
- Load the dataset
- Basic cleaning and pre-processing
- Segregation of the dataset: train/test
- Data processing
- Model training with hyper parameter tuning
- Model and processing pipeline persisting

Author: Mikel Sagardia
Date: 2023-01-16
"""
# Script to train machine learning model.
import logging
import pickle
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from .ml.data import process_data
from .ml.model import (train_model,
                       compute_model_metrics,
                       inference)

# Logging configuration
logging.basicConfig(
    filename='./logs/census_pipeline.log', # filename, where it's dumped
    level=logging.INFO, # minimum level I log
    filemode='a', # append
    format='%(name)s - %(asctime)s - %(levelname)s - train_model - %(message)s') # add function/module name for tracing
logger = logging.getLogger()

def run_setup(config_filename='config.yaml', config=None):
    """_summary_

    Args:
        config_filename (str, optional): _description_. Defaults to 'config.yaml'.
        config (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if not config:
        # Load configuration dictionary, since it's not passed
        config = dict()
        try:
            with open(config_filename) as f: # 'config.yaml'
                config = yaml.safe_load(f)
        except FileNotFoundError as e:
            logger.error("Configuration file not found: %s", config_filename)
    
    # Load dataset
    data_path = config['data_path']
    df = pd.read_csv(data_path) # './data/census.csv'
    logger.info("Dataset correctly loaded.")

    # Rename columns: remove preceding blank space: ' education' -> 'education', etc.
    df = df.rename(columns={col_name: col_name.replace(' ', '') for col_name in df.columns})

    # Drop duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info("Dataset correctly cleaned: columns renamed and duplicated dropped.")

    # Segregate
    df_train, df_test = train_test_split(df,
                                         test_size=config['test_size'], # test_size = 0.20
                                         random_state=config['random_seed'], # 42
                                         stratify=y) # if we want to keep class ratios in splits
    logger.info("Dataset correctly segregated into train/test splits.")
    
    return df_train, df_test, config

def run_processing(df, config, training=True, processing_parameters=None):
    """_summary_

    Args:
        df (_type_): _description_
        config (_type_): _description_
        training (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    # Extract parameters and load processing_parameters config dictionary
    if not processing_parameters: # We have already the processing params!
        categorical_features = processing_parameters['categorical_features']
        numerical_features = processing_parameters['numerical_features']
        label = processing_parameters['target']
    else: # We DON'T have the processing params yet!
        try:
            assert config
        except AssertionError as e:
            logger.error("Configuration dict cannot be None if processing_parameters is None.")
        categorical_features = config['features']['categorical']
        numerical_features = config['features']['numerical']
        label = config['target']
        processing_artifact = config['processing_artifact']
        #processing_parameters = None
        if not training: # PREDICTION
            label = None
            try:
                processing_parameters = pickle.load(open(processing_artifact,'rb')) # rb: read bytes
            except FileNotFoundError as e:
                logger.error("Processing parameters file not found: %s", processing_artifact)
    logger.info("Processing parameters correctly extracted.")

    # Process
    X_transformed, y_transformed, processing_parameters = process_data(
        df,
        categorical_features=categorical_features,
        numerical_features=numerical_features,    
        label=label,
        training=training,
        processing_parameters=processing_parameters
    )
    logger.info("Data processing successful.")

    # Persist
    if training:
        pickle.dump(processing_parameters, open(processing_artifact,'wb')) # wb: write bytes
    logger.info("Processing parameters correctly persisted.")

    return X_transformed, y_transformed, processing_parameters

def train_pipeline(config_filename='config.yaml'):

    # Load and clean dataset + config dictionary
    df_train, df_test, config = run_setup(config_filename=config_filename, config=None)
    
    # Process dataset: train & test splits
    X_train, y_train, processing_parameters = run_processing(df_train,
                                                             config,
                                                             training=True,
                                                             processing_parameters=None)
    
    X_test, y_test, processing_parameters = run_processing(df_test,
                                                           config,
                                                           training=False,
                                                           processing_parameters=processing_parameters)
    # Training
    config_model = config['random_forest_parameters']
    config_grid = config['random_forest_grid_search']
    model, best_params, best_score = train_model(X_train, y_train, config_model, config_grid)

    # Persist pipeline
    pickle.dump(processing_parameters, open(config['processing_artifact'],'wb')) # wb: write bytes
    pickle.dump(model, open(config['model_artifact'],'wb')) # wb: write bytes

    # Evaluation
    pred, prob = inference(model, X_test, compute_probabilities=True)
    precision, recall, fbeta, roc_auc = compute_model_metrics(y_test, pred, prob)
    
    # Persist metrics
    test_scores = (precision, recall, fbeta, roc_auc)

    return model, processing_parameters, config, test_scores

def load_pipeline(config_filename='config.yaml'):
    
    # Load configuration dictionary, since it's not passed
    config = dict()
    try:
        with open(config_filename) as f: # 'config.yaml'
            config = yaml.safe_load(f)
    except FileNotFoundError as e:
        logger.error("Configuration file not found: %s", config_filename)
    
    processing_parameters = pickle.load(open(config['processing_artifact'],'rb')) # rb: read bytes
    model = pickle.load(open(config['model_artifact'],'rb')) # rb: read bytes
    
    return model, processing_parameters, config

def predict(X, model, processing_parameters):
    
    # Process
    X_trans, _, _ = run_processing(X,
                                   config=None,
                                   training=False,
                                   processing_parameters=processing_parameters)
    # Inference
    pred, _ = inference(model, X_trans, compute_probabilities=False)
    
    return pred
