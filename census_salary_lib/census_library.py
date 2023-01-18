"""This module contains creates and persists
a machine learning model for the census dataset.

The following steps are carried out:
- Load the dataset
- Basic cleaning and pre-processing
- Segregation of the dataset: train/test
- Data processing
- Model training with hyper parameter tuning
- Model and processing pipeline persisting
   
Usage example: 

    import census_salary_lib as cs

    # Train, is not trained yet
    model, processing_parameters, config = cs.train_pipeline(config_filename='config.yaml')

    # Load pipeline, if training performed in another execution/session
    model, processing_parameters, config = cs.load_pipeline(config_filename='config.yaml')
    
    # Get and check the data.
    # Example: if we employ the dataset used for training,
    # we need to read/load it and validate it:
    # remove duplicates, rename columns and check fields with Pydantic.
    # Then, we can (optionally) drop the "salary" column (target, to be predicted).
    # Notes:
    # 1. validate_data() is originally defined for training purposes
    # and it expects the features and target as in the original dataset form.
    # Additionally, vadalidate_data() expects the target column. 
    # 2. The X dataframe must have the numerical and categorical columns
    # as defined in config.yaml or core.py
    df = pd.read_csv('./data/census.csv') # original training dataset: features & target
    df = cs.validate_data(df=df) # columns renamed, duplicates droped, etc.
    X = df.drop("salary", axis=1) # optional
    X = X.iloc[:100, :] # we take a sample
    
    # Predict salary (values already decoded)
    # This runs the complete pipeline: processing + model
    # The X dataframe must have the numerical and categorical columns
    # as defined in config.yaml or core.py
    pred = cs.predict(X, model, processing_parameters)


Author: Mikel Sagardia
Date: 2023-01-16
"""
# Script to train machine learning model.
from datetime import datetime
import logging
import pickle
import yaml
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from .ml.model import (train_model,
                       compute_model_metrics,
                       inference,
                       decode_labels)
from .ml.data import process_data
from .core.core import (load_data,
                        validate_data,
                        load_validate_config,
                        load_validate_processing_parameters,
                        load_validate_model,
                        save_processing_parameters,
                        save_model)

# Logging configuration
logging.basicConfig(
    filename='./logs/census_pipeline.log', # filename, where it's dumped
    level=logging.INFO, # minimum level I log
    filemode='a', # append
    format='%(name)s - %(asctime)s - %(levelname)s - train_model - %(message)s') # add function/module name for tracing
logger = logging.getLogger()

def run_setup(config_filename='config.yaml', config=None):
    """
    """
    if not config:
        # Load configuration dictionary, since it's not passed
        config = load_validate_config(config_filename)
        logger.info("Configuration file correctly loaded.")
    
    # Load dataset
    data_path = config['data_path']
    df = load_data(data_path=data_path)
    logger.info("Dataset correctly loaded.")

    # Validate dataset
    df, _ = validate_data(df=df)    
    logger.info("Dataset correctly validated: columns are OK, some renamed, duplicated dropped, etc.")

    # Segregate
    df_train, df_test = train_test_split(df,
                                         test_size=config['test_size'], # test_size = 0.20
                                         random_state=config['random_seed'], # 42
                                         stratify=y) # if we want to keep class ratios in splits
    logger.info("Dataset correctly segregated into train/test splits.")
    
    return df_train, df_test, config

def run_processing(df, config, training=True, processing_parameters=None):
    """
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

    # Persist pipeline: model + processing transformers
    save_processing_parameters(processing_parameters=processing_parameters,
                               processing_artifact=config['processing_artifact'])
    save_model(model=model, 
               model_artifact=config['model_artifact'])

    # Evaluation
    pred, prob = inference(model, X_test, compute_probabilities=True)
    precision, recall, fbeta, roc_auc = compute_model_metrics(y_test, pred, prob)
    
    # Persist metrics
    test_scores = (precision, recall, fbeta, roc_auc)
    report = []
    report.append(f'Training and evaluation report, {datetime.now()}')
    report.append(' ')
    report.append('# TRAINING')
    report.append(f'Best score - {} = {best_score}')
    report.append(f'Best model hyperparameters: {str(best_params)}')
    report.append(' ')
    report.append('# EVALUATION (test split)')
    report.append(f'Precision = {precision}')
    report.append(f'Recall = {recall}')
    report.append(f'F1 = {fbeta}')
    report.append(f'ROC-AUC = {roc_auc}')
    with open(config['evaluation_artifact'], 'w') as f:
        f.write('\n'.join(report))

    return model, processing_parameters, config, test_scores

def load_pipeline(config_filename='config.yaml'):
    
    config = load_validate_config(config_filename=config_filename)
    processing_parameters = load_validate_processing_parameters(processing_artifact=config['processing_artifact'])
    model = load_validate_model(model_artifact=config['model_artifact'])

    logger.info("Pipeline (model, processing, config) correctly loaded and validated.")
        
    return model, processing_parameters, config

def predict(X, model, processing_parameters):
    
    # Process
    X_trans, _, _ = run_processing(X,
                                   config=None,
                                   training=False,
                                   processing_parameters=processing_parameters)
    # Inference
    pred, _ = inference(model, X_trans, compute_probabilities=False)
    
    pred_decoded = decode_labels(pred, processing_parameters)
    
    return pred_decoded

if __name__ == "__main__":
    
    model, processing_parameters, config = train_pipeline(config_filename='config.yaml')
    