"""This module contains all the basic functions
related to the machine learning model fit to the census dataset.

It:
- defines and trains the model
- computes the model metrics on a given dataset split (test, preferably)
- and provides an inference function.

Author: Mikel Sagardia
Date: 2023-01-16
"""
import logging
#import pickle
#import numpy as np
#import pandas as pd

from sklearn.metrics import (fbeta_score,
                             precision_score,
                             recall_score,
                             roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Logging configuration
logging.basicConfig(
    filename='./logs/census_pipeline.log', # filename, where it's dumped
    level=logging.INFO, # minimum level I log
    filemode='a', # append
    format='%(name)s - %(asctime)s - %(levelname)s - model - %(message)s') # add function/module name for tracing
logger = logging.getLogger()

def train_model(X_train, y_train, config_model, config_grid):
    """
    Trains a machine learning model and returns it.
    Hyperparameter tuning is performed according to the
    passed grid values.
    The passed data must be already processed.

    Inputs
    ------
    X_train : np.array
        Training data, already processed.
    y_train : np.array
        Labels, already processed.
    config_model : dict
        Dictionary with configuration paramaters
        for the model, loaded from ROOT/config.yaml
    config_grid : dict
        Dictionary with configuration paramaters
        for the grid search, loaded from ROOT/config.yaml
    Returns
    -------
    model : sklearn model object (sklearn.ensemble.RandomForestClassifier)
        Trained machine learning model.
    best_params : dict
        Dictionary with best hyperparameters found in the gird search.
    best_score : float
        Best scoring value after the training with hyperparameter tuning
        using grid search.
    """
    # Random forest classifier
    estimator = RandomForestClassifier(**config_model)

    # Define Grid Search: parameters to try, cross-validation size
    #param_grid = {
    #    'n_estimators': [100, 150, 200],
    #    'max_features': ['sqrt', 'log2'],
    #    'criterion': ['gini', 'entropy'],
    #    'max_depth': [None]+[n for n in range(5,20,5)]
    #}
    param_grid = config_grid['hyperparameters']
    param_grid['max_depth'] = [None] + param_grid['max_depth']

    # Grid search
    search = GridSearchCV(estimator=estimator,
                        param_grid=param_grid,
                        cv=config_grid['cv'], # 3
                        scoring=config_grid['scoring']) # 'roc_auc'

    # Find best hyperparameters and best estimator pipeline
    search.fit(X_train, y_train)
    logger.info("Model successfully trained.")
    
    model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    score_string = config_grid['scoring'] + " = " + str(best_score)
    logger.info("Best score: ", %s, score_string)
    params_string = str(best_params)
    logger.info("Best hyperparameters: ", %s, params_string)

    return model, best_params, best_score


def compute_model_metrics(y, preds, probs):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    probs : np.array
        Predicted probabilities, for ROC AUC computation.
        
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    roc_auc : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    roc_auc = roc_auc_score(y, probs)
    
    return precision, recall, fbeta, roc_auc


def inference(model, X, compute_probabilities=False):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble.RandomForestClassifier object
        Trained machine learning model.
    X : np.array
        Data used for prediction.
        The values must be already processed.
    compute_probabilities : bool
        Whether probabilities need to be returned
        in addition to labels/classes (default=False).
    Returns
    -------
    preds : np.array
        Prediction labels from the model.
    probs : np.array
        Prediction probabilities from the model.
        If compute_probabilities=False (default),
        [] is returned.
    """
    preds = model.predict(X)
    probs = []
    if compute_probabilities:
        probs = model.predict_proba(X)[:, 1]

    return preds, probs