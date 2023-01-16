"""This module contains all the basic functions
related to the machine learning model fit to the census dataset.

It:
- defines and trains the model
- computes the model metrics on a given dataset split (test, preferably)
- and provides an inference function.

Author: Mikel Sagardia
Date: 2023-01-16
"""

from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train, config_model, config_grid):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    config_model : dict
        Dictionary with configuration paramaters
        for the model, loaded from ROOT/config.yaml
    config_grid : dict
        Dictionary with configuration paramaters
        for the grid search, loaded from ROOT/config.yaml
    Returns
    -------
    model
        Trained machine learning model.
    """



    pass


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
        Predicted probabilities, for ROC computation.
        
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
        None is returned.
    """
    pass
