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

    import pandas as pd
    import census_salary as cs

    # Train, is not trained yet
    model, processing_parameters, config, test_scores = cs.train_pipeline(
        config_filename='config.yaml')

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
    # Additionally, validate_data() expects the target column.
    # 2. The X dataframe must have the numerical and categorical columns
    # as defined in config.yaml or core.py (with modified names)
    df = pd.read_csv('./data/census.csv') # original training dataset: features & target
    df, _ = cs.validate_data(df=df) # columns renamed, duplicates dropped, etc.
    X = df.drop("salary", axis=1) # optional
    X = X.iloc[:100, :] # we take a sample

    # Predict salary (values already decoded)
    # This runs the complete pipeline: processing + model
    # The X dataframe must have the numerical and categorical columns
    # as defined in config.yaml or core.py
    print("Prediction:")
    print(pred)

Pylint: 7.43/10

Author: Mikel Sagardia
Date: 2023-01-16
"""
# Script to train machine learning model.
from datetime import datetime
#import logging
#import pickle
#import yaml
#import numpy as np
#import pandas as pd

from sklearn.model_selection import train_test_split

from .ml.model import (train_model,
                       compute_model_metrics,
                       inference,
                       decode_labels)
from .ml.data_processing import process_data
from .core.core import (load_data,
                        validate_data,
                        load_validate_config,
                        load_validate_processing_parameters,
                        load_validate_model,
                        save_processing_parameters,
                        save_model)

# Logging configuration: defined in core.py
from census_salary.core.core import logger

def run_setup(config_filename='config.yaml', config=None):
    """Loads dataset and validates it:
    - columns are renamed,
    - duplicates are dropped,
    - and a stratified train/test split is performed.

    Inputs
    ------
    config_filename : str
        File path of the configuration YAML.
    config : dict
        Configuration dictionary.
        If None, it is loaded using config_filename.

    Returns
    -------
    df_train : pd.DataFrame
        Train split of the dataset, already validated.
    df_test : pd.DataFrame
        Test split of the dataset, already validated.
    config : dict
        Configuration dictionary loaded using config_filename.
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
    df_validated, _ = validate_data(df=df)
    message = ("Dataset correctly validated: columns are OK, "
               "some renamed, duplicated dropped, etc.")
    logger.info(message)

    # Segregate
    target = config['target'] # 'salary'
    df_train, df_test = train_test_split(df_validated,
                                         test_size=config['test_size'], # test_size = 0.20
                                         random_state=config['random_seed'], # 42
                                         stratify=df_validated[target]) # if we want to keep class ratios in splits
    logger.info("Dataset correctly segregated into train/test splits.")

    return df_train, df_test, config

def run_processing(df, config, training=True, processing_parameters=None):
    """Runs the data processing.
    If the processing parameters are not provided,
    they are created and persisted.
    The processing is carried out calling the function process_data,
    which creates a pipeline using the following sklearn transformers:

    - SimpleImputer
    - OneHotEncoder (categorical features)
    - StandardScaler (numerical features)
    - LabelBinarizer (target/label column)

    For more information, check process_data

    Inputs
    ------
    df : pd.DataFrame
        Data frame split on which the processing needs to be carried out.
    config : dict
        Configuration dictionary.
    training : bool
        Whether the processing parameter need to be recomputed.
    processing_parameters : dict
        Dictionary with pre-computed parameters.
        The dictionary contains the pipeline with the transformers.
        Dictionary generated when training. It is equivalent to the
        class ProcessingParameters in core.py.
        It thas the following key-values:
        - features : list of all features used/processed
        - target : str of target/label column
        - categorical_features: list of all categorical features used/processed
        - numerical_features: list of all numerical features used/processed
        - final_feature_names: final list of feature names in the transformed X
        - feature_processor : trained feature processor composed of Pipeline objects
            embedded in a ColumnTransformer, which applies specific
            SimpleImputer, OneHotEncoder, StandardScaler
            depending on the type of column (categorical, numerical)
        - target_processor : trained LabelBinarizer
        When training=True or processing_parameters=None
        the dictionary is re-generated and returned.

    Returns
    -------
    X_transformed : pd.DataFrame
        Transformed data frame of features.
    y_transformed : np.array
        Array with targets, transformed.
    processing_parameters : dict
        See inputs.
    """
    # Extract parameters and load processing_parameters dictionary
    categorical_features = []
    numerical_features = []
    label = None
    if processing_parameters: # We have already the processing params!
        # Extract parameters from processing_parameters
        categorical_features = processing_parameters['categorical_features']
        numerical_features = processing_parameters['numerical_features']
        label = processing_parameters['target']
    else: # We DON'T have the processing params yet!
        try:
            assert config
        except AssertionError as e:
            logger.error("Configuration dict cannot be None if processing_parameters is None.")
            raise e
        # Extract parameters from config
        categorical_features = config['features']['categorical']
        numerical_features = config['features']['numerical']
        label = config['target']
        if not training: # PREDICTION -> processing_parameters must be there
            label = None
            processing_parameters = load_validate_processing_parameters(
                processing_artifact=config['processing_artifact'])

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
    #print(type(X_transformed))
    logger.info("Data processing successful.")

    # Persist
    if training:
        save_processing_parameters(processing_parameters=processing_parameters,
                                   processing_artifact=config['processing_artifact'])
        logger.info("Processing parameters correctly persisted.")

    return X_transformed, y_transformed, processing_parameters

def run_evaluation(model, X_test, y_test):
    """Runs the evaluation using the model
    and the true X and y variables.

    Inputs
    ------
    model : RandomForestClassifier
        Model (trained).
    X_test : pd.DataFrame
        Features dataframe; true values.
    y_tests : np.array
        Target array for X_test; true values.

    Returns
    -------
    test_scores : dict
        Dictionary with evaluation scores:
        precision, recall, fbeta (F1).
    """
    pred, prob = inference(model, X_test, compute_probabilities=True)
    #precision, recall, fbeta, roc_auc = compute_model_metrics(y_test, pred, prob)
    precision, recall, fbeta = compute_model_metrics(y_test, pred, prob)

    test_scores = {
        'precision': precision,
        'recall': recall,
        'fbeta': fbeta
        #'roc_auc': roc_auc
        }
    logger.info("Evaluation successful.")

    return test_scores

def run_slicing(model, df, config, processing_parameters):
    """Runs the data slicing.
    Numerical columns are sliced at the median (equal or above, below).
    Categorical columns are sliced at each category.
    F1 scores are shown for each slice.
    The minimum number of data points to compute slice score
    is defined in config.yaml: slicing_min_data_points.

    Inputs
    ------
    model : RandomForestClassifier
        Model (trained).
    df : pd.DataFrame
        Dataset to be sliced.
    config : dict
        Configuration dictionary.
    processing_parameters : dict
        Processing parameters;
        pipelines contained in it.

    Returns
    -------
    slicing_scores : dict
        Dictionary with slicing scores (F1).
    """
    try:
        assert config
        assert processing_parameters
    except AssertionError as e:
        logger.error("For slicing, valid config and processing_parameters must be passed!")
        raise e

    categorical_features = processing_parameters['categorical_features']
    numerical_features = processing_parameters['numerical_features']
    #label = processing_parameters['target']

    slicing_scores = {}
    X_val, y_val, processing_parameters = run_processing(df,
                                                         config,
                                                         training=False,
                                                         processing_parameters=processing_parameters)

    # Minimum number of data points to use the slice
    # to compute a score; otherwise, the score of teh slice is None
    min_data_points = config['slicing_min_data_points'] # 5
    try:
        assert min_data_points > 0
    except AssertionError as e:
        logger.warning("Minimum number of data points for slicing must be >0, current value: %s",
                       str(min_data_points))
        min_data_points = 0

    # Slice numerical columns at median value: < and >=
    for col in numerical_features:
        d_col = {}
        d_col['median'] = df[col].median()
        # Above median
        row_slice = df[col] >= d_col['median']
        f1_above = None
        if sum(row_slice) > min_data_points:
            scores_above = run_evaluation(model, X_val[row_slice], y_val[row_slice])
            f1_above = scores_above['fbeta']
        # Below median
        row_slice = df[col] < d_col['median']
        #print(col, d_col['median'], sum(row_slice))
        f1_below = None
        if sum(row_slice) > 0:
            scores_below = run_evaluation(model, X_val[row_slice], y_val[row_slice])
            f1_below = scores_below['fbeta']
        d_col['slices'] = {'median_and_above': f1_above,
                           'below_median': f1_below}
        # Append to slicing_scores
        slicing_scores[col] = d_col

    # Slice categorical columns at category levels (i.e., classes within feature)
    for col in categorical_features:
        d_col = {}
        d = {}
        for cat in list(df[col].unique()):
            row_slice = df[col] == cat
            f1_cat = None
            if sum(row_slice) > min_data_points:
                scores_cat = run_evaluation(model, X_val[row_slice], y_val[row_slice])
                f1_cat = scores_cat['fbeta']
            d[cat] = f1_cat
        d_col['num_categories'] = len(df[col].unique())
        d_col['slices'] = d
        # Append to slicing_scores
        slicing_scores[col] = d_col

    logger.info("Data slicing successful.")

    return slicing_scores

def save_evaluation_report(scoring, best_score, best_params, test_scores, evaluation_artifact):
    """Saves the evaluation report.
    The input arguments are persisted in a TXT: evaluation_artifact.

    Inputs
    ------
    scoring : string
        Type of scoring used during training/hyperparameter tuning (e.g., f1).
    best_score : float
        Best scoring value in training (grid search).
    best_params : dict
        Best hyperparameters set.
    test_scores : dict
        Scores obtained when evaluating the test split.
    evaluation_artifact: str
        File path where the evaluation report TXT should be stored.
    Returns
    -------
    None.
    """
    report_evaluation = []
    report_evaluation.append(f'Training and evaluation report, {datetime.now()}')
    report_evaluation.append(' ')
    report_evaluation.append('# TRAINING')
    report_evaluation.append(f"Best score, {scoring} = {best_score}")
    report_evaluation.append(f'Best model hyperparameters: {str(best_params)}')
    report_evaluation.append(' ')
    report_evaluation.append('# EVALUATION (test split)')
    report_evaluation.append(str(test_scores)
                             )
    with open(evaluation_artifact, 'w') as f:
        f.write('\n'.join(report_evaluation))

    logger.info("Evaluation report persisted successfully.")

def save_slicing_report(slicing_scores, slicing_artifact):
    """Saves the data slicing report.
    The slicing_scores dictionary persisted in a TXT: slicing_artifact.
    See run_slicing().

    FIXME: WARNING: If run_slicing() is changes, this function might need
    to be modified, because it checks the keys of the slicing_scores
    dictionary.

    Inputs
    ------
    slicing_scores : dict
        Scores obtained when evaluating data slices.
    slicing_artifact: str
        File path where the data slicing report TXT should be stored.
    Returns
    -------
    None.
    """
    report_slicing = []
    report_slicing.append(f'Data slicing report, {datetime.now()}')
    report_slicing.append('Numerical columns are sliced at the median (equal or above, below).')
    report_slicing.append('Categorical columns are sliced at each category.')
    report_slicing.append('F1 scores are shown for each slice.')
    report_slicing.append('Minimum number of data points to compute slice score is defined in config: slicing_min_data_points.')
    report_slicing.append(' ')
    for col, d in slicing_scores.items():
        report_slicing.append(col)
        if 'num_categories' in d:
            report_slicing.append(f"\tnum_categories: {d['num_categories']}")
        else:
            report_slicing.append(f"\tmedian: {d['median']}")
        for cat, val in d['slices'].items():
            report_slicing.append(f"\t\t{cat}: {val}")

    with open(slicing_artifact, 'w') as f:
        f.write('\n'.join(report_slicing))

    logger.info("Data slicing report persisted successfully.")

def train_pipeline(config_filename='config.yaml'):
    """Main function which runs the complete data processing
    and model training and persists the pipeline for later use.
    It executes the following functions:

    - run_setup(): config dictionary is loaded, as well a validated dataset.
    - run_processing(): dataset is processed, creating also the processing parameters.
    - train_model(): a RandomForestClassifier is fit on the processed train split.
    - save_processing_parameters(): processing pipeline is persisted.
    - save_model(): trained model is persisted.
    - run_evaluation(): evaluation is run on test split.
    - run_slicing(): data slicing is run on test split.
    - save_evaluation_report(): evaluation report is persisted.
    - save_slicing_report(): data slicing report is persisted.

    Inputs
    ------
    config_filename : str
        File path of the configuration dictionary.
    Returns
    -------
    model : RandomForestClassifier
        Trained model.
    processing_parameters : dict
        Dictionary with data processing parameters and pipeline.
    config : dict
        Configuration dictionary.
    test_scores: dict
        Dictionary with evaluation scores (test split) of the model.
    """
    print("TRAINING")
    logger.info("Training starts.")

    # Load and clean dataset + config dictionary
    print("Running setup...")
    df_train, df_test, config = run_setup(config_filename=config_filename, config=None)

    # Process dataset: train & test splits
    print("Running data processing...")
    X_train, y_train, processing_parameters = run_processing(df_train,
                                                             config,
                                                             training=True,
                                                             processing_parameters=None)

    X_test, y_test, processing_parameters = run_processing(df_test,
                                                           config,
                                                           training=False,
                                                           processing_parameters=processing_parameters)

    # Training
    print("Running model fit...")
    config_model = config['random_forest_parameters']
    config_grid = config['random_forest_grid_search']
    #print(type(X_train))
    model, best_params, best_score = train_model(X_train, y_train, config_model, config_grid)
    print("Training results:")
    print(f"- Best score, {config_grid['scoring']} = {best_score}")
    print(f'- Best model hyperparameters: {str(best_params)}')

    # Persist pipeline: model + processing transformers
    print("Persisting pipeline: model + processing...")
    save_processing_parameters(processing_parameters=processing_parameters,
                               processing_artifact=config['processing_artifact'])
    save_model(model=model,
               model_artifact=config['model_artifact'])

    # Evaluation
    print("Running evaluation with test split...")
    test_scores = run_evaluation(model, X_test, y_test)
    print(f"Evaluation results (test split): {str(test_scores)}")

    # Data slicing
    print("Running data slicing with test split...")
    #df_slicing = pd.concat([df_train, df_test], axis=0)
    df_slicing = df_test
    slicing_scores = run_slicing(model=model,
                                 df=df_slicing,
                                 config=config,
                                 processing_parameters=processing_parameters)

    # Persist training and evaluation scores
    save_evaluation_report(config_grid['scoring'],
                           best_score,
                           best_params,
                           test_scores,
                           config['evaluation_artifact'])

    # Persist data slicing scores
    save_slicing_report(slicing_scores, config['slicing_artifact'])

    print("Training successfully finished! Check exported artifacts.\n")

    return model, processing_parameters, config, test_scores

def load_pipeline(config_filename='config.yaml'):
    """Main function which loads the
    inference pipeline.
    This function should be run after the training was successful
    and all inference artifacts have been persisted according
    to config.yaml.

    Inputs
    ------
    config_filename : str
        File path of the configuration dictionary.
    Returns
    -------
    model : RandomForestClassifier
        Trained model.
    processing_parameters : dict
        Dictionary with data processing parameters and pipeline.
    config : dict
        Configuration dictionary.
    """
    print("Loading pipeline: model + processing parameters + config...")
    config = load_validate_config(config_filename=config_filename)
    processing_parameters = load_validate_processing_parameters(
        processing_artifact=config['processing_artifact'])
    model = load_validate_model(model_artifact=config['model_artifact'])

    logger.info("Pipeline (model, processing, config) correctly loaded and validated.")

    return model, processing_parameters, config

def predict(X, model, processing_parameters):
    """Real time inference function. It:
    - Processes the input data.
    - Runs the inference with the model, i.e., RandomForestClassifier.predict()
    - Decodes the model outcome to the original label format.

    Inputs
    ------
    X : pd.DataFrame
        Unprocessed features; columns and their values
        must be the same as the raw/original dataset used for training.
    model: RandomForestClassifier
        Trained model.
    processing_parameters : dict
        Dictionary with data processing parameters and pipeline.
    Returns
    -------
    pred_decoded : np.array
        Model outcome/scoring.
        Array is encoded in the original target labels.
    """
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
    """
    Main call: training.
    See the module docstring for an usage example.
    """
    model, processing_parameters, config, test_scores = train_pipeline(
        config_filename='config.yaml')
