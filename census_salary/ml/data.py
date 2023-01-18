"""This module contains all the basic functions
related to the data processing of the census dataset.

In a single function, the complete dataset
used for training is ingested.
Then, its columns are processed and returned
as a a (X, y) pair.
Additionally, an object which contains the
processing parameters is returned.

Author: Mikel Sagardia
Date: 2023-01-16
"""
import logging
import numpy as np
#import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OneHotEncoder,
                                   StandardScaler,
                                   LabelBinarizer)
from sklearn.pipeline import make_pipeline #, Pipeline 
from sklearn.compose import ColumnTransformer

# Logging configuration
logging.basicConfig(
    filename='./logs/census_pipeline.log', # filename, where it's dumped
    level=logging.INFO, # minimum level I log
    filemode='a', # append
    format='%(name)s - %(asctime)s - %(levelname)s - %(message)s') # add function/module name for tracing
logger = logging.getLogger()

def process_data(
    df,
    categorical_features=[],
    numerical_features=[],    
    label=None,
    training=True,
    processing_parameters=None
):
    """ Process the data used in the machine learning pipeline.

    Processes the data using the following sklearn transformers:
    
    - SimpleImputer
    - OneHotEncoder (categorical features)
    - StandardScaler (numerical features)
    - LabelBinarizer (target/label column)

    This function can be used in either training or
    inference/validation.

    Notes: 
    
    1. The used random forest classifier doesn't really require scaling
    the numerical features, but that transformer is added in case
    the pipeline is extended with other models which do require scaling.
    2. We could further extend the list of transformations depending on the
    dataset. We can use transformers from Scikit-Learn or custom ones.

    Inputs
    ------
    df : pd.DataFrame
        Dataframe containing the features and label.
    categorical_features: list[str]
        List containing the names of the categorical features (default=[]).
    numerical_features: list[str]
        List containing the names of the numerical features (default=[]).
    label : str
        Name of the label/target column in `X`. If None, then an empty array will be returned
        for y (default=None).
    training : bool
        Indicator if training mode or inference/validation mode (default=True).
    processing_parameters : dict
        Dictionary generated when training. It thas the following key-values:
        - features : list of all features used/processed
        - target : str of target/label column
        - categorical_features: list of all categorical features used/processed
        - numerical_features: list of all numerical features used/processed
        - feature_processor : trained feature processor composed of Pipeline objects
            embedded in a ColumnTransformer, which applies specific
            SimpleImputer, OneHotEncoder, StandardScaler
            depending on the type of column (categorical, numerical)
        - target_processor : trained LabelBinarizer
        When training=Tru or processing_parameters=None
        the dictionary is re-generated and returned.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if label is str, otherwise empty np.array.
    processing_parameters : dict
        Dictionary generated when training. See the inputs section
        for a detailed list of key-values.
        When training=True or processing_parameters=None
        the dictionary is re-generated and returned.
        Otherwise, the input dictionary is returned.
    """
    # Collection of all feature columns
    features = numerical_features + categorical_features

    try:
        X = df[features]
        y = np.array([])
    except KeyError as e:
        logger.error("A column is missing in the dataset.")
    
    #if training or label:
    if label:
        if label in df.columns:
            y = df[label]
        else:
            logger.info("No target column taken from the dataset.")

    #X_categorical = X[categorical_features].values
    #X_numerical = X[numerical_features].values

    if training is True or not processing_parameters: # TRAINING  
        ## -- 1. Features
        # Define processing for categorical columns
        # handle_unknown: label encoders need to be able to deal with unknown labels!
        categorical_transformer = make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0),
            OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        )
        # Define processing for numerical columns
        numerical_transformer = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler()
        )
        # Put the 2 tracks together into one pipeline using the ColumnTransformer
        # This also drops the columns that we are not explicitly transforming
        feature_processor = ColumnTransformer(
            transformers=[
                ("num", numerical_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ],
            remainder="drop",  # This drops the columns that we do not transform
        )
        # Train & transform
        X_transformed = feature_processor.fit_transform(X)
        
        ## -- 2. Target      
        target_processor = LabelBinarizer()
        y_transformed = target_processor.fit_transform(y).ravel()
        
        ## -- 3. Pack everything into a dictionary
        processing_parameters = dict()
        processing_parameters['features'] = categorical_features + numerical_features
        processing_parameters['target'] = label
        processing_parameters['categorical_features'] = categorical_features
        processing_parameters['numerical_features'] = numerical_features
        processing_parameters['feature_processor'] = feature_processor
        processing_parameters['target_processor'] = target_processor
        
        logger.info("Data processing pipeline trained and data transformed.")
        
    else: # PREDICTION
        ## -- 1. Extract processors
        try:
            feature_processor = processing_parameters['feature_processor']
            target_processor = processing_parameters['target_processor']
        except KeyError as e:
            logger.error("Processing parameters object has missing keys.")
        except TypeError as e:
            logger.error("Processing parameters object is not the expect object type.")
        
        ## -- 2. Transform
        # X
        try:
            X_transformed = feature_processor.transform(X)
        except ValueError as e:
            logger.error("The columns in X don't match with the columns expected by the processing pipeline.")        
        # y
        try:
            y_transformed = target_processor.transform(y).ravel()
        except ValueError as e:
            y_transformed = np.array([])
            logger.info("Empty target/label array.")
            
        logger.info("Data transformed.")

    return X_transformed, y_transformed, processing_parameters
