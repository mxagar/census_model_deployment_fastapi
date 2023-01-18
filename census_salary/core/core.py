"""_summary_

"""

import logging
from pydantic import BaseModel, ValidationError
from typing import Dict, List, Optional, Tuple # Sequence
import pickle
import yaml

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier

from census_salary.data.dataset import get_data

# Logging configuration
logging.basicConfig(
    filename='./logs/census_pipeline.log', # filename, where it's dumped
    level=logging.INFO, # minimum level I log
    filemode='a', # append
    format='%(name)s - %(asctime)s - %(levelname)s - %(message)s') # add function/module name for tracing
logger = logging.getLogger()

class ProcessingParameters(BaseModel):
    """
    Processing parameters for the data.
    Pipeline is included.
    """
    features: List[str]
    target: str
    categorical_features: List[str]
    numerical_features: List[str]
    # numerical: SimpleImputer, StandardScaler
    # categorical: SimpleImputer, OneHotEncoder
    feature_processor: ColumnTransformer
    target_processor: LabelBinarizer
    
    class Config:
        arbitrary_types_allowed = True

class ModelConfig(BaseModel):
    """
    Model configuration:
    default model parameter values.
    """
    n_estimators: int
    criterion: str
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    min_weight_fraction_leaf: float
    max_features: str
    max_leaf_nodes: None # null
    min_impurity_decrease: float
    bootstrap: bool
    oob_score: bool
    n_jobs: None # null
    random_state: int
    verbose: int
    warm_start: bool
    class_weight: str
    ccp_alpha: float
    max_samples: None # null

class TrainingConfig(BaseModel):
    """
    Training configuration, i.e.,
    hyperparameter tuning definition with grid search.
    """
    hyperparameters: Dict
    cv: int
    scoring: str

class GeneralConfig(BaseModel):
    """
    General configuration file.
    All configuration relevant to model
    training and data processing (i.e., feature engineering, etc.).
    """
    data_path: str
    test_size: float
    random_seed: int
    target: str
    features: Dict[str, List[str]]
    random_forest_parameters: ModelConfig
    random_forest_grid_search: TrainingConfig
    model_artifact: str
    processing_artifact: str
    evaluation_artifact: str

class DataRow(BaseModel):
    """
    Single dataset row.
    """
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    salary: str
    
class MultipleDataRows(BaseModel):
    """
    Multiple dataset rows,
    i.e., a dataset.
    This class is used for validation.
    """
    inputs: List[DataRow]

def load_data(data_path: str = "./data/census.csv") -> pd.DataFrame:
    """Gets and loads dataset as a dataframe.
    
    Inputs
    ------
    data_path : str
        String of to the local dataset path.
        Default: "./data/census.csv".
        
    Returns
    -------
    df: pd.DataFrame
        Loaded dataset.
    """
    # Download dataset to local file directory
    get_data(destination_file=data_path)
    
    try:
        df = pd.read_csv(data_path) # './data/census.csv'
    except FileNotFoundError as e:
        logger.error("Dataset file not found: %s.", data_path)

    return df

def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Cleans and validates training dataset.
    That implies:
    - Renaming columns: remove blank spaces and replace '-' with '_'
    - Dropping duplicates.
    Validation occurs by converting the loaded
    dictionary into the MultipleDataRows class/object defined in core.py.
        
    Inputs
    ------
    df : pd.DataFrame
        String of to the local dataset path.
        
    Returns
    -------
    df_validated: pd.DataFrame
        Validated dataset
    errors: dict
        Validation error, if there was one; otherwise None is returned.
    """
    # Rename columns
    # - remove preceding blank space: ' education' -> 'education', etc.
    # - replace - with _: 'education-num' -> 'education_num', etc.
    #df_validated = df.copy()
    #data = df.copy()
    df = df.rename(
        columns={col_name: col_name.replace(' ', '') for col_name in df.columns})
    df = df.rename(
        columns={col_name: col_name.replace('-', '_') for col_name in df.columns})

    # Drop duplicates
    df_validated = df.drop_duplicates().reset_index(drop=True)
    
    # Other checks we could do:
    # - convert types: df_validated[col] = df_validated[col].astype("O")
    # - drop NA: df_validated = df_validated.dropna()

    # Validate
    errors = None
    try:
        # Replace numpy nans so that pydantic can validate
        MultipleDataRows(
            inputs=df_validated.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return df_validated, errors

def load_validate_config(
    config_filename: str = "config.yaml") -> dict:
    """Loads and validates general configuration YAML.
    Validation occurs by converting the loaded
    dictionary into the GeneralConfig class/object defined in core.py.
    
    Inputs
    ------
    config_filename : str
        String of to the local config file path.
        
    Returns
    -------
    config: dict
        Validated configuration dictionary.
    """
    config = dict()
    try:
        with open(config_filename) as f: # 'config.yaml'
            config = yaml.safe_load(f)
            # Convert dictionary to Config class to validate it
            _ = GeneralConfig(**config)
    except FileNotFoundError as e:
        logger.error("Configuration YAML not found: %s.", config_filename)
    except ValidationError as e:
        logger.error("Configuration file validation error.")

    return config

def load_validate_processing_parameters(
    processing_artifact: str = "./exported_artifacts/processing_parameters.pickle") -> dict:
    """Loads and validates the processing parameters.
    Validation occurs by converting the loaded
    dictionary into the ProcessingParameters class/object defined in core.py.
    
    Inputs
    ------
    processing_artifact : str
        String of to the local processing parameters file path.
        
    Returns
    -------
    processing_parameters: dict
        Validated processing parameters dictionary.
    """
    processing_parameters = dict()
    try:
        with open(processing_artifact, 'rb') as f: # 'exported_artifacts/processing_parameters.pickle'
            processing_parameters = pickle.load(f)
            # Convert dictionary to ProcessingParameters class to validate it
            _ = ProcessingParameters(**processing_parameters)
    except FileNotFoundError as e:
        logger.error("Processing parameters artifact/pickle not found: %s.", processing_artifact)
    except ValidationError as e:
        logger.error("Processing parameters artifact/pickle validation error.")
        
    return processing_parameters

def load_validate_model(
    model_artifact: str = "./exported_artifacts/model.pickle") -> RandomForestClassifier:
    """Loads and validates the (trained) model.
    Validation occurs by checking that the loaded
    object type is RandomForestClassifier.
    
    Inputs
    ------
    model_artifact : str
        String of to the local model file path.
        
    Returns
    -------
    model: RandomForestClassifier
        Validated model.
    """
    try:
        with open(model_artifact, 'rb') as f: # 'exported_artifacts/model.pickle'
            model = pickle.load(f)
        # Check that the loaded model is a RandomForestClassifier to validate it
        assert isinstance(model, RandomForestClassifier)
    except FileNotFoundError as e:
        logger.error("Model artifact/pickle not found: %s.", model_artifact)
    except AssertionError as e:
        logger.error("Model artifact/pickle is not a RandomForestClassifier.")        
    
    return model

def save_processing_parameters(processing_parameters: dict,
                               processing_artifact: str = "./exported_artifacts/processing_parameters.pickle") -> None:
    """
    """
    pickle.dump(processing_parameters, open(processing_artifact,'wb')) # wb: write bytes

def save_model(model: RandomForestClassifier, 
               model_artifact: str = "./exported_artifacts/model.pickle") -> None:
    """
    """
    pickle.dump(model, open(model_artifact,'wb')) # wb: write bytes

