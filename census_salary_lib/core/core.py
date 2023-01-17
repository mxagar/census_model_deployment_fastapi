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

from census_salary_lib.data.dataset import get_data

# Logging configuration
logging.basicConfig(
    filename='./logs/census_pipeline.log', # filename, where it's dumped
    level=logging.INFO, # minimum level I log
    filemode='a', # append
    format='%(name)s - %(asctime)s - %(levelname)s - core - %(message)s') # add function/module name for tracing
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
    min_impurity_split: None # null
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

class Config(BaseModel):
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
    
    # Download dataset to local file directory
    get_data(destination_file=data_path)
    
    try:
        df = pd.read_csv(data_path) # './data/census.csv'
    except FileNotFoundError as e:
        logger.error("Dataset file not found: %s.", data_path)

    return df

def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    
    # Rename columns
    # - remove preceding blank space: ' education' -> 'education', etc.
    # - replace - with _: 'education-num' -> 'education_num', etc.
    df_validated = df.copy()
    df_validated = df_validated.rename(
        columns={col_name: col_name.replace(' ', '') for col_name in df.columns})
    df_validated = df_validated.rename(
        columns={col_name: col_name.replace('-', '_') for col_name in df.columns})

    # Drop duplicates
    df_validated = df_validated.drop_duplicates().reset_index(drop=True)
    
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
    config_filename: str = "config.yaml") -> Config:
    
    config = dict()
    try:
        with open(config_filename) as f: # 'config.yaml'
            config = yaml.safe_load(f)
            # Convert dictionary to Config class to validate it
            _ = Config(**config)
    except FileNotFoundError as e:
        logger.error("Configuration YAML not found: %s.", config_filename)
    except ValidationError as e:
        logger.error("Configuration file validation error.")

    return config

def load_validate_processing_parameters(
    processing_artifact: str = "./exported_artifacts/processing_parameters.pickle") -> ProcessingParameters:

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
