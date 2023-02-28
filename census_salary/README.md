# Census Model Library

This folder contains the census model library, which can be used as an installed package to train and use (i.e., infer) the census model. Recall that the model takes 14 features in (per censed person) and predicts the salary range associated to those features.

The package is composed of the following files:

- `core/core.py`: general data structure definitions and their respective loading, validation and saving functions. In other words, it is a data manager for all structures used in the library. Also, the logger is defined here.
- `config_example.yaml`: an example configuration file.
- `data/dataset.py`: location where data fetching functions are placed; currently not implemented.
- `ml/data_processing.py`: data processing for both training and inference; processing parameters are also returned (i.e., processing pipeline).
- `ml/model.py`: the machine learning model is defined here, along with training, inference and evaluation functions.
- `census_library.py`: high level library interfaces that use all previously mentioned functionalities; the users should use the interfaces defined in this file, which comprise all necessary steps to train and use the model:
  - Load the dataset
  - Basic cleaning and pre-processing
  - Segregation of the dataset: train/test
  - Data processing
  - Model training with hyper parameter tuning
  - Model and processing pipeline persisting
  - Inference

To install the package:

```bash
pip install -r requirements.txt
pip install .
```

The file `census_library.py` contains a usage example in its docstring, reproduced here:

```python
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
```