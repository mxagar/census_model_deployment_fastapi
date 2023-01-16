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

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.

# Add code to load in the data.

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.

# Train and save a model.
