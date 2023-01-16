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

import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X,
    categorical_features=[],
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
    X : pd.DataFrame
        Dataframe containing the features and label.
    categorical_features: list[str]
        List containing the names of the categorical features (default=[]).
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

    if label is not None:
        y = X[label]
        X = X.drop([label], axis=1)
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        X_categorical = encoder.fit_transform(X_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        X_categorical = encoder.transform(X_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb
