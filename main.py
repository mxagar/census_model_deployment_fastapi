# Put the code for your API here.

import pandas as pd
import census_salary as cs

# Train, is not trained yet
model, processing_parameters, config, test_scores = cs.train_pipeline(config_filename='config.yaml')
print("Test scores: ")
print(test_scores)

# Load pipeline, if training performed in another execution/session
model, processing_parameters, config = cs.load_pipeline(config_filename='config.yaml')

# Get and check the data
df = pd.read_csv('./data/census.csv') # original training dataset: features & target
df, _ = cs.validate_data(df=df) # columns renamed, duplicates dropped, etc.
X = df.drop("salary", axis=1) # optional
X = X.iloc[:100, :] # we take a sample

# Predict salary (values already decoded)
pred = cs.predict(X, model, processing_parameters)
print("Prediction: ")
print(pred)
