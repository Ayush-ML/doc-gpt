# This is a Preprocessing Script for the Diabetes Prediction model that focuses on Things only done to Training Data
# Examples are:
# -- Scaling
# -- Encoding
# -- Feature Engineering(Only features fit on training data)
# -- Feature Selection
# -- Resampling

# Imported Libraraies

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer, PolynomialFeatures, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load The Cleaned data that is split into train an test

X_test = pd.read_csv(r'diabetes\data\clean\test\X_test.csv')
y_test = pd.read_csv(r'diabetes\data\clean\test\y_test.csv').squeeze() # Squeeze is necessary to Cinvert a Dataframe to a Series, which is the expected format for y in scikit-learn models.

X_train = pd.read_csv(r'diabetes\data\clean\train\X_train.csv')
y_train = pd.read_csv(r'diabetes\data\clean\train\y_train.csv').squeeze()

def create_preprocessor(X: pd.DataFrame, y: pd.Series) -> ColumnTransformer: # This function is used to create a Preprocessor that can be used in a Pipeline with the model.
    # It returns a ColumnTransformer that applies the appropriate transformations to numerical and categorical features.

    # Identify Numerical and Categorical Columns

    numerical = X.select_dtypes(include=['float64', 'int64']).columns
    categorical = X.select_dtypes(include=['object']).columns
    target = y.name

    # Create Preprocessing Pipelines for Numerical and Categorical Columns

    numerical_pipeline = Pipeline(steps=[
        ('transformer', PowerTransformer(method='yeo-johnson', standardize=True)), # This is used to make the distribution of numerical features more normal
        ('feature_creation', PolynomialFeatures(interaction_only=True, include_bias=False)), # This is used to create interaction features between numerical features, which can help capture complex relationships in the data.
    ])
    categorical_pipeline = Pipeline(steps=[
        ('encoder', OrdinalEncoder(handle_unknown='ignore', sparse_output=False)) # This is used to encode categorical features into a format that can be used by machine learning models.
    ])
    target_pipeline = Pipeline(steps=[
        ('encoder', LabelEncoder()) # This is used to encode the target variable, which is necessary for classification models.
    ])

    # Create and Return a ColumnTransformer that applies all pipelines to their appropriate columns

    return ColumnTransformer(transformers=[
        ('numerical', numerical_pipeline, numerical),
        ('categorical', categorical_pipeline, categorical),
        ('target', target_pipeline, target)
    ])