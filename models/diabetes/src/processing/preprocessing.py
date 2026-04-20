# This is a Preprocessing Script for the Diabetes Prediction model that focuses on Things only done to Training Data
# Examples are:
# -- Scaling
# -- Encoding
# -- Feature Engineering(Only features fit on training data)
# -- Feature Selection

# Imported Libraraies

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from diabetes.config import (TRANSFORMER_METHOD, FEATURE_CREATION_INTERACTION, FEATURE_CREATION_BIAS)

def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer: # This function is used to create a Preprocessor that can be used in a Pipeline with the model.
    # It returns a ColumnTransformer that applies the appropriate transformations to numerical and categorical features.

    # Identify Numerical and Categorical Columns

    numerical = X.select_dtypes(include=['float64', 'int64']).columns
    categorical = X.select_dtypes(include=['object']).columns

    # Create Preprocessing Pipelines for Numerical and Categorical Columns

    numerical_pipeline = Pipeline(steps=[
        ('transformer', PowerTransformer(method=TRANSFORMER_METHOD)), # This is used to make the distribution of numerical features more normal
        ('feature_creation', PolynomialFeatures(interaction_only=FEATURE_CREATION_INTERACTION, include_bias=FEATURE_CREATION_BIAS)), # This is used to create interaction features between numerical features, which can help capture complex relationships in the data.
    ])
    categorical_pipeline = Pipeline(steps=[
        ('encoder', OrdinalEncoder()) # This is used to encode categorical features into a format that can be used by machine learning models.
    ])

    # Create and Return a ColumnTransformer that applies all pipelines to their appropriate columns

    return ColumnTransformer(transformers=[
        ('numerical', numerical_pipeline, numerical),
        ('categorical', categorical_pipeline, categorical)
    ])