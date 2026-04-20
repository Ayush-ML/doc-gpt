# This is the preporcessing script for the stroke prediction model
# This focuses mainly on creating a column transformer made up of pipelines
# This column transformer will serve as the preprocessing step in the sklearn pipeline
# The ColumnTransformer includes steps such as :
    # -- Scaling (using PowerTransformer)
    # -- Categroical Value Encoding (Seperate for 2 Unizue Values and greater than 2 Unique Values)
    # -- Feature Creation (using PolyNomialFeatures)
# Imported Libraries

import pandas as pd
import numpy as np
from models.stroke.config import VALUES_2, VALUES_OVER_2, INTERACTION_ONLY, BIAS
from sklearn.preprocessing import OrdinalEncoder, PowerTransformer, PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Function to create preprocessor using X (input features)

def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:

    num_cols = X.select_dtypes(include=np.number, exclude=np.uint8).columns.to_list()

    numerical_pipeline = Pipeline(steps=[
        ('poly', PolynomialFeatures(interaction_only=INTERACTION_ONLY, include_bias=BIAS)),
        ('transformer', PowerTransformer())
    ])
    binary_pipeline = Pipeline(steps=[
        ('encoder', OrdinalEncoder())
    ])
    categorical_pipeline = Pipeline(steps=[
        ('encoder', OneHotEncoder())
    ])

    return ColumnTransformer(transformers=[
        ('numerical', numerical_pipeline, num_cols),
        ('binary', binary_pipeline, VALUES_2),
        ('categorical', categorical_pipeline, VALUES_OVER_2)
    ])
    