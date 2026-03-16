# This is a Data Preprocessing Script for a Stroke Prediction Model
# Imported Libraries Libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

np.random.seed(42) # Set Random Seed for Reproducibility

# Load The Cleaned Dataset

X_train = pd.read_csv(r'stroke\data\processed_data\train\X_train.csv')
y_train = pd.read_csv(r'stroke\data\processed_data\train\y_train.csv')

X_test = pd.read_csv(r'stroke\data\processed_data\test\X_test.csv')
y_test = pd.read_csv(r'stroke\data\processed_data\test\y_test.csv')

# Create Preprocessing Pipelines for Numerical, Binary, and Categorical Features
# Select Numerical, Binary , Categorical and Columns with Missing Values (likely only bmi and smoking_status)

numeric_features = X_train.select_dtypes(include=['float64']).columns
binary_cols = X_train.select_dtypes(include="object").columns
binary_cols = [col for col in binary_cols if X_train[col].nunique() == 2]
categorical_cols = X_train.select_dtypes(include="object").columns
categorical_cols = [col for col in categorical_cols if X_train[col].nunique() > 2]

 # Create Pipelines for each type of feature

bmi_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])                                                   # Use 2 seperate Imputers with seperate strategies for the numerical BMI column and the categorical smoking_status column
smoking_status_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent'))
])
binary_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
numerical_transformer = Pipeline(steps=[ # Use PolyNomialFeatures to create interaction and a discretionizer to split numerical columns into multiple features
    ('poly', PolynomialFeatures(interaction_only=True, include_bias=False)),
    ('discretizer', KBinsDiscretizer(n_bins=4, encode='onehot-dense', strategy='quantile')),
])

preprocessor = ColumnTransformer(transformers=[ # Create a ColumnTransformer to combine these pipelines together
    ('bmi', bmi_transformer, ['bmi']),
    ('smoking_status', smoking_status_transformer, ['smoking_status']),
    ('numerical', numerical_transformer, numeric_features),
    ('binary', binary_transformer, binary_cols),
    ('categorical', categorical_transformer, categorical_cols),
], remainder='passthrough')