# This is a Data Preprocessing Script for a Stroke Prediction Model
# Imported Libraries Libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load The Cleaned Dataset

X_train = pd.read_csv(r'stroke\data\processed_data\train\X_train.csv')
y_train = pd.read_csv(r'stroke\data\processed_data\train\y_train.csv')

X_test = pd.read_csv(r'stroke\data\processed_data\test\X_test.csv')
y_test = pd.read_csv(r'stroke\data\processed_data\test\y_test.csv')

# Create Preprocessing Pipelines for Numerical, Binary, and Categorical Features
# Select Numerical, Binary , Categorical and Columns with Missing Values (likely only bmi and smoking_status)

numeric_features = X_train.select_dtypes(include=['float64']).columns
binary_cols = [col for col in X_train.columns if X_train[col].nunique() == 2]
categorical_cols = X_train.select_dtypes(include="object").columns
categorical_cols = [col for col in categorical_cols if X_train[col].nunique() > 2]
missing_cols = [col for col in X_train.columns if X_train[col].isnull().any()]

# Create Pipelines for each type of feature

data_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])
binary_transformer = Pipeline(steps=[
    ('encoder', OrdinalEncoder())
])
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder())
])
scaler = Pipeline(steps=[
    ('scaler', StandardScaler())
])

discretionizer = Pipeline(steps=[
    ('discretizer', KBinsDiscretizer(n_bins=4))
])

preprocessor = ColumnTransformer(transformers=[ # Create a ColumnTransformer to combine these pipelines together
    ('data', data_transformer, missing_cols),
    ('binary', binary_transformer, binary_cols),
    ('categorical', categorical_transformer, categorical_cols),
    ('scaler', scaler, numeric_features)
])



