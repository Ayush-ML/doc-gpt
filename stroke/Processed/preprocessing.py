# This is a Data Preprocessing Script for a Stroke Prediction Model
# Imported Libraries Libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load The Dataset

data = pd.read_csv('stroke/Processed/healthcare-dataset-stroke-data.csv')
data.columns = data.columns.str.strip()  # Strip whitespace from column names
data = data.drop(columns='id', axis=1)  # Drop the 'id' column as it may harm the model

# Correct The Data types of columns

data['age'] = data['age'].astype(int)
data['bmi'] = data['bmi'].astype(float)

# Remove Very Rare Categories

data = data[data['gender'] != 'Other']  # Remove the 'Other' category from the 'gender' as only 1 entry exsists

# Drop Duplicate Values

data = data.drop_duplicates()  

# Handle missing Values
# This Dataset has missing values that are represented as a string N/A for BMI and Unknown Category for Smoking Status.
# We need to replace these values with Actual NaN values in order to use fill them later

missing_index = data[data['bmi'].str.contains('N/A')]['bmi'].index.to_list() # Find the indices of The values with 'N/A'
for index in missing_index:
    data.at[index, 'bmi'] = np.nan # For each index with 'N/A', replace it with NaN

missing_index = data[data['smoking_status'].str.contains('Unknown')]['smoking_status'].index.to_list()
for index in missing_index:
    data.at[index, 'smoking_status'] = np.nan

# Split Dataset into X and y and further into train and test

X = data.drop('stroke', axis=1)  # Features
y = data['stroke']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TODO: Engineer new features

# Turn Cleaned Data into Train and Test CSV Files

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

train.to_csv("stroke/Processed/train/train.csv", index=False)
test.to_csv("stroke/Processed/test/test.csv", index=False)

# Create Preprocessing Pipelines for Numerical, Binary, and Categorical Features
# Select Numerical, Binary , Categorical and Columns with Missing Values (likely only bmi and smoking_status)

numeric_features = X.select_dtypes(include=['float64']).columns
binary_cols = [col for col in X_train.columns if X_train[col].nunique() == 2]
categorical_cols = [col for col in X_train.columns if X_train[col].nunique() > 2]
categorical_cols = categorical_cols.select_dtypes(include="object")
missing_cols = [col for col in X_train.columns if X_train[col].isnull().any()]

# Create Pipelines for each type of feature

data_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
])
binary_transformer = Pipeline(steps=[
    ('encoder', LabelEncoder())
])
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder())
])
scaler = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[ # Create a ColumnTransformer to combine these pipelines together
    ('data', data_transformer, missing_cols),
    ('binary', binary_transformer, binary_cols),
    ('categorical', categorical_transformer, categorical_cols),
    ('scaler', scaler, numeric_features)
])



