# This is a Data Preprocessing Script for a Stroke Prediction Model
# Imported Libraries Libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

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

# Turn Cleaned Data into Train and Test CSV Files

train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

train.to_csv("stroke/Processed/train/train.csv", index=False)
test.to_csv("stroke/Processed/test/test.csv", index=False)

# Use SimpleImputer respectively to fill missing values 

def imputation(X_train, X_test):
    bmi_imputer = SimpleImputer(strategy='mean')
    smoking_imputer = SimpleImputer(strategy='most_frequent')

    bmi_imputer.fit(X_train[['bmi']])
    smoking_imputer.fit(X_train[['smoking_status']])

    X_train['bmi'] = bmi_imputer.transform(X_train[['bmi']]).ravel()
    X_test['bmi'] = bmi_imputer.transform(X_test[['bmi']]).ravel()

    X_train['smoking_status'] = smoking_imputer.transform(X_train[['smoking_status']])
    X_test['smoking_status'] = smoking_imputer.transform(X_test[['smoking_status']])

    return X_train, X_test

# TODO: Engineer new features

# Encode Categorical Variables

def encode(X_train, X_test):

    binary_cols = [col for col in X_train.columns if X_train[col].nunique() == 2]
    categorical_cols = [col for col in X_train.columns if X_train[col].nunique() > 2]
    categorical_cols = X.select_dtypes(include="object")

    for col in binary_cols:
        encoder = LabelEncoder()
        X_train[col] = encoder.fit_transform(X_train[col])
        X_test[col] = encoder.transform(X_test[col])

    X_train = pd.get_dummies(X_train, columns=[categorical_cols], drop_first=True)
    X_test = pd.get_dummies(X_test, columns=[categorical_cols], drop_first=True)

    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0) # Align the train and test sets to have the same columns after one-hot encoding

    return X_train, X_test

# Scale Numerical Features

def scale(X_train, X_test):
    numerical_features = X.select_dtypes(include=['float64']).columns

    scaler = StandardScaler()
    scaler.fit(X_train[numerical_features])

    X_train[numerical_features] = scaler.transform(X_train[numerical_features])
    X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    return X_train, X_test

# Beacuse Class Imbalance is very High, We will use SMOTE
def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    return X_train, y_train