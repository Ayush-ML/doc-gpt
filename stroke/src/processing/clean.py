# This is a Data Cleaning Script for Diabetes Dataset
# It handles things like :
# -- Handling Missing Values
# -- Removing Outliers
# -- Fixing Skew
# -- Feature Engineering (Some)
# -- Fixing Data Types
# -- Standardizing Text
# -- Splitting into Train, Test and Validation sets

# Imported Libararies 

import pandas as pd
import numpy as np
from scipy.stats import boxcox
from stroke.config import RAW, BINARY_COLS, HIGHLY_RIGHT_SKEWED, MODERATELY_RIGHT_SKEWED, RANDOM_STATE, X_TEST, X_TRAIN, X_VAL, Y_TEST, Y_TRAIN, Y_VAL, VAL_SIZE, TEST_SIZE
from stroke.utils import engineer_features
from sklearn.model_selection import train_test_split

np.random.seed(RANDOM_STATE) # Set Seed for Reproducability

df = pd.read_csv(RAW) # Load Dataset

df.columns = df.columns.str.strip().str.lower() # Remove Trailing Whitespaces

# Fix Data Types

df['bmi'] = pd.to_numeric(df['bmi'].replace('N/A', np.nan), errors='coerce') # Also replace N/A string with Nan Value
for cols in BINARY_COLS:
    df[cols] = df[cols].astype('uint8') # Convert Binary Cols to uint8 to save memory
df['age'] = df['age'].astype('int64') # Convert Age to Int

df = df.drop(columns=['id']) # Drop Unnecesary Columns

# Define Numerical and Categorical Columns

num_cols = df.select_dtypes(include=np.number, exclude=np.uint8).columns.to_list()
cat_cols = df.select_dtypes(include=['object']).columns.to_list()

# Standerdize Text Data

for cols in cat_cols:
    df[cols] = df[cols].str.strip().str.lower()

# Remove All Duplicate Entries (0 duplicates were detected in analysis but removal of id column may have created duplicates so just in case)

df = df.drop_duplicates()

# Remove physically impossible values

df = df[(df['age'] > 0) & (df['age'] <= 100)]
df = df[(df['bmi'] >= 10) & (df['bmi'] <= 80)]
df = df[(df['avg_glucose_level'] >= 50) & (df['avg_glucose_level'] <= 500)]

# Fix Skewness and Outliers

for cols in HIGHLY_RIGHT_SKEWED:
    df[cols], _ = boxcox(df[cols] + 1) # Use Scipy's BoxCox for columns with heavy right skew
for cols in MODERATELY_RIGHT_SKEWED:
    df[cols] = np.sqrt(df[cols]) # Use Square Root transformation for columns with moderate right skew

# Engineer New Features

df = engineer_features(df=df)

# Define Input's and Target Feature

X = df.drop(columns=['stroke'])
y = df['stroke']


# Divide into Train, Test and Val splits

X_temp, X_test, y_temp, y_test = train_test_split(X, y ,test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp)

X_train.to_csv(X_TRAIN, index=False)
X_test.to_csv(X_TEST, index=False)
y_train.to_csv(Y_TRAIN, index=False)
y_test.to_csv(Y_TEST, index=False)
X_val.to_csv(X_VAL, index=False)
y_val.to_csv(Y_VAL, index=False)