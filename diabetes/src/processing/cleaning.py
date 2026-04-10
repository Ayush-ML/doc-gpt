# This is a Data Cleaning Script for Diabetes Dataset which includes things that are performed on the entire dataset before splitting into train and test sets.
# Examples are:
# -- Handling Missing Values
# -- Handling Outliers
# -- Removing Duplicates
# -- Fixing Data Types
# -- Fixing Inconsistent Categorical Values
# -- Feature Engineering that is done on the entire dataset (e.g. creating new features based on existing ones)
# Import Libararies 

from diabetes.config import (RAW_DATA, X_TEST, X_TRAIN, X_VAL, Y_TEST, Y_TRAIN, Y_VAL,
DROP_COLUMNS, EXTREMELY_RIGHT_SKEWED, HIGHLY_RIGHT_SKEWED,
UPPER, LOWER, RANDOM_STATE, TEST_SIZE, VAL_SIZE)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from diabetes.utils import engineer_features

df = pd.read_csv(RAW_DATA) # Load Dataset

df.columns = df.columns.str.strip().str.lower()  # Remove leading and trailing whitespace from column names and convert to lower case
df = df.drop(columns=DROP_COLUMNS)  # Drop unnecessary columns

categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    df[column] = df[column].str.strip().str.lower()  # Remove leading and trailing whitespace and Convert To lower case in categorical columns

df['class'] = df['class'].map({'n': 0, 'p': 1, 'y': 2})

# Removed Duplicates

df.drop_duplicates()

# All Data Types are correct, so no need to fix data types

# One Row has Cholesterol (chol) value of 0, which is not possible, so we will treat it as a missing value and impute it with the median value of the column

chol_median = df['chol'].median()
df['chol'] = df['chol'].replace(0, chol_median)  

# hbA1c has values of 0.9, 2.0, 3.0, 3.7
# These values are physiologically impossible for living adults and are likely data entry errors.
# To be safe, we will keep values that are above 3.0 and drop rows with hbA1c values of 0.9, 2.0, which are never before seen

# df = df[df['hba1c'] > 3.0]  # Keep rows where hba1c > 3.0, drop invalid rows

# Fix Outliers, this dataset has mainly Right Skewed Distributions
# So use a transformation based approach to treat outliers instead of dropping them
# Columns that are skewed and have outliers: 
    # --'cr' (Extremely right skewed)
    # -- age' (left skewed)
    # -- 'hdl' (Extremely right skewed)
    # -- 'ldl' (Moderately right skewed)
    # -- 'vldl' (Extremely right skewed)
    # -- 'urea' (Highly right skewed)
    # -- 'tg' (Highly right skewed)

# Use Log Transform for Extremely Right Skewed Columns
for column in EXTREMELY_RIGHT_SKEWED:
    df[column] = np.log1p(df[column]) # log1p is used to handle zero values
# Use Box-Cox Transform for Highly Right Skewed Columns
for column in HIGHLY_RIGHT_SKEWED:
    df[column], _ = boxcox(df[column] + 1)
# Use Square Root Transform for Moderately Right Skewed Columns
df['ldl'] = np.sqrt(df['ldl'])

# Remove Leftover Outliers of Extremely Right Skewed Columns using IQR Method

for column in EXTREMELY_RIGHT_SKEWED:
    upper = df[column].quantile(UPPER)
    lower = df[column].quantile(LOWER)
    outliers = ((df[column] > upper) | (df[column] < lower)).sum()
    if outliers > 0:
        df[column] = df[column].clip(lower=lower, upper=upper)  # Cap outliers to the 1st and 99th percentiles

# Engineering New Features (Only Self Made Features that are created directly and do not use Sklearn tools are made in the cleaning script
                        #   Features that are fit only in training data are made in preprocessing script)

df = engineer_features(df)

# Split Dataset into X and y and further into train and test

X = df.drop('class', axis=1)  # Features
y = df['class']  # Target variable

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)  # Stratify to maintain class distribution in train and test sets
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_temp)  # Stratify to maintain class distribution in train and validation sets (also use 0.1111 as val size beacuse 0.1111 * 0.9 = 0.1)

# Turn Cleaned Data into Train and Test CSV Files

X_train.to_csv(X_TRAIN, index=False)
X_test.to_csv(X_TEST, index=False)
y_train.to_csv(Y_TRAIN, index=False)
y_test.to_csv(Y_TEST, index=False)
X_val.to_csv(X_VAL, index=False)
y_val.to_csv(Y_VAL, index=False)