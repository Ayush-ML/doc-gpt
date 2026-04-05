# This is a Data Cleaning Script for Diabetes Dataset
# Import Libararies 

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox, mstats

df = pd.read_csv(r'diabetes\data\raw\Dataset of Diabetes .csv', on_bad_lines='skip') # Load Dataset

df.columns = df.columns.str.strip().str.lower()  # Remove leading and trailing whitespace from column names and convert to lower case
df = df.drop(columns=['id', 'no_pation'])  # Drop unnecessary columns

categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    df[column] = df[column].str.strip().str.lower()  # Remove leading and trailing whitespace and Convert To lower case in categorical columns

# Did not remove duplicates because of Dataset Size and the fact that they may represent different patients with similar medical records, which is common in medical datasets.

df['cr'] = df['cr'].astype('float64') # Convert 'cr' column to Float64

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
for column in ['cr', 'hdl', 'vldl']:
    df[column] = np.log1p(df[column]) # log1p is used to handle zero values
# Use Box-Cox Transform for Highly Right Skewed Columns
for column in ['urea', 'tg']:
    df[column], _ = boxcox(df[column] + 1)
# Use Square Root Transform for Moderately Right Skewed Columns
df['ldl'] = np.sqrt(df['ldl'])

# Remove Leftover Outliers of Extremely Right Skewed Columns using IQR Method

for column in ['cr', 'hdl', 'vldl']:
    upper = df[column].quantile(0.99)
    lower = df[column].quantile(0.01)
    outliers = ((df[column] > upper) | (df[column] < lower)).sum()
    if outliers > 0:
        df[column] = df[column].clip(lower=lower, upper=upper)  # Cap outliers to the 1st and 99th percentiles

# Split Dataset into X and y and further into train and test

X = df.drop('class', axis=1)  # Features
y = df['class']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # Stratify to maintain class distribution in train and test sets

# Turn Cleaned Data into Train and Test CSV Files

X_train.to_csv(r"diabetes\data\clean\train\X_train.csv", index=False)
X_test.to_csv(r"diabetes\data\clean\test\X_test.csv", index=False)
y_train.to_csv(r"diabetes\data\clean\train\y_train.csv", index=False)
y_test.to_csv(r"diabetes\data\clean\test\y_test.csv", index=False)
