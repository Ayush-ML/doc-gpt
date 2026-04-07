# This is a Data Cleaning Script for Diabetes Dataset which includes things that are performed on the entire dataset before splitting into train and test sets.
# Examples are:
# -- Handling Missing Values
# -- Handling Outliers
# -- Removing Duplicates
# -- Fixing Data Types
# -- Fixing Inconsistent Categorical Values
# -- Feature Engineering that is done on the entire dataset (e.g. creating new features based on existing ones)
# Import Libararies 

from shutil import which

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox

df = pd.read_csv(r'diabetes\data\raw\Dataset of Diabetes .csv', on_bad_lines='skip') # Load Dataset

df.columns = df.columns.str.strip().str.lower()  # Remove leading and trailing whitespace from column names and convert to lower case
df = df.drop(columns=['id', 'no_pation'])  # Drop unnecessary columns

categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    df[column] = df[column].str.strip().str.lower()  # Remove leading and trailing whitespace and Convert To lower case in categorical columns

# Did not remove duplicates because of Dataset Size and the fact that they may represent different patients with similar medical records, which is common in medical datasets.

# All Data Types are correct, so no need to fix data types

# One Row has Cholesterol (chol) value of 0, which is not possible, so we will treat it as a missing value and impute it with the median value of the column

chol_median = df['chol'].median()
df['chol'] = df['chol'].replace(0, chol_median)  

# hbA1c has values of 0.9, 2.0, 3.0, 3.7
# These values are physiologically impossible for living adults and are likely data entry errors.
# To be safe, we will keep values that are above 3.0 and drop rows with hbA1c values of 0.9, 2.0, which are never before seen

df = df[df['hba1c'] > 3.0]  # Keep rows where hba1c > 3.0, drop invalid rows

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

# Engineering New Features (Only Self Made Features that are created directly and do not use Sklearn tools are made in the cleaning script
                        #   Features that are fit only in training data are made in preprocessing script)

df['chol_hdl_ratio'] = (df['chol'] / df['hdl']).round(4) # These are Lipid Ratios that are commonly used in medical practice to assess cardiovascular risk,
df['ldl_hdl_ratio'] = (df['ldl'] / df['hdl']).round(4)   # which is a common comorbidity in diabetes patients. 
df['tg_hdl_ratio'] = (df['tg'] / df['hdl']).round(4)
df['non_hdl_chol'] = (df['chol'] - df['hdl']).round(4)

df['urea_cr_ratio'] = ((df['urea'] * 1000) / df['cr'].replace(0, np.nan)).round(4) # These are Kidney Functions
df['cr_elevated'] = (df['cr'] > 120).astype('uint8') # Diabetic nephropathy is a major complication.
df['cr_elevated'] = (df['cr'] > 120).astype('uint8') # Elevated creatinine levels can indicate impaired kidney function, which is a common complication of diabetes.
_cr_mgdl = df['cr'] / 88.4
_gender_factor = df['gender'].map({'f': 0.742, 'm': 0.9}).fillna(1.0)
df['egfr'] = (186 * (_cr_mgdl ** -1.154) * (df['age'] ** -0.203) * _gender_factor).round(1)

df['hba1c_diabetic'] = (df['hba1c'] >= 6.5).astype('uint8') # HbA1c thresholds for high or low risk patients, this is a MAJOR factor in predicting diabetes
df['hba1c_controlled'] = ((df['hba1c'] >= 6.5) & (df['hba1c'] < 7.0)).astype('uint8') # Controlled or Uncontrolled Diabetes is an important factor in predicting complications and outcomes in diabetes patients

df['Obese'] = (df['bmi'] >= 30).astype('uint8') # Binary Obesesity Flag based on BMI, important factor for diabetes risk

df['metsyn_score'] = ( # Metabolic Syndrome is a cluster of conditions that combine multiple signals into one score
    (df['bmi']   >= 30.0).astype(int) +
    (df['tg']    >= 1.70).astype(int) +
    (df['hdl']   <  1.00).astype(int) +
    (df['hba1c'] >= 6.50).astype(int)
)
df['metsyn_flag'] = (df['metsyn_score'] > 2).astype('uint8') # Binary Metabolic Syndrome Flag

def minmax(x): # A MinMax function to combine all features into a final Risk Score Features between 0 and 1
    return (x - x.min()) / (x.max() - x.min())
df['risk_score'] = (
    0.35 * minmax(df['hba1c']) +
    0.25 * minmax(df['bmi'])   +
    0.15 * minmax(df['age'])   +
    0.15 * minmax(df['tg'])    +
    0.10 * minmax(df['chol'])
).round(4)

# Split Dataset into X and y and further into train and test

X = df.drop('class', axis=1)  # Features
y = df['class']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # Stratify to maintain class distribution in train and test sets

# Turn Cleaned Data into Train and Test CSV Files

X_train.to_csv(r"diabetes\data\clean\train\X_train.csv", index=False)
X_test.to_csv(r"diabetes\data\clean\test\X_test.csv", index=False)
y_train.to_csv(r"diabetes\data\clean\train\y_train.csv", index=False)
y_test.to_csv(r"diabetes\data\clean\test\y_test.csv", index=False)
