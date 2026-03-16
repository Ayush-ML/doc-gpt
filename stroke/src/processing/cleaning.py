# This is a Data Cleaning Script for a Stroke Prediction Model
# Imported Libraries Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

np.random.seed(42) # Set Random Seed for Reproducibility

# Load The Dataset

data = pd.read_csv(r'stroke\data\raw_data\stroke-dataset.csv')
data.columns = data.columns.str.strip()  # Strip whitespace from column names
data = data.drop(columns='id', axis=1)  # Drop the 'id' column as it may harm the model

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

# Correct The Data types of columns

data['bmi'] = data['bmi'].astype(float)

# Split Dataset into X and y and further into train and test

X = data.drop('stroke', axis=1)  # Features
y = data['stroke']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)  # Stratify to maintain class distribution in train and test sets

# Turn Cleaned Data into Train and Test CSV Files

X_train.to_csv(r"stroke\data\processed_data\train\X_train.csv", index=False)
y_train.to_csv(r"stroke\data\processed_data\train\y_train.csv", index=False)

X_test.to_csv(r"stroke\data\processed_data\test\X_test.csv", index=False)
y_test.to_csv(r"stroke\data\processed_data\test\y_test.csv", index=False)