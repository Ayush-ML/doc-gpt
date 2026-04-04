# This is a Data Cleaning Script for Diabetes Dataset
# Import Libararies 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load the dataset

df = pd.read_csv('diabetes/data/diabetes.csv')
df.columns = df.columns.str.strip()  # Remove leading and trailing whitespace from column names
for col in df.columns:
    df[col] = df[col].str.strip().str.lower() # Standardize Categorical Data by stripping whitespace, converting to lowercase
df.drop(columns=['clinical_notes']) # Drop Columns Unecessary for cleaning

# Analyze The Dataset

print(df.head()) # View First 5 rows of the dataset
print(df.info()) # Get information about the dataset
print(df.describe(include='all')) # Get statistical summary of the dataset
print(df.shape) # Get the shape of the dataset
print(df.dtypes) # Get data types of each column
print(df.select_dtypes(include=[np.number]).corr())
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    print(f"{column}: \n\t -- Unique values:{df[column].unique()} \n\t -- No. of Unique Values: {df[column].nunique()} \n\t -- Count of Each Unique Values: {df[column].value_counts().to_dict()}") # Get unique values, their counts and distributions in each column
for column in df.columns:
    missing_values = df[column].isnull()
    print(f"{column}: \n\t -- No. of Missing Values: {missing_values.sum()} \n\t -- Percentage of Missing Values: {missing_values.mean() * 100}%") # Get number and percentage of missing values in each column
# Use Graphs to Detect Anomalies and Identify Skew
for column in df.select_dtypes(include=[np.number]).columns:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram + KDE
    sns.histplot(df[column], bins=30, ax=axes[0], alpha=0.7, edgecolor='black', kde=True)
    axes[0].set_title(f'{column} - Distribution')
    
    # Q-Q Plot
    stats.probplot(df[column], dist="norm", plot=axes[1])
    axes[1].set_title(f'{column} - Q-Q Plot')
    
    plt.tight_layout()
    plt.savefig()
    plt.show()

assert df['age'].between(0, 120).all(), "Age column contains Too High or Negative values, which is invalid." # Check for Negative Values in Age Column
assert df['bmi'].between(10, 60).all(), "BMI column contains Too High or Too low values, which is invalid." # Check for Negative Values in BMI Column
assert df['hbA1c_level'].between(1, 20).all(), "HbA1c Level column contains Too High or Too low values, which is invalid." # Check for Negative Values in HbA1c Level Column
assert df['blood_glucose_level'].between(50, 500).all(), "Blood Glucose Level column contains Too High or Too low values, which is invalid." # Check for Negative Values in Blood Glucose Level Column