# This Script contains Helper Functions
# These Functions are used more than once across the Model
# Imported Libraraies

import pandas as pd
import numpy as np

# Engineer New Features (Used in both Cleaning Data and Prediction)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Discretionized Features

    df['age_group'] = pd.cut(
        df['age'],
        bins=[0, 18, 40, 60, 100],
        labels=['child', 'young_adult', 'middle_aged', 'senior']
    )

    df['bmi_category'] = pd.cut(
        df['bmi'],
        bins=[0, 18.5, 24.9, 29.9, 39.9, np.inf],
        labels=['underweight', 'normal', 'overweight', 'obese', 'morbidly_obese']
    )

    df['glucose_category'] = pd.cut(
        df['avg_glucose_level'],
        bins=[0, 70, 99, 125, np.inf],
        labels=['low', 'normal', 'prediabetic', 'diabetic']
    )

    # Cardio Vascular Risk Score, Signal Combination

    df['cardiovascular_risk'] = (
        df['hypertension'] +
        df['heart_disease'] +
        (df['avg_glucose_level'] > 125).astype(int) +  
        (df['bmi'] > 30).astype(int) +                  
        (df['age'] > 60).astype(int)                     
    )

    # Interaction Type Features

    df['age_hypertension'] = df['age'] * df['hypertension']

    df['age_heart_disease'] = df['age'] * df['heart_disease']

    df['age_glucose'] = df['age'] * df['avg_glucose_level']

    df['bmi_glucose'] = df['bmi'] * df['avg_glucose_level']

    # Lifestyle Type Features
   
    df['is_smoker'] = (df['smoking_status'] == 'smokes').astype('uint8')

    df['ever_smoked'] = (
        df['smoking_status'].isin(['smokes', 'formerly smoked'])
    ).astype('uint8')
    
    df['is_employed'] = (
        df['work_type'].isin(['Private', 'Self-employed', 'Govt_job'])
    ).astype('uint8')

    df['lifestyle_risk'] = (
        df['is_smoker'] +
        (df['bmi'] > 30).astype(int) +
        (df['avg_glucose_level'] > 125).astype(int)
    )

    df['married_adult'] = (
        (df['ever_married'] == 'Yes') & (df['age'] >= 18)
    ).astype('uint8')

    return df

