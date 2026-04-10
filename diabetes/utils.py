# This Script contains all functions that are used more than once across scripts in the model in the Diabetes Prediction Model, 
# such as feature engineering functions, data preprocessing functions
# Imported Libararies

import pandas as pd
import numpy as np

# A MinMax function for The engineered feature risk_score 

def minmax(x):
    if x.max() == x.min():
        return 0.5
    else:
        return (x - x.min()) / (x.max() - x.min())

# A function to engineer features for the dataset, this is used in both the cleaning and prediction scripts of the model

def engineer_features(df): 

    # Lipid Ratios
    df['chol_hdl_ratio'] = (df['chol'] / df['hdl']).round(4)
    df['ldl_hdl_ratio'] = (df['ldl'] / df['hdl']).round(4)
    df['tg_hdl_ratio'] = (df['tg'] / df['hdl']).round(4)
    df['non_hdl_chol'] = (df['chol'] - df['hdl']).round(4)

    # Kidney Functions
    df['urea_cr_ratio'] = ((df['urea'] * 1000) / df['cr'].replace(0, np.nan)).round(4)
    df['cr_elevated'] = (df['cr'] > 120).astype('uint8')
    
    _cr_mgdl = df['cr'] / 88.4
    _gender_factor = df['gender'].map({'f': 0.742, 'm': 0.9}).fillna(1.0)
    df['egfr'] = (186 * (_cr_mgdl ** -1.154) * (df['age'] ** -0.203) * _gender_factor).round(1)

    df['Obese'] = (df['bmi'] >= 30).astype('uint8') # Obesity Flag

    df['risk_score'] = (
    0.35 * minmax(df['bmi'])   +
    0.25 * minmax(df['age'])   +
    0.20 * minmax(df['tg'])    +
    0.20 * minmax(df['chol'])
    ).round(4)

    df['metsyn_score'] = ( # Metabolic Syndrome is a cluster of conditions that combine multiple signals into one score
    (df['bmi']   >= 30.0).astype(int) +
    (df['tg']    >= 1.70).astype(int) +
    (df['hdl']   <  1.00).astype(int) 
    )
    df['metsyn_flag'] = (df['metsyn_score'] >= 2).astype('uint8') # Binary Metabolic Syndrome Flag

    return df