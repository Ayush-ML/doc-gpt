# This script contains the prediction function for the Diabetes Prediction Model
# It used the trained model pipeline to make predictions on new data
# Imported Libraries

import mlflow
import pandas as pd
import numpy as np
from diabetes.utils import engineer_features
from diabetes.config import MODEL

pipeline = mlflow.sklearn.load_model(MODEL)

def predict_diabetes(input_data: dict, threshold: float=0.3) -> dict:
    """"""
    # Convert input data to a pandas DataFrame
    df = pd.DataFrame([input_data])
    
    # Engineer features
    df = engineer_features(df)
    
    # Make prediction using Custom Class 1 Threshold
    probability = pipeline.predict_proba(df)[0]
    if probability[1] > threshold:
        prediction = 1
    else:
        prediction = int(np.argmax(probability))
    label_map = {
        0: 'Non-Diabetic',
        1: 'Pre-Diabetic',
        2: 'Diabetic'
    }
    
    return {
        "prediction": label_map[prediction],
        "confidence of predicted class(in percentage)": float((probability[prediction] * 100).round(3)),
        'all probabilities for classes': {
            'Non-Diabetic': float((probability[0] * 100).round(3)),
            'Pre-Diabetic': float((probability[1] * 100).round(3)),
            'Diabetic': float((probability[2] * 100).round(3))
        }
    }