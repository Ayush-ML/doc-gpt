# This script contains the prediction function for the Diabetes Prediction Model
# It used the trained model pipeline to make predictions on new data
# Imported Libraries

import mlflow
import pandas as pd
import numpy as np
from diabetes.utils import engineer_features
from diabetes.config import MODEL

pipeline = mlflow.sklearn.load_model(MODEL)

def predict_diabetes(input_data: dict) -> dict:
    # Convert input data to a pandas DataFrame
    df = pd.DataFrame([input_data])
    
    # Engineer features
    df = engineer_features(df)
    
    # Make prediction
    prediction = pipeline.predict(df)
    probability = pipeline.predict_proba(df)[:, 1]
    
    return {
        "prediction": "Diabetic" if prediction[0] == 1 else "Non-Diabetic",
        "confidence": float(probability[0])
    }