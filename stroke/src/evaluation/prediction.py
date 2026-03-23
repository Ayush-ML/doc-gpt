# This is a Prediction Script For the Model
# Required Imports

import joblib
import pandas as pd
import numpy as np

# Load the Model

model = joblib.load(r'stroke\src\models\final_model.joblib')

# Define Function for Prediction

def predict_stroke(patient_data: dict) -> dict:
    patient_data = pd.DataFrame([patient_data]) # Convert from Dictionary to Dataframe
    
    prediction = model.predict(patient_data)
    probability = model.predict_proba(patient_data)

    return {
        "Model Prediction": prediction,
        "Chance of Stroke": probability
    }

