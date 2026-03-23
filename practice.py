from stroke.src.evaluation.prediction import predict_stroke

high_risk_patient = {
    'gender': 'Male',
    'age': 78.0,
    'hypertension': 1,
    'heart_disease': 1,
    'ever_married': 'Yes',
    'work_type': 'Private',
    'Residence_type': 'Urban',
    'avg_glucose_level': 219.84,
    'bmi': 30.5,
    'smoking_status': 'formerly smoked'
}

response = predict_stroke(high_risk_patient)
print(response)