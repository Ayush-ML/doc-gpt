from diabetes.src.inference.pred import predict_diabetes
import pandas as pd

subtle_case = {
    'age': 58,
    'gender': 'm',
    'bmi': 33.5,
    'hba1c': 6.2,        # BELOW the 6.5 threshold
    'chol': 6.2,
    'tg': 2.8,           # High Triglycerides
    'hdl': 0.85,         # Low "Good" Cholesterol
    'ldl': 4.1,
    'vldl': 1.2,
    'urea': 5.5,
    'cr': 95.0,
}

print(predict_diabetes(input_data=subtle_case))
