# This is the Final Training Script for the project
# Imported Libraries

import warnings
warnings.filterwarnings('ignore', category=FutureWarning) # Ignore Certain Warnings to reduce Terminal Clutter
warnings.filterwarnings('ignore', category=UserWarning)
from stroke.src.training.tune_model import pipeline
from sklearn.model_selection import StratifiedKFold, TunedThresholdClassifierCV
import pandas as pd
import joblib, json
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, recall_score, average_precision_score, f1_score, matthews_corrcoef
from sklearn.base import clone

np.random.seed(42) # Set Random Seed for Reproducibility

# Load The Preprocessed Data for Training

X_train = pd.read_csv(r'stroke\data\clean\train\X_train.csv')
X_train = pd.DataFrame(X_train, columns=X_train.columns) # Ensure X_train is a DataFrame with the correct column names
y_train = pd.read_csv(r'stroke\data\clean\train\y_train.csv').squeeze()

X_test = pd.read_csv(r'stroke\data\clean\test\X_test.csv')
X_test = pd.DataFrame(X_test, columns=X_test.columns) # Ensure X_test is a DataFrame with the correct column names
y_test = pd.read_csv(r'stroke\data\clean\test\y_test.csv').squeeze()

# Load the Tuned Hyperparameters

with open(r'stroke\src\models\tuned_hyperparameters.json', 'r') as f:
    tuned_hyperparameters = json.load(f)

# Load the Pipeline with the Tuned Hyperparameters

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Define StratifiedKFold for Cross Validation
pipeline = clone(pipeline) # Clone the pipeline to avoid any issues with fitting the same pipeline multiple times in case of re-runs
pipeline.set_params(**tuned_hyperparameters) 
model = TunedThresholdClassifierCV(estimator=pipeline, cv=5, scoring='f1', n_jobs=-1) # Create a TunedThresholdClassifierCV to find the optimal threshold for classification

print("Training Phase...")
model.fit(X_train, y_train) # Train the Model with the Tuned Hyperparameters

print(f"Best Threshold: {model.best_threshold_}") # Print the Best Threshold found by TunedThresholdClassifierCV

X_test.columns = X_test.columns.str.strip() # Ensure there are no leading/trailing spaces in column names of X_test to avoid issues during prediction
X_test = X_test.replace(r'^\s*$', np.nan, regex=True) # Replace any empty strings with NaN to ensure the model can handle missing values correctly during prediction
X_test = X_test.reindex(columns=X_train.columns, fill_value=0) # Align X_test columns with X_train to prevent feature mismatch errors


# View Metrics to score Performance of the Model on the Test Set

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Predicted Positives:", y_pred.sum()) # Helps to catch any issues with the model predicting all 0s or all 1s

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("ROC AUC Score:")
print(roc_auc_score(y_test, y_prob))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))

print("Recall Score:")
print(recall_score(y_test, y_pred))

print("Average Precision Score:")
print(average_precision_score(y_test, y_prob))

print("F1 Score:")
print(f1_score(y_test, y_pred))

print("Matthews Correlation Coefficient (MCC):")
print(matthews_corrcoef(y_test, y_pred))


# Save the Trained Model for Future Use

joblib.dump(model, r'stroke\src\models\final_model.joblib')