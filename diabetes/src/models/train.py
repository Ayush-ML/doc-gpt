# This is the Training Script for The diabetes Prediction Model
# This is used to run a final training script on the model which is fitted with the best hyper parameters
# The model is then tested on the test set and the metrics are saved in the reports directory
# These Hyperparameters are obtained used Bayesian Optimization with Optuna
# Imported Libraries

import pandas as pd
from diabetes.src.models.build import create_pipe
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
from sklearn.calibration import CalibrationDisplay
import json, mlflow, os
from sklearn import set_config
from xgboost.callback import EarlyStopping

# Load Train, Test and Validation Data

X_train = pd.read_csv(r'diabetes\data\clean\train\X_train.csv')
y_train = pd.read_csv(r'diabetes\data\clean\train\y_train.csv').squeeze()
X_test = pd.read_csv(r'diabetes\data\clean\test\X_test.csv')
y_test = pd.read_csv(r'diabetes\data\clean\test\y_test.csv').squeeze()
X_val = pd.read_csv(r'diabetes\data\clean\val\X_val.csv')
y_val = pd.read_csv(r'diabetes\data\clean\val\y_val.csv').squeeze()

mlflow.sklearn.autolog() # Enable MLflow Autologging for scikit-learn, which automatically logs parameters, metrics, and models during training.
set_config(enable_metadata_routing=True) # Enable Metadata Routing , which helps to customize your models params further
mlflow.set_tracking_uri(r"sqlite:///diabetes/reports/training_logs/mlflow.db")
mlflow.set_experiment("model_training") # Set an Experiment before starting run to prevent issues

# Load Model Pipeline

pipeline = create_pipe()

# Load Tuned Hyperparameters

with open(r"diabetes\models\tuned_hyperparams.json", 'r') as f:
    best_params = json.load(f)

# Define my XGBoost Callbacks
# Here we will use Early Stopping to stop training when the model stops improving on the validation set

earlystopping = EarlyStopping(rounds=50, metric_name='auc', data_name='validation_0', save_best=True) # This will monitor the AUC metric on the validation set and stop training if it doesn't improve for 50 rounds, also it will save the best model during training

# Set the Best Hyperparameters to the Model

pipeline.named_steps['model'].set_params(**best_params)
pipeline.named_steps['model'].set_params(callbacks=[earlystopping]) # Enable Callbacks in the model, which is necessary for Optimal and Efficent Training
pipeline.named_steps['model'].set_fit_request(eval_set=True) # Enable eval_set in the model, which is important for the model to not guess one class and get good metrics with learning anything

X_val_transformed = pipeline[:-1].fit_transform(X_train, y_train) # Fit the Preprocessor and Selector on the training data and transform the validation data
X_val = pipeline[:-1].transform(X_val)                 # Transform the validation data with the fitted preprocessor and selector, this is necessary to get the correct feature selection and transformation for the validation set 

with mlflow.start_run(run_name="model_training"):
    pipeline.fit(X_train, y_train, eval_set=[(X_val, y_val)]) # Fit the model on the training data with eval set and calls backs
    mlflow.sklearn.save_model(sk_model=pipeline, path=r"diabetes\models\final_model") # Save the final model as an artifact in MLflow

y_pred = pipeline.predict(X_test) # Predict on the test set
y_proba = pipeline.predict_proba(X_test)[:, 1] # Get the predicted probabilities for the positive class

# Calculate Metrics

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
avg_precision = average_precision_score(y_test, y_proba)
classification_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Create and Save Plots

roc = RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
mlflow.log_figure(roc.figure_, "garphed_metrics/roc_curve.png") # Log the ROC curve as an artifact in MLflow

conf = ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test, cmap='Blues')
mlflow.log_figure(conf.figure_, "garphed_metrics/confusion_matrix.png") # Log the confusion matrix as an artifact in MLflow

cal = CalibrationDisplay.from_estimator(pipeline, X_test, y_test, n_bins=10)
mlflow.log_figure(cal.figure_, "garphed_metrics/calibration_curve.png") # Log the calibration curve as an artifact in MLflow

per = PrecisionRecallDisplay.from_estimator(pipeline, X_test, y_test)
mlflow.log_figure(per.figure_, "garphed_metrics/precision_recall_curve.png") # Log the precision-recall curve as an artifact in MLflow

# Save Metrics to a txt file

with open(r"diabetes\reports\training_logs\metrics.txt", 'w') as f:
    f.write(f"Final Model Performance Metrics on Test Set\n")
    f.write(f"=========================================\n\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"MCC: {mcc}\n")
    f.write(f"ROC AUC: {roc_auc}\n")
    f.write(f"Average Precision: {avg_precision}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(conf_matrix))

mlflow.log_artifact(r"diabetes\reports\training_logs\metrics.txt") # Log the metrics file as an artifact in MLflow
