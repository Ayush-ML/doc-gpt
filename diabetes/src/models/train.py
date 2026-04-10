# This is the Training Script for The diabetes Prediction Model
# This is used to run a final training script on the model which is fitted with the best hyper parameters
# The model is then tested on the test set and the metrics are saved in the reports directory
# These Hyperparameters are obtained used Bayesian Optimization with Optuna
# Imported Libraries

import pandas as pd
import numpy as np
from diabetes.src.models.build import create_pipe
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, average_precision_score, cohen_kappa_score, log_loss, balanced_accuracy_score
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
import json, mlflow
from diabetes.config import X_TEST, X_TRAIN, X_VAL, Y_TEST, Y_TRAIN, Y_VAL, HYPERPARAMS, EXPERIEMENT, METRICS, MODEL, TRACKING, ROUNDS, METRIC_NAME, N_SPLITS, AVERAGE, CALIBRATION
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

# Load Train, Test and Validation Data

X_train = pd.read_csv(X_TRAIN)
y_train = pd.read_csv(Y_TRAIN).squeeze()
X_test = pd.read_csv(X_TEST)
y_test = pd.read_csv(Y_TEST).squeeze()

mlflow.set_tracking_uri(TRACKING)
mlflow.sklearn.autolog() # Enable MLflow Autologging for scikit-learn, which automatically logs parameters, metrics, and models during training.

# Load Model Pipeline

pipeline = create_pipe()

# Load Tuned Hyperparameters

with open(HYPERPARAMS, 'r') as f:
    best_params = json.load(f)

# Set the Best Hyperparameters to the Model

pipeline.named_steps['model'].set_params(**best_params)

# Wrap Claibrated Classifier around the Pipeline to imporve Calibration

pipeline = CalibratedClassifierCV(estimator=pipeline, method='sigmoid', cv=StratifiedKFold(n_splits=N_SPLITS))

with mlflow.start_run(run_name=EXPERIEMENT):
    pipeline.fit(X_train, y_train) # Fit the model on the training data with eval set and calls backs
    mlflow.sklearn.save_model(sk_model=pipeline, path=MODEL) # Save the final model as an artifact in MLflow

y_pred = pipeline.predict(X_test) # Predict on the test set
y_proba = pipeline.predict_proba(X_test) # Get the predicted probabilities 

# Calculate Metrics

accuracy = accuracy_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average=AVERAGE)
recall = recall_score(y_test, y_pred, average=AVERAGE)
f1 = f1_score(y_test, y_pred, average=AVERAGE)
macro_f1 = f1_score(y_test, y_pred, average='macro')

kappa = cohen_kappa_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average=AVERAGE)
log_loss = log_loss(y_test, y_proba)

avg_precision = average_precision_score(y_test, y_proba, average=AVERAGE)
classification_report = classification_report(y_test, y_pred, target_names=['N', 'P', 'Y'])
conf_matrix = confusion_matrix(y_test, y_pred)

# Create and Save Plots

classes = [0, 1, 2]
class_names = ['N', 'P', 'Y']
y_test_bin = label_binarize(y_test, classes=classes) # Binarize Y_test

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, (class_name, ax) in enumerate(zip(class_names, axes)):
    CalibrationDisplay.from_predictions(
        y_test_bin[:, i],   # true binary labels for this class
        y_proba[:, i],      # predicted probabilities for this class
        n_bins=10,
        name=f'Class {class_name}',
        ax=ax
    )
    ax.set_title(f'Calibration — Class {class_name}')

plt.suptitle('Calibration Curves (One-vs-Rest)', fontsize=14)
plt.tight_layout()

mlflow.log_figure(fig, CALIBRATION)

# Save Metrics to a txt file

with open(METRICS, 'w') as f:
    f.write(f"Final Model Performance Metrics on Test Set\n")
    f.write(f"=========================================\n\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"MCC: {mcc}\n")
    f.write(f"Log Loss: {log_loss}\n")
    f.write(f"Cohen Kappa: {kappa}\n")
    f.write(f"Balanced accuracy: {balanced_accuracy}\n")
    f.write(f"Macro F1: {macro_f1}\n")
    f.write(f"ROC AUC: {roc_auc}\n")
    f.write(f"Average Precision: {avg_precision}\n\n")
    f.write("Classification Report:\n")
    f.write(classification_report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(conf_matrix))

mlflow.log_artifact(METRICS) # Log the metrics file as an artifact in MLflow
