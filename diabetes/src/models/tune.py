# This is a model Tuning Script for the Diabetes Prediction Model
# This is used to tune the hyperparameters of the model using Optuna's Bayesian Optimization
# Import Libraraies

from diabetes.src.models.build import create_pipe
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from optuna import create_study, Trial
from optuna.samplers import TPESampler
from optuna.visualization import plot_parallel_coordinate, plot_param_importances, plot_slice, plot_optimization_history
from optuna.integration import XGBoostPruningCallback
import xgboost as xgb
from xgboost import DMatrix
import json
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Load Training Data

X_train = pd.read_csv(r'diabetes\data\clean\train\X_train.csv')
y_train = pd.read_csv(r'diabetes\data\clean\train\y_train.csv').squeeze()
X_val = pd.read_csv(r'diabetes\data\clean\val\X_val.csv')
y_val = pd.read_csv(r'diabetes\data\clean\val\y_val.csv').squeeze()

# Setup Logging Directory
logs_dir = r'diabetes\reports\tuning_logs'

# Configure Logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(logs_dir, f'tuning_{timestamp}.log')

logger = logging.getLogger('optuna')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Define The Objective Function that Optuna will Optimize

def objective(trial: Trial) -> float:
    # Encode categorical columns (gender) to numeric before creating DMatrix
    
    le = LabelEncoder()
    X_train['gender'] = le.fit_transform(X_train['gender'])
    X_val['gender'] = le.transform(X_val['gender'])

    dtrain = DMatrix(X_train, label=y_train) # Create DMatrix for XGBoost, which is a more efficient data structure for training
    dval = DMatrix(X_val, label=y_val)
    
    # Define the Hyperparameters to Tune and their Search Space
    param_grid = {
        'max_depth': trial.suggest_int('max_depth', 4, 8),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 3),
        'gamma': trial.suggest_float('gamma', 0, 2, step=0.5),
        'lambda': trial.suggest_float('lambda', 0.5, 2, step=0.5),
        'alpha': trial.suggest_float('alpha', 0, 1, step=0.1),
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.7, 1.0, step=0.1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0, step=0.1),
        'objective': 'binary:logistic',  
        'eval_metric': 'auc',  
        'random_state': 42,
    }
        
    # Setup pruning callback with correct metric name for binary
    callback = XGBoostPruningCallback(trial, 'validation-auc')
    
    # Train XGBoost model
    num_round = trial.suggest_int('n_estimators', 100, 300)
    bst = xgb.train(
        param_grid, 
        dtrain, 
        num_boost_round=num_round,
        evals=[(dval, 'validation')], 
        callbacks=[callback],
        verbose_eval=False
    )
    
    # Get predictions and calculate AUC 
    y_pred = bst.predict(dval)  # Returns probabilities [0, 1]
    auc_score = roc_auc_score(y_val, y_pred)
    return auc_score

if __name__ == "__main__":
    study = create_study(direction='maximize', sampler=TPESampler(seed=42), study_name='xgb_tuning') # Create Optuna Study
    study.optimize(
        objective,
        n_trials=50,
        show_progress_bar=True,
        gc_after_trial=True
    )
        
    # Save Best Hyperparameters
    with open(r"diabetes\models\tuned_hyperparams.json", 'w') as f:
        json.dump(study.best_params, f, indent=4)
    
    # Save All Trials History as CSV
    csv_path = r"diabetes\reports\tuning_logs\trial_history"
    trials_df = study.trials_dataframe()
    trials_df.to_csv(csv_path, index=False)
    
    # Save All Trials as JSON
    trials_data = {
        'trials': [
            {
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state)
            }
            for trial in study.trials
        ]
    }
    with open(r"diabetes\reports\tuning_logs\trial_data.json", 'w') as f:
        json.dump(trials_data, f, indent=4)
    
    # Save Study Summary
    with open(r"diabetes\reports\tuning_logs\trial_summary.txt", 'w') as f:
        f.write(f"Optuna Tuning Study Summary\n")
        f.write(f"===========================\n\n")
        f.write(f"Study Name: {study.study_name}\n")
        f.write(f"Direction: maximize\n")
        f.write(f"Metric: ROC-AUC (binary classification)\n")
        f.write(f"Total Trials: {len(study.trials)}\n")
        f.write(f"Best Trial Number: {study.best_trial.number}\n")
        f.write(f"Best AUC Score: {study.best_value:.6f}\n\n")
        f.write(f"Best Hyperparameters:\n")
        for param, value in study.best_params.items():
            f.write(f"  {param}: {value}\n")
        f.write(f"\nTimestamp: {timestamp}\n")
    
    # Save Optuna Visualizations

    # Optimization progress over trials
    fig1 = plot_optimization_history(study)
    fig1.write_html(r"diabetes\reports\tuning_logs\optimization_history.html")

    # Which hyperparameters matter most?
    fig2 = plot_param_importances(study)
    fig2.write_html(r"diabetes\reports\tuning_logs\param_importances.html")

    # Relationship between each param and objective
    fig3 = plot_slice(study)
    fig3.write_html(r"diabetes\reports\tuning_logs\param_obj_relation.html")

    # High-dimensional parameter relationships
    fig4 = plot_parallel_coordinate(study)
    fig4.write_html(r"diabetes\reports\tuning_logs\high_dim_param_relations.html")
