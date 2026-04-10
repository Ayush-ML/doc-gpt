# This is a model Tuning Script for the Diabetes Prediction Model
# This is used to tune the hyperparameters of the model using Optuna's Bayesian Optimization
# Import Libraraies

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
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
from sklearn.model_selection import StratifiedKFold
from diabetes.config import X_VAL, X_TRAIN, Y_TRAIN, Y_VAL, LOGS, RANDOM_STATE, DIRECTION, HYPERPARAMS, TRIAL_DATA, TRIAL_HISTROY, TRIAL_PLOTS, TRIAL_SUMMARY, N_TRIALS, STUDY_NAME, METRIC_NAME, OBJECTIVE, NUM_CLASS, SCORING, N_SPLITS

# Load Training Data

X_train = pd.read_csv(X_TRAIN)
y_train = pd.read_csv(Y_TRAIN).squeeze()
X_val = pd.read_csv(X_VAL)
y_val = pd.read_csv(Y_VAL).squeeze()

X_train['gender'] = X_train['gender'].str.strip()
X_val['gender'] = X_val['gender'].str.strip()
X_train.columns = X_train.columns.str.strip()
X_val.columns = X_val.columns.str.strip()


# Setup Logging Directory

# Configure Logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(LOGS, f'tuning_{timestamp}.log')

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
    
    le = OrdinalEncoder(handle_unknown='error')
    X_train['gender'] = le.fit_transform(X_train[['gender']])
    X_val['gender'] = le.transform(X_val[['gender']])

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
        'objective': OBJECTIVE,  
        'eval_metric': METRIC_NAME,  
        'random_state': RANDOM_STATE,
        'num_class': NUM_CLASS
    }
        
    # Setup pruning callback with correct metric name for binary
    callback = XGBoostPruningCallback(trial, f'validation-{METRIC_NAME}')
    
    # Build XGBoost model

    model = xgb.XGBClassifier(**param_grid)
    
    # Calculate Cross Validation Score
    cross_val = cross_val_score(model, X_train, y_train, scoring=SCORING, cv=StratifiedKFold(n_splits=N_SPLITS)).mean()
    return cross_val

if __name__ == "__main__":
    sampler = TPESampler(seed=RANDOM_STATE)

    study = create_study(direction=DIRECTION, sampler=sampler, study_name=STUDY_NAME) # Create Optuna Study
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
        gc_after_trial=True
    )
        
    # Save Best Hyperparameters
    with open(HYPERPARAMS, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    
    # Save All Trials History as CSV
    trials_df = study.trials_dataframe()
    trials_df.to_csv(TRIAL_HISTROY, index=False)
    
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
    with open(TRIAL_DATA, 'w') as f:
        json.dump(trials_data, f, indent=4)
    
    # Save Study Summary
    with open(TRIAL_SUMMARY, 'w') as f:
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
    fig1.write_html(f"{TRIAL_PLOTS}\optimization_history.html")

    # Which hyperparameters matter most?
    fig2 = plot_param_importances(study)
    fig2.write_html(f"{TRIAL_PLOTS}\param_importances.html")

    # Relationship between each param and objective
    fig3 = plot_slice(study)
    fig3.write_html(f"{TRIAL_PLOTS}\param_obj_relation.html")

    # High-dimensional parameter relationships
    fig4 = plot_parallel_coordinate(study)
    fig4.write_html(f"{TRIAL_PLOTS}\high_dim_param_relations.html")