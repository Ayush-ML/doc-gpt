# This is a Hyperparameter Tuning Script for the Diabetes Model
# This modifies the params of the create_pipeline function from build.py
# Mainly for Paramters of LightGBM, RFECV and its internal params
# Imported Libraries

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from optuna.visualization import plot_parallel_coordinate, plot_param_importances, plot_slice, plot_optimization_history
from optuna import Trial, create_study, TrialPruned
from optuna.samplers import TPESampler
import logging, datetime, os, json
from models.stroke.config import LOGS, X_TRAIN, Y_TRAIN, SCORING, HYPERPARAMS, TRIAL_DATA, TRIAL_HISTROY, TRIAL_PLOTS, TRIAL_SUMMARY, SHUFFLE, RANDOM_STATE, INTERVAL, DIRECTION, N_TRIALS, EVAL_DATA, N_SPLITS, N_JOBS
from models.stroke.src.models.build import create_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
import lightgbm as lgbm
from sklearn import set_config

# Enable Metadata Routing to be able to pass extra parameters

set_config(enable_metadata_routing=True)

# Setup Logging Directory

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(LOGS, f'tuning_{timestamp}.log')

logger = logging.getLogger('optuna')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

lgbm.register_logger(logger=logger) # Register Our Logger to collect LightGBM Data

# Load Train

X_train = pd.read_csv(X_TRAIN)
y_train = pd.read_csv(Y_TRAIN).squeeze()

# Create a Dict to Store All Eval Data

all_eval_data = []

# Define Objective Function that handles HyperParameter Tuning

def objective(trial: Trial) -> float:

    # Create a Dict to Store Trial Tuning Evaluation Data

    trial_data = {}

    params = { # Create Parameter Dictionary
        # ── LightGBM 
        'estimator__estimator__model__n_estimators':          trial.suggest_int('n_estimators', 200, 1000),
        'estimator__estimator__model__max_depth':             trial.suggest_int('max_depth', 3, 7),
        'estimator__estimator__model__learning_rate':         trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'estimator__estimator__model__min_child_samples':     trial.suggest_int('min_child_samples', 5, 20),
        'estimator__estimator__model__subsample':             trial.suggest_float('subsample', 0.6, 1.0),
        'estimator__estimator__model__colsample_bytree':      trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'estimator__estimator__model__reg_alpha':             trial.suggest_float('reg_alpha', 0.0, 0.5),
        'estimator__estimator__model__reg_lambda':            trial.suggest_float('reg_lambda', 1.0, 5.0),
        'estimator__estimator__model__scale_pos_weight':      trial.suggest_float('scale_pos_weight', 15.0, 30.0),
        'estimator__estimator__model__num_leaves':            trial.suggest_int('num_leaves', 20, 150),
        'estimator__estimator__model__min_split_gain':        trial.suggest_float('min_split_gain', 0.0, 1.0),
        'estimator__estimator__model__subsample_freq':        trial.suggest_int('subsample_freq', 1, 10),

        # ── RFECV 
        'estimator__estimator__selector__step':                        trial.suggest_int('step', 1, 10),
        'estimator__estimator__selector__min_features_to_select':      trial.suggest_int('min_features_to_select', 5, 20),

        # ── RFECV Estimator (LogisticRegression) 
        'estimator__estimator__selector__estimator__C':                trial.suggest_float('C', 0.01, 1.0, log=True),
        'estimator__estimator__selector__estimator__max_iter':         trial.suggest_int('max_iter', 500, 2000),
        'estimator__estimator__selector__estimator__tol':              trial.suggest_float('tol', 1e-5, 1e-2, log=True),
        'estimator__estimator__selector__estimator__fit_intercept':    trial.suggest_categorical('fit_intercept', [True, False]),
        'estimator__estimator__selector__estimator__warm_start':       trial.suggest_categorical('warm_start', [True, False]),
    }

    pipeline = create_pipeline() # Create Pipeline
    pipeline.set_params(**params) # Set Parameters
    pipeline.fit(X_train, y_train)

    scores = cross_val_score( # Calculate Cross Validation Score
        pipeline,
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_STATE),
        scoring='average_precision',
        n_jobs=N_JOBS,
    ).mean()

    all_eval_data.append({'trial_no': trial.number, 'eval': trial_data}) # Add to All Evaluation Data

    trial.report(scores, step=0) # Report to Optuna

    if trial.should_prune(): # Prune Trial if needed
        raise TrialPruned()
    
    return scores

study = create_study(direction=DIRECTION, sampler=TPESampler(seed=RANDOM_STATE)) # Create Study
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, gc_after_trial=True) # Start Optimization

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
    f.write(f"Metric: PR-AUC (binary classification)\n")
    f.write(f"Total Trials: {len(study.trials)}\n")
    f.write(f"Best Trial Number: {study.best_trial.number}\n")
    f.write(f"Best PR-AUC Score: {study.best_value:.6f}\n\n")
    f.write(f"Best Hyperparameters:\n")
    for param, value in study.best_params.items():
        f.write(f"  {param}: {value}\n")
    f.write(f"\nTimestamp: {timestamp}\n")
    

# Optimization progress over trials

fig1 = plot_optimization_history(study)
fig1.write_html(os.path.join(TRIAL_PLOTS, "optimization_history.html"))

# Which hyperparameters matter most?

fig2 = plot_param_importances(study)
fig2.write_html(os.path.join(TRIAL_PLOTS, "param_importances.html"))

# Relationship between each param and objective

fig3 = plot_slice(study)
fig3.write_html(os.path.join(TRIAL_PLOTS, "param_obj_relation.html"))

# High-dimensional parameter relationships

fig4 = plot_parallel_coordinate(study)
fig4.write_html(os.path.join(TRIAL_PLOTS, "high_dim_param_relations.html"))

# Write Evaluation Data to JSON

with open(EVAL_DATA, 'w') as f:
    json.dump(all_eval_data, f, indent=4)


