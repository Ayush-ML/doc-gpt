# This is the Model Build for The Project
# Imported Libraries
import warnings
warnings.filterwarnings('ignore', category=FutureWarning) # Ignore Certain Warnings to reduce Terminal Clutter
warnings.filterwarnings('ignore', category=UserWarning)
from stroke.src.processing.preprocessing import preprocessor
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import numpy as np
from imblearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold

np.random.seed(42) # Set Random Seed for Reproducibility

# Load The Preprocessed Data for Tuning

X_train = pd.read_csv(r'stroke\data\clean\train\X_train.csv')
X_train = pd.DataFrame(X_train, columns=X_train.columns) # Ensure X_train is a DataFrame with the correct column names
y_train = pd.read_csv(r'stroke\data\clean\train\y_train.csv').squeeze() 

# We will build an Ensemble Model for this Dataset using Stacking
# Define our base models

base_models = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('lgbm', LGBMClassifier(random_state=42, verbosity=-1, min_child_samples=20, num_leaves=31)),
    ('svc', SVC(probability=True, random_state=42, kernel='linear')),
    ('xgb', XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0))
]

smote = SMOTE(random_state=42) # Define SMOTE for handling class imbalance in the dataset
skb = SelectKBest(score_func=f_classif) # Define SelectKBest for feature selection
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Define StratifiedKFold for Cross Validation

# Define our Meta Model

meta_model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')

# Create a Stacking Classifier

stack = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3, passthrough=True, stack_method='predict_proba')

# Build Our pipeline

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('skb', skb),
    ('stack', stack)
])
print("Testing Phase...")
pipeline.fit(X_train[:100], y_train[:100]) # Fit the pipeline on a subset of the data to test it before tuning

# Lets use Bayesian Optimization for Hyperparameter Tuning
# Define the search space for the hyperparameters of the base models and the meta model

search_space = {

    # SelectKBest
    'skb__k': Integer(10, 20),

    # Random Forest
    'stack__rf__n_estimators': Integer(100, 400),
    'stack__rf__max_depth': Integer(3, 15),

    # SVC
    'stack__svc__C': Real(0.1, 10, prior='log-uniform'),

    # XGBoost
    'stack__xgb__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'stack__xgb__max_depth': Integer(3, 10),
    'stack__xgb__n_estimators': Integer(100, 400),

    # LightGBM
    'stack__lgbm__num_leaves': Integer(20, 60),
    'stack__lgbm__min_child_samples': Integer(5, 50),

    # Logistic Regression
    'stack__final_estimator__C': Real(0.01, 10, prior='log-uniform'),
    'stack__final_estimator__solver': Categorical(['lbfgs', 'liblinear'])
    
}
bayes_search = BayesSearchCV(estimator=pipeline, search_spaces=search_space, n_iter=25, cv=cv, n_jobs=-1, verbose=2, random_state=42, scoring='roc_auc')
print("Tuning Phase...")
bayes_search.fit(X_train, y_train) # Tune Hyperparameters
print(bayes_search.best_score_)
tuned_hyperparameters = bayes_search.best_params_ # Get the Best Hyperparameters after Tuning

# Save the Tuned Hyperparameters for Future Use
print("Tuning Completed.Saving Hyperparameters...")
with open(r'stroke\src\training\tuned_hyperparameters.json', 'w') as f:
    json.dump(tuned_hyperparameters, f, indent=4)
