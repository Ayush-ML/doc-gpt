# This is the Model Tuning for The Project
# Imported Libraries
import warnings
warnings.filterwarnings('ignore', category=FutureWarning) # Ignore Certain Warnings to reduce Terminal Clutter
warnings.filterwarnings('ignore', category=UserWarning)
from stroke.src.processing.preprocessing import preprocessor
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import numpy as np
from imblearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, make_scorer

np.random.seed(42) # Set Random Seed for Reproducibility

# Load The Preprocessed Data for Tuning

X_train = pd.read_csv(r'stroke\data\clean\train\X_train.csv')
X_train = pd.DataFrame(X_train, columns=X_train.columns) # Ensure X_train is a DataFrame with the correct column names
y_train = pd.read_csv(r'stroke\data\clean\train\y_train.csv').squeeze() 

# We will build an Ensemble Model for this Dataset using Stacking
# Define our base models

base_models = [
    ('rf', RandomForestClassifier(random_state=42, class_weight='balanced')),
    ('lgbm', LGBMClassifier(random_state=42, verbosity=-1, class_weight='balanced')),
    ('svc', SVC(probability=True, random_state=42, kernel='rbf', class_weight='balanced')),
    ('xgb', XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0, scale_pos_weight=19)),
    ('ada', AdaBoostClassifier(random_state=42))
]

smotetomek = SMOTETomek(random_state=42, sampling_strategy=0.5) # Define SMOTE for handling class imbalance in the dataset
skb = SelectKBest(score_func=mutual_info_classif) # Define SelectKBest for feature selection
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # Define StratifiedKFold for Cross Validation
mcc = make_scorer(matthews_corrcoef) # Define Matthews Correlation Coefficient as a scoring metric for evaluation during hyperparameter tuning, As it is the most reliable metric for Medial datasets such as this one

# Define our Meta Model

meta_model = LogisticRegression(random_state=42, max_iter=1000)

# Create a Stacking Classifier

stack = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=3, passthrough=False, stack_method='predict_proba')

# Build Our pipeline

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('skb', skb),
    ('stack', stack)
])
stroke_percentage = y_train.mean() * 100
pipeline.set_params(stack__xgb__scale_pos_weight=(100 / stroke_percentage) - 1) # Set scale_pos_weight for XGBoost to handle class imbalance according to the ratio of classes in the dataset
                                                     # use a set formula to calculate the optimal weight to make it adaptable across datasets with different levels of imbalance

if __name__ == "__main__": # Only Run trainig code if the file is ran directly not on any imports
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
        'stack__rf__bootstrap': ([True, False]),
        'stack__rf__max_features': Real(0.1, 1.0),
        'stack__rf__min_samples_leaf': Integer(1, 20),

        # SVC
        'stack__svc__C': Real(0.1, 10, prior='log-uniform'),
        'stack__svc__gamma': Real(0.001, 1, prior='log-uniform'),

        # XGBoost
        'stack__xgb__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'stack__xgb__max_depth': Integer(3, 10),
        'stack__xgb__n_estimators': Integer(100, 400),
        'stack__xgb__subsample': Real(0.5, 1.0),
        'stack__xgb__colsample_bytree': Real(0.5, 1.0),
        'stack__xgb__min_child_weight': Integer(1, 10),

        # LightGBM
        'stack__lgbm__num_leaves': Integer(20, 100),
        'stack__lgbm__min_child_samples': Integer(5, 50),
        'stack__lgbm__learning_rate': Real(0.01, 0.2),
        'stack__lgbm__n_estimators': Integer(100, 400),
        'stack__lgbm__feature_fraction': Real(0.4, 1.0),
        'stack__lgbm__bagging_fraction': Real(0.4, 1.0),

        # Logistic Regression
        'stack__final_estimator__C': Real(0.1, 2.0, prior='log-uniform'),
        'stack__final_estimator__solver': Categorical(['saga', 'liblinear']),
        'stack__final_estimator__penalty': Categorical(['l1', 'l2'])
    }
    bayes_search = BayesSearchCV(estimator=pipeline, search_spaces=search_space, n_iter=100, cv=cv, n_jobs=-1, verbose=2, random_state=42, scoring=mcc)
    print("Tuning Phase...")
    bayes_search.fit(X_train, y_train) # Tune Hyperparameters
    print(bayes_search.best_score_)
    tuned_hyperparameters = bayes_search.best_params_ # Get the Best Hyperparameters after Tuning

# Save the Tuned Hyperparameters for Future Use

    print("Tuning Completed.Saving Hyperparameters...")
    with open(r'stroke\src\models\tuned_hyperparameters.json', 'w') as f:
        json.dump(tuned_hyperparameters, f, indent=4)
