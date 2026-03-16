# This is the Model Build for The Project
# Imported Libraries

from stroke.src.processing.preprocessing import preprocessor
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import numpy as np
from imblearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from sklearn.feature_selection import SelectKBest, f_classif

np.random.seed(42) # Set Random Seed for Reproducibility

# We will build an Ensemble Model for this Dataset using Stacking
# Define our base models

base_models = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('gb', GradientBoostingClassifier(random_state=42)),
    ('ada', AdaBoostClassifier(random_state=42)),
    ('cat', CatBoostClassifier(verbose=1, random_state=42)),
    ('lgbm', LGBMClassifier(class_weight='balanced', random_state=42))
]

smote = SMOTE(random_state=42) # Define SMOTE for handling class imbalance in the dataset
skb = SelectKBest(score_func=f_classif) # Define SelectKBest for feature selection

# Define our Meta Model

meta_model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

# Create a Stacking Classifier

stack = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5, passthrough=True, stack_method='predict_proba')

# Build Our pipeline

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('skb', skb),
    ('smote', smote),
    ('stacking', stack)
])

# Lets use Bayesian Optimization for Hyperparameter Tuning
# Define the search space for the hyperparameters of the base models and the meta model

search_space = {

    # SelectKBest
    'skb__k': Integer(10, 50),

    # Random Forest
    'stacking__rf__n_estimators': Integer(100, 400),
    'stacking__rf__max_depth': Integer(3, 15),

    # Gradient Boosting
    'stacking__gb__n_estimators': Integer(100, 300),
    'stacking__gb__learning_rate': Real(0.01, 0.3, prior='log-uniform'),

    # AdaBoost
    'stacking__ada__n_estimators': Integer(50, 200),
    'stacking__ada__learning_rate': Real(0.01, 0.3, prior='log-uniform'),

    # CatBoost
    'stacking__cat__depth': Integer(4, 10),
    'stacking__cat__learning_rate': Real(0.01, 0.3, prior='log-uniform'),

    # LightGBM
    'stacking__lgbm__num_leaves': Integer(20, 60),
    'stacking__lgbm__min_child_samples': Integer(5, 50),

    # Logistic Regression
    'stacking__final_estimator__C': Real(0.01, 10, prior='log-uniform'),
    'stacking__final_estimator__solver': Categorical(['lbfgs', 'liblinear'])
    
}
bayes_search = BayesSearchCV(estimator=pipeline, search_spaces=search_space, n_iter=50, cv=5, n_jobs=-1, verbose=2, random_state=42, scoring='roc_auc')