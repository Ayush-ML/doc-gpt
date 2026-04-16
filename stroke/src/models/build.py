# This is the model building Script for the Stroke Prediction model
# It consists of the following parts :
    # -- Preprocessor (from preprocess.py)
    # -- Feature Selector (RFECV)
    # -- SMOTE
    # -- LightGBM model
# Imported Libraries

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from stroke.config import X_TRAIN, RANDOM_STATE, CLASS_WEIGHT, SHUFFLE, SOLVER, SCORING, N_JOBS, REFIT, CV, METHOD
from stroke.src.processing.preprocess import create_preprocessor

# Load X_train data

X_train = pd.read_csv(X_TRAIN)

# Create Preprocessor

preprocessor = create_preprocessor(X=X_train)

# Create Feature Selector

estimator = LogisticRegression(random_state=RANDOM_STATE, class_weight=CLASS_WEIGHT, solver=SOLVER)
cv = StratifiedKFold(shuffle=SHUFFLE, random_state=RANDOM_STATE)

selector = RFECV(estimator=estimator, cv=cv, scoring=SCORING, n_jobs=N_JOBS)

# Create SMOTE

smote = SMOTE(random_state=RANDOM_STATE)

# Create Model

model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)

# Function for Pipeline Creation

def create_pipeline() -> TunedThresholdClassifierCV:

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('smote', smote)
        ('model', model)
    ])

    calibrated_pipeline = CalibratedClassifierCV(estimator=pipeline, cv=CV, method=METHOD)

    return TunedThresholdClassifierCV(estimator=calibrated_pipeline, cv=CV, refit=REFIT, scoring=SCORING)