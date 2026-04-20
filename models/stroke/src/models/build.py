# This is the model building Script for the Stroke Prediction model
# It consists of the following parts :
    # -- Preprocessor (from preprocess.py)
    # -- Feature Selector (RFECV)
    # -- SMOTE
    # -- LightGBM model
# Imported Libraries

from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from models.stroke.config import X_TRAIN, RANDOM_STATE, CLASS_WEIGHT, SHUFFLE, SOLVER, SCORING, N_JOBS, REFIT, CV, METHOD, N_SPLITS
from models.stroke.src.processing.preprocess import create_preprocessor
from sklearn import set_config

# Load X_train data and Validation Set

X_train = pd.read_csv(X_TRAIN)
    
# Create a Function for Pipeline Creation

def create_pipeline() -> TunedThresholdClassifierCV:
    # Create Preprocessor

    preprocessor = create_preprocessor(X=X_train)

    # Create Feature Selector

    estimator = LogisticRegression(random_state=RANDOM_STATE, class_weight=CLASS_WEIGHT, solver=SOLVER)
    cv = StratifiedKFold(shuffle=SHUFFLE, random_state=RANDOM_STATE, n_splits=N_SPLITS)

    selector = RFECV(estimator=estimator, cv=cv, scoring=SCORING, n_jobs=N_JOBS)

    # Create Model

    model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS, verbosity=-1)


    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', model)
    ])
    calibrated_pipeline = CalibratedClassifierCV(estimator=pipeline, cv=CV, method=METHOD)

    final_pipeline = TunedThresholdClassifierCV(estimator=calibrated_pipeline, cv=CV, refit=REFIT, scoring='recall')

    return final_pipeline
