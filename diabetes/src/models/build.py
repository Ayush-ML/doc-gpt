# This is the Model Building script for the Diabetes Prediction Model
# This is used to build the pipeline of the model
# Imported Libraries

from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn import set_config
from diabetes.src.processing.preprocessing import create_preprocessor
from xgboost import XGBClassifier
import pandas as pd
from imblearn.combine import SMOTETomek
from diabetes.config import X_TRAIN, Y_TRAIN, RFECV_STEP, RFECV_MIN_FEATURES, RANDOM_STATE, SHUFFLE, MAX_ITER, C, CLASS_WEIGHT, N_SPLITS, SCORING

# Load Train Data

X_train = pd.read_csv(X_TRAIN)
y_train = pd.read_csv(Y_TRAIN).squeeze() 

# Create Preprocessor

preprocessor = create_preprocessor(X=X_train, y=y_train)

# Create the Selector used for Feature Selection

estimator = LogisticRegression(max_iter=MAX_ITER, C=C, class_weight=CLASS_WEIGHT, random_state=RANDOM_STATE)
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=SHUFFLE, random_state=RANDOM_STATE)

selector = RFECV(estimator=estimator,
                step=RFECV_STEP,
                cv=cv,
                scoring=SCORING,
                min_features_to_select=RFECV_MIN_FEATURES)

smote = SMOTETomek(random_state=RANDOM_STATE)

# Create the Pipeline that combines the Preprocessor, Selector and the Model

def create_pipe() -> Pipeline:
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smotetomek', smote),
        ('selector', selector),
        ('model', XGBClassifier())
    ]) 