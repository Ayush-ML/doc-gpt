# This is the Model Building script for the Diabetes Prediction Model
# This is used to build the pipeline of the model
# Imported Libraries

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn import set_config
from diabetes.src.processing.preprocessing import create_preprocessor
from xgboost import XGBClassifier
import pandas as pd

# Load Train Data

X_train = pd.read_csv(r'diabetes\data\clean\train\X_train.csv')
y_train = pd.read_csv(r'diabetes\data\clean\train\y_train.csv').squeeze() 

# Create Preprocessor

preprocessor = create_preprocessor(X=X_train, y=y_train)

# Create the Selector used for Feature Selection

selector = RFECV(estimator=LogisticRegression(max_iter=1000, C=0.1, class_weight='balanced', random_state=42),
                step=1,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc',
                min_features_to_select=7)

# Create the Pipeline that combines the Preprocessor, Selector and the Model

def create_pipe() -> Pipeline:
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('selector', selector),
        ('model', XGBClassifier())
    ]) 