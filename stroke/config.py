# This is a script that contains all variables that are used across the model that are not specific to the script
# For example,
# file paths, column names, etc. that are used across multiple scripts in the model are defined here to avoid hardcoding them in multiple places and to make it easier to update them in the future if needed.

# File Paths

RAW = r"stroke\data\raw\healthcare-dataset-stroke-data.csv"
X_TRAIN = r"stroke\data\clean\train\X_train.csv"
Y_TRAIN = r"stroke\data\clean\train\y_train.csv"
X_TEST = r"stroke\data\clean\test\X_test.csv"
Y_TEST = r"stroke\data\clean\test\y_test.csv"
X_VAL = r"stroke\data\clean\val\X_val.csv"
Y_VAL = r"stroke\data\clean\val\y_val.csv"


# Lists

HIGHLY_RIGHT_SKEWED = ['avg_glucose_level']
MODERATELY_RIGHT_SKEWED = ['bmi']
BINARY_COLS = ['hypertension', 'heart_disease', 'stroke']
VALUES_2 = ['gender', 'ever_married', 'residence_type']
VALUES_OVER_2 = ['work_type', 'smoking_status']
TARGET = ['stroke']
USELESS = ['id']

# Values

RANDOM_STATE = 42
TEST_SIZE = 0.1
VAL_SIZE = 0.1111
SCORING = 'average_percision'

# Parameters

CLASS_WEIGHT = 'balanced'
INTERACTION_ONLY = True
BIAS = False
SHUFFLE = True
SOLVER = 'lbfgs'
N_JOBS = -1
CV = 5
METHOD = 'sigmoid'
REFIT = True
