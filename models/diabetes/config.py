# This is a script that contains all variables that are used across the model that are not specific to the script
# For example, file paths, column names, etc. that are used across multiple scripts in the model are defined here to avoid hardcoding them in multiple places and to make it easier to update them in the future if needed.

# File Paths 

RAW_DATA = r'models\diabetes\data\raw\Dataset of Diabetes .csv'
X_TRAIN = r"models\diabetes\data\clean\train\X_train.csv"
Y_TRAIN = r"models\diabetes\data\clean\train\y_train.csv"
X_TEST = r"models\diabetes\data\clean\test\X_test.csv"
Y_TEST = r"models\diabetes\data\clean\test\y_test.csv"
X_VAL = r"models\diabetes\data\clean\val\X_val.csv"
Y_VAL = r"models\diabetes\data\clean\val\y_val.csv"
LOGS = r'models\diabetes\reports\tuning_logs'
HYPERPARAMS = r"models\diabetes\models\tuned_hyperparams.json"
TRIAL_HISTROY = r"models\diabetes\reports\tuning_logs\trial_history"
TRIAL_DATA = r"models\diabetes\reports\tuning_logs\trial_data.json"
TRIAL_SUMMARY = r"models\diabetes\reports\tuning_logs\trial_summary.txt"
TRIAL_PLOTS = r"models\diabetes\reports\tuning_logs\tuning_graphs"
TRACKING = r"sqlite:///models\diabetes/reports/training_logs/mlflow.db"
MODEL = r"models\diabetes\models\final_model"
METRICS = r"models\diabetes\reports\training_logs\metrics.txt"
CALIBRATION = r"models\garphed_metrics/calibration_curve.png"

# Lists

DROP_COLUMNS = ['id', 'no_pation', 'hba1c']
EXTREMELY_RIGHT_SKEWED = ['cr', 'hdl', 'vldl']
HIGHLY_RIGHT_SKEWED = ['urea', 'tg']

# Values

UPPER = 0.99
LOWER = 0.01
RANDOM_STATE = 42
TEST_SIZE = 0.1
VAL_SIZE = 0.1111
EXPERIEMENT = "model_training"

# Parameters

TRANSFORMER_METHOD = 'yeo-johnson'
FEATURE_CREATION_INTERACTION = True
FEATURE_CREATION_BIAS = False
RFECV_STEP = 1
RFECV_SCORING = 'roc_auc'
RFECV_MIN_FEATURES = 7
SHUFFLE = True
MAX_ITER = 1000
C = 0.1
CLASS_WEIGHT = 'balanced'
N_SPLITS = 5
DIRECTION = 'maximize'
STUDY_NAME = 'xgb_tuning'
N_TRIALS = 50
ROUNDS = 50
METRIC_NAME = 'mlogloss'
OBJECTIVE = 'multi:softprob'
NUM_CLASS = 3
AVERAGE = 'weighted'
SCORING = f"f1_{AVERAGE}"