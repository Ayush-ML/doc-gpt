import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from diabetes.config import Y_TRAIN

y_train = pd.read_csv(Y_TRAIN).squeeze()
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weight_dict = dict(zip(classes, weights))
sample_weights = np.array([class_weight_dict[y] for y in y_train])