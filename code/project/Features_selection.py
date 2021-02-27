print("[INFO] importing libraries....")


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import config as cfg
import pickle
import tsfel

from scipy import interpolate
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier



print("[INFO] importing pickles files....")
with open(os.path.join(cfg.pickle_dir, "df_sum_temporal_features.pickle"), "rb") as handle:
    df_sum_temporal_features = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_sum_statistical_features.pickle"), "rb") as handle:
    df_sum_statistical_features = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_sum_spectral_features.pickle"), "rb") as handle:
    df_sum_spectral_features = pickle.load(handle)


with open(os.path.join(cfg.pickle_dir, "df_max_temporal_features.pickle"), "rb") as handle:
    df_max_temporal_features = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_max_statistical_features.pickle"), "rb") as handle:
    df_max_statistical_features = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_max_spectral_features.pickle"), "rb") as handle:
    df_max_spectral_features = pickle.load(handle)


with open(os.path.join(cfg.pickle_dir, "df_xCe_temporal_features.pickle"), "rb") as handle:
    df_xCe_temporal_features = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_xCe_statistical_features.pickle"), "rb") as handle:
    df_xCe_statistical_features = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_xCe_spectral_features.pickle"), "rb") as handle:
    df_xCe_spectral_features = pickle.load(handle)


with open(os.path.join(cfg.pickle_dir, "df_yCe_temporal_features.pickle"), "rb") as handle:
    df_yCe_temporal_features = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_yCe_statistical_features.pickle"), "rb") as handle:
    df_yCe_statistical_features = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_yCe_spectral_features.pickle"), "rb") as handle:
    df_yCe_spectral_features = pickle.load(handle)



# Highly correlated features are removed
corr_features = tsfel.correlated_features(df_yCe_temporal_features)
df_yCe_temporal_features.drop(corr_features, axis=1, inplace=True)



# Remove low variance features
selector = VarianceThreshold()
X_train = selector.fit_transform(x_train_feat)
X_test = selector.transform(x_test_feat)

# Normalising Features
scaler = preprocessing.StandardScaler()
nX_train = scaler.fit_transform(X_train)
nX_test = scaler.transform(X_test)


classifier = RandomForestClassifier(n_estimators = 20, min_samples_split=10)

activities = ['walking', 'jogging', 'stairs', 'sitting', 'standing', 'typing', 
              'brushing teeth', 'eating soup', 'eating chips', 'eating pasta', 
              'drinking', 'eating sandwich', 'kicking', 'playing catch', 
              'dribblinlg', 'writing', 'clapping', 'folding clothes']

# Train The Classifier
classifier.fit(X_train, y_train.ravel())

# Predict Test Data
y_predict = classifier.predict(X_test)

# Get the Classification Report
accuracy = accuracy_score(y_test, y_predict)*100
print(classification_report(y_test, y_predict, target_names = activities))
print('Accuracy: ' + str(accuracy) + '%')