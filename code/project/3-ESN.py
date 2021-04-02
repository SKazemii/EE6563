from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import VarianceThreshold
import sklearn


from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import sklearn.model_selection
from sklearn import preprocessing
from sklearn import feature_selection

import seaborn as sns
import tsfel
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys

# Custom imports
import config as cfg
from modules import RC_model


print("[INFO] importing libraries....")


# df_sum ##################################################################################
print("[INFO] importing pickles files....")
with open(os.path.join(cfg.pickle_dir, "df_sum.pickle"), "rb") as handle:
    df_sum = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_max.pickle"), "rb") as handle:
    df_max = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_xCe.pickle"), "rb") as handle:
    df_xCe = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_yCe.pickle"), "rb") as handle:
    df_yCe = pickle.load(handle)

with open(os.path.join(cfg.pickle_dir, "df_label.pickle"), "rb") as handle:
    df_label = pickle.load(handle)


df_max = np.expand_dims(df_max, axis=0)
df_max = df_max.T

df_sum = np.expand_dims(df_sum, axis=0)
df_sum = df_sum.T

df_xCe = np.expand_dims(df_xCe, axis=0)
df_xCe = df_xCe.T

df_yCe = np.expand_dims(df_yCe, axis=0)
df_yCe = df_yCe.T


dataset = np.concatenate((df_sum, df_max, df_xCe, df_yCe), axis=2)

print("[INFO] splitting the training and testing sets...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
    dataset,
    np.array(df_label),
    stratify=np.array(df_label),
    test_size=cfg.test_size,
    random_state=cfg.seed,
)


# One-hot encoding for labels
onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
OneHot_trainLabels = onehot_encoder.fit_transform(trainLabels)
OneHot_testLabels = onehot_encoder.transform(testLabels)


print(dataset.shape)
print(trainData.shape)
print(testData.shape)
print(trainLabels.shape)
print(testLabels.shape)
print(OneHot_trainLabels.shape)
print(OneHot_testLabels.shape)


classifier = RC_model(
    reservoir=None,
    n_internal_units=cfg.esn_config["n_internal_units"],
    spectral_radius=cfg.esn_config["spectral_radius"],
    leak=cfg.esn_config["leak"],
    connectivity=cfg.esn_config["connectivity"],
    input_scaling=cfg.esn_config["input_scaling"],
    noise_level=cfg.esn_config["noise_level"],
    circle=cfg.esn_config["circ"],
    n_drop=cfg.esn_config["n_drop"],
    bidir=cfg.esn_config["bidir"],
    dimred_method=cfg.esn_config["dimred_method"],
    n_dim=cfg.esn_config["n_dim"],
    mts_rep=cfg.esn_config["mts_rep"],
    w_ridge_embedding=cfg.esn_config["w_ridge_embedding"],
    readout_type=cfg.esn_config["readout_type"],
    w_ridge=cfg.esn_config["w_ridge"],
    mlp_layout=cfg.esn_config["mlp_layout"],
    num_epochs=cfg.esn_config["num_epochs"],
    w_l2=cfg.esn_config["w_l2"],
    nonlinearity=cfg.esn_config["nonlinearity"],
    svm_gamma=cfg.esn_config["svm_gamma"],
    svm_C=cfg.esn_config["svm_C"],
)

tr_time = classifier.train(trainData, OneHot_trainLabels)
print("Training time = %.2f seconds" % tr_time)

accuracy, f1 = classifier.test(testData, OneHot_testLabels)
print("Accuracy = %.3f, F1 = %.3f" % (accuracy, f1))
