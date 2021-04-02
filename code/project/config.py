import os
import numpy as np
import datetime
import matplotlib.pyplot as plt


plt.rcParams["figure.figsize"] = (14, 7)
plt.rcParams["figure.dpi"] = 128


classifier_name = "lda"  # {lda, knn, svm, reg, tree}
features_name = "all"  # {temporal, statistical, spectral, all, AR}
transform = "standardization"  # {standardization, normalization, none}


VarianceThresholdflag = True
Highcorrelatedflag = True

test_size = 0.1
seed = 10
Grid_n_jobs = 3
Grid_refit = True

# outer_n_splits = 10
# outer_shuffle = True

inner_n_splits = 10
inner_shuffle = True


# define search spaces
knnspace = {
    "model__n_neighbors": np.arange(1, 22, 2),
    "model__metric": ["euclidean", "manhattan", "chebyshev"],
    "model__weights": ["distance", "uniform"],
    "sfs__k_features": [100, 200, 300],
}

# knnspace = [
#     {"n_neighbors": np.arange(1, 30, 2), "metric": ["euclidean", "manhattan", "chebyshev"], "weights": ["distance", "uniform"],},
#     {"n_neighbors": np.arange(1, 30, 2), "metric": ["mahalanobis", "seuclidean"],
#      "metric_params": [{"V": np.cov(X_train)}],"weights": ["distance", "uniform"],}
# ]

treespace = {
    "model__max_depth": np.arange(3, 33, 2),
    "model__criterion": ["gini", "entropy"],
    "sfs__k_features": [100, 200, 300],
}

svmspace = {
    "model__probability": [True],
    "model__kernel": ["rbf", "linear"],
    "model__decision_function_shape": ["ovr", "ovo"],
    "model__C": [0.1, 10, 1000],
    "model__gamma": [1, 0.01, 0.0001],
    "model__random_state": [seed],
    "sfs__k_features": [100, 200, 300],
}

ldaspace = {
    # "model__n_components": [10,20,30],
    "sfs__k_features": [100],
}

regspace = {
    "C": [1, 10, 100, 1000],
}

print("[INFO] Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "manuscript", "src", "figures", "project")
tbl_dir = os.path.join(project_dir, "manuscript", "src", "tables", "project")
data_dir = os.path.join(project_dir, "Dataset", "project", "Step Scan Dataset", "[H5]")
dataset_file = os.path.join(data_dir, "footpressures_align.h5")
pickle_dir = os.path.join(project_dir, "Dataset", "project", "pickle")
output_dir = os.path.join(project_dir, "code", "project", "output")


result_name_file = (
    datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    + "-"
    + classifier_name
    + "-"
    + features_name
    + "-"
    + transform
)

# ============ RC model configuration and hyperparameter values ============
esn_config = {}
esn_config["dataset_name"] = "JpVow"

esn_config["seed"] = seed
np.random.seed(esn_config["seed"])

# Hyperarameters of the reservoir
esn_config["n_internal_units"] = 450  # size of the reservoir
esn_config["spectral_radius"] = 0.59  # largest eigenvalue of the reservoir
esn_config[
    "leak"
] = 0.6  # amount of leakage in the reservoir state update (None or 1.0 --> no leakage)
esn_config["connectivity"] = 0.25  # percentage of nonzero connections in the reservoir
esn_config["input_scaling"] = 0.1  # scaling of the input weights
esn_config["noise_level"] = 0.01  # noise in the reservoir state update
esn_config["n_drop"] = 5  # transient states to be dropped
esn_config["bidir"] = True  # if True, use bidirectional reservoir
esn_config["circ"] = False  # use reservoir with circle topology

# Dimensionality reduction hyperparameters
esn_config[
    "dimred_method"
] = "tenpca"  # options: {None (no dimensionality reduction), 'pca', 'tenpca'}
esn_config[
    "n_dim"
] = 75  # number of resulting dimensions after the dimensionality reduction procedure

# Type of MTS representation
esn_config[
    "mts_rep"
] = "reservoir"  # MTS representation:  {'last', 'mean', 'output', 'reservoir'}
esn_config[
    "w_ridge_embedding"
] = 10.0  # regularization parameter of the ridge regression

# Type of readout
esn_config[
    "readout_type"
] = "lin"  # readout used for classification: {'lin', 'mlp', 'svm'}

# Linear readout hyperparameters
esn_config["w_ridge"] = 5.0  # regularization of the ridge regression readout

# SVM readout hyperparameters
esn_config["svm_gamma"] = 0.005  # bandwith of the RBF kernel
esn_config["svm_C"] = 5.0  # regularization for SVM hyperplane

# MLP readout hyperparameters
esn_config["mlp_layout"] = (10, 10)  # neurons in each MLP layer
esn_config["num_epochs"] = 2000  # number of epochs
esn_config["w_l2"] = 0.001  # weight of the L2 regularization
esn_config[
    "nonlinearity"
] = "relu"  # type of activation function {'relu', 'tanh', 'logistic', 'identity'}

print(esn_config)
