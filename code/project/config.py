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
Highcorrelatedflag = False

test_size = 0.1
seed = 10
Grid_n_jobs = 3
Grid_refit = True

outer_n_splits = 10
outer_shuffle = True

inner_n_splits = 10
inner_shuffle = True


# define search spaces
knnspace = {
    "n_neighbors": np.arange(1, 22, 2),
    "metric": ["euclidean", "manhattan", "chebyshev"],
    "weights": ["distance", "uniform"],
}

# knnspace = [
#     {"n_neighbors": np.arange(1, 30, 2), "metric": ["euclidean", "manhattan", "chebyshev"], "weights": ["distance", "uniform"],},
#     {"n_neighbors": np.arange(1, 30, 2), "metric": ["mahalanobis", "seuclidean"],
#      "metric_params": [{"V": np.cov(X_train)}],"weights": ["distance", "uniform"],}
# ]

treespace = {
    "max_depth": np.arange(3, 33, 2),
    "criterion": ["gini", "entropy"],
}

svmspace = {
    "probability": [True],
    "kernel": ["rbf", "linear"],
    "decision_function_shape": ["ovr", "ovo"],
    "C": [0.1, 10, 1000],
    "gamma": [1, 0.01, 0.0001],
    "random_state": [seed],
}

ldaspace = {"n_components": [10, 15, 20, 25, 30], "sfs__k_features": [1, 4]}

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
    + ".txt"
)
