import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (14, 7)
a = "Ass1_D2_"

print("[INFO] Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "manuscript", "src", "figures", "project")
tbl_dir = os.path.join(project_dir, "manuscript", "src", "tables", "project")
data_dir = os.path.join(project_dir, "Dataset", "project", "Step Scan Dataset", "[H5]")
dataset_file = os.path.join(data_dir, "footpressures_align.h5")
pickle_dir = os.path.join(project_dir, "Dataset", "project", "pickle")
