"""https://www.machinelearningplus.com/time-series/time-series-analysis-python/"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pywt

from pandas.plotting import lag_plot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa import seasonal, stattools
from statsmodels.tsa.ar_model import AutoReg

plt.rcParams["figure.figsize"] = (14, 7)
a = "Ass1_D1_"

print("[INFO] Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "manuscript", "src", "figures")
tbl_dir = os.path.join(project_dir, "manuscript", "src", "tables")
data_dir = os.path.join(project_dir, "Dataset")
# dataset_file = os.path.join(data_dir, "monthly-sunspots.csv")
dataset_file = os.path.join(data_dir, "temperatures.csv")


print("[INFO] Reading the first dataset")
series = pd.read_csv(
    dataset_file,
    header=0,
    index_col=0,
).dropna()

series.index = pd.to_datetime(series.index)

coeffs = pywt.wavedec(series["1989"], "sym2", mode="periodic", level=3)
# Load image

print(len(coeffs))
# print(np.array(coeffs).shape())
# Wavelet transform of image, and plot approximation and details
titles = ["Approximation", " Horizontal detail", "Vertical detail", "Diagonal detail"]

A, B, C, D = coeffs
fig = plt.figure(figsize=(18, 9))
for i, a in enumerate([A, B, C, D]):
    ax = fig.add_subplot(4, 1, i + 1)
    ax.plot(a)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()