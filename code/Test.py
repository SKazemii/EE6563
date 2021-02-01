import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

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
# split dataset
X = series.values
train, test = X[1 : len(X) - 7], X[len(X) - 7 :]
# train autoregression
model = AutoReg(train, lags=29)
model_fit = model.fit()
print("Coefficients: %s" % model_fit.params)
# make predictions
predictions = model_fit.predict(
    start=len(train), end=len(train) + len(test) - 1, dynamic=False
)
for i in range(len(predictions)):
    print("predicted=%f, expected=%f" % (predictions[i], test[i]))
rmse = np.sqrt(mean_squared_error(test, predictions))
print("Test RMSE: %.3f" % rmse)

plt.figure()
plt.plot(test)
plt.plot(predictions, color="blue")

plt.figure()
xpos = np.arange(len(X))
plt.plot(X, "r--", linewidth=0.2)
plt.plot(xpos[len(X) - 7 : len(X)], predictions[:], color="blue")
plt.show()