# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 10:31:10 2021

@author: pkumar1
"""

# loading important libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# reading the dataset
series = pd.read_csv(
    "/Users/saeedkazemi/OneDrive - University of New Brunswick/The fall Term 2020/ECE6563/codes/AirPassengers.csv"
)
# the dataset contains number of passengers per month between Jan 1949-Dec. 1960
# preprocessing
series.timestamp = pd.to_datetime(series.Month, format="%Y-%m")
series.index = series.timestamp
series.drop("Month", axis=1, inplace=True)

# looking at the first few rows
series.head()
# Visual Test
series["#Passengers"].plot(label="Passengers")
plt.legend(loc="upper left")

# Model Fitting
X = [i for i in range(0, len(series))]
X = np.reshape(X, (len(X), 1))
y = series.values
model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)
# plot trend
plt.plot(y, label="Series")
plt.plot(trend, label="trend line")
plt.legend(loc="upper left")
# detrend
detrended = [y[i] - trend[i] for i in range(0, len(series))]
# plot detrended
plt.plot(detrended, label="detrended")
plt.legend(loc="upper left")


###############################################################Eliminate
# 1. Differencing yt‘ = yt – y(t-1)
series["#Passengers_diff"] = series["#Passengers"] - series["#Passengers"].shift(1)
series["#Passengers_diff"].dropna().plot(label="Diff Passengers")
series["#Passengers"].plot(label="Passengers")
plt.legend(loc="upper left")
# train['#Passengers'].plot()

# 2. Seasonal Differencing
n = 7
series["#Passengers_diff_season"] = series["#Passengers"] - series["#Passengers"].shift(
    n
)
series["#Passengers_diff_season"].dropna().plot(label="Seasonal Diff Passengers")
series["#Passengers_diff"].dropna().plot(label="Diff Passengers")
series["#Passengers"].plot(label="Passengers")
plt.legend(loc="upper left")

# 3. Transformation
series["#Passengers_log"] = np.log(series["#Passengers"])
series["#Passengers_log_diff"] = series["#Passengers_log"] - series[
    "#Passengers_log"
].shift(1)
series["#Passengers_log_diff"].dropna().plot(label="Log Difference")
plt.legend(loc="upper left")

# 4. Check the mean and variance (Stationary?)
X = series["#Passengers"].values  # original data
split = int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print("mean1=%f, mean2=%f" % (mean1, mean2))
print("variance1=%f, variance2=%f" % (var1, var2))

X = series["#Passengers_log"].values  # original data
split = int(len(X) / 2)
X1, X2 = X[0:split], X[split:]
mean1, mean2 = X1.mean(), X2.mean()
var1, var2 = X1.var(), X2.var()
print("mean1=%f, mean2=%f" % (mean1, mean2))
print("variance1=%f, variance2=%f" % (var1, var2))

# 5. Augmented Dickey-Fuller test??
