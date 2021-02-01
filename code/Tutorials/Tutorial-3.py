# Random Walk:random walk is different from a list of random numbers because the next value
# in the sequence is a modification of the previous value in the sequence. (y(t) = B0 + B1*X(t-1) + e(t))
# loading important libraries
import pandas as pd
from random import seed
from random import random
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt

seed(1)  # uses the seed value as a base to generate a random number.
random_walk = list()
random_walk.append(-1 if random() < 0.5 else 1)
for i in range(1, 1000):
    movement = -1 if random() < 0.5 else 1
    value = random_walk[i - 1] + movement
    random_walk.append(value)
pyplot.plot(random_walk)
pyplot.show()

# ACF plot
# autocorrelation_plot(random_walk)
plot_acf(random_walk, lags=100)


# 2. ADF Test
# The null hypothesis of the test is that the time series can be represented by a unit root,
# that it is not stationary (has some time-dependent structure).
# The alternate hypothesis (rejecting the null hypothesis) is that the time series is stationary.
# We interpret this result using the p-value from the test.
# A p-value below a threshold (such as 5% or 1%) suggests we reject the null hypothesis (stationary),
# otherwise a p-value above the threshold suggests we fail to reject the null hypothesis (non-stationary).
result = adfuller(random_walk)
print(result)
print("ADF Statistic: %f" % result[0])
print("p-value: %f" % result[1])
for key, value in result[4].items():
    print("\t%s: %.3f" % (key, value))

# 3. Take the difference of the series to make it stationary
random_walk1 = np.diff(np.array(random_walk))
result = adfuller(random_walk1)
print("ADF Statistic: %f" % result[0])
print("p-value: %f" % result[1])
for key, value in result[4].items():
    print("\t%s: %.3f" % (key, value))

# 4. Now compute the ACF for the stationary time series
plot_acf(random_walk1, lags=100)


train, test = (
    random_walk[0 : len(random_walk) - 150],
    random_walk[len(random_walk) - 150 :],
)
# train autoregression
model = AutoReg(train, lags=1)
model_fit = model.fit()
print("Coefficients: %s" % model_fit.params)

###############################################


# reading the dataset
series = pd.read_csv(
    "/Users/saeedkazemi/OneDrive - University of New Brunswick/The fall Term 2020/ECE6563/codes/AirPassengers.csv"
)
# the dataset contains number of passengers per month between Jan 1949-Dec. 1960
X = series["#Passengers"].values


plt.plot(X, label="Passengers")
plt.legend(loc="upper left")


result = adfuller(X)
print("ADF Statistic: %f" % result[0])
print("p-value: %f" % result[1])
for key, value in result[4].items():
    print("\t%s: %.3f" % (key, value))

# ACF Plot
plot_acf(X, lags=50)
# x = pd.plotting.autocorrelation_plot(X)

# check the transformed series
# 1. Transformation
series["#Passengers_log"] = np.log(series["#Passengers"])
# series['#Passengers_log'].dropna().plot(label="Log Difference")
X_lg = series["#Passengers_log"]
result = adfuller(X_lg)
print("ADF Statistic: %f" % result[0])
print("p-value: %f" % result[1])
for key, value in result[4].items():
    print("\t%s: %.3f" % (key, value))

plot_acf(X_lg, lags=50)

# 2. Lets take first difference
X_lg_1diff = np.diff(X_lg, 1)
result = adfuller(X_lg_1diff)
print("ADF Statistic: %f" % result[0])
print("p-value: %f" % result[1])
for key, value in result[4].items():
    print("\t%s: %.3f" % (key, value))
plot_acf(X_lg_1diff, lags=50)

# 3. Lets take 12th difference
X_lg_12diff = np.diff(X_lg_1diff, 12)
result = adfuller(X_lg_12diff)
print("ADF Statistic: %f" % result[0])
print("p-value: %f" % result[1])
for key, value in result[4].items():
    print("\t%s: %.3f" % (key, value))

plot_acf(X_lg_12diff, lags=50)


# 4. AR model

train, test = (
    X_lg_12diff[0 : len(X_lg_12diff) - 15],
    X_lg_12diff[len(X_lg_12diff) - 15 :],
)
# train autoregression
model = AutoReg(train, lags=1)
model_fit = model.fit()
print("Coefficients: %s" % model_fit.params)
# make predictions
predictions = model_fit.predict(
    start=len(X_lg_12diff) - 14, end=len(X_lg_12diff), dynamic=False
)
for i in range(len(predictions)):
    print("predicted=%f, expected=%f" % (predictions[i], test[i]))
rmse = sqrt(mean_squared_error(test, predictions))
print("Test RMSE: %.3f" % rmse)
# plot results
pyplot.plot(test)
pyplot.plot(predictions, color="red")
pyplot.show()
