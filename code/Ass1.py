# import seaborn as sns
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from pandas import read_csv
from statsmodels.tsa import stattools
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa import seasonal
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error


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
series = read_csv(
    dataset_file,
    header=0,
    parse_dates=[0],
    index_col=0,
    squeeze=True,
)


############################################################################
#########      Saving and showing the plot of the raw signal      ##########
############################################################################

print("[INFO] Saving and showing the plot of the second dataset")
plt.figure(0)
fig = series.plot()
plt.savefig(os.path.join(fig_dir, a + "raw_signal.png"))

print("[INFO] Saving and printing the head of the first dataset")
with open(os.path.join(tbl_dir, a + "raw_signal.tex"), "w") as tf:
    tf.write(series.head(7).to_latex(index=True))


############################################################################
#########        Decompositions: seasonal_decompose method        ##########
############################################################################
"""
## seasonal_decompose(x[, model, filt, period, …])   Seasonal decomposition using moving averages.
## STL(endog[, period, seasonal, trend, …])          Season-Trend decomposition using LOESS.
## Multiplicative seasonality is not appropriate for zero and negative values
"""
print("[INFO] Plot the decomposition by seasonal_decompose method...")
decomposition_sd = seasonal.seasonal_decompose(
    series, model="additive", extrapolate_trend="freq", period=365
)
fig = decomposition_sd.plot()
plt.savefig(os.path.join(fig_dir, a + "seasonal_decompose.png"))


############################################################################
#########               Decompositions: STL method                ##########
############################################################################

print("[INFO] Plot the decomposition by STL method...")
decomposition_STL = seasonal.STL(series, period=365).fit()
fig = decomposition_STL.plot()
plt.savefig(os.path.join(fig_dir, a + "STL.png"))


############################################################################
#########         Decompositions: LinearRegression method         ##########
############################################################################

print("[INFO] Plot the decomposition by LinearRegression method...")
X = [i for i in range(0, len(series))]
X = np.reshape(X, (len(X), 1))
y = series.values
model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)

# plot the raw signal
plt.subplot(411)
plt.plot(y)
plt.ylabel("the raw signal")

# plot trend
plt.subplot(412)
plt.plot(trend)
plt.ylabel("trend")
# detrending
detrended = [y[i] - trend[i] for i in range(0, len(series))]
# plot Detrended
plt.subplot(413)
plt.plot(detrended)
plt.ylabel("seasonal")

############################################################################
#########          Decompositions: Differencing method            ##########
############################################################################
"""https://machinelearningmastery.com/time-series-seasonality-with-python/"""
diff = list()
period = 365
for i in range(period, len(X)):
    value = detrended[i] - detrended[i - period]
    diff.append(value)

# plot Residual
plt.subplot(414)
plt.plot(diff)
plt.ylabel("residual")
plt.savefig(os.path.join(fig_dir, a + "LinearRegression_diff.png"))


############################################################################
#########               stationary test: ADF method               ##########
############################################################################

"""residual = {decomposition_sd.resid, decomposition_STL.resid, diff}"""
residual = diff

print("[INFO] ACF plot for residual component...")
plot_acf(residual, lags=100)
plt.savefig(os.path.join(fig_dir, a + "ADF.png"))

print("[INFO] Results of Dickey-Fuller Test:")
result = stattools.adfuller(residual, autolag="AIC")
dfoutput = pd.Series(
    result[0:4],
    index=["ADF Statistic", "p-value", "#Lags Used", "Number of Observations Used"],
)
for key, value in result[4].items():
    dfoutput["Critical Value (%s)" % key] = value
print(dfoutput)

print("[INFO] saving Results of Dickey-Fuller Test on file...")
with open(os.path.join(tbl_dir, a + "ADF.tex"), "w") as tf:
    tf.write(dfoutput.to_latex(index=True))

############################################################################
#########              stationary test: KPSS method               ##########
############################################################################
"""
## https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html

The p-value is obtained is greater than significance level of 0.05 and
the ADF statistic is higher than any of the critical values.
Clearly, there is no reason to reject the null hypothesis. So,
the time series is in fact non-stationary.

The p-value is very less than the significance level of 0.05 and
hence we can reject the null hypothesis and take that the series is stationary.

Case 1: Both tests conclude that the series is not stationary - The series is not stationary
Case 2: Both tests conclude that the series is stationary - The series is stationary
Case 3: KPSS indicates stationarity and ADF indicates non-stationarity - The series is trend stationary. Trend needs to be removed to make series strict stationary. The detrended series is checked for stationarity.
Case 4: KPSS indicates non-stationarity and ADF indicates stationarity - The series is difference stationary. Differencing is to be used to make series stationary. The differenced series is checked for stationarity.
"""
print("[INFO] Results of KPSS Test:")
Results = stattools.kpss(residual, regression="c", nlags="auto")
kpss_output = pd.Series(Results[0:3], index=["KPSS Statistic", "p-value", "Lags Used"])
for key, value in Results[3].items():
    kpss_output["Critical Value (%s)" % key] = value
print(kpss_output)

print("[INFO] saving Results of KPSS Test on file...")
with open(os.path.join(tbl_dir, a + "KPSS.tex"), "w") as tf:
    tf.write(kpss_output.to_latex(index=True))


############################################################################
#########               stationary test: PACF method              ##########
############################################################################
### Lag Plots, and/or other approaches
### https://towardsdatascience.com/detecting-stationarity-in-time-series-data-d29e0a21e638
"""You see how ACF is declining in amplitude exponentially, while PACF cuts off after lag 1.
This may suggest that you're dealing with AR(1) process.
How do I know this? If you derive the ACF and PACF assuming that your process is AR(1),
which includes constant variance of errors, then you'd come up with a similar shaped curves.
Hence, if your underlying series are not stationary, you're breaking the assumptions that 
are base for the heuristics that I mentioned about ACF/PACF. It's pointless to apply these on 
non-stationary series, since you can't make any conclusions about the lag structure anymore.
"""
print("[INFO] PACF plot for residual component...")
PACF_output = stattools.pacf(residual)
plt.stem(PACF_output)
plt.savefig(os.path.join(fig_dir, a + "PACF.png"))

############################################################################
#########           Predict Approach of Auto Regression           ##########
############################################################################
print("[INFO] spliting dataset to train and test set...")
npseries = series.to_numpy()
train, test = (
    npseries[0 : int(len(npseries) - (len(npseries) * 0.10))],
    npseries[int(len(npseries) - (len(npseries) * 0.10) + 1) :],
)

print("[INFO] train autoregression...")
model = AutoReg(train, lags=1)
model_fit = model.fit()
print("[INFO] Coefficients: %s" % model_fit.params)


print("[INFO] make predictions on test set...")
predictions = model_fit.predict(
    start=int(len(npseries) - (len(npseries) * 0.10) + 1),
    end=len(npseries),
    dynamic=False,
)
print(predictions.shape)
print(test.shape)
for i in range(len(predictions) - 1):
    print("predicted=%f, expected=%f" % (predictions[i], test[i]))
rmse = math.sqrt(mean_squared_error(test, predictions))
print("Test RMSE: %.3f" % rmse)

# plot results
plt.plot(test)
plt.plot(predictions, color="red")
plt.show()
##https://towardsdatascience.com/trend-seasonality-moving-average-auto-regressive-model-my-journey-to-time-series-data-with-edc4c0c8284b