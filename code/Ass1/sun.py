"""https://www.machinelearningplus.com/time-series/time-series-analysis-python/"""
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
a = "Ass1_D2_"

print("[INFO] Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "manuscript", "src", "figures", "Ass1")
tbl_dir = os.path.join(project_dir, "manuscript", "src", "tables", "Ass1")
data_dir = os.path.join(project_dir, "Dataset", "Ass1")
dataset_file = os.path.join(data_dir, "monthly-sunspots.csv")
# dataset_file = os.path.join(data_dir, "temperatures.csv")


print("[INFO] Reading the first dataset")
series = pd.read_csv(
    dataset_file,
    header=0,
    index_col=0,
).dropna()

series.index = pd.to_datetime(series.index)


############################################################################
#########      Saving and showing the plot of the raw signal      ##########
############################################################################

print("[INFO] Saving and showing the plot of the first dataset")
plt.figure(0)
fig = series.plot()
plt.savefig(os.path.join(fig_dir, a + "raw_signal.png"))

# plt.figure()
# plt.plot(series["1990":"1991"])
# plt.savefig(os.path.join(fig_dir, a + "raw_signal_1990.png"))

# plt.figure()
# plt.plot(series.loc["1986"])
# plt.savefig(os.path.join(fig_dir, a + "raw_signal_1986.png"))


print("[INFO] Saving and printing the head of the first dataset")
with open(os.path.join(tbl_dir, a + "raw_signal.tex"), "w") as tf:
    tf.write(series.head(5).to_latex())


print(series.describe())
with open(os.path.join(tbl_dir, a + "raw_signal_summary_statistics.tex"), "w") as tf:
    tf.write(series.describe().to_latex())

############################################################################
#########         Decompositions: Moving Avrage function          ##########
############################################################################
print("[INFO] Saving and showing the plot of Moving Avrage function")

# r.agg, r.apply, r.count, r.exclusions, r.max, r.median, r.name, r.quantile, r.kurt, r.cov, r.corr, r.aggregate, r.std, r.skew, r.sum, r.var
r = series.rolling(window=100)


r.mean().plot(style="b", linewidth=3)
plt.plot(series, "r--")
plt.legend(["input data", "Seasonal component"])
plt.savefig(os.path.join(fig_dir, a + "Moving_Avrage.png"))


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
#########          Seasonal Modeling (fiting polynomial)          ##########
############################################################################
print("[INFO] Plot the decomposition by fiting polynomial method...")
X = [i % 365 for i in range(0, len(series))]
y = series.values

degree = 5
coef = np.polyfit(X, y, degree)
print("[INFO] polynomial Coefficients are :\n%s\n" % coef)


curve = list()
for i in range(len(X)):
    value = 13
    for d in range(degree):
        value += X[i] ** (degree - d) * coef[d]
    curve.append(value)

plt.figure()
plt.plot(y)
plt.plot(curve, color="red", linewidth=3)
plt.legend(["input data", "Seasonal component"])
plt.savefig(os.path.join(fig_dir, a + "fiting_polynomial.png"))


############################################################################
#########          Decompositions: Differencing method            ##########
############################################################################
print("[INFO] Plot the detrending signal by Differencing method...")
diff = list()
period = 1
for i in range(period, len(X)):
    value = series.values[i] - series.values[i - period]
    diff.append(value)

plt.figure()
plt.plot(series.values[:-1] - diff)
plt.ylabel("detrend Signal")
plt.savefig(os.path.join(fig_dir, a + "one_diff.png"))


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
print("[INFO] Plot the seasonal signal by Differencing method...")

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
residual = decomposition_sd.resid

print("[INFO] ACF plot for residual component...")
plt.figure()
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
plt.figure()
plt.stem(PACF_output)
plt.savefig(os.path.join(fig_dir, a + "PACF.png"))

plt.figure()
fig, axes = plt.subplots(2, 1)
plot_acf(residual, lags=50, ax=axes[0])
plot_pacf(residual, lags=50, ax=axes[1])
plt.savefig(os.path.join(fig_dir, a + "PACF_ACF.png"))


############################################################################
#########               stationary test: Lag Plots                ##########
############################################################################

'''(Points get wide and scattered with increasing lag -> lesser correlation)\n"'''
print("[INFO] Lag plot for residual component...")

fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(series, lag=i + 1, ax=ax, c="firebrick")
    ax.set_title("Lag " + str(i + 1))

fig.suptitle("Lag Plots of the Dataset")
plt.savefig(os.path.join(fig_dir, a + "Lag_Plots.png"))


fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(residual, lag=i + 1, ax=ax, c="firebrick")
    ax.set_title("Lag " + str(i + 1))

fig.suptitle("Lag Plots of the residual component")
plt.savefig(os.path.join(fig_dir, a + "Lag_Plots_residual.png"))

############################################################################
#########           Predict Approach of Auto Regression           ##########
############################################################################
"""
https://towardsdatascience.com/trend-seasonality-moving-average-auto-regressive-model-my-journey-to-time-series-data-with-edc4c0c8284b
"""
X = decomposition_STL.resid.values

print("[INFO] spliting dataset to train and test set...")
n_predict = 14
train, test = X[1 : len(X) - n_predict], X[len(X) - n_predict :]


print("[INFO] train autoregression...")
model = AutoReg(train, lags=2)
model_fit = model.fit()


print("[INFO] Coefficients:\n%s\n" % (model_fit.params))
# make predictions
predictions = model_fit.predict(
    start=len(train), end=len(train) + len(test) - 1, dynamic=False
)

print("[INFO] AutoReg Model Results (summary): ")
print(model_fit.summary())

for i in range(len(predictions)):
    print("\tPredicted={:1.2f},\t\tExpected={:1.2f}".format(predictions[i], test[i]))
rmse = np.sqrt(mean_squared_error(test, predictions))
print("\n[INFO] RMS Error: {:1.2f}".format(rmse))


plt.figure()
plt.plot(test, color="red")
plt.plot(predictions, color="blue")
plt.legend(["Test", "Predictions"])
plt.savefig(os.path.join(fig_dir, a + "AR.png"))


plt.figure()
xpos = np.arange(len(X))
plt.plot(X, "r", linewidth=0.5)
plt.plot(xpos[len(X) - n_predict : len(X)], predictions[:], color="blue")
plt.legend(["train+Test", "Predictions"])
plt.savefig(os.path.join(fig_dir, a + "AR1.png"))

plt.figure()
xpos = np.arange(len(X))
plt.plot(X[3500:], "r", linewidth=0.5)
plt.plot(xpos[len(X) - n_predict - 3500 : len(X) - 3500], predictions[:], color="blue")
plt.legend(["train+Test", "Predictions"])
plt.savefig(os.path.join(fig_dir, a + "AR2.png"))
