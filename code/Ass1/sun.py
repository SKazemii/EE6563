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

plt.close("all")
############################################################################
#########      Saving and showing the plot of the raw signal      ##########
############################################################################

print("[INFO] Saving and showing the plot of the first dataset")
plt.figure(0)
fig = series.plot()
plt.savefig(os.path.join(fig_dir, a + "raw_signal.png"))


plt.figure()
plt.plot(series["1749":"1800"])
plt.savefig(os.path.join(fig_dir, a + "raw_signal_1990.png"))


series_diff = series.copy(deep=True)
series_diff["Sunspots"] = series.diff().dropna()
series_diff.dropna(inplace=True)
plt.figure()
fig = plt.plot(series_diff)
plt.savefig(os.path.join(fig_dir, a + "1_diff_raw_signal.png"))


series_2diff = series.copy(deep=True)
series_2diff["Sunspots"] = series.diff().diff().dropna()
series_2diff.dropna(inplace=True)
fig = plt.plot(series_2diff)
plt.savefig(os.path.join(fig_dir, a + "2_diff_raw_signal.png"))


print("[INFO] Saving and printing the head of the first dataset")
with open(os.path.join(tbl_dir, a + "raw_signal.tex"), "w") as tf:
    tf.write(series.head(5).to_latex())


print(series.describe())
with open(os.path.join(tbl_dir, a + "raw_signal_summary_statistics.tex"), "w") as tf:
    tf.write(series.describe().to_latex())

############################################################################
#########        Decompositions: Moving Avarage function          ##########
############################################################################
print("[INFO] Saving and showing the plot of Moving Avarage function")

# r.agg, r.apply, r.count, r.exclusions, r.max, r.median, r.name, r.quantile, r.kurt, r.cov, r.corr, r.aggregate, r.std, r.skew, r.sum, r.var
r = series.rolling(window=12)

axes = plt.axes()
series.plot(color="red", ax=axes)
r.mean().plot(style="b", linewidth=3, ax=axes)

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
    series, model="additive", extrapolate_trend="freq"  # , period=120
)

fig = decomposition_sd.plot()
plt.savefig(os.path.join(fig_dir, a + "seasonal_decompose.png"))


############################################################################
#########          Seasonal Modeling (fitting polynomial)         ##########
############################################################################
resample = series.resample("AS").mean()
plt.figure()
plt.plot(resample)
plt.savefig(os.path.join(fig_dir, a + "resample.png"))


print("[INFO] Plot the decomposition by fitting polynomial method...")
X = [i % 120 for i in range(0, len(series))]
y = series.values

degree = 2
coef = np.polyfit(X, y, degree)
print("[INFO] polynomial Coefficients are :\n%s\n" % coef)


curve = list()
for i in range(len(X)):
    value = 88
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
decomposition_STL = seasonal.STL(series, period=130).fit()
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
period = 120
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
residual = decomposition_STL.resid

print("[INFO] ACF plot for residual component...")
plt.figure()
plot_acf(residual, lags=100)
plt.savefig(os.path.join(fig_dir, a + "ACF.png"))

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


print("[INFO] ACF plot for the first order diff...")
plt.figure()
plot_acf(series_diff, lags=100)
plt.savefig(os.path.join(fig_dir, a + "ACF_1_diff.png"))

print("[INFO] Results of Dickey-Fuller Test:")
result = stattools.adfuller(series_diff, autolag="AIC")
dfoutput = pd.Series(
    result[0:4],
    index=["ADF Statistic", "p-value", "#Lags Used", "Number of Observations Used"],
)
for key, value in result[4].items():
    dfoutput["Critical Value (%s)" % key] = value
print(dfoutput)

print("[INFO] saving Results of Dickey-Fuller Test on file...")
with open(os.path.join(tbl_dir, a + "ADF_1_diff.tex"), "w") as tf:
    tf.write(dfoutput.to_latex(index=True))


print("[INFO] ACF plot for the second order diff...")
plt.figure()
plot_acf(series_2diff, lags=100)
plt.savefig(os.path.join(fig_dir, a + "ACF_2_diff.png"))

print("[INFO] Results of Dickey-Fuller Test:")
result = stattools.adfuller(series_2diff, autolag="AIC")
dfoutput = pd.Series(
    result[0:4],
    index=["ADF Statistic", "p-value", "#Lags Used", "Number of Observations Used"],
)
for key, value in result[4].items():
    dfoutput["Critical Value (%s)" % key] = value
print(dfoutput)

print("[INFO] saving Results of Dickey-Fuller Test on file...")
with open(os.path.join(tbl_dir, a + "ADF_2_diff.tex"), "w") as tf:
    tf.write(dfoutput.to_latex(index=True))


############################################################################
#########              stationary test: KPSS method               ##########
############################################################################
"""
## https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
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


print("[INFO] Results of KPSS Test:")
Results = stattools.kpss(series_diff, regression="c", nlags="auto")
kpss_output = pd.Series(Results[0:3], index=["KPSS Statistic", "p-value", "Lags Used"])
for key, value in Results[3].items():
    kpss_output["Critical Value (%s)" % key] = value
print(kpss_output)

print("[INFO] saving Results of KPSS Test on file...")
with open(os.path.join(tbl_dir, a + "KPSS_1_diff.tex"), "w") as tf:
    tf.write(kpss_output.to_latex(index=True))

print("[INFO] Results of KPSS Test:")
Results = stattools.kpss(series_2diff, regression="c", nlags="auto")
kpss_output = pd.Series(Results[0:3], index=["KPSS Statistic", "p-value", "Lags Used"])
for key, value in Results[3].items():
    kpss_output["Critical Value (%s)" % key] = value
print(kpss_output)

print("[INFO] saving Results of KPSS Test on file...")
with open(os.path.join(tbl_dir, a + "KPSS_2_diff.tex"), "w") as tf:
    tf.write(kpss_output.to_latex(index=True))

############################################################################
#########               stationary test: PACF method              ##########
############################################################################
""" https://towardsdatascience.com/detecting-stationarity-in-time-series-data-d29e0a21e638
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

plt.figure()
fig, axes = plt.subplots(2, 1)
plot_acf(series_diff, lags=50, ax=axes[0])
plot_pacf(series_diff, lags=50, ax=axes[1])
plt.savefig(os.path.join(fig_dir, a + "PACF_ACF_1_diff.png"))

plt.figure()
fig, axes = plt.subplots(2, 1)
plot_acf(series_2diff, lags=50, ax=axes[0])
plot_pacf(series_2diff, lags=50, ax=axes[1])
plt.savefig(os.path.join(fig_dir, a + "PACF_ACF_2_diff.png"))
############################################################################
#########               stationary test: Lag Plots                ##########
############################################################################

'''(Points get wide and scattered with increasing lag -> lesser correlation)\n"'''
print("[INFO] Lag plot for residual component...")

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(series, lag=i + 1, ax=ax, c="firebrick")
    ax.set_title("Lag " + str(i + 1))

fig.suptitle("Lag Plots of the Dataset")
plt.savefig(os.path.join(fig_dir, a + "Lag_Plots.png"))


fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(residual, lag=i + 1, ax=ax, c="firebrick")
    ax.set_title("Lag " + str(i + 1))

fig.suptitle("Lag Plots of the residual component")
plt.savefig(os.path.join(fig_dir, a + "Lag_Plots_residual.png"))

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(series_diff, lag=i + 1, ax=ax, c="firebrick")
    ax.set_title("Lag " + str(i + 1))

fig.suptitle("Lag Plots of the first order diff")
plt.savefig(os.path.join(fig_dir, a + "Lag_Plots_1_diff.png"))

fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, dpi=100)
for i, ax in enumerate(axes.flatten()[:4]):
    lag_plot(series_2diff, lag=i + 1, ax=ax, c="firebrick")
    ax.set_title("Lag " + str(i + 1))

fig.suptitle("Lag Plots of the second order diff")
plt.savefig(os.path.join(fig_dir, a + "Lag_Plots_2_diff.png"))
############################################################################
#########           Predict Approach of Auto Regression           ##########
############################################################################
"""
https://towardsdatascience.com/trend-seasonality-moving-average-auto-regressive-model-my-journey-to-time-series-data-with-edc4c0c8284b
"""
X = decomposition_STL.resid.values

print("[INFO] spliting dataset to train and test set...")
n_predict = 30
train, test = X[1 : len(X) - n_predict], X[len(X) - n_predict :]


print("[INFO] train autoregression...")
model = AutoReg(train, lags=6)
model_fit = model.fit()


print("[INFO] Coefficients:\n%s\n" % (model_fit.params))
# make predictions
predictions = model_fit.predict(
    start=len(train), end=len(train) + len(test) - 1, dynamic=False
)
# predictions = model_fit.forecast(n_predict, dynamic=False)

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
plt.plot(X[2700:], "r", linewidth=0.5)
plt.plot(xpos[len(X) - n_predict - 2700 : len(X) - 2700], predictions[:], color="blue")
plt.legend(["train+Test", "Predictions"])
plt.savefig(os.path.join(fig_dir, a + "AR2.png"))
