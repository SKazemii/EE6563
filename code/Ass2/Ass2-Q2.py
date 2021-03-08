# Autoregressive Integrated Moving Averages
# The general process for ARIMA models is the following:

# Visualize the Time Series Data
# Make the time series data stationary
# Plot the Correlation and AutoCorrelation Charts
# Construct the ARIMA Model or Seasonal ARIMA based on the data
# Use the model to make predictions
# Let's go through these steps!


# Final Thoughts on Autocorrelation and Partial Autocorrelation¶
# Identification of an AR model is often best done with the PACF.
# For an AR model, the theoretical PACF “shuts off” past the order of the model. The phrase “shuts off” means that in theory the partial autocorrelations are equal to 0 beyond that point. Put another way, the number of non-zero partial autocorrelations gives the order of the AR model. By the “order of the model” we mean the most extreme lag of x that is used as a predictor.
# Identification of an MA model is often best done with the ACF rather than the PACF.

# For an MA model, the theoretical PACF does not shut off, but instead tapers toward 0 in some manner. A clearer pattern for an MA model is in the ACF. The ACF will have non-zero autocorrelations only at lags involved in the model.

# p,d,q p AR model lags d differencing q MA lags


# (Null hypothesis) H0: Xt does not granger causes Yt.
# (Alternate hypothesis) H1: Xt granger causes Yt.
# If P-value is less than 5% (or 0.05), then we can reject the Null hypothesis (H0), and can conclude that Xt granger causes Yt.

# So where ever your P-value is less than 0.05, you can consider those features.
# Grange causality means that past values of x2 have a statistically significant effect
# on the current value of x1, taking past values of x1 into account as regressors.

# df["lag"] = df["diff2M"].shift().shift()
# df.dropna(inplace=True)
# model3 = ARIMA(endog=df["Sales"], exog=df[["lag"]], order=[1, 2, 0])
# results3 = model3.fit()
# print(results3.summary())
import warnings

warnings.filterwarnings("ignore")

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime
import pmdarima as pm

from pandas.plotting import lag_plot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa import seasonal, stattools
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA, ARIMA


plt.rcParams["figure.figsize"] = (14, 7)
a = "Ass2_Q2_"

print("[INFO] Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "manuscript", "src", "figures", "Ass2")
tbl_dir = os.path.join(project_dir, "manuscript", "src", "tables", "Ass2")
data_dir = os.path.join(project_dir, "Dataset", "Ass2")
dataset_file = os.path.join(data_dir, "Processed_NYSE.csv")


def parser(x):
    return datetime.strptime(x, "%Y-%m-%d")


print("[INFO] Reading the first dataset")
series = pd.read_csv(
    dataset_file,
    header=0,
    index_col=0,
    parse_dates=True,
    squeeze=True,
    date_parser=parser,
).dropna()


############################################################################
#########      Saving and showing the plot of the raw signal      ##########
############################################################################

print("[INFO] Saving and showing the plot of the dataset")
plt.figure()
fig = series["Close"].plot(label="Close price")
plt.legend()
plt.savefig(os.path.join(fig_dir, a + "raw_signal.png"))

############################################################################
#########               stationary test: PACF method              ##########
############################################################################
print("[INFO] Saving and showing the PACF and ACF plot ")
plt.figure()
fig, axes = plt.subplots(2, 1)
plot_acf(series["Close"], lags=250, ax=axes[0])
plot_pacf(series["Close"], lags=250, ax=axes[1])
plt.savefig(os.path.join(fig_dir, a + "PACF_ACF.png"))


series_diff = series["Close"].diff()
series_diff.dropna(inplace=True)
plt.figure()
fig = plt.plot(series_diff)
plt.savefig(os.path.join(fig_dir, a + "1diff_signal.png"))


plt.figure()
fig, axes = plt.subplots(2, 1)
plot_acf(series_diff, lags=50, ax=axes[0])
plot_pacf(series_diff, lags=50, ax=axes[1])
plt.savefig(os.path.join(fig_dir, a + "PACF_ACF_1diff.png"))

print("[INFO] Results of Dickey-Fuller Test:")
result = stattools.adfuller(series_diff, autolag="AIC")
dfoutput = pd.Series(
    result[0:4],
    index=["ADF Statistic", "p-value", "#Lags Used", "Number of Observations Used"],
)


for key, value in result[4].items():
    dfoutput["Critical Value (%s)" % key] = value

print("[INFO] saving Results of Dickey-Fuller Test on file...")
with open(os.path.join(tbl_dir, a + "ADF_1diff.tex"), "w") as tf:
    tf.write(dfoutput.to_latex(index=True))


############################################################################
#########                     fitting ARIMA                       ##########
############################################################################
nobs = 35
orders = [1, 1, 1]

print("[INFO] Create Training and Test set...")
train = series["Close"][:-nobs]
test = series["Close"][-nobs:]

model = ARIMA(train, order=orders)
model_fitted = model.fit(disp=-1)

# print(model_fitted.summary())


print("[INFO] saving the plot of residual...")
residuals = pd.Series(model_fitted.resid, index=train.index)
fig, ax = plt.subplots(2, 1)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind="kde", title="Density", ax=ax[1])
# plot_acf(residuals, lags=50, title="ACF", ax=ax[2])
plt.savefig(os.path.join(fig_dir, a + "residual_plot.png"))


with open(os.path.join(tbl_dir, a + "residuals_statistics.tex"), "w") as tf:
    tf.write(residuals.describe().to_latex())

plt.figure()
model_fitted.plot_predict()
plt.savefig(os.path.join(fig_dir, a + "0000.png"))


############################################################################
#########                         Forecast                         #########
############################################################################
fc, se, conf = model_fitted.forecast(nobs, alpha=0.05)

fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)


print("[INFO] saving the plot of forecast...")
plt.figure()
plt.plot(train, label="Training set")
plt.plot(test, label="Testing set")
plt.plot(fc_series, label="Forecast")
plt.fill_between(lower_series.index, lower_series, upper_series, color="k", alpha=0.1)
plt.title("Forecast vs Actuals")
plt.legend(loc="upper left", fontsize=12)
plt.savefig(os.path.join(fig_dir, a + "Forecast_vs_Actuals.png"))

print("[INFO] evaluate forecasts...")
rmse = np.sqrt(mean_squared_error(test, fc_series))
print("[INFO] Test RMS error: %.3f" % rmse)
############################################################################
#########       Automatically find the order of ARIMA model.      ##########
############################################################################
best_model = pm.auto_arima(
    train,
    start_p=1,
    start_q=1,
    test="adf",
    max_p=4,
    max_q=4,
    m=1,
    seasonal=False,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True,
)
# print(best_model.summary())

# How to interpret the residual plots in ARIMA model
best_model.plot_diagnostics()
plt.savefig(os.path.join(fig_dir, a + "best_model_residual.png"))
# Bottom left:Normal Q-Q plot. All the dots should fall perfectly in line with the red line.
# Any significant deviations would imply the distribution is skewed.


############################################################################
#########                         Forecast                         #########
############################################################################
fc, confint = best_model.predict(n_periods=nobs, return_conf_int=True)
index_of_fc = np.arange(len(train), len(train) + nobs)


fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(confint[:, 0], index=test.index)
upper_series = pd.Series(confint[:, 1], index=test.index)


print("[INFO] saving the plot of forecast...")
plt.figure()
train.plot(label="Train")
test.plot(label="Test")
plt.plot(fc_series, color="darkgreen", label="Predicted")
plt.fill_between(
    lower_series.index,
    lower_series,
    upper_series,
    color="k",
    alpha=0.1,
    label="Confidence Interval",
)
plt.legend()
plt.savefig(os.path.join(fig_dir, a + "Automatic_model_forcasting.png"))

print("[INFO] evaluate forecasts...")
rmse = np.sqrt(mean_squared_error(test, fc_series))
print("[INFO] Test RMS error: %.3f" % rmse)

############################################################################
#########            Rolling Forecast of ARIMA model.             ##########
############################################################################

X = series["Close"].values
size = len(X) - nobs
window = 570  # = 3 * 365
train, test = X[window:size], X[size : len(X)]
history = [x for x in train]
predictions = list()


plt.figure()
plt.plot(series["Close"][window:-nobs], label="train set")
plt.legend()
plt.savefig(os.path.join(fig_dir, a + "Rolling_trainset.png"))


for t in range(len(test)):
    model = ARIMA(history, order=orders)
    model_fit = model.fit(disp=-1)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    history = history[1:]


print("[INFO] evaluate forecasts...")
rmse = np.sqrt(mean_squared_error(test, predictions))
print("[INFO] Test RMS error: %.3f" % rmse)

print("[INFO] saving the plot of forecast...")
plt.figure()
plt.plot(test, label="test set")
plt.plot(predictions, color="red", label="predictions")
plt.legend()
plt.savefig(os.path.join(fig_dir, a + "Rolling_Forecast.png"))


plt.figure()
plt.plot(np.arange(0, len(train)), train, label="Train")
plt.plot(
    np.arange(len(train), len(train) + len(test)),
    test,
    color="red",
    label="Test",
)
plt.plot(
    np.arange(len(train), len(train) + len(test)),
    predictions,
    color="darkgreen",
    label="Predicted",
    dashes=[6, 2],
)
plt.legend()
plt.savefig(os.path.join(fig_dir, a + "Rolling_Forecast_1.png"))
