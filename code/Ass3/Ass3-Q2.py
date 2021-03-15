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
a = "Ass3_Q2_"

print("[INFO] Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "manuscript", "src", "figures", "Ass3")
tbl_dir = os.path.join(project_dir, "manuscript", "src", "tables", "Ass3")
data_dir = os.path.join(project_dir, "Dataset", "Ass3")
dataset_file = os.path.join(data_dir, "1.csv")


print("[INFO] Reading the first dataset")
df = pd.read_csv(
    dataset_file,
    header=0,
    names=[
        "sequential_number",
        "x_acceleration",
        "y_acceleration",
        "z_acceleration",
        "label",
    ],
    index_col=0,
).dropna()
# df["color"] = 0
# colors = [
#     "#1f77b4",
#     "#ff7f0e",
#     "#2ca02c",
#     "#f61600",
#     "#7834a9",
#     "#17becf",
#     "#684427",
#     "#fa5deb",
#     "#17becf",
#     "#17becf",
# ]
# df1 = pd.DataFrame(
#     {
#         "label": range(10),
#         "colors": colors,
#     }
# )
# df2 = pd.merge(df, df1, on="label", how="left")

# print(df1.head())
# print(df2.head())


# # Creating figure
# fig = plt.figure()
# ax = plt.axes(projection="3d")

# # Add x, y gridlines
# ax.grid(b=True, color="grey", linestyle="-.", linewidth=0.3, alpha=0.2)


# # for data, color, group in zip(data, colors, groups):
# # x, y = data
# # Creating plot
# print()
# sctt = ax.scatter3D(
#     df2["x_acceleration"],
#     df2["y_acceleration"],
#     df2["z_acceleration"],
#     alpha=0.8,
#     c=df2["colors"],
#     marker="+",
# )

# plt.title("simple 3D scatter plot")
# ax.set_xlabel("X-axis", fontweight="bold")
# ax.set_ylabel("Y-axis", fontweight="bold")
# ax.set_zlabel("Z-axis", fontweight="bold")
# fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)

# # show plot
# plt.show()
# plt.plot()
# 1 / 0
############################################################################
#########      Saving and showing the plot of the raw signal      ##########
############################################################################

print("[INFO] Saving and showing the plot of the dataset")
plt.figure()
fig = df["x_acceleration"].plot(label="x_acceleration")
fig = df["y_acceleration"].plot(label="y_acceleration")
fig = df["z_acceleration"].plot(label="z_acceleration")
fig = df["label"].plot(label="label")
plt.legend()
plt.savefig(os.path.join(fig_dir, a + "raw_signal.png"))
1 / 0
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
print(train.shape)

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
