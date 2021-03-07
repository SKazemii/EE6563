import warnings

warnings.filterwarnings("ignore")

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from datetime import datetime
import pmdarima as pm

from pandas.plotting import lag_plot
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa import seasonal, stattools
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.api import VAR


plt.rcParams["figure.figsize"] = (14, 7)
a = "Ass2_Q4_"

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
#########                      Standardzation                     ##########
############################################################################
Standard_scaler = preprocessing.StandardScaler()
series_scaled = Standard_scaler.fit_transform(series.iloc[:, 0:9].values)
df_scaled = pd.DataFrame(series_scaled, index=series.index, columns=series.columns[0:9])

############################################################################
#########      Saving and showing the plot of the raw signal      ##########
############################################################################

print("[INFO] Saving and showing the plot of the first dataset")
df = list()
fig, axes = plt.subplots(nrows=3, ncols=3, dpi=120, figsize=(10, 6))

ADF_result = pd.DataFrame(
    np.zeros([7, 9]),
    columns=series.columns[0:9],
    index=[
        "ADF Statistic",
        "p-value",
        "#Lags Used",
        "Number of Observations Used",
        "Critical Value (1%)",
        "Critical Value (5%)",
        "Critical Value (10%)",
    ],
)
for i, ax in enumerate(axes.flatten()):
    data = df_scaled[series.columns[i]]
    ax.plot(data, color="blue", linewidth=1)
    ax.set_title(series.columns[i])

    result1 = stattools.adfuller(data, autolag="AIC")
    ADF_result[series.columns[i]] = pd.Series(
        result1[0:4],
        index=["ADF Statistic", "p-value", "#Lags Used", "Number of Observations Used"],
    )
    for j in enumerate(result1[4].items()):
        ADF_result.iloc[4 + j[0], i] = round(j[1][1], 4)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, a + "standard_data.png"))

print("[INFO] saving Results of Dickey-Fuller Test on file...")
with open(os.path.join(tbl_dir, a + "ADF_results.tex"), "w") as tf:
    tf.write(ADF_result.to_latex(index=True))


############################################################################
######### Making stationary dataset and the Granger Causality test #########
############################################################################
print("[INFO] calculating the first difference...")
series_diff = df_scaled["Close"].copy(deep=True)
series_diff = df_scaled["Close"].diff()
series_diff.dropna(inplace=True)
plt.figure()
fig = plt.plot(series_diff)
plt.savefig(os.path.join(fig_dir, a + "1diff_Close_signal.png"))

print("[INFO] 'df' is now a stationary dataset...")
df = df_scaled[series.columns[0:9]]
df["Close"] = series_diff
df.dropna(inplace=True)

print("[INFO] calculating the Granger Causality test...")
maxlag = 30
test = "ssr_chi2test"


def grangers_causation_matrix(data, variables, test="ssr_chi2test", verbose=False):
    df = pd.DataFrame(
        np.zeros((3, len(variables))),
        columns=variables,
        index=["min_p_value", "lag", "Causality"],
    )
    for c in df.columns:
        test_result = stattools.grangercausalitytests(
            data[["Close", c]], maxlag=maxlag, verbose=False
        )

        p_values = list()
        lag = list()
        for i in range(maxlag):
            p_values.append(round(test_result[i + 1][0][test][1], 4))
            lag.append(test_result[i + 1][0][test][2])

        if verbose:
            print(f"Y = Close, X = {c}, P Values = {p_values}")

        min_p_value = np.min(p_values)
        lagm = lag[p_values.index(min_p_value)]
        df.loc["min_p_value", c] = min_p_value
        df.loc["lag", c] = lagm
        if min_p_value < 0.05:
            df.loc["Causality", c] = 1

    return df


res = grangers_causation_matrix(df, variables=df.columns)
print("[INFO] saving Results of Granger Causality test on file...")
with open(os.path.join(tbl_dir, a + "Granger_results.tex"), "w") as tf:
    tf.write(res.to_latex(index=True))


############################################################################
#########                     Making VAR model                     #########
############################################################################
print("[INFO] spliting dataset into train and test set...")
nobs = 100
df_train, df_test = df[0:-nobs], df[-nobs:]

AIC = list()
model = VAR(df_train)
for i in np.arange(1, 14):
    result = model.fit(i)
    AIC.append(result.aic)


plt.figure()
min_p_value = np.min(AIC)
lagm = AIC.index(min_p_value)
plt.plot(np.arange(1, 14), AIC, label="AIC")
plt.plot(lagm + 1, min_p_value, marker=(5, 0), color="red", label="min value")
plt.legend()
plt.savefig(os.path.join(fig_dir, a + "AIC_plot.png"))


model_fitted = model.fit(lagm)
# print(model_fitted.summary())


############################################################################
#########                    Durbin Watson Test                    #########
############################################################################

out = durbin_watson(model_fitted.resid)

for col, val in zip(df.columns, out):
    print(col, "\t:\t ", round(val, 2))


############################################################################
#########                    Forecast VAR model                    #########
############################################################################
lag_order = model_fitted.k_ar
print("[INFO] lag order: " + str(lag_order))


# Input data for forecasting
forecast_input = df_train.values[-lag_order:]

# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=df_test.index, columns=df.columns + "_forecast")

############################################################################
#########               Inverting the transformation               #########
############################################################################

print("[INFO] Invert the transformation to get the real forecast... ")
df_fc = df_forecast.copy()
columns = df_train.columns
i = 1
for col in columns:
    if col == "Close":
        df_fc[str(col) + "_original_forecast"] = (
            series["Close"].iloc[-100]
            + Standard_scaler.scale_[0] * df_fc[str(col) + "_forecast"].cumsum()
        )
    else:
        df_fc[str(col) + "_original_forecast"] = (
            Standard_scaler.scale_[i] * df_fc[str(col) + "_forecast"]
            + Standard_scaler.mean_[i]
        )
        i = i + 1


df_results = df_fc[[str(col) + "_original_forecast" for col in columns]]


print("[INFO] saving the plot of forecast...")
fig, axes = plt.subplots(nrows=3, ncols=3, dpi=150, figsize=(12, 7))
for i, (col, ax) in enumerate(zip(df.columns, axes.flatten())):
    df_results[col + "_original_forecast"].plot(legend=True, ax=ax).autoscale(
        axis="x", tight=True
    )
    series[col][-nobs:].plot(legend=True, ax=ax)
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, a + "00_Forecast_vs_Actuals.png"))


fig = plt.figure(dpi=150, figsize=(12, 7))
df_results["Close_original_forecast"].plot(legend=True).autoscale(axis="x", tight=True)
series["Close"][-nobs:].plot(legend=True)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, a + "00_Forecast_vs_Actuals_of_close.png"))


rmse = np.sqrt(
    mean_squared_error(series["Close"][-nobs:], df_results["Close_original_forecast"])
)
print("[INFO] RMS error: %.3f" % rmse)
############################################################################
#########            Rolling Forecast of ARIMA model.             ##########
############################################################################


window = 500  # = 3 * 365

df_train, df_test = df[window:-nobs], df[-nobs:]

history_train = df_train
predictions = list()

plt.figure()
plt.plot(series["Close"][window:-nobs], label="train set")
plt.legend()
plt.savefig(os.path.join(fig_dir, a + "Rolling_trainset.png"))


for t in range(nobs):
    model = VAR(history_train)
    model_fit = model.fit(lagm)

    lag_order = model_fit.k_ar

    # Input data for forecasting
    # print(df_train)
    forecast_input = df_train.values[-lag_order:]
    # print(forecast_input)
    # print(forecast_input.shape)

    # Forecast
    fc = model_fit.forecast(y=forecast_input, steps=1)
    # print((fc.squeeze().tolist()))

    predictions.append(fc.squeeze().tolist())
    # print(df_test.iloc[t, :])

    history_train = history_train.append(df_test.iloc[t, :])
    # print(history_train)

    history_train = history_train[1:]
    # print(history_train)


# print(predictions.shape)
# print(predictions)

print("[INFO] Invert the transformation to get the real forecast... ")
df_forecast = pd.DataFrame(
    predictions, index=df_test.index, columns=df.columns + "_forecast"
)
df_fc = df_forecast.copy()

columns = df_train.columns
i = 1
for col in columns:
    if col == "Close":
        df_fc[str(col) + "_original_forecast"] = (
            series["Close"].iloc[-100]
            + (Standard_scaler.scale_[0]) * df_fc[str(col) + "_forecast"].cumsum()
        )
    else:
        df_fc[str(col) + "_original_forecast"] = (
            Standard_scaler.scale_[i] * df_fc[str(col) + "_forecast"]
        ) + Standard_scaler.mean_[i]

        i = i + 1


df_results = df_fc[[str(col) + "_original_forecast" for col in columns]]


print("[INFO] saving the plot of forecast...")
fig, axes = plt.subplots(nrows=3, ncols=3, dpi=150, figsize=(12, 7))
for i, (col, ax) in enumerate(zip(df.columns, axes.flatten())):
    df_results[col + "_original_forecast"].plot(legend=True, ax=ax).autoscale(
        axis="x", tight=True
    )
    series[col][-nobs:].plot(legend=True, ax=ax)
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, a + "0_Rolling_Forecast_vs_Actuals.png"))


fig = plt.figure(dpi=150, figsize=(12, 7))
df_results["Close_original_forecast"].plot(legend=True).autoscale(axis="x", tight=True)
series["Close"][-nobs:].plot(legend=True)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, a + "0_Rolling_Forecast_vs_Actuals_of_close.png"))


# df_fc["original_forecast"] = (
#     series["Close"].iloc[-nobs]
#     + (-Standard_scaler.scale_[0]) * df_fc["close_forecast"].cumsum()
# )
# df_results = df_fc["original_forecast"]

# print("[INFO] saving the plot of forecast...")
# plt.figure(dpi=150)
# plt.plot(series["Close"], label="training and testing data")
# plt.plot(df_results.T, label="forecast")
# plt.title("Forecast vs Actuals")
# plt.legend(loc="upper left", fontsize=8)
# plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, a + "0_Rolling_Forecast_vs_Actuals.png"))

rmse = np.sqrt(
    mean_squared_error(series["Close"][-nobs:], df_results["Close_original_forecast"])
)
print("[INFO] RMS error: %.3f" % rmse)
