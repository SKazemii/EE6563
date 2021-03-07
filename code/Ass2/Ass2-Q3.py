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
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa import seasonal, stattools
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.tsa.api import VAR


plt.rcParams["figure.figsize"] = (14, 7)
a = "Ass2_Q3_"
maxlag = 30
nobs = 100
orders = [1, 0, 1]

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

fig, axes = plt.subplots(nrows=3, ncols=3, dpi=120, figsize=(10, 6))

for i, ax in enumerate(axes.flatten()):
    data = series[series.columns[i]]
    ax.plot(data, color="blue", linewidth=1)
    ax.set_title(series.columns[i])

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, a + "raw_signal.png"))


############################################################################
#########                      Standardzation                     ##########
############################################################################
Standard_scaler = preprocessing.StandardScaler()
series_scaled = Standard_scaler.fit_transform(series.iloc[:, 0:9].values)
df_scaled = pd.DataFrame(series_scaled, index=series.index, columns=series.columns[0:9])


############################################################################
#########      Saving and showing the plot of the raw signal      ##########
############################################################################

print("[INFO] Saving and showing the plot of the dataset")
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

plt.figure()
fig, axes = plt.subplots(2, 1)
plot_acf(series_diff, lags=50, ax=axes[0])
plot_pacf(series_diff, lags=50, ax=axes[1])
plt.savefig(os.path.join(fig_dir, a + "PACF_ACF_1diff.png"))


print("[INFO] 'df' is now a stationary dataset...")
df = series[series.columns[0:9]]
df["Close"] = series_diff
df.dropna(inplace=True)

print("[INFO] calculating the Granger Causality test...")

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
#########                 Making exog and endog DF                 #########
############################################################################

exog = pd.DataFrame(index=df.index)
for c in df.columns[1:]:
    if res.loc["Causality", c] == 1:
        exog[c + "_shifted"] = df[c].shift(int(res.loc["lag", c]))

exog.drop(exog.index[0:maxlag], inplace=True)
df.drop(df.index[0:maxlag], inplace=True)


############################################################################
#########                   Making ARIMAX model                    #########
############################################################################
print("[INFO] spliting dataset into train and test set...")

df_train, df_test = df[0:-nobs], df[-nobs:]
exog_train, exog_test = exog[0:-nobs], exog[-nobs:]


model = ARIMA(endog=df_train["Close"], exog=exog_train, order=orders)
model_fit = model.fit(disp=-1)

print("[INFO] saving the plot of residual...")
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(3, 1)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind="kde", title="Density", ax=ax[1])
plot_acf(residuals, title="ACF", ax=ax[2])
plt.savefig(os.path.join(fig_dir, a + "residual_plot.png"))

plt.figure()
model_fit.plot_predict()
plt.savefig(os.path.join(fig_dir, a + "0000.png"))


############################################################################
#########                         Forecast                         #########
############################################################################
fc, se, conf = model_fit.forecast(nobs, exog=exog_test, alpha=0.05)
df_forecast = pd.DataFrame(fc, index=df_test.index, columns=["close_forecast"])


print("[INFO] Invert the transformation to get the real forecast... ")
df_fc = df_forecast.copy()

df_fc["original_forecast"] = (
    series["Close"].iloc[-nobs]
    + Standard_scaler.scale_[0] * df_fc["close_forecast"].cumsum()
)
df_results = df_fc["original_forecast"]

print("[INFO] saving the plot of forecast...")
plt.figure(dpi=150)
plt.plot(series["Close"], label="training and testing data")
plt.plot(df_results, label="forecast")
plt.title("Forecast vs Actuals")
plt.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, a + "Forecast_vs_Actuals.png"))

rmse = np.sqrt(mean_squared_error(series["Close"][-nobs:], df_fc["original_forecast"]))
print("[INFO] RMS error: %.3f" % rmse)


############################################################################
#########            Rolling Forecast of ARIMA model.             ##########
############################################################################

X = series["Close"].values
size = len(X) - nobs
window = 500  # = 3 * 365

df_train, df_test = df[window:-nobs], df[-nobs:]
exog_train, exog_test = exog[window:-nobs], exog[-nobs:]
history_train = df_train["Close"].to_list()
history_exog = exog_train
predictions = list()

plt.figure()
plt.plot(series["Close"][window:size], label="train set")
plt.legend()
plt.savefig(os.path.join(fig_dir, a + "Rolling_trainset.png"))


for t in range(nobs):

    model = ARIMA(endog=history_train, exog=history_exog, order=orders)

    model_fit = model.fit(disp=-1)
    output = model_fit.forecast(1, exog=exog_test.iloc[t, :], alpha=0.05)
    predictions.append(output[0])

    history_train.append(df_test["Close"][t])
    history_train = history_train[1:]

    history_exog = history_exog.append(exog_test.iloc[t, :])
    history_exog = history_exog.iloc[1:, :]

    # print(str(df_test["Close"][t]) + "===>" + str(output[0]))


print("[INFO] Invert the transformation to get the real forecast... ")
df_forecast = pd.DataFrame(predictions, index=df_test.index, columns=["close_forecast"])
df_fc = df_forecast.copy()

df_fc["original_forecast"] = (
    series["Close"].iloc[-nobs]
    + (-Standard_scaler.scale_[0]) * df_fc["close_forecast"].cumsum()
)
df_results = df_fc["original_forecast"]

print("[INFO] saving the plot of forecast...")
plt.figure(dpi=150)
plt.plot(series["Close"], label="training and testing data")
plt.plot(df_results.T, label="forecast")
plt.title("Forecast vs Actuals")
plt.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, a + "Rolling_Forecast_vs_Actuals.png"))

rmse = np.sqrt(mean_squared_error(series["Close"][-nobs:], df_fc["original_forecast"]))
print("[INFO] RMS error: %.3f" % rmse)
