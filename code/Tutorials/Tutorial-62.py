import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import Statsmodels
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import grangercausalitytests

# filepath = 'https://raw.githubusercontent.com/selva86/datasets/master/Raotbl6.csv'
# Wage Growth and the Inflation Process
# This dataset has the following 8 quarterly time series 1959-89:
# 1. rgnp  : Real GNP.
# 2. pgnp  : Potential real GNP.
# 3. ulc   : Unit labor cost.
# 4. gdfco : Fixed weight deflator for personal consumption expenditure excluding food and energy.
# 5. gdf   : Fixed weight GNP deflator.
# 6. gdfim : Fixed weight import deflator.
# 7. gdfcf : Fixed weight deflator for food in personal consumption expenditure.
# 8. gdfce : Fixed weight deflator for energy in personal consumption expenditure.

df = pd.read_csv(
    "/Users/saeedkazemi/Documents/Python/EE6563/Dataset/Tutorial/tutorial62.csv",
    parse_dates=["date"],
    index_col="date",
)
print(df.shape)  # (123, 8)
df.tail()


# Plot
fig, axes = plt.subplots(nrows=4, ncols=2, dpi=120, figsize=(10, 6))
for i, ax in enumerate(axes.flatten()):
    data = df[df.columns[i]]
    ax.plot(data, color="red", linewidth=1)
    # Decorations
    ax.set_title(df.columns[i])
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()


# 2. Granger Causality test
maxlag = 12
test = "ssr_chi2test"


def grangers_causation_matrix(data, variables, test="ssr_chi2test", verbose=False):
    """Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table
    are the P-Values. P-Values lesser than the significance level (0.05), implies
    the Null Hypothesis that the coefficients of the corresponding past values is
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(
        np.zeros((len(variables), len(variables))), columns=variables, index=variables
    )
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(
                data[[r, c]], maxlag=maxlag, verbose=False
            )
            p_values = [round(test_result[i + 1][0][test][1], 4) for i in range(maxlag)]
            if verbose:
                print(f"Y = {r}, X = {c}, P Values = {p_values}")
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + "_x" for var in variables]
    df.index = [var + "_y" for var in variables]
    return df


grangers_causation_matrix(df, variables=df.columns)


# 3. Cointegration Test
# Cointegration test helps to establish the presence of a statistically
# significant connection between two or more time series.
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def adjust(val, length=6):
    return str(val).ljust(length)


def cointegration_test(df, alpha=0.05):
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(df, -1, 5)
    d = {"0.90": 0, "0.95": 1, "0.99": 2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1 - alpha)]]

    # Summary
    print("Name   ::  Test Stat > C(95%)    =>   Signif  \n", "--" * 20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(
            adjust(col),
            ":: ",
            adjust(round(trace, 2), 9),
            ">",
            adjust(cvt, 8),
            " =>  ",
            trace > cvt,
        )


cointegration_test(df)


# 4.Split the Series into Training and Testing Data
nobs = 4
df_train, df_test = df[0:-nobs], df[-nobs:]

# Check size
print(df_train.shape)  # (119, 8)
print(df_test.shape)  # (4, 8)

# 5. Checking the Stationarity
def adfuller_test(series, signif=0.05, name="", verbose=False):
    """Perform ADFuller to test for Stationarity of given series and print report"""
    r = adfuller(series, autolag="AIC")
    output = {
        "test_statistic": round(r[0], 4),
        "pvalue": round(r[1], 4),
        "n_lags": round(r[2], 4),
        "n_obs": r[3],
    }
    p_value = output["pvalue"]

    def adjust(val, length=6):
        return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", "-" * 47)
    print(f" Null Hypothesis: Data has unit root. Non-Stationary.")
    print(f" Significance Level    = {signif}")
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key, val in r[4].items():
        print(f" Critical value {adjust(key)} = {round(val, 3)}")

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")


# ADF Test on each column
for name, column in df_train.iteritems():
    adfuller_test(column, name=column.name)
    print("\n")

# 1st difference
df_differenced = df_train.diff().dropna()
# Second Differencing
df_differenced = df_differenced.diff().dropna()


# ADF Test on each column
for name, column in df_differenced.iteritems():
    adfuller_test(column, name=column.name)
    print("\n")


# select the order of VAR model
AIC = list()
BIC = list()
model = VAR(df_differenced)
for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
    result = model.fit(i)
    AIC.append(result.aic)
    BIC.append(result.bic)
    print("Lag Order =", i)
    print("AIC : ", result.aic)
    print("BIC : ", result.bic)
    print("FPE : ", result.fpe)
    print("HQIC: ", result.hqic, "\n")
# plt.plot(np.arange(1, 10), AIC, label="AIC")
plt.plot(np.arange(1, 10), AIC, label="BIC")
plt.plot(4, AIC[4], marker=(5, 0), label="BIC")

# x = model.select_order(maxlags=12)
# x.summary()
plt.plot(AIC)
plt.show()
model_fitted = model.fit(4)
print(model_fitted.summary())

# 6. Check for Serial Correlation of Residuals (Errors) using Durbin Watson Statistic
# Serial correlation of residuals is used to check if there is any leftover pattern in the residuals (errors).

# The value of this statistic can vary between 0 and 4.
# The closer it is to the value 2, then there is no significant serial
# correlation. The closer to 0, there is a positive serial correlation,
# and the closer it is to 4 implies negative serial correlation.

from statsmodels.stats.stattools import durbin_watson

out = durbin_watson(model_fitted.resid)

for col, val in zip(df.columns, out):
    print(adjust(col), ":", round(val, 2))

# Forecast VAR model using statsmodels
# Get the lag order
lag_order = model_fitted.k_ar
print(lag_order)  # > 4

# Input data for forecasting
forecast_input = df_differenced.values[-lag_order:]
forecast_input
# Forecast
fc = model_fitted.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=df.index[-nobs:], columns=df.columns + "_2d")
df_forecast

# Invert the transformation to get the real forecast


def invert_transformation(df_train, df_forecast, second_diff=False):
    """Revert back the differencing to get the forecast to original scale."""
    df_fc = df_forecast.copy()
    columns = df_train.columns
    for col in columns:
        # print(col)
        # Roll back 2nd Diff
        if second_diff:
            df_fc[str(col) + "_1d"] = (
                df_train[col].iloc[-1] - df_train[col].iloc[-2]
            ) + df_fc[str(col) + "_2d"].cumsum()
        # Roll back 1st Diff
        df_fc[str(col) + "_forecast"] = (
            df_train[col].iloc[-1] + df_fc[str(col) + "_1d"].cumsum()
        )
    return df_fc


df_results = invert_transformation(df_train, df_forecast, second_diff=True)
df_results.loc[
    :,
    [
        "rgnp_forecast",
        "pgnp_forecast",
        "ulc_forecast",
        "gdfco_forecast",
        "gdf_forecast",
        "gdfim_forecast",
        "gdfcf_forecast",
        "gdfce_forecast",
    ],
]

# plot on the original scale
fig, axes = plt.subplots(
    nrows=int(len(df.columns) / 2), ncols=2, dpi=150, figsize=(10, 10)
)
for i, (col, ax) in enumerate(zip(df.columns, axes.flatten())):
    df_results[col + "_forecast"].plot(legend=True, ax=ax).autoscale(
        axis="x", tight=True
    )
    df_test[col][-nobs:].plot(legend=True, ax=ax)
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout()

# Evaluate the Forecasts
from statsmodels.tsa.stattools import acf


def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)  # MEQ
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mpe = np.mean((forecast - actual) / actual)  # MPE
    rmse = np.mean((forecast - actual) ** 2) ** 0.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0, 1]  # corr
    mins = np.amin(np.hstack([forecast[:, None], actual[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:, None], actual[:, None]]), axis=1)
    minmax = 1 - np.mean(mins / maxs)  # minmax
    return {
        "mape": mape,
        "me": me,
        "mae": mae,
        "mpe": mpe,
        "rmse": rmse,
        "corr": corr,
        "minmax": minmax,
    }


print("Forecast Accuracy of: rgnp")
accuracy_prod = forecast_accuracy(df_results["rgnp_forecast"].values, df_test["rgnp"])
for k, v in accuracy_prod.items():
    print(adjust(k), ": ", round(v, 4))

print("\nForecast Accuracy of: pgnp")
accuracy_prod = forecast_accuracy(df_results["pgnp_forecast"].values, df_test["pgnp"])
for k, v in accuracy_prod.items():
    print(adjust(k), ": ", round(v, 4))

print("\nForecast Accuracy of: ulc")
accuracy_prod = forecast_accuracy(df_results["ulc_forecast"].values, df_test["ulc"])
for k, v in accuracy_prod.items():
    print(adjust(k), ": ", round(v, 4))

print("\nForecast Accuracy of: gdfco")
accuracy_prod = forecast_accuracy(df_results["gdfco_forecast"].values, df_test["gdfco"])
for k, v in accuracy_prod.items():
    print(adjust(k), ": ", round(v, 4))

print("\nForecast Accuracy of: gdf")
accuracy_prod = forecast_accuracy(df_results["gdf_forecast"].values, df_test["gdf"])
for k, v in accuracy_prod.items():
    print(adjust(k), ": ", round(v, 4))

print("\nForecast Accuracy of: gdfim")
accuracy_prod = forecast_accuracy(df_results["gdfim_forecast"].values, df_test["gdfim"])
for k, v in accuracy_prod.items():
    print(adjust(k), ": ", round(v, 4))

print("\nForecast Accuracy of: gdfcf")
accuracy_prod = forecast_accuracy(df_results["gdfcf_forecast"].values, df_test["gdfcf"])
for k, v in accuracy_prod.items():
    print(adjust(k), ": ", round(v, 4))

print("\nForecast Accuracy of: gdfce")
accuracy_prod = forecast_accuracy(df_results["gdfce_forecast"].values, df_test["gdfce"])
for k, v in accuracy_prod.items():
    print(adjust(k), ": ", round(v, 4))
