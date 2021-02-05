import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import acf, plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

plt.rcParams.update({"figure.figsize": (9, 7), "figure.dpi": 120})

# Import data
# https://vincentarelbundock.github.io/Rdatasets/doc/datasets/WWWusage.html
# A time series of the numbers of users connected to the Internet through a server every minute.
df = pd.read_csv(
    "/Users/saeedkazemi/Documents/Python/EE6563/Dataset/Tutorial/wwwusage.csv",
    names=["value"],
    header=0,
)
plt.plot(df.values)
# Original Series
fig, axes = plt.subplots(3, 2, sharex=False)
axes[0, 0].plot(df.value)
axes[0, 0].set_title("Original Series")
plot_acf(df.value, ax=axes[0, 1], lags=50)

# 1st Differencing
axes[1, 0].plot(df.value.diff())
axes[1, 0].set_title("1st Order Differencing")
plot_acf(df.value.diff().dropna(), ax=axes[1, 1])


# 2nd Differencing
axes[2, 0].plot(df.value.diff().diff())
axes[2, 0].set_title("2nd Order Differencing")
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])
result = adfuller(df.value.diff().diff().dropna())

# ADF test for 1st and 2nd differences
result = adfuller(df.value.diff().dropna())
print("ADF Statistic of 1st Order Differencing: %f" % result[0])
print("p-value: %f" % result[1])
for key, value in result[4].items():
    print("\t%s: %.3f" % (key, value))

print("ADF Statistic of 2nd Order Differencing: %f" % result[0])
result = adfuller(df.value.diff().diff().dropna())
print("p-value: %f" % result[1])
for key, value in result[4].items():
    print("\t%s: %.3f" % (key, value))
plt.show()

# 2.How to find the order of the AR term (p)
# PACF plot of 1st differenced series
plt.rcParams.update({"figure.figsize": (9, 3), "figure.dpi": 120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff())
axes[0].set_title("1st Differencing")
axes[1].set(ylim=(0, 5))
plot_pacf(df.value.diff().dropna(), ax=axes[1])

plot_pacf(df.value.diff().dropna())
plt.show()

# 3.How to find the order of the MA term (q)
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff())
axes[0].set_title("1st Differencing")
axes[1].set(ylim=(0, 1.2))
plot_acf(df.value.diff().dropna(), ax=axes[1])

plt.show()

# If your series is slightly under differenced, adding one or more additional AR terms usually makes it up.
# Likewise, if it is slightly over-differenced, try adding an additional MA term.

# 4. How to build the ARIMA Model (ARIMA(p,d,q))
# 1,1,2 ARIMA Model
model = ARIMA(df.value, order=(1, 1, 2))
model_fit = model.fit(disp=0)
print(model_fit.summary())
print(model_fit.aic)
# Notice here the coefficient of the MA2 term is close to zero and the P-Value in ‘P>|z|’ column
# is highly insignificant.
# It should ideally be less than 0.05 for the respective X to be significant.
# 5 Rebuild ARIMA
# 1,1,1 ARIMA Model
model = ARIMA(df.value, order=(1, 1, 1))
model_fit = model.fit(disp=0)
print(model_fit.summary())

# 5. # Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1, 3)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind="kde", title="Density", ax=ax[1])
plot_acf(residuals, ax=ax[2])
plt.show()
# In statistics, kernel density estimation (KDE) is a non-parametric way to estimate the probability density function (PDF)
# of a random variable.

# 6. # Actual vs Fitted
model_fit.plot_predict()
plt.show()

# 7. Lets do forecasting
# Create Training and Test
train = df.value[:85]
test = df.value[85:]
# Build Model
# model = ARIMA(train, order=(3,2,1))
model = ARIMA(train, order=(1, 1, 1))
fitted = model.fit(disp=-1)

# Forecast
fc, se, conf = fitted.forecast(15, alpha=0.05)  # 95% conf
# predictions = fitted.predict(start=len(df.value)-15, end=len(df.value)-1,typ='levels')
# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12, 5), dpi=100)
plt.plot(train, label="training")
plt.plot(test, label="actual")
plt.plot(fc_series, label="forecast")
plt.fill_between(lower_series.index, lower_series, upper_series, color="k", alpha=0.15)
plt.title("Forecast vs Actuals")
plt.legend(loc="upper left", fontsize=8)
plt.show()


# But each of the predicted forecasts is consistently below the actuals.
# That means, by adding a small constant to our forecast, the accuracy will certainly improve.
# So, there is definitely scope for improvement.
# So, what I am going to do is to increase the order of differencing to two,
# that is set d=2 and iteratively increase p to up to 5 and then q up to 5 to see
# which model gives least AIC and also look for a
# chart that gives closer actuals and forecasts.
# 8# Build Model
model = ARIMA(train, order=(3, 2, 1))
fitted = model.fit(disp=-1)
print(fitted.summary())
# Forecast
fc, se, conf = fitted.forecast(15, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12, 5), dpi=100)
plt.plot(train, label="training")
plt.plot(test, label="actual")
plt.plot(fc_series, label="forecast")
plt.fill_between(lower_series.index, lower_series, upper_series, color="k", alpha=0.15)
plt.title("Forecast vs Actuals")
plt.legend(loc="upper left", fontsize=8)
plt.show()

# 9 Accuracy Metrics for Time Series Forecast
# 1. Mean Absolute Percentage Error (MAPE)
# 2. Mean Error (ME)
# 3. Mean Absolute Error (MAE)
# 4. Mean Percentage Error (MPE)
# 5. Root Mean Squared Error (RMSE)
# 6. Lag 1 Autocorrelation of Error (ACF1)
# 7. Correlation between the Actual and the Forecast (corr)
# 8. Min-Max Error (minmax)
# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)  # ME
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mpe = np.mean((forecast - actual) / actual)  # MPE
    rmse = np.mean((forecast - actual) ** 2) ** 0.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0, 1]  # corr
    mins = np.amin(np.hstack([forecast[:, None], actual[:, None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:, None], actual[:, None]]), axis=1)
    minmax = 1 - np.mean(mins / maxs)  # minmax
    acf1 = acf(fc - test)[1]  # ACF1
    return {
        "mape": mape,
        "me": me,
        "mae": mae,
        "mpe": mpe,
        "rmse": rmse,
        "acf1": acf1,
        "corr": corr,
        "minmax": minmax,
    }


forecast_accuracy(fc, test.values)


# 10. How to do Auto Arima Forecast in Python
import pmdarima as pm

# conda install -c saravji pmdarima
# pip install pmdarima

model = pm.auto_arima(
    df.value,
    start_p=1,
    start_q=1,
    test="adf",  # use adftest to find optimal 'd'
    max_p=3,
    max_q=3,  # maximum p and q
    m=1,  # frequency of series
    d=None,  # let model determine 'd'
    seasonal=False,  # No Seasonality
    start_P=0,
    D=0,
    trace=True,  # to print status on the fits.
    error_action="ignore",
    suppress_warnings=True,  # Whether to use the stepwise algorithm outlined in Hyndman and Khandakar (2008) to identify the optimal model parameters.
    stepwise=True,
)
# https://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html
print(model.summary())

# How to interpret the residual plots in ARIMA model
model.plot_diagnostics(figsize=(7, 5))
plt.show()
# Bottom left:Normal Q-Q plot. All the dots should fall perfectly in line with the red line.
# Any significant deviations would imply the distribution is skewed.
# Forecast
n_periods = 24
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df.value), len(df.value) + n_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)
# Plot
plt.plot(df.value)
plt.plot(fc_series, color="darkgreen")
plt.fill_between(lower_series.index, lower_series, upper_series, color="k", alpha=0.15)

plt.title("Final Forecast of WWW Usage")
plt.show()
