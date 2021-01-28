# import seaborn as sns
import os

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from pandas import read_csv
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa import seasonal


print("[INFO] Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "manuscript", "src", "figures")
tbl_dir = os.path.join(project_dir, "manuscript", "src", "tables")
data_dir = os.path.join(project_dir, "Dataset")
dataset1_sun_file = os.path.join(data_dir, "monthly-sunspots.csv")
dataset2_temp_file = os.path.join(data_dir, "temperatures.csv")


print("[INFO] Reading the first dataset")
series_temp = read_csv(
    dataset2_temp_file,
    header=0,
    parse_dates=[0],
    index_col=0,
    squeeze=True,
)

print("[INFO] Saving and showing the plot of the second dataset")
series_temp.plot()
plt.savefig(os.path.join(fig_dir, "temp.png"))
# plt.show()

print("[INFO] Saving and printing the head of the first dataset")
print(series_temp.head(15))
with open(os.path.join(tbl_dir, "temp_head.tex"), "w") as tf:
    tf.write(series_temp.head(7).to_latex(index=True))


print("[INFO] Reading the second dataset")
series_sun = read_csv(
    dataset1_sun_file,
    header=0,
    parse_dates=[0],
    index_col=0,
    squeeze=True,
)

print("[INFO] Saving and showing the plot of the second dataset")
series_sun.plot()
plt.savefig(os.path.join(fig_dir, "sun.png"))
# plt.show()

print("[INFO] Saving and printing the head of the second dataset")
with open(os.path.join(tbl_dir, "sun_head.tex"), "w") as tf:
    tf.write(series_sun.head(7).to_latex(index=True))

# Decompositions

# seasonal_decompose(x[, model, filt, period, …])   Seasonal decomposition using moving averages.
# STL(endog[, period, seasonal, trend, …])          Season-Trend decomposition using LOESS.


## Multiplicative Decomposition
## Multiplicative seasonality is not appropriate for zero and negative values

# Plot Decomposition by seasonal_decompose method
decomposition_sd = seasonal.seasonal_decompose(
    series_temp, model="additive", extrapolate_trend="freq", period=365
)
fig = decomposition_sd.plot()
fig.set_size_inches(14, 7)
plt.savefig(os.path.join(fig_dir, "1-temp_seasonal_decompose.png"))
# plt.show()

# Plot Decomposition by STL method
decomposition_STL = seasonal.STL(series_temp, period=365).fit()
fig = decomposition_STL.plot()
fig.set_size_inches(14, 7)
plt.savefig(os.path.join(fig_dir, "1-temp_STL.png"))
# plt.show()


# ACF plot
plot_acf(decomposition_sd.resid, lags=100)

# ADF Test
result = adfuller(series_temp)
print(result)
print("ADF Statistic: %f" % result[0])
print("p-value: %f" % result[1])
for key, value in result[4].items():
    print("\t%s: %.3f" % (key, value))

# The p-value is obtained is greater than significance level of 0.05 and
# the ADF statistic is higher than any of the critical values.
# Clearly, there is no reason to reject the null hypothesis. So,
# the time series is in fact non-stationary.

# The p-value is very less than the significance level of 0.05 and
# hence we can reject the null hypothesis and take that the series is stationary.
