# import seaborn as sns
import os

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import STL
from matplotlib import pyplot as plt
from pandas import read_csv

# from pandas import datetime


# def parser(x):
#     return datetime.strptime(
#         "190" + x, "%Y-%m"
#     )  # method creates a datetime object from the given string.

print("[INFO] Setting directories")
project_dir = os.getcwd()  # os.path.dirname
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
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "temp.png"))
# plt.show()

print("[INFO] Saving and printing the head of the first dataset")
print(series_temp.head())  # used to print 5 top values
temp_head = series_temp.head()
with open(os.path.join(tbl_dir, "temp_head.tex"), "w") as tf:
    tf.write(temp_head.to_latex(index=True))


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
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "sun.png"))
# plt.show()

print("[INFO] Saving and printing the head of the second dataset")
print(series_sun.head())
sun_head = series_sun.head()
with open(os.path.join(tbl_dir, "sun_head.tex"), "w") as tf:
    tf.write(sun_head.to_latex(index=True))

print(series_temp.shape)
series_temp = series_temp.resample("Y").mean().ffill()
print(series_sun.head(15))

res = STL(series_sun).fit()
res.plot()
plt.show()