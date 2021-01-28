import matplotlib.pyplot as plt
import pandas as pd

from pandas import read_csv

# from pandas import datetime

from matplotlib import pyplot as plt

# from influxdb import InfluxDBClient
# import seaborn as sns
import os


# def parser(x):
#     return datetime.strptime(
#         "190" + x, "%Y-%m"
#     )  # method creates a datetime object from the given string.


project_dir = os.getcwd()  # os.path.dirname
fig_dir = os.path.join(project_dir, "manuscript", "src", "figures")
tbl_dir = os.path.join(project_dir, "manuscript", "src", "tables")
data_dir = os.path.join(project_dir, "Dataset")
dataset1_sun_file = os.path.join(data_dir, "monthly-sunspots.csv")
dataset2_temp_file = os.path.join(data_dir, "temperatures.csv")

series_temp = read_csv(
    dataset2_temp_file,
    header=0,
    parse_dates=[0],
    index_col=0,
    squeeze=True,
)
print(series_temp.head())  # used to print 5 top values
series_temp.plot()
plt.show()
plt.savefig(os.path.join(fig_dir, "temp.png"))

sun_head = series_temp.head()
with open(os.path.join(tbl_dir, "sun_head.tex"), "w") as tf:
    tf.write(sun_head.to_latex(index=False))

series_sun = read_csv(
    dataset1_sun_file,
    header=0,
    parse_dates=[0],
    index_col=0,
    squeeze=True,
)
print(series_sun.head())  # used to print 5 top values
series_sun.plot()
plt.show()
plt.savefig(os.path.join(fig_dir, "sun.png"))
