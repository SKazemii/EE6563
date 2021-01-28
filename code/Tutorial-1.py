# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 10:52:07 2021

@author: pkumar1
"""

#Dataset:https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv
#This dataset describes the monthly number of sales of shampoo over a 3 year period. The units are a sales count and there are 36 observations.


from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot as plt
 
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')#method creates a datetime object from the given string.
 
series = read_csv('/Users/saeedkazemi/Downloads/shampoo.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
print(series.head())#used to print 5 top values
series.plot()
plt.show()

#decompose the time-series for further analysis
import statsmodels.api as sm

def seasonal_decompose (y):
    #decomposition = sm.tsa.seasonal_decompose(y, model='additive',extrapolate_trend='freq')
    decomposition = sm.tsa.seasonal_decompose(y, extrapolate_trend='freq')
    fig = decomposition.plot()
    fig.set_size_inches(14,7)
    plt.show()

seasonal_decompose(series)