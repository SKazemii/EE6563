# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 12:51:01 2021

@author: pkumar1
"""


import numpy as np
import matplotlib.pyplot as plt

from hmmlearn import hmm
from pandas import read_csv

series = read_csv('/Users/saeedkazemi/Documents/Python/EE6563/Dataset/Tutorial/gait_train.csv', header=0, parse_dates=True, squeeze=True)

X = series.values
X=np.expand_dims(X,1)
#train, test = X[1:len(X)-200], X[len(X)-200:]

model1 = hmm.GaussianHMM(n_components=2, covariance_type="full")
mdl=model1.fit(X)
dec=mdl.decode(X)[1]
#dec=1-dec
plt.plot(X, linewidth=2,label='Stride')
plt.plot(dec,label='states',color='red')
plt.legend(loc='upper left', fontsize=8)
plt.show()

#from sklearn import preprocessing
#scaler = preprocessing.StandardScaler()
series2 = read_csv('/Users/saeedkazemi/Documents/Python/EE6563/Dataset/Tutorial/gait_test.csv', header=0, parse_dates=True, squeeze=True)
s2 = series2.values;
s2=np.expand_dims(s2,1)
#s2 = scaler.fit_transform(s2);mdl=model1.fit(s2)
d2=mdl.decode(s2)[1]
#d2=1-d2
plt.plot(s2, linewidth=2,label='Stride')
plt.plot(d2,label='states',color='red')
plt.legend(loc='upper left', fontsize=8)
plt.show()

