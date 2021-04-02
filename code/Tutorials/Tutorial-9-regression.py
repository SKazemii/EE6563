# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 11:33:20 2021

@author: pkumar1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input data
data = pd.read_csv('/Users/saeedkazemi/Downloads/data.csv')
X = data.iloc[:, 0].values
train=X[:85];test=X[85:]
Y = data.iloc[:, 1].values
y_train=Y[:85];y_test=Y[85:]

plt.scatter(train, y_train)
plt.show()

# Building the model
m = 0
c = 0

L = 0.0001  # The learning Rate
epochs = 1000  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
error=[]
for i in range(epochs): 
    Y_pred = m*train + c  # The current predicted value of Y
    D_m = (-2/n) * sum(train * (y_train - Y_pred))  # Derivative wrt m
    #(y-y_hat)^2=>2(y-y^hat)*d/dm(y-mx+c)
    D_c = (-2/n) * sum(y_train - Y_pred)  # Derivative wrt c
    m = m - L * D_m  # Update m
    c = c - L * D_c  # Update c
    error.append(sum(train * (y_train - Y_pred))/n )
print (m, c)

# Making predictions
tr_pred = m*train + c
rmse = sqrt(mean_squared_error(train, tr_pred))
print('Train RMSE: %.3f' % rmse)
plt.scatter(train, y_train)
#plt.plot([min(train), max(train)], [min(tr_pred), max(tr_pred)], color='red') # predicted
plt.plot(train, tr_pred, color='red')
plt.show()
te_pred = m*test + c
for i in range(len(te_pred)):
	print('predicted=%f, expected=%f' % (te_pred[i], test[i]))
rmse = sqrt(mean_squared_error(test, te_pred))
print('Test RMSE: %.3f' % rmse)


r2_score(y_test, te_pred)#coefficient of determination

####################
#Sklearn 
#################
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(np.expand_dims(train,1), y_train)
regressor.coef_
regressor.intercept_
y_pred = regressor.predict(np.expand_dims(test,1))
rmse = sqrt(mean_squared_error(test, y_pred))
print('Test RMSE: %.3f' % rmse)

r2_score(y_test, y_pred)