from pandas import read_csv
import pandas as pd
from random import seed
from random import random
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf,acf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt
# load dataset
df = pd.read_csv("/Users/saeedkazemi/Documents/Python/EE6563/Dataset/Tutorial/wwwusage.csv", names=['value'], header=0)
plt.plot(df.values)

# Original Series
fig, axes = plt.subplots(3, 2, sharex=False)
axes[0, 0].plot(df.value); axes[0, 0].set_title('Original Series')
plot_acf(df.value, ax=axes[0, 1],lags=50)

# 1st Differencing
axes[1, 0].plot(df.value.diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df.value.diff().dropna(), ax=axes[1, 1])
# 2nd Differencing
axes[2, 0].plot(df.value.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])
result = adfuller(df.value.diff().diff().dropna())

#ADF test for 1st and 2nd differences
result = adfuller(df.value.diff().dropna())
print('ADF Statistic of 1st Order Differencing: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df.value.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(df.value.diff().dropna(), ax=axes[1])
plt.show()

#create training and testing data
X=np.array(df.value.diff().dropna())
train = X[:85]
test = X[85:]
# train autoregression
model = AutoReg(train, lags=3)
model_fit = model.fit()
print('Coefficients: %s' % model_fit.params)
#5. # Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,3)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plot_acf(residuals, ax=ax[2])
plt.show()

# make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+13)
#test=test.tolist();predictions=predictions.tolist()
for i in range(len(predictions)):
	print('predicted=%f, expected=%f' % (predictions[i], test[i]))
rmse = sqrt(mean_squared_error(test, predictions))
print('Train RMSE: %.3f' % rmse)

# plot results
xpos=np.arange(len(X))
x_total=X
pyplot.plot(x_total, linewidth=2,label='actual')
pyplot.plot(xpos[85:99],predictions, color='red', label='forecast')
pyplot.legend(loc='upper left', fontsize=8)
pyplot.show()


#%%model updation

#1. First we update the model as we get the observation
#df = pd.read_csv(r'file:///C:/Users/pkumar1/OneDrive - University of New Brunswick/time_series_course/Tutorial/totorial-1/wwwusage.csv', names=['value'], header=0)
#X = df.values
#size = len(X) - 15
#train, test = X[0:size], X[size:]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = AutoReg(history, lags=3)
    model_fit = model.fit()
    yhat = model_fit.predict(start=len(history),end=len(history))
    
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    #print(len(history))
    print('>predicted=%.3f, expected=%.3f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
x_total=X
xpos=np.arange(len(X))
pyplot.plot(x_total, linewidth=2,label='Actual')
pyplot.plot(xpos[85:99],predictions, color='red',label='Forecast')
pyplot.legend(loc='upper left', fontsize=8)
pyplot.show()


###############################
#2 Model updation using cofficients(Rolling Window Forecasts)
window = 3

model = AutoReg(train, lags=3)
model_fit = model.fit()
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]#get the last three data values
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
	length = len(history)
	lag = [history[i] for i in range(length-window,length)]#creating a lag list
	yhat = coef[0]#constant term
	for d in range(window):
		yhat += coef[d+1] * lag[window-d-1]#y=c0+c1x1+c2x2+c3x3
	obs = test[t]#observation
	predictions.append(yhat)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot
x_total=X
xpos=np.arange(len(X))
pyplot.plot(x_total, linewidth=2,label='Actual')
pyplot.plot(xpos[85:99],predictions, color='red',label='Forecast')
pyplot.legend(loc='upper left', fontsize=8)
pyplot.show()

### Map to the Original Domain
original=df.values
ind=0;pred=[]
for val in range(84,98):
    last_tr=original[val]
    pred.append(last_tr+predictions[ind])
    ind=ind+1
xpos=np.arange(len(original))
pyplot.plot(original, linewidth=2, label='Original')
pyplot.plot(xpos[85:99],pred, color='red', label='Forecast')
pyplot.legend(loc='upper left', fontsize=8)
pyplot.show()    