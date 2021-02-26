# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:55:19 2021

@author: pkumar1
"""


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf,acf
from sklearn.metrics import mean_squared_error
from math import sqrt

df=pd.read_csv('/Users/saeedkazemi/Documents/Python/EE6563/Dataset/Tutorial/salesdata_tutorial61.csv')
print(df)

df[['Marketing','Sales']].plot()
plt.show()


#ADF test 
result = adfuller(df['Marketing'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
    
result = adfuller(df['Sales'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))    
    
#check the 1st difference ADF
df['diffM']=df['Marketing'].diff()
result = adfuller(df['diffM'].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

df['diffS']=df['Sales'].diff()    
result = adfuller(df['diffS'].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))    

#check 2nd difference
    
#check the 2nd difference ADF
df['diff2M']=df['diffM'].diff()
result = adfuller(df['diff2M'].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

df['diff2S']=df['diffS'].diff()   
result = adfuller(df['diff2S'].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value)) 
    
    
#Check the Granger Causality:whether the first variable is caused by 2nd variable
res=sm.tsa.stattools.grangercausalitytests(df[['diff2S','diff2M']].dropna(),2)
#other way around
res=sm.tsa.stattools.grangercausalitytests(df[['diff2M','diff2S']].dropna(),2)

#df['diffM']=df['Marketing'].diff()
df['lag']=df['diff2M'].shift().shift()
df.dropna(inplace=True)
model3=ARIMA(endog=df['Sales'],exog=df[['lag']],order=[1,2,0])
results3=model3.fit()
print(results3.summary())

 # Plot residual errors
residuals = pd.DataFrame(results3.resid)
fig, ax = plt.subplots(1,3)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plot_acf(residuals, ax=ax[2])
plt.show()
results3.plot_predict()
plt.show()


# Forecast
train=df['Sales'][:65];trainexo=df['lag'][:65]
test=df['Sales'][-7:];testexo=df['lag'][-7:]
model=ARIMA(endog=train,exog=trainexo,order=[1,2,0])
Arx=model.fit()
fc, se, conf = Arx.forecast(7,exog=testexo, alpha=0.05)  # 95% conf
# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

rmse = sqrt(mean_squared_error(test, fc))
print('Test RMSE: %.3f' % rmse)

#compare it with the Sales model only

modelS=ARIMA(train,order=[1,2,0])
Ar=modelS.fit()
fc, se, conf = Ar.forecast(7, alpha=0.05)  # 95% conf
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

rmse = sqrt(mean_squared_error(test, fc))
print('Test RMSE: %.3f' % rmse)



