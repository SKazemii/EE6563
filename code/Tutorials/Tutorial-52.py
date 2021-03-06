import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf,acf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA


def detect_classify_anomalies(df,window):    
    #df=pd.Series(test-fc_series)
    df['error']=pd.Series(df)
    df['meanval'] = df['error'].rolling(window=window).mean()
    df['deviation'] = df['error'].rolling(window=window).std()
    df['-3s'] = df['meanval'] - (2 * df['deviation'])
    df['3s'] = df['meanval'] + (2 * df['deviation'])
    df['-2s'] = df['meanval'] - (1.75 * df['deviation'])
    df['2s'] = df['meanval'] + (1.75 * df['deviation'])
    df['-1s'] = df['meanval'] - (1.5 * df['deviation'])
    df['1s'] = df['meanval'] + (1.5 * df['deviation'])
    return df


plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

# Import data
#https://vincentarelbundock.github.io/Rdatasets/doc/datasets/WWWusage.html
#A time series of the numbers of users connected to the Internet through a server every minute.
df = pd.read_csv("/Users/saeedkazemi/Documents/Python/EE6563/Dataset/Tutorial/wwwusage_Anomaly.csv", names=['value'], header=0)
#plt.plot(df.values)
# Original Series
plt.figure(figsize=(12,5), dpi=100)
plt.plot(df, label='actual')
plt.show()
# Create Training and Test
train = df.value[:80]
test = df.value[80:]


#8# Build Model
model = ARIMA(train, order=(3, 2, 1))  
fitted = model.fit(disp=-1)  
print(fitted.summary())
# Forecast
fc, se, conf = fitted.forecast(20, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

###Lets train the model repeatively 
confidence=[];testt=test.values
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(3,2,1))
	model_fit = model.fit()
	fc, se, conf = model_fit.forecast(1, alpha=0.05)  # 95% conf
	confidence.append(conf)
	predictions.append(fc)
	obs = testt[t]
	history.append(obs)
conf=np.vstack(confidence)

fc_series = pd.Series(predictions, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)
###
df=detect_classify_anomalies(test-fc_series,7)
upper_bound=pd.Series(df['1s'].values, index=test.index)
lower_bound=pd.Series(df['-1s'].values, index=test.index)
# Plot
plt.figure(figsize=(12,7), dpi=100)
#plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')
plt.plot((test-fc_series), label='Error')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)

plt.fill_between(lower_bound.index, lower_bound, upper_bound, 
                 color='k', alpha=.15)

plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()




def detect_classify_anomalies(df,window):    
    #df=pd.Series(test-fc_series)
    df['error']=pd.Series(df)
    df['meanval'] = df['error'].rolling(window=window).mean()
    df['deviation'] = df['error'].rolling(window=window).std()
    df['-3s'] = df['meanval'] - (2 * df['deviation'])
    df['3s'] = df['meanval'] + (2 * df['deviation'])
    df['-2s'] = df['meanval'] - (1.75 * df['deviation'])
    df['2s'] = df['meanval'] + (1.75 * df['deviation'])
    df['-1s'] = df['meanval'] - (1.5 * df['deviation'])
    df['1s'] = df['meanval'] + (1.5 * df['deviation'])
    return df
