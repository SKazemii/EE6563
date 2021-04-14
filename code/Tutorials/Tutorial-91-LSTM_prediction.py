# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 15:31:56 2021

@author: pkumar1
"""
#This is a problem where, given a year and a month, the task is to predict the number of 
#international airline passengers in units of 1,000. 
#The data ranges from January 1949 to December 1960, or 12 years, with 144 observations.
# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('file:///C:/Users/pkumar1/OneDrive - University of New Brunswick/time_series_course/Tutorial/totorial-1/AirPassengers.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY, epochs=250, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

#%%
#######################################
#Model Saving
######################################
# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.summary()
model.compile(loss='mean_squared_error', optimizer='adam')

weight_path="{}_weights.best.hdf5".format('LSTM')
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min', save_weights_only = False)

history=model.fit(trainX, trainY,batch_size=1,validation_data=(testX,testY),callbacks = [checkpoint],epochs=250)
#%%Plot the Loss 
# summarize history for loss
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

############
#Load the saved Model
############
from keras.models import load_model
model1 = load_model('C:/Users/pkumar1/OneDrive - University of New Brunswick/time_series_course/Tutorial/totorial-1/LSTM_weights.best.hdf5')
# make predictions
trainPredict = model1.predict(trainX)
testPredict = model1.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore_mdl1 = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score before saved model: %.2f RMSE' % (trainScore))
print('Train Score on saved model: %.2f RMSE' % (trainScore_mdl1))
testScore_mdl1 = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score before saved Model: %.2f RMSE' % (testScore))
print('Test Score on saved Model: %.2f RMSE' % (testScore_mdl1))

#%%
###########
#With Time-steps
###########
# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(trainX, trainY,batch_size=1,validation_data=(testX,testY),callbacks = [checkpoint],epochs=250)
# load the saved model
model2 = load_model('C:/Users/pkumar1/OneDrive - University of New Brunswick/time_series_course/Tutorial/totorial-1/LSTM_weights.best.hdf5')
# make predictions
testPredict = model2.predict(testX)
# invert predictions
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
testScore_time_mdl = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score of time step saved Model: %.2f RMSE' % (testScore_time_mdl))


#comparison among three approaches
p_data = [testScore,testScore_mdl1, testScore_time_mdl]
#ind = np.arange(3)
#fig, axs = plt.subplots()
plt.figure()
plt.title('Comparison b/w three LSTMs approaches')
plt.bar(['RMSE (Basic approach)','RMSE (Save & Load)', 'RMSE (Time Step)'], p_data, color = 'b', width = 0.25)
plt.ylim(30, 100)
plt.ylabel('RMSE Scores')
plt.legend(labels=['RMSE'])

#%%Comparison with AR model
import pandas as pd
from random import seed
from random import random
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
from math import sqrt

#reading the dataset
series = pd.read_csv(r'C:/Users/pkumar1/OneDrive - University of New Brunswick/time_series_course/Tutorial/totorial-1/AirPassengers.csv')
#the dataset contains number of passengers per month between Jan 1949-Dec. 1960
X = series['#Passengers'].values



#check the transformed series
#1. Transformation
series['#Passengers_log'] = np.log(series['#Passengers'])
#series['#Passengers_log'].dropna().plot(label="Log Difference")
X_lg=series['#Passengers_log']

#2. Lets take first difference
X_lg_1diff  =np.diff(X_lg,1)

#3. Lets take 12th difference
X_lg_12diff  =np.diff(X_lg_1diff,12)
result = adfuller(X_lg_12diff)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


#4. AR model
train_size = int(len(X_lg_12diff) * 0.67)
test_size = len(X_lg_12diff) - train_size
X_lg_12diff=np.expand_dims(X_lg_12diff,1)
train, test = X_lg_12diff[0:train_size,:], X_lg_12diff[train_size:len(X_lg_12diff),:]



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
ar_rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % ar_rmse)

#comparison among four approaches
p_data = [testScore,testScore_mdl1, testScore_time_mdl,ar_rmse]
#ind = np.arange(3)
#fig, axs = plt.subplots()
plt.figure()
plt.title('Comparison b/w four approaches')
plt.bar(['RMSE (Basic approach)','RMSE (Save & Load)', 'RMSE (Time Step)', 'ar_rmse'], p_data, color = 'b', width = 0.25)
plt.ylim(10, 100)
plt.ylabel('RMSE Scores')
plt.legend(labels=['RMSE'])




