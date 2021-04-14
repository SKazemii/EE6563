import warnings

warnings.filterwarnings("ignore")

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime

from pandas.plotting import lag_plot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa import seasonal, stattools
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARMA, ARIMA

from sklearn.preprocessing import MinMaxScaler


from keras.callbacks import (
    ModelCheckpoint,
    LearningRateScheduler,
    EarlyStopping,
    ReduceLROnPlateau,
)
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model

from keras.wrappers.scikit_learn import KerasClassifier


plt.rcParams["figure.figsize"] = (14, 7)
a = "Ass4_Q2_"

print("[INFO] Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "manuscript", "src", "figures", "Ass4")
tbl_dir = os.path.join(project_dir, "manuscript", "src", "tables", "Ass4")
data_dir = os.path.join(project_dir, "Dataset", "Ass2")
dataset_file = os.path.join(data_dir, "Processed_NYSE.csv")


def parser(x):
    return datetime.strptime(x, "%Y-%m-%d")


print("[INFO] Reading the first dataset")
series = pd.read_csv(
    dataset_file,
    header=0,
    index_col=0,
    parse_dates=True,
    squeeze=True,
    date_parser=parser,
).dropna()


############################################################################
#########      Saving and showing the plot of the raw signal      ##########
############################################################################

print("[INFO] Saving and showing the plot of the dataset")
plt.figure()
fig = series["Close"].plot(label="Close price")
plt.legend()
plt.savefig(os.path.join(fig_dir, a + "raw_signal.png"))


############################################################################
#########               LSTM              ##########
############################################################################
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


dataset = series["Close"].values
dataset = dataset.astype("float32")
dataset = np.expand_dims(dataset, axis=1)


window = 570  # = 3 * 365
df_train, df_test = dataset[:window, :], dataset[window:, :]


scaler = MinMaxScaler()
train = scaler.fit_transform(df_train)
test = scaler.transform(df_test)


look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

trainY = np.expand_dims(trainY, axis=1)
testY = np.expand_dims(testY, axis=1)


# create and fit the LSTM network
# LSTM units, no.  of layers, batch size, learning rate,etc.
# def create_model(LSTM_units=4):
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.summary()
model.compile(loss="mean_squared_error", optimizer="adam")


weight_path = "{}_weights.best.hdf5".format("LSTM")
checkpoint = ModelCheckpoint(
    weight_path,
    monitor="val_loss",
    verbose=1,
    save_best_only=True,
    mode="min",
    save_weights_only=False,
)

history = model.fit(
    trainX,
    trainY,
    batch_size=1,
    validation_data=(testX, testY),
    callbacks=[checkpoint],
    epochs=5,
)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
testScore = math.sqrt(mean_squared_error(testY, testPredict))


model1 = load_model("/Users/saeedkazemi/Documents/Python/EE6563/LSTM_weights.best.hdf5")
# make predictions
trainPredict = model1.predict(trainX)
testPredict = model1.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
testPredict = scaler.inverse_transform(testPredict)

# calculate root mean squared error
trainScore_mdl1 = math.sqrt(mean_squared_error(trainY, trainPredict))
print("Train Score before saved model: %.2f RMSE" % (trainScore))
print("Train Score on saved model: %.2f RMSE" % (trainScore_mdl1))
testScore_mdl1 = math.sqrt(mean_squared_error(testY, testPredict))
print("Test Score before saved Model: %.2f RMSE" % (testScore))
print("Test Score on saved Model: %.2f RMSE" % (testScore_mdl1))

plt.figure()

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back : len(trainPredict) + look_back, :] = trainPredict


# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[
    len(trainPredict) + (look_back * 2) + 1 : len(dataset) - 1, :
] = testPredict


# plot baseline and predictions
plt.plot(series["Close"])
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# ############################################################################
# #########               stationary test: PACF method              ##########
# ############################################################################
# print("[INFO] Saving and showing the PACF and ACF plot ")
# plt.figure()
# fig, axes = plt.subplots(2, 1)
# plot_acf(series["Close"], lags=250, ax=axes[0])
# plot_pacf(series["Close"], lags=250, ax=axes[1])
# plt.savefig(os.path.join(fig_dir, a + "PACF_ACF.png"))


# series_diff = series["Close"].diff()
# series_diff.dropna(inplace=True)
# plt.figure()
# fig = plt.plot(series_diff)
# plt.savefig(os.path.join(fig_dir, a + "1diff_signal.png"))


# plt.figure()
# fig, axes = plt.subplots(2, 1)
# plot_acf(series_diff, lags=50, ax=axes[0])
# plot_pacf(series_diff, lags=50, ax=axes[1])
# plt.savefig(os.path.join(fig_dir, a + "PACF_ACF_1diff.png"))

# print("[INFO] Results of Dickey-Fuller Test:")
# result = stattools.adfuller(series_diff, autolag="AIC")
# dfoutput = pd.Series(
#     result[0:4],
#     index=["ADF Statistic", "p-value", "#Lags Used", "Number of Observations Used"],
# )


# for key, value in result[4].items():
#     dfoutput["Critical Value (%s)" % key] = value

# print("[INFO] saving Results of Dickey-Fuller Test on file...")
# with open(os.path.join(tbl_dir, a + "ADF_1diff.tex"), "w") as tf:
#     tf.write(dfoutput.to_latex(index=True))


# ############################################################################
# #########                     fitting ARIMA                       ##########
# ############################################################################
# nobs = 35
# orders = [1, 1, 1]

# print("[INFO] Create Training and Test set...")
# train = series["Close"][:-nobs]
# test = series["Close"][-nobs:]

# model = ARIMA(train, order=orders)
# model_fitted = model.fit(disp=-1)

# # print(model_fitted.summary())


# print("[INFO] saving the plot of residual...")
# residuals = pd.Series(model_fitted.resid, index=train.index)
# fig, ax = plt.subplots(2, 1)
# residuals.plot(title="Residuals", ax=ax[0])
# residuals.plot(kind="kde", title="Density", ax=ax[1])
# # plot_acf(residuals, lags=50, title="ACF", ax=ax[2])
# plt.savefig(os.path.join(fig_dir, a + "residual_plot.png"))


# with open(os.path.join(tbl_dir, a + "residuals_statistics.tex"), "w") as tf:
#     tf.write(residuals.describe().to_latex())

# plt.figure()
# model_fitted.plot_predict()
# plt.savefig(os.path.join(fig_dir, a + "0000.png"))


# ############################################################################
# #########                         Forecast                         #########
# ############################################################################
# fc, se, conf = model_fitted.forecast(nobs, alpha=0.05)

# fc_series = pd.Series(fc, index=test.index)
# lower_series = pd.Series(conf[:, 0], index=test.index)
# upper_series = pd.Series(conf[:, 1], index=test.index)


# print("[INFO] saving the plot of forecast...")
# plt.figure()
# plt.plot(train, label="Training set")
# plt.plot(test, label="Testing set")
# plt.plot(fc_series, label="Forecast")
# plt.fill_between(lower_series.index, lower_series, upper_series, color="k", alpha=0.1)
# plt.title("Forecast vs Actuals")
# plt.legend(loc="upper left", fontsize=12)
# plt.savefig(os.path.join(fig_dir, a + "Forecast_vs_Actuals.png"))

# print("[INFO] evaluate forecasts...")
# rmse = np.sqrt(mean_squared_error(test, fc_series))
# print("[INFO] Test RMS error: %.3f" % rmse)


# ############################################################################
# #########            Rolling Forecast of ARIMA model.             ##########
# ############################################################################

# X = series["Close"].values
# size = len(X) - nobs
# window = 570  # = 3 * 365
# train, test = X[window:size], X[size : len(X)]
# history = [x for x in train]
# predictions = list()
# print(train.shape)

# plt.figure()
# plt.plot(series["Close"][window:-nobs], label="train set")
# plt.legend()
# plt.savefig(os.path.join(fig_dir, a + "Rolling_trainset.png"))


# for t in range(len(test)):
#     model = ARIMA(history, order=orders)
#     model_fit = model.fit(disp=-1)
#     output = model_fit.forecast()
#     yhat = output[0]
#     predictions.append(yhat)
#     obs = test[t]
#     history.append(obs)
#     history = history[1:]


# print("[INFO] evaluate forecasts...")
# rmse = np.sqrt(mean_squared_error(test, predictions))
# print("[INFO] Test RMS error: %.3f" % rmse)

# print("[INFO] saving the plot of forecast...")
# plt.figure()
# plt.plot(test, label="test set")
# plt.plot(predictions, color="red", label="predictions")
# plt.legend()
# plt.savefig(os.path.join(fig_dir, a + "Rolling_Forecast.png"))


# plt.figure()
# plt.plot(np.arange(0, len(train)), train, label="Train")
# plt.plot(
#     np.arange(len(train), len(train) + len(test)), test, color="red", label="Test",
# )
# plt.plot(
#     np.arange(len(train), len(train) + len(test)),
#     predictions,
#     color="darkgreen",
#     label="Predicted",
#     dashes=[6, 2],
# )
# plt.legend()
# plt.savefig(os.path.join(fig_dir, a + "Rolling_Forecast_1.png"))
