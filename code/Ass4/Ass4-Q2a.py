import warnings

warnings.filterwarnings("ignore")

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from datetime import datetime
import pprint

from pandas.plotting import lag_plot
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold

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
from keras import metrics
from keras.wrappers.scikit_learn import KerasClassifier
import keras

plt.rcParams["figure.figsize"] = (14, 7)
a = "Ass4_Q2a_"

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
#########                           LSTM                          ##########
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


window = 50  # = 3 * 365
df_train, df_test = dataset[:-window, :], dataset[-window:, :]


scaler = MinMaxScaler()
train = scaler.fit_transform(df_train)
test = scaler.transform(df_test)


look_back = 15
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

trainY = np.expand_dims(trainY, axis=1)
testY = np.expand_dims(testY, axis=1)


def create_model(LSTM_units=4, learning_rate=0.001, flag=True):
    model = Sequential()
    model.add(LSTM(LSTM_units, input_shape=(look_back, 1)))
    model.add(Dense(1))
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    if flag:
        model.compile(
            loss="mean_squared_error", optimizer=opt, metrics=["accuracy"],
        )
    else:
        model.compile(
            loss="mean_squared_error", optimizer=opt,
        )

    return model


############################################################################
#########                       grid search                       ##########
############################################################################
model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [16, 32, 64]
epochs = [10]
LSTM_units = [5, 15, 10]
learning_rate = [0.0001, 0.001, 0.01]
param_grid = dict(
    batch_size=batch_size,
    epochs=epochs,
    LSTM_units=LSTM_units,
    learning_rate=learning_rate,
)
pprint.pprint(param_grid)
kfold = KFold(n_splits=5, random_state=None, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=kfold)
grid_result = grid.fit(trainX, trainY)
# summarize results
print(
    "Best accuracy: %f by using %s"
    % (grid_result.best_score_, grid_result.best_params_)
)
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]
for mean, stdev, param in zip(means, stds, params):
    print(
        "mean accuracy: %f (standard accuracy:%f) with parameters: %r"
        % (mean, stdev, param)
    )


############################################################################
#########                       best model                        ##########
############################################################################
best_model = create_model(
    LSTM_units=grid_result.best_params_["LSTM_units"],
    learning_rate=grid_result.best_params_["learning_rate"],
    flag=False,
)
weight_path = "Ass4_Q2a_weights.best.hdf5"
checkpoint = ModelCheckpoint(
    weight_path,
    monitor="val_loss",
    verbose=0,
    save_best_only=True,
    mode="min",
    save_weights_only=False,
)
history = best_model.fit(
    trainX,
    trainY,
    batch_size=grid_result.best_params_["batch_size"],
    validation_split=0.2,
    callbacks=[checkpoint],
    epochs=100,
)

trainPredict = best_model.predict(trainX)
testPredict = best_model.predict(testX)


# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform(testY)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
testScore = math.sqrt(mean_squared_error(testY, testPredict))


best_model1 = load_model(
    "/Users/saeedkazemi/Documents/Python/EE6563/Ass4_Q2a_weights.best.hdf5"
)
# make predictions
trainPredict = best_model1.predict(trainX)
testPredict = best_model1.predict(testX)
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


def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(fig_dir, a + title + ".png"))


visualize_loss(history, "Training and Validation Loss")


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
plt.plot(dataset)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.savefig(os.path.join(fig_dir, a + "forcasted.png"))


# plt.savefig(os.path.join(fig_dir, a + "residual_plot.png"))


# with open(os.path.join(tbl_dir, a + "residuals_statistics.tex"), "w") as tf:
#     tf.write(residuals.describe().to_latex())

