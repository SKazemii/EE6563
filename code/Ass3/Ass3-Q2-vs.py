import warnings

warnings.filterwarnings("ignore")

import math
import os
import glob
import scipy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime

from hmmlearn import hmm
from hmmlearn.hmm import GaussianHMM
from numpy.lib.stride_tricks import as_strided
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


plt.rcParams["figure.figsize"] = (14, 7)
a = "Ass3_Q2_"
random_state = 0


print("[INFO] Setting directories")
project_dir = os.getcwd()
fig_dir = os.path.join(project_dir, "manuscript", "src", "figures", "Ass3")
tbl_dir = os.path.join(project_dir, "manuscript", "src", "tables", "Ass3")
data_dir = os.path.join(project_dir, "Dataset", "Ass3")


############################################################################
#########                       Reading data                      ##########
############################################################################
print("[INFO] Reading all files...")
file_paths = glob.glob(data_dir + "/" + "*.csv")
file_paths.sort(key=os.path.getmtime)
users = list(range(15))
print(users)
activities = {
    "Working at Computer": 1,
    "Standing Up, Walking and Going updown stairs": 2,
    "Standing": 3,
    "Walking": 4,
    "Going UpDown Stairs": 5,
    "Walking and Talking with Someone": 6,
    "Talking while Standing": 7,
}
directions = ["x_acceleration", "y_acceleration", "z_acceleration"]


print("[INFO] Reading the first file...")
sequences = np.zeros((315, 84000))
columns = ["user", "activity", "activity_code", "direction_sensor"]

DF = pd.DataFrame(np.zeros((315, 4)), columns=columns)
DF["activity"].astype(str)
DF["user"].astype(str)
DF["direction_sensor"].astype(str)

# print(DF.head())
row = 0
maxa = list()
for user in users:
    df = pd.read_csv(
        file_paths[user],
        header=0,
        names=[
            "sequential_number",
            "x_acceleration",
            "y_acceleration",
            "z_acceleration",
            "label",
        ],
        index_col=0,
    ).dropna()
    for activity, label in activities.items():
        for direct in directions:
            temp = df[df["label"] == label]

            DF.loc[row, "user"] = str(user)
            DF.loc[row, "activity"] = activity
            DF.loc[row, "activity_code"] = label
            DF.loc[row, "direction_sensor"] = direct

            # Expand the array up to 84000 with zeros.
            temp1 = temp[direct].values.T

            sequences[row, 0 : len(temp1)] = temp1
            row = row + 1


columns = ["seq_" + str(i) for i in range(84000)]
DF2 = pd.DataFrame(sequences, columns=columns)


DF1 = pd.concat([DF, DF2], axis=1)
print(DF1.head())


############################################################################
#########      Saving and showing the plot of the raw signal      ##########
############################################################################
# i = 9
# print("[INFO] Saving and showing the plot of the dataset")
# plt.figure()
# fig = DF1.iloc[i, 4:84004].plot(label="x_acceleration")
# fig = DF1.iloc[i + 1, 4:84004].plot(label="y_acceleration")
# fig = DF1.iloc[i + 2, 4:84004].plot(label="z_acceleration")
# plt.title("Person: " + str(DF1.iloc[i, 0]) + ", activity: " + DF1.iloc[i, 1])
# plt.legend()
# plt.savefig(os.path.join(fig_dir, a + "raw_signal.png"))

# print("[INFO] Saving and showing the plot of the dataset")
# plt.figure()
# fig = df["x_acceleration"].plot(label="x_acceleration")
# fig = df["y_acceleration"].plot(label="y_acceleration")
# fig = df["z_acceleration"].plot(label="z_acceleration")
# fig = (df["label"] * 500).plot(label="label")
# plt.title("Person: " + str(DF1.iloc[314, 0]))
# plt.legend()
# plt.savefig(os.path.join(fig_dir, a + "raw_signal_2.png"))

# print("[INFO] Saving and showing the histgram of the observations")
# plt.figure()
# fig = df["x_acceleration"].hist(bins=70, label="x_acceleration")
# fig = df["y_acceleration"].hist(bins=70, label="y_acceleration")
# fig = df["z_acceleration"].hist(bins=70, label="z_acceleration")
# plt.legend()
# plt.savefig(os.path.join(fig_dir, a + "hist.png"))


############################################################################
#########                    feature engineering                  ##########
############################################################################
def stft(x, fftsize=256, overlap_pct=0.2):
    # Modified from http://stackoverflow.com/questions/2459295/stft-and-istft-in-python
    hop = int(fftsize * (1 - overlap_pct))

    w = scipy.hanning(fftsize + 1)[:-1]

    raw = np.array(
        [np.fft.rfft(w * x[i : i + fftsize]) for i in range(0, len(x) - fftsize, hop)]
    )
    # print(raw.shape)
    # print((fftsize // 2))
    # print(raw[:, : (fftsize // 2)].shape)
    # 1 / 0
    return raw[:, : (fftsize // 2)]


# Peak detection using the technique described here: http://kkjkok.blogspot.com/2013/12/dsp-snippets_9.html
def peakfind(x, n_peaks, l_size=3, r_size=3, c_size=3, f=np.mean):
    win_size = l_size + r_size + c_size
    shape = (x.shape[-1] - win_size + 1, win_size)
    strides = x.strides + (x.strides[-1],)
    xs = as_strided(x, shape=shape, strides=strides)

    def is_peak(x):
        centered = np.argmax(x) == l_size + int(c_size / 2)
        l = x[:l_size]
        c = x[l_size : l_size + c_size]
        r = x[-r_size:]
        passes = np.max(c) > np.max([f(l), f(r)])
        if centered and passes:
            return np.max(c)
        else:
            return -1

    r = np.apply_along_axis(is_peak, 1, xs)
    top = np.argsort(r, None)[::-1]
    heights = r[top[:n_peaks]]
    # Add l_size and half - 1 of center size to get to actual peak location
    top[top > -1] = top[top > -1] + l_size + int(c_size / 2.0)
    return heights, top[:n_peaks]


# all_obs = list()
data = DF1.iloc[:, 4:].values

# for i in range(DF1.shape[0]):
#     d = np.abs(stft(data[i, :]))

#     n_dim = 6
#     obs = np.zeros((n_dim, d.shape[0]))
#     for r in range(d.shape[0]):
#         _, t = peakfind(d[r, :], n_peaks=n_dim)
#         obs[:, r] = t.copy()
#     if i % 10 == 0:
#         print("Processed obs %s" % i)
#     all_obs.append(obs)

# all_obs = np.atleast_3d(all_obs)
# print(all_obs)

# np.save(os.path.join(data_dir, "all_obs"), all_obs)
all_obs = np.load(os.path.join(data_dir, "all_obs.npy"))


plot_data = np.abs(stft(data[0, :]))[1, :]
values, locs = peakfind(plot_data, n_peaks=6)
fp = locs[values > -1]
fv = values[values > -1]
plt.plot(np.log(plot_data), color="steelblue")
plt.plot(fp, np.log(fv), "x", color="darkred")
plt.title("Peak location example")
plt.xlabel("Frequency (bins)")
plt.ylabel("Amplitude")
# plt.show()


all_obs = all_obs.transpose(0, 2, 1)  # .reshape(315, -1)
all_obs = np.vstack(all_obs)
print(all_obs.shape)
# print(all_obs)

all_labels = DF1["activity_code"].values
# print(all_labels)
print("[INFO] encoding labels...")
le = preprocessing.LabelEncoder()
labels_encoder = le.fit_transform(all_labels)
# print(labels_encoder)
print(labels_encoder.shape)
q = [labels_encoder[i // 411] for i in range(315 * 411)]
# print(q[-1400:])
print(len(q))
# print(labels_encoder)

# 1 / 0
print("[INFO] splitting the training and testing sets...")
(trainData, testData, trainLabels, testLabels) = train_test_split(
    all_obs,
    q,
    stratify=q,
    test_size=0.1,
    random_state=random_state,
)

scaler = preprocessing.MinMaxScaler()
trainData = scaler.fit_transform(trainData)
testData = scaler.transform(testData)

from collections import Counter

print(Counter(testLabels).keys())  # equals to list(set(words))
print(Counter(testLabels).values())  # counts the elements' frequency
temp_2 = [[trainLabels[i]] for i in range(len(trainLabels))]
temp_arr = np.concatenate((trainData, np.array(temp_2)), axis=1)
df_tr = pd.DataFrame(
    temp_arr, columns=["f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "label"]
)


temp_3 = [[testLabels[i]] for i in range(len(testLabels))]
temp_arr = np.concatenate((testData, np.array(temp_3)), axis=1)
df_te = pd.DataFrame(
    temp_arr, columns=["f_1", "f_2", "f_3", "f_4", "f_5", "f_6", "label"]
)
print(df_te.shape)
states = np.arange(3, 10, 1)
models = list()
acc = list()
for state in states:

    mdls = list()
    for act in range(7):
        temp = df_tr[df_tr["label"] == act]
        hmm = GaussianHMM(
            n_components=state, covariance_type="diag"  # , random_state=random_state
        )
        mdls.append(hmm.fit(temp.iloc[:, :-1].values))
    pred = []
    for i in range(len(df_te)):
        mdl_p = []
        for act in range(7):

            mdl_p.append(mdls[act].score(df_te.iloc[i, :-1].values.reshape(-1, 1)))
        pred.append(np.array(mdl_p))

    pred = np.vstack(pred)
    y_hat = np.argmax(pred, axis=1)
    missed = y_hat != np.array(testLabels)
    print(
        "Test accuracy of {} : {} percent".format(state, (100 * (1 - np.mean(missed))))
    )
    acc.append(100 * (1 - np.mean(missed)))
print(acc)
# 1 / 0
# print(mdls[0].score(df_te.iloc[0, :-1].values.reshape(-1, 1)))

# labels_predict = []
# for i in range(len(df_te)):
#     log_likelihood_value = np.array(
#         [
#             mdls.score(df_te.iloc[i, :-1].values),
#             mdls.score(df_te.iloc[i, :-1].values),
#             mdls.score(df_te.iloc[i, :-1].values),
#             mdls.score(df_te.iloc[i, :-1].values),
#             mdls.score(df_te.iloc[i, :-1].values),
#             mdls.score(df_te.iloc[i, :-1].values),
#         ]
#     )
#     labels_predict.append(np.argmax(log_likelihood_value))
# labels_predict = np.array(labels_predict)

# F1 = f1_score(labels_true, labels_predict, average="micro")
# acc = accuracy_score(labels_true, labels_predict)
# models.append(mdls)

#     1 / 0

#     print(df_tr.head())


# 1 / 0

# from sklearn import metrics


# hmm_clf = hmm.GaussianHMM(n_components=7, covariance_type="full", n_iter=1000).fit(
#     trainData
# )

# pred_train_hmm = hmm_clf.predict(trainData)
# pred_test_hmm = hmm_clf.predict(testData)

# print("\nPrediction accuracy for the training dataset")
# print("{:.2%}\n".format(metrics.accuracy_score(trainLabels, pred_train_hmm)))

# print("Prediction accuracy for the test dataset")
# print("{:.2%}\n".format(metrics.accuracy_score(testLabels, pred_test_hmm)))

# print("Confusion Matrix")
# print(metrics.confusion_matrix(testLabels, pred_test_hmm))


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


############################################################################
#########                     fitting ARIMA                       ##########
############################################################################
