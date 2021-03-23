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

activities = {
    # "Working at Computer": 1,
    # "Standing Up, Walking and Going updown stairs": 2,
    "Walking": 0,
    "Standing": 1,
    "Going UpDown Stairs": 1,
    # "Walking and Talking with Someone": 6,
    # "Talking while Standing": 7,
}

directions = ["x_acceleration", "y_acceleration", "z_acceleration"]


print("[INFO] Reading the first file...")
aaa = 84000  # 2657
sequences = np.zeros((135, aaa))
columns = ["user", "activity", "activity_code", "direction_sensor", "len"]

DF = pd.DataFrame(np.zeros((135, 5)), columns=columns)
DF["activity"].astype(str)
DF["user"].astype(str)
DF["direction_sensor"].astype(str)
from scipy import signal

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
            DF.loc[row, "len"] = len(temp1)

            # sequences[row, 0 : len(temp1)] = temp1
            # sequences[row, 0 : aaa] = temp1[0 : aaa]
            sequences[row, :] = signal.resample(temp1, aaa)
            maxa.append(len(temp1))
            row = row + 1


columns = ["seq_" + str(i) for i in range(aaa)]
DF2 = pd.DataFrame(sequences, columns=columns)

DF1 = pd.concat([DF, DF2], axis=1)


############################################################################
#########      Saving and showing the plot of the raw signal      ##########
############################################################################
if False:
    i = 24
    print("[INFO] Saving and showing the plot of the dataset")
    plt.figure()
    fig = DF1.iloc[i, 5:].plot(label="x_acceleration")
    fig = DF1.iloc[i + 1, 5:].plot(label="y_acceleration")
    fig = DF1.iloc[i + 2, 5:].plot(label="z_acceleration")
    plt.title("Person: " + str(DF1.iloc[i, 0]) + ", activity: " + DF1.iloc[i, 1])
    plt.legend()
    plt.savefig(os.path.join(fig_dir, a + "raw_signal.png"))

    print("[INFO] Saving and showing the plot of the dataset")
    plt.figure()
    fig = df["x_acceleration"].plot(label="x_acceleration")
    fig = df["y_acceleration"].plot(label="y_acceleration")
    fig = df["z_acceleration"].plot(label="z_acceleration")
    fig = (df["label"] * 500).plot(label="label")
    plt.title("Person: " + str(DF1.iloc[134, 0]))
    plt.legend()
    plt.savefig(os.path.join(fig_dir, a + "raw_signal_2.png"))

    print("[INFO] Saving and showing the histgram of the observations")
    plt.figure()
    fig = df["x_acceleration"].hist(bins=70, label="x_acceleration")
    fig = df["y_acceleration"].hist(bins=70, label="y_acceleration")
    fig = df["z_acceleration"].hist(bins=70, label="z_acceleration")
    plt.legend()
    plt.savefig(os.path.join(fig_dir, a + "hist.png"))


############################################################################
#########                    feature engineering                  ##########
############################################################################
if False:

    def stft(x, fftsize=256, overlap_pct=0.2):
        hop = int(fftsize * (1 - overlap_pct))

        w = scipy.hanning(fftsize + 1)[:-1]

        raw = np.array(
            [
                np.fft.rfft(w * x[i : i + fftsize])
                for i in range(0, len(x) - fftsize, hop)
            ]
        )
        return raw[:, : (fftsize // 2)]

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

    def var(x, varsize=256, overlap_pct=0.2):
        hop = int(varsize * (1 - overlap_pct))

        raw = [
            [np.var(x[i : i + varsize]) for i in range(0, len(x) - varsize, hop)],
            [np.mean(x[i : i + varsize]) for i in range(0, len(x) - varsize, hop)],
            [np.std(x[i : i + varsize]) for i in range(0, len(x) - varsize, hop)],
        ]

        # print(np.array(raw).shape)
        # print(raw)
        # 1 / 0
        return np.array(raw)

    data = DF1.iloc[:, 5:].values

    all_obs = list()
    all_varobs = list()

    for i in range(1, DF1.shape[0]):
        win = data[i, :]
        d = np.abs(stft(win))
        d1 = var(win)

        n_dim = 6
        obs = np.zeros((n_dim, d.shape[0]))
        for r in range(d.shape[0]):
            _, t = peakfind(d[r, :], n_peaks=n_dim)
            obs[:, r] = t.copy()
        if i % 10 == 0:
            print("Processed obs %s" % i)
        all_varobs.append(d1)
        all_obs.append(obs)

    all_obs = np.atleast_3d(all_obs)
    all_varobs = np.atleast_3d(all_varobs)

    all_obs = np.moveaxis(all_obs, 2, 1)
    all_varobs = np.moveaxis(all_varobs, 2, 1)
    all_obs = np.concatenate((all_obs, all_varobs), axis=2)


# print(all_obs)
# np.save(os.path.join(data_dir, "all_obs_ver1"), all_obs)
all_obs = np.load(os.path.join(data_dir, "all_obs_ver1.npy"))

############################################################################
#########                     Showing features                    ##########
############################################################################
if False:
    plot_data = np.abs(stft(data[0, :]))[99, :]
    values, locs = peakfind(plot_data, n_peaks=6)
    fp = locs[values > -1]
    fv = values[values > -1]
    plt.plot(np.log(plot_data), color="steelblue", label="fft")
    plt.plot(fp, np.log(fv), "x", color="darkred", label="Peak values")
    plt.title("Peak location example")
    plt.xlabel("Frequency (bins)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.savefig(os.path.join(fig_dir, a + "Peak_freq.png"))


############################################################################
#########                   compacting features                   ##########
############################################################################
print("[INFO] Compacting features...")


all_att = list()
all_labels = list()
for i in range(0, 132, 3):

    q1 = all_obs[i, :, :]
    q2 = all_obs[i + 1, :, :]
    q3 = all_obs[i + 2, :, :]

    q = np.concatenate((q1, q2, q3), axis=1)
    all_att.append(q)
    all_labels.append(DF1["activity_code"].values[i])

all_att = np.atleast_3d(all_att)


print("[INFO] encoding labels...")
le = preprocessing.LabelEncoder()
labels_encoder = le.fit_transform(all_labels)


############################################################################
#########                      making models                      ##########
############################################################################
print("[INFO] splitting the training and testing sets...")
aa = 410
trainData = all_att[:30, 0:aa, :]
testData = all_att[30:, 0:aa, :]
trainLabels = labels_encoder[:30]
testLabels = labels_encoder[30:]

lengths = []
for y in set(labels_encoder):
    occ = (trainLabels == y).sum()
    lengths.append([aa] * occ)

ys = set(labels_encoder)
acc = list()
for state in np.arange(2, 6):

    models = hmm.GaussianHMM(n_components=state).fit(
        np.vstack(trainData[trainLabels == 0, :, :]), lengths[int(0)]
    )  # for y in ys]

    # ff = [
    #     m.fit(np.vstack(trainData[trainLabels == y, :, :]), lengths[int(y)])
    #     for m, y in zip(models, ys)
    # ]
    # ff = [m.fit(trainData[trainLabels == y, :, :].mean(axis=1)) for m, y in zip(models, ys)]

    pred = []
    for x in testData:
        mdl_p = []
        # for m in models:
        mdl_p.append(models.score(x))
        # mdl_p.append(m.score(np.expand_dims(x, 0)))
        pred.append(np.array(mdl_p))

    pred = np.vstack(pred)
    #pred = np.exp(pred)
    for i in range(10000, 71000, 10000):
        q = [-i] * 14

        predi = np.column_stack((pred, np.array(q)))

        y_hat = np.argmax(predi, axis=1)
        missed = y_hat != testLabels
        print(
            "Test accuracy of {} states HMM for theresold -{} : {:.2f}\%".format(
                state, i, (100 * (1 - np.mean(missed)))
            )
        )
        acc.append(100 * (1 - np.mean(missed)))

        # fpr, tpr, threshold = metrics.roc_curve(testLabels, y_hat)
        # roc_auc = metrics.auc(fpr, tpr)

        # # method I: plt
        # plt.title("Receiver Operating Characteristic")
        # plt.plot(fpr, tpr, "b", label="AUC = %0.2f" % roc_auc)
        # plt.legend(loc="lower right")
        # plt.plot([0, 1], [0, 1], "r--")
        # plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.ylabel("True Positive Rate")
        # plt.xlabel("False Positive Rate")
        # plt.show()

print(acc)
# logpred= np.log(pred)


# import sklearn.metrics as metrics
# # calculate the fpr and tpr for all thresholds of the classification
# probs = model.predict_proba(X_test)
# preds = probs[:,1]
# fpr, tpr, threshold = metrics.roc_curve(testLabels, y_hat)
# roc_auc = metrics.auc(fpr, tpr)

# # method I: plt
# import matplotlib.pyplot as plt
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()


# from collections import Counter

# print(Counter(testLabels).keys())  # equals to list(set(words))
# print(Counter(testLabels).values())  # counts the elements' frequency
