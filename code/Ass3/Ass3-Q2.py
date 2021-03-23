import warnings

warnings.filterwarnings("ignore")

import math
import os
import glob
import scipy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    "Standing Up, Walking and Going updown stairs": 2,
    # "Standing": 3,
    # "Walking": 4,
    # "Going UpDown Stairs": 5,
    # "Walking and Talking with Someone": 6,
    # "Talking while Standing": 7,
}
directions = ["x_acceleration", "y_acceleration", "z_acceleration"]


print("[INFO] Reading the first file...")
aaa = 84000  # 2657
aa = 45
sequences = np.zeros((aa, aaa))
columns = ["user", "activity", "activity_code", "direction_sensor", "len"]

DF = pd.DataFrame(np.zeros((aa, 5)), columns=columns)
DF["activity"].astype(str)
DF["user"].astype(str)
DF["direction_sensor"].astype(str)
from scipy import signal

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
            sequences[row, :] = signal.resample(temp1, aaa)

            # sequences[row, 0 : aaa] = temp1[0 : aaa]
            maxa.append(len(temp1))
            row = row + 1


columns = ["seq_" + str(i) for i in range(aaa)]
DF2 = pd.DataFrame(sequences, columns=columns)


DF1 = pd.concat([DF, DF2], axis=1)


############################################################################
#########      Saving and showing the plot of the raw signal      ##########
############################################################################
if False:
    i = 3
    print("[INFO] Saving and showing the plot of the dataset")
    plt.figure()
    fig = DF1.iloc[i, 5:].plot(label="x_acceleration")
    fig = DF1.iloc[i + 1, 5:].plot(label="y_acceleration")
    fig = DF1.iloc[i + 2, 5:].plot(label="z_acceleration")
    plt.title("Person: " + str(DF1.iloc[i, 0]) + ", activity: " + DF1.iloc[i, 1])
    plt.legend()
    plt.savefig(os.path.join(fig_dir, a + "raw_signal.png"))


############################################################################
#########                      decoding problem                   ##########
############################################################################

data1 = DF1.iloc[:, 5:].values

from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
data1 = scaler.fit_transform(data1)
    
    
data = np.moveaxis(data1, 1, 0)

XX = np.array([data[:, i : i + 3] for i in range(0, 45, 3)])

lengths = DF1.iloc[:, 4].values
lengths = [int(lengths[i]) for i in range(0, 28, 3)]
# lengths = [84000] * 10
w=0000
ww =40000
model = hmm.GaussianHMM(n_components=3).fit(np.vstack(XX[0:10, :, :]), lengths)
dec = list()
for i in range(0, 43, 3):
    dec.append(model.decode(np.array(data[:, i : i + 3]))[1])  # *50 +1850

    plt.figure()
    plt.plot(data[w:ww, i], linewidth=2, label="x_acceleration")
    plt.plot(data[w:ww, i + 1], linewidth=2, label="y_acceleration")
    plt.plot(data[w:ww, i + 2], linewidth=2, label="z_acceleration")

    plt.plot(dec[int(i / 3)][w:ww], label="states", color="Blue")
    plt.legend(fontsize=11)
    plt.savefig(os.path.join(fig_dir, a + "states_user_" + str(int(i / 3)) + "N.png"))
