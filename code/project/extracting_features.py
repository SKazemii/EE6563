print("[INFO] importing libraries....")


import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import config as cfg

print("[INFO] defining functions....")


def centriod(img):
    wx = list()
    wy = list()
    for j in range(60):
        non_zero_elements = np.nonzero(img[j, :, :])
        x = 0
        y = 0
        for i in range(len(non_zero_elements[0])):
            x += non_zero_elements[1][i]
            y += non_zero_elements[0][i]

        if x == 0:
            # x = 0
            pass
        else:
            x /= len(non_zero_elements[0])

        if y == 0:
            # y = 0
            pass
        else:
            y /= len(non_zero_elements[0])

        wx.append(x)
        wy.append(y)

    return wx, wy


print("[INFO] reading dataset....")
with h5py.File(cfg.dataset_file, "r") as hdf:
    barefoots = hdf.get("/barefoot/data")[:]

print("[INFO] saving 3 samples....")
img = np.squeeze(barefoots[1, :, :, :])


ss = np.concatenate(
    (img[18, 21:65, 17:42], img[25, 21:65, 17:42], img[32, 21:65, 17:42]), axis=1
)
i = plt.imshow(ss)
plt.tight_layout()
plt.savefig(os.path.join(cfg.fig_dir, "frame.png"))

print("[INFO] storing in pandas dataframe ....")
Data_sum = list()
Data_max = list()
Data_xCe = list()
Data_yCe = list()

for sample in range(barefoots.shape[0]):
    img = np.squeeze(barefoots[sample, :, :, :])
    aa = np.sum(img, axis=2)
    bb = np.sum(aa, axis=1)
    Data_sum.append(bb[:60])

    img = np.squeeze(barefoots[sample, :, :, :])
    aa = np.max(img, axis=2)
    bb = np.max(aa, axis=1)
    Data_max.append(bb[:60])

    wx, wy = centriod(img)
    Data_xCe.append(wx)
    Data_yCe.append(wy)

index = ["Sample_" + str(i) for i in np.arange(barefoots.shape[0])]
column = [i for i in np.arange(60)]
df_sum = pd.DataFrame(Data_sum, columns=column, index=index).T
df_sum.to_pickle(os.path.join(cfg.pickle_dir, "df_sum.pickle"))

df_max = pd.DataFrame(Data_max, columns=column, index=index).T
df_max.to_pickle(os.path.join(cfg.pickle_dir, "df_max.pickle"))

df_xCe = pd.DataFrame(Data_xCe, columns=column, index=index).T
df_xCe.to_pickle(os.path.join(cfg.pickle_dir, "df_xCe.pickle"))

df_yCe = pd.DataFrame(Data_yCe, columns=column, index=index).T
df_yCe.to_pickle(os.path.join(cfg.pickle_dir, "df_yCe.pickle"))


print("[INFO] Saving and showing the plot of the some samples...")
plt.figure()
axes = plt.axes()
df_sum["Sample_1"].plot(ax=axes)
df_sum["Sample_100"].plot(ax=axes)
df_sum["Sample_200"].plot(ax=axes)
df_sum["Sample_150"].plot(ax=axes)
df_sum["Sample_400"].plot(ax=axes)

plt.legend(["Sample 1", "Sample 100", "Sample 200", "Sample 300", "Sample 400"])
plt.savefig(os.path.join(cfg.fig_dir, "df_sum.png"))


print("[INFO] Saving and showing the plot of the some samples...")
plt.figure()
axes = plt.axes()
df_max["Sample_1"].plot(ax=axes)
df_max["Sample_100"].plot(ax=axes)
df_max["Sample_200"].plot(ax=axes)
df_max["Sample_150"].plot(ax=axes)
df_max["Sample_400"].plot(ax=axes)

plt.legend(["Sample 1", "Sample 100", "Sample 200", "Sample 300", "Sample 400"])
plt.savefig(os.path.join(cfg.fig_dir, "df_max.png"))

print("[INFO] Saving and showing the plot of the some samples...")
plt.figure()
axes = plt.axes()
df_xCe["Sample_1"].plot(ax=axes)
df_xCe["Sample_100"].plot(ax=axes)
df_xCe["Sample_200"].plot(ax=axes)
df_xCe["Sample_150"].plot(ax=axes)
df_xCe["Sample_400"].plot(ax=axes)

plt.legend(["Sample 1", "Sample 100", "Sample 200", "Sample 300", "Sample 400"])
plt.savefig(os.path.join(cfg.fig_dir, "df_xCe.png"))

print("[INFO] Saving and showing the plot of the some samples...")
plt.figure()
axes = plt.axes()
df_yCe["Sample_1"].plot(ax=axes)
df_yCe["Sample_100"].plot(ax=axes)
df_yCe["Sample_200"].plot(ax=axes)
df_yCe["Sample_150"].plot(ax=axes)
df_yCe["Sample_400"].plot(ax=axes)

plt.legend(["Sample 1", "Sample 100", "Sample 200", "Sample 300", "Sample 400"])
plt.savefig(os.path.join(cfg.fig_dir, "df_yCe.png"))