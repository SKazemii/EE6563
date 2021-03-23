# 3. Learning HMM parameters
import os

# import urllib
# from urllib.request import urlopen
import numpy as np
from numpy.lib.stride_tricks import as_strided
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
from sklearn.model_selection import StratifiedShuffleSplit
from hmmlearn import hmm


#%matplotlib inline

#######################
## import audio file ##
#######################

# link = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
# link = 'https://github.com/yashiro32/speech_recognition/blob/master/audio.tar.gz'
# dlname = 'audio.tar.gz'
# testfile = urllib.request.URLopener()
# testfile.retrieve(link, dlname)
# os.system('tar xzf %s' % dlname)

fpaths = []
labels = []
spoken = []
path = "/Users/saeedkazemi/Documents/Python/EE6563/Dataset/Tutorial/"
for f in os.listdir(path + "audio"):
    for w in os.listdir(path + "audio" + "/" + f):
        fpaths.append("audio/" + f + "/" + w)
        labels.append(f)
        if f not in spoken:
            spoken.append(f)
print("Words spoken:", spoken)
# ('Words spoken:', ['apple', 'banana', 'kiwi', 'lime', 'orange', 'peach', 'pineapple'])


data = np.zeros((len(fpaths), 32000))
maxsize = -1
for n, file in enumerate(fpaths):
    _, d = wavfile.read(path + file)
    data[n, : d.shape[0]] = d
    if d.shape[0] > maxsize:
        maxsize = d.shape[0]
data = data[:, :maxsize]

all_labels = np.zeros(data.shape[0])
for n, l in enumerate(set(labels)):
    # print(l)
    all_labels[np.array([i for i, _ in enumerate(labels) if _ == l])] = n

#######################
##  plot audio file  ##
#######################

plt.plot(data[0, :], color="steelblue")
plt.title("Timeseries example for %s" % labels[0])
#plt.xlim(0, 3500)
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude (signed 16 bit)")
# plt.figure()

############################################################################################
#######################
# feature engineering #
#######################


def stft(x, fftsize=64, overlap_pct=0.5):
    # Modified from http://stackoverflow.com/questions/2459295/stft-and-istft-in-python
    hop = int(fftsize * (1 - overlap_pct))
    w = scipy.hanning(fftsize + 1)[:-1]
    raw = np.array(
        [np.fft.rfft(w * x[i : i + fftsize]) for i in range(0, len(x) - fftsize, hop)]
    )
    return raw[:, : (fftsize // 2)]

# Peak detection using the technique described here: http://kkjkok.blogspot.com/2013/12/dsp-snippets_9.html
def peakfind(x, n_peaks, l_size=3, r_size=3, c_size=3, f=np.mean):
    win_size = l_size + r_size + c_size
    shape = x.shape[:-1] + (x.shape[-1] - win_size + 1, win_size)
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


# print(data)
print(data.shape)
all_obs = []
for i in range(data.shape[0]):
    d = np.abs(stft(data[i, :]))
    n_dim = 6
    obs = np.zeros((n_dim, d.shape[0]))
    for r in range(d.shape[0]):
        _, t = peakfind(d[r, :], n_peaks=n_dim)
        obs[:, r] = t.copy()
    if i % 10 == 0:
        print("Processed obs %s" % i)
    all_obs.append(obs)

all_obs = np.atleast_3d(all_obs)  # to convert into 3D
# print(all_obs)
print(all_obs.shape)
#######################
## plot new features ##
#######################

plot_data = np.abs(stft(data[0, :]))[15, :]
values, locs = peakfind(plot_data, n_peaks=6)
fp = locs[values > -1]
fv = values[values > -1]
plt.plot(plot_data, color="steelblue")
plt.plot(fp, fv, "x", color="darkred")
plt.title("Peak location example")
plt.xlabel("Frequency (bins)")
plt.ylabel("Amplitude")


#############################################################################
#######################
# split training data #
#######################

sss = StratifiedShuffleSplit(n_splits=2, test_size=0.1, random_state=0)
sss.get_n_splits(all_obs, all_labels)
print(all_obs.shape)
print(all_labels.shape)

for n, i in enumerate(all_obs):  # normalization
    all_obs[n] /= all_obs[n].sum(axis=0)  # row-wise sum

for train_index, test_index in sss.split(all_obs, all_labels):
    X_train = [all_obs[i] for i in train_index]
    X_test = [all_obs[i] for i in test_index]
    # X_train, X_test = all_obs[train_index, ...], all_obs[test_index, ...]
    y_train, y_test = all_labels[train_index], all_labels[test_index]

X_train = np.array(X_train)
X_test = np.array(X_test)
print("Size of training matrix:", X_train.shape)
# ('Size of training matrix:', (94, 6, 216))
print("Size of testing matrix:", X_test.shape)
# ('Size of testing matrix:', (11, 6, 216))

#######################
##  estimate model   ##
#######################


ys = set(all_labels)
models = [hmm.GaussianHMM(n_components=6) for y in ys]
# _ = [m.fit(X_train[y_train == y, :, :].mean(axis=2)) for m, y in zip(models, ys)]
################################
# Training
##################
X_tr = np.moveaxis(X_train, 2, 1)
X_te = np.moveaxis(X_test, 2, 1)
lengths = []
for y in range(7):
    occ = (y_train == y).sum()
    lengths.append([216] * occ)

ff = [m.fit(np.vstack(X_tr[y_train == y, :, :]), lengths[int(y)]) for m, y in zip(models, ys)]


##########Prediction
# logprob = np.array([[m.score(i) for i in np.expand_dims(X_test.mean(axis=2),axis=0)] for m in models])
pred = []
for x in X_te:
    print(x)
    mdl_p = []
    for m in models:
        mdl_p.append(m.score(x))
    pred.append(np.array(mdl_p))

pred = np.vstack(pred)


y_hat = np.argmax(pred, axis=1)
missed = y_hat != y_test
print("Test accuracy: %.2f percent" % (100 * (1 - np.mean(missed))))


#######################
## Method 2 ##
#######################
ys = set(all_labels)
models = [hmm.GaussianHMM(n_components=6) for y in ys]
_ = [m.fit(X_train[y_train == y, :, :].mean(axis=2)) for m, y in zip(models, ys)]
# logprob = np.array([[m.score(i) for i in np.expand_dims(X_test.mean(axis=2),axis=0)] for m in models])
pred = []
Xte = X_test.mean(axis=2)
for x in Xte:
    mdl_p = []
    for m in models:
        mdl_p.append(m.score(np.expand_dims(x, 0)))
    pred.append(np.array(mdl_p))

pred = np.vstack(pred)


y_hat = np.argmax(pred, axis=1)
missed = y_hat != y_test
print("Test accuracy: %.2f percent" % (100 * (1 - np.mean(missed))))
