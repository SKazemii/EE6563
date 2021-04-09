# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:05:52 2021

@author: pkumar1
"""

# all combined
from itertools import cycle
from scipy import interp

plt.figure()

colors = cycle(["aqua", "darkorange", "cornflowerblue", "black", "red"])
for i, color in zip(
    range(len(UdistanceSVM_mv_FP)), colors
):  # len(FP_score) serving no. of users
    #    if i==2:
    #        continue;
    #    else:

    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of User {0} (area = {1:0.2f})" "".format(i, roc_auc[i]),
    )

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(UdistanceSVM_mv_FP))]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(UdistanceSVM_mv_FP)):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#    if i==2:
#        continue;
#    else:
#        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= len(UdistanceSVM_mv_FP)

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="Average ROC curve (area = {0:0.2f})" "".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)
plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("All users ROC")
plt.legend(loc="lower right")
plt.show()

fnr = 1 - mean_tpr
# eer_threshold = th[np.nanargmin(np.absolute((fnr - lr_fpr)))]
EER = all_fpr[np.nanargmin(np.absolute((fnr - all_fpr)))]
EER1 = fnr[np.nanargmin(np.absolute((fnr - all_fpr)))]
print(str(EER1) + "   " + str(EER))
# i=3
# plt.figure()
# glist=['Wrist-Ext','Handopen','Pinky','Thumb']
##fpr1[i]=all_fpr; tpr1[i]=mean_tpr; roc_auc1[i]=roc_auc["macro"]
# colors = cycle(['aqua', 'darkorange', 'cornflowerblue','black',])#'red'])
# for i, color in zip(range(len(UdistanceSVM_mv_FP)), colors):#len(FP_score) serving no. of users
#     plt.plot(fpr1[i], tpr1[i], color=color, lw=lw,
#             label='ROC False Class {0} (area = {1:0.2f})'''.format(glist[i], roc_auc1[i]))
