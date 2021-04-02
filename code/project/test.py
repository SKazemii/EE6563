from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_selection import VarianceThreshold
import sklearn


from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import sklearn.model_selection
from sklearn import preprocessing
from sklearn import feature_selection

import seaborn as sns
import tsfel
import pickle
import config as cfg
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, sys

sys.path.append("code/project/")

project_dir = os.getcwd()


f = open(os.path.join(cfg.output_dir, "result.txt"), "w")

testLabels = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
preds = [1, 1, 1, 3, 3, 2, 2, 2, 4, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 1]
# write the classification report to file
f.write("{}\n".format(classification_report(testLabels, preds)))


# display the confusion matrix
print("[INFO] confusion matrix")


# plot the confusion matrix

cm = confusion_matrix(testLabels, preds)
sns.heatmap(cm, annot=True, cmap="Set2")
plt.savefig(os.path.join(cfg.fig_dir, cfg.result_name_file + ".png"))
plt.savefig(os.path.join(cfg.fig_dir, cfg.result_name_file + ".png"))

FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)

FP = FP.astype(float)
FN = FN.astype(float)
TP = TP.astype(float)
TN = TN.astype(float)

# Sensitivity, hit rate, recall, or true positive rate
# reflects the classifier’s ability to detect members of the positive class (pathological state)
TPR = TP / (TP + FN)
# Specificity or true negative rate
# reflects the classifier’s ability to detect members of the negative class (normal state)
TNR = TN / (TN + FP)
# Precision or positive predictive value
PPV = TP / (TP + FP)
# Negative predictive value
NPV = TN / (TN + FN)
# Fall out or false positive rate
# reflects the frequency with which the classifier makes a mistake by classifying normal state as pathological
FPR = FP / (FP + TN)
# False negative rate
# reflects the frequency with which the classifier makes a mistake by classifying pathological state as normal
FNR = FN / (TP + FN)
# False discovery rate
FDR = FP / (TP + FP)
# Overall accuracy
ACC = (TP + TN) / (TP + FP + FN + TN)

f.write("\n###########################################################")
f.write("\n###########################################################\n")
f.write("False Positive (FP):\t\t\t\n {}\n\n".format(FP))
f.write("False Negative (FN):\t\t\t\n {}\n\n".format(FN))
f.write("True Positive (TP):\t\t\t\n {}\n\n".format(TP))
f.write("True Negative (TN):\t\t\t\n {}\n\n".format(TN))
f.write("True Positive Rate (TPR)(Recall):\t\t\n {} \n\n".format(TPR))
f.write("True Negative Rate (TNR)(Specificity):\t\t\n {} \n\n".format(TNR))
f.write("Positive Predictive Value (PPV)(Precision):\n {} \n\n".format(PPV))
f.write("Negative Predictive Value (NPV):\n {} \n\n".format(NPV))
f.write(
    "False Positive Rate (FPR)(False Match Rate (FMR))(False Acceptance Rate (FAR)):\t\t\n {} \n\n".format(
        FPR
    )
)
f.write(
    "False Negative Rate (FNR)(False Non-Match Rate (FNMR))(False Rejection Rate (FRR)):\t\t\n {} \n\n".format(
        FNR
    )
)
f.write("False Discovery Rate (FDR):\t\t\n {} \n\n".format(FDR))
f.write("Overall accuracy (ACC):\t\t\t\n {} \n\n".format(ACC))
f.write("\n###########################################################")
f.write("\n###########################################################")
f.write("\nConfusion Matrix (CM): \n{}".format(cm))


f.close()
