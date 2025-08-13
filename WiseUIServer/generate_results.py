import os, sys
import pickle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10), max_n=None, flag_show=True, flag_part=False, acc_array=None, std_array=None):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        # change category codes or labels to new labels
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    # calculate a confusion matrix with the new labels
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # calculate row sums (for calculating % & plot annotations)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    # calculate proportions
    cm_perc = cm / float(max_n) * 100
    # empty array for holding annotations for each cell in the heatmap
    annot = np.empty_like(cm).astype(str)
    # get the dimensions
    nrows, ncols = cm.shape
    # cycle over cells and create annotations for each cell
    for i in range(nrows):
        for j in range(ncols):
            # get the count for the cell
            c = cm[i, j]
            # get the percentage for the cell
            p = cm_perc[i, j]

            if i == j:
                s = cm_sum[i]
                if acc_array is not None and std_array is not None:
                    annot[i, j] = '%d\nsd:%.1f\n(%.1f%%)' % (c, std_array[i, j], p)
                elif not flag_part:
                    # convert the proportion, count, and row sum to a string with pretty formatting
                    annot[i, j] = '%d\n(%.1f%%)' % (c, p)
                else:
                    annot[i, j] = '%.1f' % p
            elif c == 0:
                if not flag_part:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f' % 0.0
            else:
                if acc_array is not None and std_array is not None:
                    annot[i, j] = '%d\nsd:%.1f\n(%.1f%%)' % (c, std_array[i, j], p)
                elif not flag_part:
                    annot[i, j] = '%d\n(%.1f%%)' % (c, p)
                else:
                    annot[i, j] = '%.1f' % p


    # convert the array to a dataframe. To plot by proportion instead of number, use cm_perc in the DataFrame instead of cm
    cm = pd.DataFrame(cm, index=labels_str, columns=labels_str)
    cm.index.name = 'Actural Gesture'
    cm.columns.name = 'Predicted Gesture\n(multiple predictions during action)'

    if flag_show:
        # create empty figure with a specified size
        fig, ax = plt.subplots(figsize=figsize)
        # plot the data using the Pandas dataframe. To change the color map, add cmap=..., e.g. cmap = 'rocket_r'
        sns.heatmap(cm, annot=annot, fmt='', ax=ax)#, vmin=0.0, vmax=1.0)
        plt.savefig(f"./pkl_test/cm_{SUBJECT}.png")
        plt.show()

    return annot



SUBJECT_list = [0, 1, 3, 4]
OBJ_list = ['key_0', 'cyl_0', 'app_0']

## TP, FP, ...
# pkl_path = f"./pkl_test/result_{SUBJECT}.pkl"
# with open(pkl_path, 'rb') as f:
#     data = pickle.load(f)
#
# eps = sys.float_info.epsilon
#
# prec_ = 0.0
# recall_ = 0.0
# f1score = 0.0
# for i in range(8):
#     TP = data[i, i]
#     FN = sum(data[i, :]) - data[i, i]
#     FP = sum(data[:, i]) - data[i, i]
#
#     prec = TP / (TP + FP+eps)
#     recall = TP / (TP + FN+eps)
#     f1score += 2*(prec*recall)/(prec+recall+eps)
#     prec_ += prec
#     recall_ += recall
#
# print("Precision Score: ", prec_/8.0)
# print("Recall Score: ", recall_/8.0)
# print("F1 Score: ", f1score/8.0)

## confusion matrix
y_test_sum = []
y_pred_sum = []

test_subj = []
pred_subj = []
for SUBJECT in SUBJECT_list:
    y_test_subj = []
    y_pred_subj = []

    for OBJ in OBJ_list:
        pkl_path = f"./pkl_test/results_ours/result_{SUBJECT}_{OBJ}_cm.pkl"

        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        [y_test, y_pred] = data
        y_test_sum.append(y_test)
        y_pred_sum.append(y_pred)

        y_test_subj.append(y_test)
        y_pred_subj.append(y_pred)

    test_subj.append(y_test_subj)
    pred_subj.append(y_pred_subj)
# print("F1 Score: ", f1_score(y_test, y_pred, average="macro"))
# print("Precision Score: ", precision_score(y_test, y_pred, average="macro"))
# print("Recall Score: ", recall_score(y_test, y_pred, average="macro"))

y_test_sum = np.asarray(y_test_sum).flatten()
y_pred_sum = np.asarray(y_pred_sum).flatten()

mask = y_pred_sum == 7
np.delete(y_test_sum, mask)
np.delete(y_pred_sum, mask)

labels_str = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'CLOCK', 'C-CLOCK', 'TAP']
labels = [0,1,2,3,4,5,6]

anno_list = []
## calculate mean for each subject and std
for i in range(4):
    perc_subj = np.zeros((7, 7))
    gt_np = np.asarray(test_subj[i]).flatten()
    pred_np = np.asarray(pred_subj[i]).flatten()

    anno = cm_analysis(gt_np, pred_np, labels, ymap=None, figsize=(10, 10), max_n=15, flag_show=False, flag_part=True)
    anno_list.append(anno)

std_array = np.zeros((7,7))
acc_array = np.empty_like(std_array).astype(str)

per_subj_all = []
per_subj_acc = []
per_subj_std = []
for i in range(7):
    for j in range(7):
        acc_list = []
        for anno in anno_list:  # for each subject results
            acc = anno[i, j]
            acc_list.append(float(acc))
        acc_np = np.asarray(acc_list)
        acc_mean = np.mean(acc_np)
        acc_std = np.std(acc_np)

        std_array[i, j] = acc_std
        acc_array[i, j] = f'({acc_list[0]}%, {acc_list[1]}% \n {acc_list[2]}%, {acc_list[3]}%)'

        if i == j:
            per_subj_acc.append(acc_mean)
            per_subj_std.append(acc_std)
            per_subj_all.append(acc_np)



per_subj_all = np.asarray(per_subj_all)

for i in range(4):
    s_mean = np.mean(per_subj_all[:, i])
    s_std = np.std(per_subj_all[:, i])

    print(f"subject{i} - mean: {s_mean}, std: {s_std}")

for i in range(7):
    g_mean = np.mean(per_subj_all[i, :])
    g_std = np.std(per_subj_all[i, :])
    print(f"gesture{i} - mean: {g_mean}, std: {g_std}")

print(".")




cm_analysis(y_test_sum, y_pred_sum, labels, ymap=None, figsize=(10,10), max_n=60, acc_array=acc_array,std_array=std_array)