import os
import sys
import asyncio
import time
from collections import deque
import cv2
import numpy as np
import websockets
import struct
import json
from PIL import Image, ImageDraw, ImageFont

from modules import HandTracker_mp, ObjTracker, GestureClassfier, HandTracker_our, recog_contact
from handtracker.utils.visualize import draw_2d_skeleton
import pickle
from sklearn.metrics import confusion_matrix
import tqdm

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


"""
pen은 object 인식 안됨. object 유무 관계 없이 action 인식 해보고,
안되면 pen용 flag 따로, apple, cup, key용 따로 세팅.

test dataset 수집해서 TP, FP, TN, FN counting해서 시각화?
"""

## args ##
flag_mediapipe = True

flag_gesture = True
flag_saveimg = True
flag_contact = True


TARGET_SUBJECTS = [0, 1, 3, 4]

"""
4 : down에 200frame대 하나 있고, right에 두번 모션한거있음
"""

OBJ_list = ['key_0', 'cyl_0', 'app_0']

# HoloLens address
host = '192.168.1.31'

# Calibration path (must exist but can be empty)
calibration_path = 'calibration'

# Front RGB camera parameters
pv_width = 1280
pv_height = 720
pv_fps = 30

# Buffer length in seconds
buffer_size = 2

# process depth image per n frame
num_depth_count = 1     # 0 for only rgb

contact_que = deque([], maxlen=10)

labels_str = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'CLOCK', 'C-CLOCK', 'TAP', 'NATURAL']
labels = [0,1,2,3,4,5,6,7]

# subject별 clip range 다름. 정확히는 wifi 상태 때문에 seq마다 차이가 나지만.. 연속적으로 촬영한 subject 안에선 일괄 기준 지정.
clip_threshold_per_subject = [[75, 0],
                              [85, 0],
                              [],
                              [85, 30],
                              [75, 0]]

target_class = 3    # -1 for none


def main_single(SUBJECT=0):
    ###################### init models ######################
    if flag_mediapipe:
        track_hand = HandTracker_mp()
        img_save_path = "./pkl_test/vis_mp"
    else:
        track_hand = None #HandTracker_our()
        img_save_path = "./pkl_test/vis_ours"
    #
    # track_obj = ObjTracker()
    track_gesture = GestureClassfier(img_width=640, img_height=360)




    ###################### load test dataset ######################
    for OBJ in OBJ_list:
        pkl_folder = os.path.join("./pkl_test/", f"subject_{SUBJECT}_{OBJ}")
        pkls = os.listdir(pkl_folder)

        per_gesture_result = np.zeros((8, 8))

        stat_dict = dict()
        y_pred = []
        y_gt = []

        for i in range(8):
            stat_dict[i] = dict()

        for pkl_name in tqdm.tqdm(pkls):
            pkl_path = os.path.join(pkl_folder, pkl_name)

            gesture_gt = int(pkl_name.split('_')[3])

            ## check natural class
            # if gesture_gt != target_class:
            #     continue

            with open(pkl_path, 'rb') as f:
                frame_list = pickle.load(f)

            record_gesture_pred = np.zeros(8)
            track_gesture.init_que()
            check_gesture = np.zeros(8)
            clip_len = len(frame_list)

            pred_gesture = None
            # print("clip_len : ", clip_len)

            # my seq error. init frames are added in first seq. fixed in recording.
            if clip_len > 200:
                frame_list = frame_list[90:]
                clip_len = len(frame_list)

            start_idx = clip_threshold_per_subject[SUBJECT][0]
            end_idx = clip_threshold_per_subject[SUBJECT][1]

            flag_visgesture = True
            for idx, frame in enumerate(frame_list):
                if idx < start_idx:
                    continue
                if idx > (clip_len - end_idx):
                    continue

                color, depth, result_hand = frame["color"], frame["depth"], frame["joint"]
                if depth.size > 1:
                    flag_depth = True
                else:
                    flag_depth = False

                pred_contact = True

                if flag_mediapipe:
                    result_hand = track_hand.run(np.copy(color))
                    if not isinstance(result_hand, np.ndarray):
                        continue


                ## with depth info. & tip location
                if flag_depth and flag_contact:
                    tip = result_hand[4, :2] #, result_hand[8, :2]

                    contact = recog_contact(np.copy(depth), tip)
                    contact_que.append(contact)

                    if sum(list(contact_que)) > 2:
                        pred_contact = True
                        # print("contact")
                    else:
                        pred_contact = False
                        # print("no contact")

                ###################### process gesture ######################
                if flag_gesture:
                    gesture_pred_idx, gesture = track_gesture.run(result_hand)
                else:
                    gesture = None

                ###################### visualize ######################
                img_cv = draw_2d_skeleton(color, result_hand)
                if gesture != None and gesture != "Natural" and pred_contact and flag_visgesture:
                    cv2.putText(img_cv, f'{gesture.upper()}',
                                org=(20, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255),
                                thickness=3)

                tip_history = track_gesture.tip_history
                if tip_history.shape[0] > 1:
                    for tip_idx in range(tip_history.shape[0]):
                        tip = tip_history[tip_idx, :2]
                        cv2.circle(img_cv, (int(tip[0]), int(tip[1])), int(tip_idx), (255, 255, 0), -1, cv2.LINE_AA)

                # cv2.imshow("Output", img_cv)
                # cv2.waitKey(0)

                if flag_saveimg:
                    save_path_img = os.path.join(img_save_path, str(SUBJECT), str(OBJ))
                    os.makedirs(save_path_img, exist_ok=True)
                    save_path_img = os.path.join(save_path_img, str(pkl_name) + '_' + str(idx)+'.png')
                    cv2.imwrite(save_path_img, img_cv)

                if gesture != None and gesture != "Natural" and gesture_pred_idx != pred_gesture:       #### CAUTION. contact
                    record_gesture_pred[gesture_pred_idx] += 1
                    y_pred.append(gesture_pred_idx)
                    y_gt.append(gesture_gt)

                    if check_gesture[gesture_gt] == 0:
                        check_gesture[gesture_gt] = 1
                    # break
                    flag_visgesture = False

                pred_gesture = gesture_pred_idx

            # if fail to find gt gesture, set natural class
            if check_gesture[gesture_gt] == 0 and OBJ != 'natural':
                y_pred.append(7)
                y_gt.append(gesture_gt)
            # # 10 sec per 3 object. split 1 sec for gt anno
            # if OBJ == 'natural':
            #     for i in range(30):
            #         y_pred.append(7)
            #         y_gt.append(gesture_gt)

            # print(record_gesture_pred)
            per_gesture_result[gesture_gt, :] += record_gesture_pred

        print("final : ", per_gesture_result)     # if natural, count FP and how to set TP?

        # result_pkl = f"./pkl_test/results/result_{SUBJECT}_{OBJ}.pkl"
        # with open(result_pkl, 'wb') as f:
        #     pickle.dump(per_gesture_result, f)
        #
        # y_for_cm = [y_gt, y_pred]
        # result_pkl = f"./pkl_test/results/result_{SUBJECT}_{OBJ}_cm.pkl"
        # with open(result_pkl, 'wb') as f:
        #     pickle.dump(y_for_cm, f)
        #
        # cm_analysis(y_gt, y_pred, labels, OBJ, ymap=None, figsize=(10, 10))


def cm_analysis(y_true, y_pred, labels, OBJ, ymap=None, figsize=(10,10)):
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
    cm_perc = cm / cm_sum.astype(float) * 100
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
                # convert the proportion, count, and row sum to a string with pretty formatting
                annot[i, j] = '%d\n(%.1f%%)' % (c, p)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%d\n(%.1f%%)' % (c, p)

    # convert the array to a dataframe. To plot by proportion instead of number, use cm_perc in the DataFrame instead of cm
    cm = pd.DataFrame(cm, index=labels_str, columns=labels_str)
    cm.index.name = 'Actural Gesture'
    cm.columns.name = 'Predicted Gesture\n(multiple predictions during action)'
    # create empty figure with a specified size
    fig, ax = plt.subplots(figsize=figsize)
    # plot the data using the Pandas dataframe. To change the color map, add cmap=..., e.g. cmap = 'rocket_r'
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)#, vmin=0.0, vmax=1.0)
    plt.savefig(f"./pkl_test/results/cm_{SUBJECT}_{OBJ}.png")
    print("saved confusion matrix")
    # plt.show()


if __name__ == '__main__':
    for subj in TARGET_SUBJECTS:
        main_single(SUBJECT=subj)