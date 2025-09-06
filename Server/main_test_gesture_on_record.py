import os
import argparse
import numpy as np
import time

import torch
import torch.nn as nn
import cv2
from modules import HandTracker_our_v2, GestureClassfier
from collections import deque
from utils.visualize import draw_2d_skeleton
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib
matplotlib.use('TkAgg')  # 또는 'Qt5Agg', 'QtAgg'


# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='DeepGRU Training')
parser.add_argument('--ckpt', type=str, default="checkpoint.tar")
parser.add_argument('--use-cuda', action='store_true',
                    help='use CUDA if available',
                    default=True)


# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()
use_cuda = torch.cuda.is_available() and args.use_cuda

dataset_path = f"./dataset"

seq_len = 16
threshold_num = 4

flag_vis = False
# record 2.5 sec ready, 1.5 sec record
# ----------------------------------------------------------------------------------------------------------------------
def main():
    results = []
    result_path = "result.pkl"

    if not os.path.exists(result_path):
        # Load the dataset
        dataset = []
        for sname in sorted(os.listdir(dataset_path)):
            spath = os.path.join(dataset_path, sname)  # test/subject_0

            for tname in sorted(os.listdir(spath)):
                tpath = os.path.join(spath, tname)  # test/subject_0/trial_0

                for fname in sorted(os.listdir(tpath)):
                    fpath = os.path.join(tpath, fname)  # test/subject_0/trial_0/thumb      # or index

                    rgb_path = os.path.join(fpath, "rgb")
                    depth_path = os.path.join(fpath, "depth")
                    # label_path = os.path.join(fpath, "label.txt")

                    datset_small = []
                    for seq in sorted(os.listdir(rgb_path)):  # seq : 1_CClock
                        rgb_file_path = os.path.join(rgb_path, seq)
                        rgb_list = os.listdir(rgb_file_path)
                        depth_file_path = os.path.join(depth_path, seq)
                        depth_list = os.listdir(depth_file_path)

                        # rgb
                        rgb_files = {}
                        for r_path in rgb_list:
                            frame_idx = int(r_path.split('_')[1].split('.')[0])
                            rgb_files[int(frame_idx)] = os.path.join(rgb_file_path, r_path)

                        # depth
                        depth_files = {}
                        for d_path in depth_list:
                            frame_idx = int(d_path.split('_')[1].split('.')[0])
                            depth_files[int(frame_idx)] = os.path.join(depth_file_path, d_path)

                        datset_small.append([seq.split('_')[1], rgb_files, depth_files])

                    # # label
                    # label_file = {}
                    # with open(label_path, 'r', encoding='utf-8') as f:
                    #     for line in f:
                    #         key, value = line.strip().split()
                    #         label_file[int(key)] = value

                    dataset.append([sname, tname, fname, datset_small])

        # Instantiate models
        track_hand = HandTracker_our_v2()
        track_gesture = GestureClassfier(seq_len=seq_len)

        if flag_vis:
            cv2.namedWindow('Prompt')
            cv2.resizeWindow(winname='Prompt', width=500, height=500)
            cv2.moveWindow(winname='Prompt', x=2000, y=200)
        for db in dataset:
            sname, tname, fname, datset_small = db

            for label, rgb_files, depth_files in tqdm(datset_small, desc=f"testing on {sname}, {tname}, {fname}"):
                # print(f"current gt : {label}")

                len_clip = len(rgb_files)
                start_idx = int(len_clip * 0.6) - seq_len

                queue_righthand = deque([], maxlen=seq_len)

                valid_pred_list = []
                valid_gt = f"{label}_{fname}"

                prev_gesture = None
                gesture_cnt = 0

                for idx, frame_idx in enumerate(rgb_files):
                    if idx < start_idx:
                        continue

                    pose_path_split = rgb_files[frame_idx].split('\\')
                    pose_path = os.path.join(*pose_path_split[:-3], 'pose', pose_path_split[-2])
                    pose_file_path = os.path.join(pose_path, pose_path_split[-1].split('.')[0] + '.npy')

                    if os.path.exists(pose_file_path):
                        data = np.load(pose_file_path)
                        queue_righthand.append(data)
                    else:
                        color = cv2.imread(rgb_files[frame_idx])
                        # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        # cv2.imshow("color", color)

                        # if frame_idx in depth_files:
                        #     depth = np.load(depth_files[frame_idx])
                        # depth_vis = depth / depth.max()
                        # cv2.imshow("depth", depth_vis)  # 정규화해서 보기 좋게

                        ## preprocess color image
                        # color = cv2.resize(color, dsize=(640, 360), interpolation=cv2.INTER_AREA)

                        ## run hand tracker
                        outs = track_hand.run(np.copy(color))
                        if not outs:
                            continue

                        all_right, all_uvds, all_verts, all_cam_t = outs

                        ## process only right hand visible
                        indices = np.where(np.asarray(all_right) == 1)[0]  ### check. 0: left, 1: right
                        if len(indices) > 0:
                            uvd_right = np.squeeze(np.asarray(all_uvds)[indices[0]])

                            # preprocess joint pose
                            angle_label = track_gesture._compute_ang_from_joint(uvd_right)
                            data = np.concatenate([uvd_right.flatten(), angle_label])

                            os.makedirs(pose_path, exist_ok=True)
                            np.save(pose_file_path, data)

                            # save pose data
                            queue_righthand.append(data)


                    if len(queue_righthand) < seq_len:
                        continue

                    gesture_idx, gesture = track_gesture.run(queue_righthand)       # queue (seq_len, 63+15)
                    # print(f"pred : {gesture}")

                    ## valid gesture if same gesture continously detected
                    if prev_gesture == gesture and gesture != 'Natural':
                        gesture_cnt += 1
                    else:
                        gesture_cnt = 0
                    prev_gesture = gesture

                    if gesture_cnt > threshold_num:
                        valid_pred_list.append(gesture)


                    #### visualize
                    if flag_vis:
                        for uvd_hand in all_uvds:
                            color = draw_2d_skeleton(color, uvd_hand)
                        cv2.imshow("Prompt", color)
                        cv2.waitKey(1)

                results.append([sname, valid_gt, valid_pred_list])

        with open(result_path, 'wb') as f:  # wb = write binary
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(result_path, 'rb') as f:  # rb = read binary
            results = pickle.load(f)

    ## check results
    # TP : prediction == gt
    # FP : prediction != gt
    # FN : model didn't predict anything (only natural class detected)
    # TN : Not exist (GT-sequence data doesn't contain gesture)
    TP, FP, FN, TN = 0, 0, 0, 0

    perSubj = {}
    for sname in os.listdir(dataset_path):
        perSubj[sname] = [0, 0, 0, 0, [], [], [], []]   # TP, FP, FN, TN, Thumb_gt, Thumb_pred, Intex_gt, Index_pred

    perGesture = {}
    gesture_list = ['Up_thumb', 'Down_thumb', 'Left_thumb', 'Right_thumb', 'Tap_thumb', 'Clock_thumb', 'CClock_thumb',
                    'Up_index', 'Down_index', 'Left_index', 'Right_index', 'Tap_index', 'Clock_index', 'CClock_index']
    for gesture in gesture_list:
        perGesture[gesture] = [0, 0, 0, 0]   # TP, FP, FN, TN

    cnt_diff_finger = 0

    for sname, valid_gt, valid_pred_list in results:
        valid_label = valid_gt.split('_')[0]
        valid_finger = valid_gt.split('_')[1]

        if len(valid_pred_list) == 0:
            FN += 1
            perSubj[sname][2] += 1
            perGesture[valid_gt][2] += 1
            if valid_finger == 'thumb':
                perSubj[sname][4].append(valid_label)
                perSubj[sname][5].append('Natural')
            else:
                perSubj[sname][6].append(valid_label)
                perSubj[sname][7].append('Natural')
        else:
            for pred in valid_pred_list:
                pred_label = pred.split('_')[0]
                pred_finger = pred.split('_')[1]
                if valid_finger != pred_finger:
                    cnt_diff_finger += 1
                elif valid_finger == 'thumb':
                    perSubj[sname][4].append(valid_label)
                    perSubj[sname][5].append(pred_label)
                else:
                    perSubj[sname][6].append(valid_label)
                    perSubj[sname][7].append(pred_label)

                if pred == valid_gt:
                    TP += 1
                    perSubj[sname][0] += 1
                    perGesture[valid_gt][0] += 1
                elif pred != valid_gt:
                    FP += 1
                    perSubj[sname][1] += 1
                    perGesture[valid_gt][1] += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Overall F1 Score: {f1:.3f}")


    for sname in perSubj:
        TP, FP, FN, TN, _, _, _, _ = perSubj[sname]
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        print(f"{sname} F1 Score: {f1:.3f}")

    for gesture in perGesture:
        TP, FP, FN, TN = perGesture[gesture]
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        print(f"{gesture} F1 Score: {f1:.3f}")

    ## Confusion Matrix
    labels = ['Up', 'Down', 'Left', 'Right', 'Tap', 'Clock', 'CClock']

    for finger_idx in range(2):
        print(f"Make confusion matrix for finger {finger_idx}")

        y_true_subj = []
        y_pred_subj = []
        for sname in perSubj:
            y_true_subj.append(perSubj[sname][4 + finger_idx * 2])
            y_pred_subj.append(perSubj[sname][5 + finger_idx * 2])

        cms = []
        for y_true, y_pred in zip(y_true_subj, y_pred_subj):
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            cms.append(cm)

        cms = np.array(cms)  # shape: (subj 수, 클래스수, 클래스수)

        cm_mean = np.mean(cms, axis=0)
        cm_std = np.std(cms, axis=0)
        cm_percent = cm_mean / cm_mean.sum(axis=1, keepdims=True) * 100

        annot = np.empty_like(cm_mean, dtype=object)
        for i in range(cm_mean.shape[0]):
            for j in range(cm_mean.shape[1]):
                if float(cm_mean[i, j]) < 0.1:
                    annot[i, j] = ""  # 0이면 비움
                else:
                    annot[i, j] = f"{cm_mean[i, j]:.1f}\nsd: {cm_std[i, j]:.1f}\n{cm_percent[i, j]:.1f}%"

        df_cm = pd.DataFrame(cm_mean, index=labels, columns=labels)
        plt.figure(figsize=(10, 10))
        sns.heatmap(df_cm, annot=annot, fmt='', cmap='magma', vmin=0, vmax=50,
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix of all subjects (Mean ± SD)')
        plt.show()

    print(f"cnt_diff_finger = {cnt_diff_finger}")

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
