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
import statistics
matplotlib.use('TkAgg')  # 또는 'Qt5Agg', 'QtAgg'


# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='DeepGRU Training')
parser.add_argument('--ckpt', type=str, default="checkpoint-40.tar")
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

    # for confusion matrix
    perSubj = {}
    for sname in os.listdir(dataset_path):
        perSubj[sname] = [[], [], [], []]   # TP, FP, FN, TN, Thumb_gt, Thumb_pred, Intex_gt, Index_pred


    gesture_list = ['Up_thumb', 'Down_thumb', 'Left_thumb', 'Right_thumb', 'Tap_thumb', 'Clock_thumb', 'CClock_thumb',
                    'Up_index', 'Down_index', 'Left_index', 'Right_index', 'Tap_index', 'Clock_index', 'CClock_index']

    per_Subj_Gesture = {}
    for sname in os.listdir(dataset_path):
        per_Subj_Gesture[sname] = {}
        for gesture in gesture_list:
            per_Subj_Gesture[sname][gesture] = [0, 0, 0, 0]   # TP, FP, FN, TN

    cnt_diff_finger = 0

    for sname, valid_gt, valid_pred_list in results:
        valid_label = valid_gt.split('_')[0]
        valid_finger = valid_gt.split('_')[1]

        if len(valid_pred_list) == 0:
            FN += 1
            per_Subj_Gesture[sname][valid_gt][2] += 1

            if valid_finger == 'thumb':
                perSubj[sname][0].append(valid_label)
                perSubj[sname][1].append('Natural')
            else:
                perSubj[sname][2].append(valid_label)
                perSubj[sname][3].append('Natural')
        else:
            for pred in valid_pred_list:
                pred_label = pred.split('_')[0]
                pred_finger = pred.split('_')[1]
                if valid_finger != pred_finger:
                    cnt_diff_finger += 1
                elif valid_finger == 'thumb':
                    perSubj[sname][0].append(valid_label)
                    perSubj[sname][1].append(pred_label)
                else:
                    perSubj[sname][2].append(valid_label)
                    perSubj[sname][3].append(pred_label)

                if pred == valid_gt:
                    TP += 1
                    per_Subj_Gesture[sname][valid_gt][0] += 1
                elif pred != valid_gt:
                    FP += 1
                    per_Subj_Gesture[sname][valid_gt][1] += 1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Overall F1 Score: {f1:.3f}")

    sum_per_sname = {}
    for sname, gestures in per_Subj_Gesture.items():
        total = [0, 0, 0, 0]
        for arr in gestures.values():
            total = [a + b for a, b in zip(total, arr)]
        sum_per_sname[sname] = total

    sum_per_gesture = {gesture: [0, 0, 0, 0] for gesture in gesture_list}
    for gestures in per_Subj_Gesture.values():
        for gesture, arr in gestures.items():
            sum_per_gesture[gesture] = [a + b for a, b in zip(sum_per_gesture[gesture], arr)]


    for sname in sum_per_sname:
        TP, FP, FN, TN = sum_per_sname[sname]

        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        print(f"{sname} F1 Score: {f1:.4f}")        #- 마이크로 평균(micro-F1): TP/FP/FN을 먼저 합계한 뒤 F1을 계산.


    for gesture in sum_per_gesture:
        TP, FP, FN, TN = sum_per_gesture[gesture]
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        print(f"{gesture} F1 Score: {f1:.4f}")


    # sname별 F1 리스트 저장
    f1_per_sname = {sname: [] for sname in per_Subj_Gesture}
    for sname, gestures in per_Subj_Gesture.items():
        for arr in gestures.values():
            TP, FP, FN, TN = arr
            precision = TP / (TP + FP) if (TP + FP) else 0
            recall = TP / (TP + FN) if (TP + FN) else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
            f1_per_sname[sname].append(f1)

    # gesture별 F1 리스트 저장
    f1_per_gesture = {gesture: [] for gesture in gesture_list}
    for gestures in per_Subj_Gesture.values():
        for gesture, arr in gestures.items():
            TP, FP, FN, TN = arr
            precision = TP / (TP + FP) if (TP + FP) else 0
            recall = TP / (TP + FN) if (TP + FN) else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
            f1_per_gesture[gesture].append(f1)

    # sname별 평균, 표준편차 출력
    for sname, f1_list in f1_per_sname.items():
        mean_f1 = statistics.mean(f1_list)
        std_f1 = statistics.stdev(f1_list) if len(f1_list) > 1 else 0
        print(f"{sname} 표준편차: {std_f1:.3f}")      # 매크로 평균(macro-F1): 각 항목에서 F1을 계산한 후 평균.


    # gesture별 평균, 표준편차 출력
    for gesture, f1_list in f1_per_gesture.items():
        mean_f1 = statistics.mean(f1_list)
        std_f1 = statistics.stdev(f1_list) if len(f1_list) > 1 else 0
        print(f"{gesture} 표준편차: {std_f1:.3f}")



    f1_matrix = {}
    for sname, gestures in per_Subj_Gesture.items():
        f1_matrix[sname] = {}
        for gesture, arr in gestures.items():
            TP, FP, FN, TN = arr
            precision = TP / (TP + FP) if (TP + FP) else 0
            recall = TP / (TP + FN) if (TP + FN) else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
            f1_matrix[sname][gesture] = round(f1, 4)  # 소수점 4자리

    # DataFrame으로 변환 (행: sname, 열: gesture)
    df_f1 = pd.DataFrame.from_dict(f1_matrix, orient='index')

    print(df_f1)  # 콘솔 출력
    # LaTeX 표로 변환하려면:
    print(df_f1.to_latex(float_format="%.4f"))





    ## Confusion Matrix
    labels = ['Up', 'Down', 'Left', 'Right', 'Tap', 'Clock', 'CClock']
    fingers = ['Thumb', 'Index']

    for finger_idx in range(2):
        print(f"Make confusion matrix for finger {finger_idx}")

        y_true_subj = []
        y_pred_subj = []
        for sname in perSubj:
            y_true_subj.append(perSubj[sname][finger_idx * 2])
            y_pred_subj.append(perSubj[sname][1 + finger_idx * 2])

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

        sns.set(font_scale=0.9)

        df_cm = pd.DataFrame(cm_mean, index=labels, columns=labels)
        plt.figure(figsize=(10, 10))
        sns.heatmap(df_cm, annot=annot, fmt='', cmap='magma', vmin=0, vmax=50,
                    xticklabels=labels, yticklabels=labels)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('Actual', fontsize=14)
        plt.title(f"Confusion Matrix across subjects - {fingers[finger_idx]}", fontsize=14)
        plt.show()

    print(f"cnt_diff_finger = {cnt_diff_finger}")

# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
