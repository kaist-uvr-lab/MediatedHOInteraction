import os
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import cv2
from modules import GestureClassfier
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

# Use 'Agg' backend for non-interactive environments
matplotlib.use('Agg')

# --- Configuration Constants ---
# Argument Parser Setup
parser = argparse.ArgumentParser(description='Gesture Recognition Test Script')
parser.add_argument('--ckpt', type=str, default="checkpoint.tar-paper")  # Checkpoint filename
parser.add_argument('--model', type=int, default=-1)  # Model option (0: base, 1:ours, 2~6:ablation)
parser.add_argument('--use-cuda', action='store_true',
                    help='Use CUDA if available',
                    default=True)

# --- Global Settings ---
SEQ_LEN = 16  # Sequence length for the gesture classifier
THRESHOLD_NUM = 3  # Number of consecutive identical predictions to be considered 'valid'
DATASET_PATH = "./dataset"
F1_EPSILON = 1e-10  # Epsilon for F1 calculation to avoid division by zero
GESTURE_LIST = ['Up_thumb', 'Down_thumb', 'Left_thumb', 'Right_thumb', 'Tap_thumb', 'Clock_thumb', 'CClock_thumb',
                'Up_index', 'Down_index', 'Left_index', 'Right_index', 'Tap_index', 'Clock_index', 'CClock_index']
LABELS_CM = ['Up', 'Down', 'Left', 'Right', 'Tap', 'Clock', 'CClock', 'Natural']
LABELS_CM_ = ['Up', 'Down', 'Left', 'Right', 'Tap', 'Clock', 'CClock']
FINGERS = ['Thumb', 'Index']


# ----------------------------------------------------------------------------------------------------------------------
def setup_models(args):
    """Initializes hand tracker and gesture classifier."""
    ckpt_path = os.path.join("./gestureclassifier/checkpoints", args.ckpt)

    # Assuming the HandTracker_our_v2 wrapper was removed and replaced by the original HandTracker_wilor.
    try:
        from handtracker_wilor.module_WILOR import HandTracker_wilor
        track_hand = HandTracker_wilor()
    except ImportError:
        # Fallback or Error handling if the path is incorrect. User should ensure correct import.
        # This will raise an error if HandTracker_wilor is not found.
        raise ImportError("HandTracker_wilor (v2 tracker) not found. Please check your project structure.")

    track_gesture = GestureClassfier(ckpt=ckpt_path, seq_len=SEQ_LEN, model_opt=args.model)
    return track_hand, track_gesture


def load_dataset_metadata(dataset_path):
    """Loads file paths and metadata for the dataset."""
    dataset = []
    for sname in sorted(os.listdir(dataset_path)):
        spath = os.path.join(dataset_path, sname)

        for tname in sorted(os.listdir(spath)):
            tpath = os.path.join(spath, tname)

            for fname in sorted(os.listdir(tpath)):
                fpath = os.path.join(tpath, fname)

                rgb_path = os.path.join(fpath, "rgb")

                datset_small = []
                # Use os.scandir for potentially faster directory iteration if needed, but os.listdir is fine.
                for seq in sorted(os.listdir(rgb_path)):
                    # seq: e.g., 1_CClock
                    rgb_file_path = os.path.join(rgb_path, seq)
                    rgb_list = os.listdir(rgb_file_path)

                    rgb_files = {}
                    for r_path in rgb_list:
                        try:
                            # Assumes filename format is 'rgb_FRAMEIDX.png'
                            frame_idx = int(r_path.split('_')[1].split('.')[0])
                            rgb_files[frame_idx] = os.path.join(rgb_file_path, r_path)
                        except:
                            # Skip files that don't match the expected naming convention
                            continue

                    if rgb_files:
                        datset_small.append([seq.split('_')[1], rgb_files])

                if datset_small:
                    dataset.append([sname, tname, fname, datset_small])
    return dataset


def run_evaluation(dataset, track_hand, track_gesture, flag_vis):
    """Runs the hand tracking and gesture classification on the dataset."""
    results = []

    if flag_vis:
        cv2.namedWindow('Prompt')
        cv2.resizeWindow(winname='Prompt', width=500, height=500)
        cv2.moveWindow(winname='Prompt', x=2000, y=200)

    for db in dataset:
        sname, tname, fname, datset_small = db

        for label, rgb_files in tqdm(datset_small, desc=f"Testing on {sname}, {tname}, {fname}"):
            len_clip = len(rgb_files)
            # Calculate start index to only evaluate the gesture execution phase
            start_idx = int(len_clip * 0.6) - SEQ_LEN

            queue_righthand = deque([], maxlen=SEQ_LEN)
            valid_pred_list = []
            valid_gt = f"{label}_{fname}"

            prev_gesture = None
            gesture_cnt = 0

            # Sort by frame index to process chronologically
            sorted_frame_indices = sorted(rgb_files.keys())

            for idx, frame_idx in enumerate(sorted_frame_indices):
                if idx < start_idx:
                    continue

                # Construct pose path based on RGB path
                rgb_path_parts = rgb_files[frame_idx].split(os.sep)

                # Structure: .../dataset/sname/tname/fname/pose/1_CClock/rgb_1.npy
                pose_path = os.path.join(os.sep.join(rgb_path_parts[:-2]), 'pose', rgb_path_parts[-2])
                pose_file_name = rgb_path_parts[-1].split('.')[0] + '.npy'
                pose_file_path = os.path.join(pose_path, pose_file_name)

                data = None
                if os.path.exists(pose_file_path):
                    # Load pre-saved pose data
                    data = np.load(pose_file_path)
                    queue_righthand.append(data)
                else:
                    # Run hand tracking if pose data is not found
                    color = cv2.imread(rgb_files[frame_idx])
                    if color is None:
                        continue

                    outs = track_hand.run(np.copy(color))
                    if not outs:
                        continue

                    # outs structure: all_right, all_uvds, all_verts, all_cam_t
                    all_right, all_uvds, _, _ = outs

                    # Process only right hand
                    indices = np.where(np.asarray(all_right) == 1)[0]
                    if len(indices) > 0:
                        uvd_right = np.squeeze(np.asarray(all_uvds)[indices[0]])

                        # Preprocess joint pose
                        angle_label = track_gesture._compute_ang_from_joint(uvd_right)
                        data = np.concatenate([uvd_right.flatten(), angle_label])

                        # Save pose data
                        os.makedirs(pose_path, exist_ok=True)
                        np.save(pose_file_path, data)
                        queue_righthand.append(data)

                if data is None:
                    continue

                if len(queue_righthand) < SEQ_LEN:
                    continue

                gesture_idx, gesture = track_gesture.run(queue_righthand)

                # Valid gesture if same gesture is continuously detected
                if prev_gesture == gesture and gesture != 'Natural':
                    gesture_cnt += 1
                else:
                    gesture_cnt = 0
                prev_gesture = gesture

                if gesture_cnt > THRESHOLD_NUM:
                    valid_pred_list.append(gesture)

                # Visualize
                if flag_vis:
                    color = cv2.imread(rgb_files[frame_idx])
                    if color is not None:
                        uvd_hand = data[:-15].reshape(21, 3)
                        color = draw_2d_skeleton(color, uvd_hand)
                        cv2.imshow("Prompt", color)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

                            # Post-processing: collect unique predictions from the valid list
            valid_pred_list = list(set(valid_pred_list))
            results.append([sname, valid_gt, valid_pred_list])

    if flag_vis:
        cv2.destroyAllWindows()

    return results


def calculate_metrics(results):
    """Calculates overall metrics (Micro-F1) and per-subject/per-gesture metrics."""
    TP, FP, FN = 0, 0, 0
    per_subj_gesture = {sname: {gesture: [0, 0, 0, 0] for gesture in GESTURE_LIST}
                        for sname in os.listdir(DATASET_PATH)}
    # per_subj_cm structure: [Thumb_gt, Thumb_pred, Index_gt, Index_pred]
    per_subj_cm = {sname: [[], [], [], []] for sname in os.listdir(DATASET_PATH)}

    for sname, valid_gt, valid_pred_list in results:
        valid_label = valid_gt.split('_')[0]
        valid_finger = valid_gt.split('_')[1]

        if len(valid_pred_list) == 0:
            FN += 1
            per_subj_gesture[sname][valid_gt][2] += 1  # FN

            # For Confusion Matrix: Predicted as 'Natural'
            if valid_finger == 'thumb':
                per_subj_cm[sname][0].append(valid_label)
                per_subj_cm[sname][1].append('Natural')
            else:
                per_subj_cm[sname][2].append(valid_label)
                per_subj_cm[sname][3].append('Natural')
        else:
            for pred in valid_pred_list:
                pred_label = pred.split('_')[0]
                pred_finger = pred.split('_')[1]

                # For Confusion Matrix
                if valid_finger == 'thumb':
                    per_subj_cm[sname][0].append(valid_label)
                    per_subj_cm[sname][1].append(pred_label)
                else:
                    per_subj_cm[sname][2].append(valid_label)
                    per_subj_cm[sname][3].append(pred_label)

                if pred == valid_gt:
                    TP += 1
                    per_subj_gesture[sname][valid_gt][0] += 1  # TP
                else:
                    FP += 1
                    per_subj_gesture[sname][valid_gt][1] += 1  # FP

    # Calculate Overall Micro-F1
    precision_micro = TP / (TP + FP + F1_EPSILON)
    recall_micro = TP / (TP + FN + F1_EPSILON)
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + F1_EPSILON)

    return f1_micro, per_subj_gesture, per_subj_cm


def calculate_f1_summary(per_subj_gesture):
    """Calculates Micro and Macro F1 scores per subject and per gesture."""

    # Calculate sum of TP/FP/FN per subject (Micro-F1 per subject)
    sum_per_sname = {}
    for sname, gestures in per_subj_gesture.items():
        total = [0, 0, 0, 0]  # TP, FP, FN, TN
        for arr in gestures.values():
            total = [a + b for a, b in zip(total, arr)]
        sum_per_sname[sname] = total

    # Calculate sum of TP/FP/FN per gesture (Micro-F1 per gesture)
    sum_per_gesture = {gesture: [0, 0, 0, 0] for gesture in GESTURE_LIST}
    for gestures in per_subj_gesture.values():
        for gesture, arr in gestures.items():
            sum_per_gesture[gesture] = [a + b for a, b in zip(sum_per_gesture[gesture], arr)]

    eps = F1_EPSILON

    # Micro-F1 per Subject
    print("\n--- Micro-F1 per Subject (Combined Gestures) ---")
    for sname in sum_per_sname:
        TP, FP, FN, TN = sum_per_sname[sname]
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        print(f"{sname} F1 Score: {f1:.4f}")

    # Micro-F1 per Gesture
    print("\n--- Micro-F1 per Gesture (Combined Subjects) ---")
    for gesture in sum_per_gesture:
        TP, FP, FN, TN = sum_per_gesture[gesture]
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        print(f"{gesture} F1 Score: {f1:.4f}")

    # Calculate Macro-F1 (Average F1 per instance)
    f1_matrix = {}
    f1_per_sname = {sname: [] for sname in per_subj_gesture}
    f1_per_gesture = {gesture: [] for gesture in GESTURE_LIST}

    for sname, gestures in per_subj_gesture.items():
        f1_matrix[sname] = {}
        for gesture, arr in gestures.items():
            TP, FP, FN, TN = arr
            precision = TP / (TP + FP + eps)
            recall = TP / (TP + FN + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)
            f1_matrix[sname][gesture] = round(f1, 4)
            f1_per_sname[sname].append(f1)
            f1_per_gesture[gesture].append(f1)

    # Macro-F1 (Mean and Std Dev) per Subject
    print("\n--- Macro-F1 per Subject (Mean F1 across gestures) ---")
    for sname, f1_list in f1_per_sname.items():
        mean_f1 = statistics.mean(f1_list)
        std_f1 = statistics.stdev(f1_list) if len(f1_list) > 1 else 0
        print(f"{sname} Macro F1: {mean_f1:.3f}, Std Dev: {std_f1:.3f}")

    # Macro-F1 (Mean and Std Dev) per Gesture
    print("\n--- Macro-F1 per Gesture (Mean F1 across subjects) ---")
    for gesture, f1_list in f1_per_gesture.items():
        mean_f1 = statistics.mean(f1_list)
        std_f1 = statistics.stdev(f1_list) if len(f1_list) > 1 else 0
        print(f"{gesture} Macro F1: {mean_f1:.3f}, Std Dev: {std_f1:.3f}")

    # Print F1 Matrix
    df_f1 = pd.DataFrame.from_dict(f1_matrix, orient='index')
    print("\n--- F1 Score Matrix (Rows: Subject, Columns: Gesture) ---")
    print(df_f1)
    print("\n--- LaTeX Table ---")
    print(df_f1.to_latex(float_format="%.4f"))

    return per_subj_gesture


def plot_confusion_matrices(per_subj_cm):
    """Generates and displays Confusion Matrices for Thumb and Index fingers."""

    for finger_idx in range(2):
        print(f"\n--- Making Confusion Matrix for {FINGERS[finger_idx]} ---")

        # Get true and predicted labels for the current finger
        y_true_subj = [data[finger_idx * 2] for data in per_subj_cm.values()]
        y_pred_subj = [data[1 + finger_idx * 2] for data in per_subj_cm.values()]

        cms = []
        for y_true, y_pred in zip(y_true_subj, y_pred_subj):
            # Create crosstab: index=Actual, columns=Predicted
            cm = pd.crosstab(index=pd.Series(y_true, name="Actual"),
                             columns=pd.Series(y_pred, name="Predicted"))

            # Reindex to ensure consistent order and include 'Natural'
            cm = cm.reindex(index=LABELS_CM_, columns=LABELS_CM, fill_value=0)
            cms.append(cm.values)

        cms = np.array(cms)

        cm_mean = np.mean(cms, axis=0)
        cm_std = np.std(cms, axis=0)
        cm_percent = cm_mean / cm_mean.sum(axis=1, keepdims=True) * 100

        # Create annotation text
        annot = np.empty_like(cm_mean, dtype=object)
        for i in range(cm_mean.shape[0]):
            for j in range(cm_mean.shape[1]):
                if float(cm_mean[i, j]) < 0.1:
                    annot[i, j] = ""
                else:
                    annot[i, j] = f"{cm_mean[i, j]:.1f}\nsd: {cm_std[i, j]:.1f}\n{cm_percent[i, j]:.1f}%"

        # DataFrame for Heatmap
        df_cm = pd.DataFrame(cm_mean, index=LABELS_CM_, columns=LABELS_CM)

        sns.set(font_scale=0.9)
        plt.figure(figsize=(10, 10))
        sns.heatmap(df_cm, annot=annot, fmt='', cmap='magma', vmin=0, vmax=10,
                    xticklabels=LABELS_CM, yticklabels=LABELS_CM_)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('Actual', fontsize=14)
        plt.title(f"Confusion Matrix (mean ± sd) - {FINGERS[finger_idx]}", fontsize=16)

        # Save figure instead of showing it
        save_path = f"confusion_matrix_{FINGERS[finger_idx]}_{args.ckpt}.png"
        plt.savefig(save_path)
        print(f"Saved confusion matrix to {save_path}")
        plt.close()  # Close plot figure


# ----------------------------------------------------------------------------------------------------------------------
def main():
    args = parser.parse_args()

    # Use 'Agg' backend for non-interactive environments
    matplotlib.use('Agg')

    result_path = os.path.join("log", f"result-{args.ckpt}.pkl")

    if not os.path.exists(result_path):

        # 1. Setup Models
        track_hand, track_gesture = setup_models(args)

        # 2. Load Dataset Metadata
        dataset = load_dataset_metadata(DATASET_PATH)

        # 3. Run Evaluation and Save Results
        # flag_vis is hardcoded to False for automation
        results = run_evaluation(dataset, track_hand, track_gesture, False)

        # Save results to disk
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Load pre-calculated results
        print(f"Loading results from {result_path}")
        with open(result_path, 'rb') as f:
            results = pickle.load(f)

    # 4. Calculate Metrics
    f1_micro_overall, per_subj_gesture, per_subj_cm = calculate_metrics(results)

    # 5. Print Summary Metrics
    print(f"\n--- Overall Micro-F1 Score ---")
    print(f"Overall F1 Score: {f1_micro_overall:.3f}")

    # Calculate and Print F1 Summary (Micro per group, Macro)
    per_subj_gesture = calculate_f1_summary(per_subj_gesture)

    # 6. Generate Confusion Matrices (saved as files)
    plot_confusion_matrices(per_subj_cm)


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()