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

# ----------------------------------------------------------------------------------------------------------------------
def main():
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
                label_path = os.path.join(fpath, "label.txt")

                rgb_list = os.listdir(rgb_path)
                depth_list = os.listdir(depth_path)

                # depth
                rgb_file = {}
                for r_path in rgb_list:
                    frame_idx = int(r_path.split('_')[1].split('.')[0])
                    rgb_file[int(frame_idx)] = r_path

                # depth
                depth_file = {}
                for d_path in depth_list:
                    frame_idx = int(d_path.split('_')[1].split('.')[0])
                    depth_file[int(frame_idx)] = d_path

                # label
                label_file = {}
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        key, value = line.strip().split()
                        label_file[int(key)] = value

                dataset.append([rgb_path, depth_path, rgb_file, depth_file, label_file])


    # Instantiate models

    track_hand = HandTracker_our_v2()
    track_gesture = GestureClassfier()

    queue_righthand = deque([], maxlen=10)

    cv2.namedWindow('Prompt')
    cv2.resizeWindow(winname='Prompt', width=500, height=500)
    cv2.moveWindow(winname='Prompt', x=2000, y=200)

    for db in dataset:
        rgb_path, depth_path, rgb_file, depth_file, label_file = db
        db_len = len(rgb_file)

        print(f"start test on {rgb_path}")

        for idx in range(db_len):
            color = cv2.imread(os.path.join(rgb_path, rgb_file[idx]))
            # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            cv2.imshow("color", color)

            if idx in depth_file:
                depth = np.load(os.path.join(depth_path, depth_file[idx]))
                depth_vis = depth / depth.max()
                cv2.imshow("depth", depth_vis)  # 정규화해서 보기 좋게
                                                #### train 에서 input numpy로 변환해서 color space확인.

            ## label recording has changed
            # label = label_file[idx]
            if idx in label_file:
                label = label_file[idx]

            ## preprocess color image
            color = cv2.resize(color, dsize=(640, 360), interpolation=cv2.INTER_AREA)

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

                queue_righthand.append(data)

                if len(queue_righthand) < 10:
                    continue

                gesture_idx, gesture = track_gesture.run(queue_righthand)       # queue (10, 63+15)

            for uvd_hand in all_uvds:
                color = draw_2d_skeleton(color, uvd_hand)

            cv2.imshow("Prompt", color)
            cv2.waitKey(1)



# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    main()
