import os
import sys
import cv2
import time
import numpy as np
# from ultralytics import YOLO

import torch
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from enum import Enum, IntEnum
import copy

# from tensorflow.keras.models import load_model
from collections import deque


# from handtracker.module_SARTE import HandTracker
from handtracker_wilor.module_WILOR import HandTracker_wilor
from gestureclassifier.model_update import create_model


finger_joints = {
        'thumb': [1, 2, 3, 4],
        'index': [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring': [13, 14, 15, 16],
        'pinky': [17, 18, 19, 20]
}
tip_joints = [4, 8, 12, 16, 20]
baseline_variance = None


class GestureClassfier():
    def __init__(self, ckpt="./gestureclassifier/checkpoints/checkpoint.tar"):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self.model_gesture = create_model(num_features=60, num_classes=15)

        checkpoint = torch.load(ckpt)
        state_dict = checkpoint['model_state_dict']

        # "module." prefix가 있는지 확인
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
        # prefix 제거
        if has_module_prefix:
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        else:
            new_state_dict = state_dict

        self.model_gesture.load_state_dict(new_state_dict)
        self.model_gesture = torch.nn.DataParallel(self.model_gesture).cuda()
        self.model_gesture.eval()


        # default args
        self.seq_len = 10
        self.idx_to_class = {0: 'CClock_index', 1: 'CClock_thumb',
                             2: 'Clock_index', 3: 'Clock_thumb',
                             4: 'Down_index', 5: 'Down_thumb',
                             6: 'Left_index', 7: 'Left_thumb',
                             8: 'Natural',
                             9: 'Right_index', 10: 'Right_thumb',
                             11: 'Tap_index', 12: 'Tap_thumb',
                             13: 'Up_index', 14: 'Up_thumb'}
        self.partial_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 20]



    def run(self, input):
        # input : queue (10, 63+15)  -> ndarray (10, 78)
        input = np.array(input).reshape(10, -1)

        input = self._normalize(input)
        input = self._extract_partialhand(input)

        input = torch.from_numpy(input).to(self.device).unsqueeze(0).float()

        with torch.no_grad():
            output = self.model_gesture(input)

        pred = output.argmax(1).cpu().numpy()
        gesture = self.idx_to_class[pred[0]]

        return pred[0], gesture

    def _normalize(self, pts, norm_ratio_x=180.0, norm_ratio_y=180.0, norm_ratio_z=100.0):
        """
        Normalize a single sample

        :param sample: the sample to normalize
        :return: the normalized sample
        """

        pts = np.asarray(pts)

        pts_norm = np.zeros((pts.shape[0], pts.shape[1]))

        for frame_idx in range(pts.shape[0]):
            target_pose = pts[frame_idx, :63].reshape(21, 3)
            target_angle = pts[frame_idx, 63:]

            # norm 2d pose
            if frame_idx == 0:
                root_pose = target_pose[0, :]
            norm_pose = target_pose - root_pose

            norm_pose[:, 0] = norm_pose[:, 0] / norm_ratio_x
            norm_pose[:, 1] = norm_pose[:, 1] / norm_ratio_y
            norm_pose[:, 2] = norm_pose[:, 2] / norm_ratio_z

            # update pose and angle
            pts_norm[frame_idx, :63] = norm_pose.flatten()
            pts_norm[frame_idx, 63:] = target_angle / 180.0

        return pts_norm

    def _extract_partialhand(self, pts_norm):
        # set partial pts
        # 0~4   5~8   9 12   13 16   17 20
        # pts_norm : (seq_len, 63+15) -> (seq_len, 45+15)
        pts_norm = np.asarray(pts_norm)
        pts_norm_part = []
        for frame_idx in range(pts_norm.shape[0]):
            target_pose = pts_norm[frame_idx, :63].reshape(21, 3)
            target_angle = pts_norm[frame_idx, 63:]

            target_pose = target_pose[self.partial_idx, :]
            target_pose = target_pose.flatten()

            pts_ = np.concatenate((target_pose, target_angle), axis=0)
            pts_norm_part.append(pts_)

        return np.array(pts_norm_part)

    def _compute_ang_from_joint(self, joint):  # joint : (21, 3)
        # Compute angles between joints
        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
        v = v2 - v1  # [20, 3]
        # Normalize v
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        angle = np.arccos(np.einsum('nt,nt->n',
                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

        angle = np.degrees(angle)  # Convert radian to degree

        return angle


class HandTracker_our_v2():
    def __init__(self):
        self.model_hand = HandTracker_wilor()

    def run(self, input):
        return self.model_hand.run(input)



# class HandTracker_our():
#     def __init__(self):
#         self.track_hand = HandTracker()
#
#     def run(self, input):
#         result_hand = self.track_hand.Process_single_newroi(input)
#
#         return result_hand


class HandTracker_mp():
    def __init__(self, ckpt=None):

        # self.mp_drawing = mp.solutions.drawing_utils
        # self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        print("init hand tracker")
        torch.backends.cudnn.benchmark = True
        self.mediahand = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

    def run(self, input):
        img_height = input.shape[0]
        img_width = input.shape[1]

        input = cv2.flip(input, 1)
        results = self.mediahand.process(cv2.cvtColor(input, cv2.COLOR_BGR2RGB))

        result_hand = []
        if results.multi_hand_landmarks == None:
            return None

        for hand_landmarks in results.multi_hand_landmarks:
            for _, landmark in enumerate(hand_landmarks.landmark):
                x = img_width - int(landmark.x * img_width)
                y = int(landmark.y * img_height)
                z = landmark.z
                result_hand.append([x, y, z])
        result_hand = np.asarray(result_hand)

        return result_hand


# class ObjTracker():
#     def __init__(self):
#         self.model = YOLO("./objecttracker/yolo11n.yaml")
#         self.model = YOLO("./objecttracker/yolo11n.pt").to('cuda')
#         self.idx = 0
#
#     def run(self, img, flag_vis=False): # input : img_cv
#         # imgSize = (img.shape[0], img.shape[1])  # (360, 640)
#
#         # results = self.model(img, conf=0.4, device=0)
#         results = self.model(img, conf=0.4, device=0, verbose=False, classes=[0, 39, 41, 43, 44, 46, 47, 64, 65, 67])
#         result = results[0]
#
#         if flag_vis:
#             plots = result.plot()
#             cv2.imshow("object tracker results", plots)
#             cv2.waitKey(1)
#
#         boxes = result.boxes
#
#         center_dict = {}
#         flag_hand = False
#         for box in boxes:
#             bbox = np.squeeze(box.xyxy.cpu().numpy())
#             cls = int(box.cls.cpu().numpy()[0])
#             if cls == 0:
#                 # hand detected
#                 flag_hand = True
#             if cls != 0:
#                 center_x = int((bbox[0] + bbox[2]) / 2)
#                 center_y = int((bbox[1] + bbox[3]) / 2)
#                 center_dict[cls] = [center_x, center_y]
#
#         ## visualize obj centers
#         # if flag_vis:
#             # debug = np.copy(img)
#             # for center in center_list:
#             #     cv2.circle(debug, center, 5, color=[255, 255, 0], thickness=-1, lineType=cv2.LINE_AA)
#             # cv2.imshow("object centers", debug)
#             # cv2.waitKey(1)
#
#         return flag_hand, center_dict
#


# 0: 'person', 41: 'cup',
"""
{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 
9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 
25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 
33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
  49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch',
   58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 
   66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 
   73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
"""

