import os
import sys
import cv2
import time
import numpy as np
from ultralytics import YOLO

import torch
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from enum import Enum, IntEnum
import copy

from handtracker.module_SARTE import HandTracker
from tensorflow.keras.models import load_model
from collections import deque


class GestureClassfier():
    def __init__(self, img_width=640, img_height=360):
        self.actions = ['Up', 'Down', 'Left', 'Right', 'Clock', 'CClock', 'Tap', 'Natural']
        self.seq_length = 13
        self.model = load_model('./gestureclassifier/model.h5')

        self.seq = deque([], maxlen=15)
        self.action_list = []

        self.norm_ratio_x = img_width / 2.0
        self.norm_ratio_y = img_height / 2.0
        self.tip_history = np.zeros(1)

    def init_que(self):
        self.seq = deque([], maxlen=15)
        self.action_list = []
        self.tip_history = np.zeros(1)

    def run(self, joint):
        pred_idx = -1
        if joint[0, 0] == joint[1, 0] and joint[0, 1] == joint[1, 1]:
            return pred_idx, None

        angle_label = self.compute_ang_from_joint(joint)
        d = np.concatenate([joint.flatten(), angle_label])
        self.seq.append(d)
        if len(self.seq) < self.seq_length:
            return pred_idx, None

        input_norm, self.tip_history = self.preprocess(self.seq)

        pred = self.model.predict(input_norm, verbose=0).squeeze()
        pred_idx = int(np.argmax(pred))
        conf = pred[pred_idx]

        if conf < 0.8:
            this_action = "Natural"
            pred_idx = 7
        else:
            this_action = self.actions[pred_idx]
            # print(this_action)

        # print(f"{this_action} with {conf}")

        if len(self.action_list) == 0:
            self.action_list.append(this_action)
        elif self.action_list[-1] == this_action:
            self.action_list.append(this_action)
        else:
            self.action_list = []

        if len(self.action_list) > 4:
            return pred_idx, this_action
        else:
            return -1, None

    def preprocess(self, seq):
        seq_list = [*seq]
        input_data = np.expand_dims(np.array(seq_list[-self.seq_length:], dtype=np.float32), axis=0)  # (1, 10, 78)
        input_norm = np.zeros((input_data.shape[0], input_data.shape[1], 51))
        for frame_idx in range(input_data.shape[1]):
            target_pose = input_data[0, frame_idx, :63].reshape(21, 3)
            target_angle = input_data[0, frame_idx, 63:]

            # norm 2d pose
            if frame_idx == 0:
                root_pose = target_pose[0, :]
            norm_pose = target_pose - root_pose
            norm_pose[:, 0] = norm_pose[:, 0] / self.norm_ratio_x
            norm_pose[:, 1] = norm_pose[:, 1] / self.norm_ratio_y

            norm_angle = target_angle / 180.0

            nan_indice = np.argwhere(np.isnan(norm_angle))
            for nan_idx in nan_indice:
                norm_angle[nan_idx[0]] = 0.0

            # update pose and angle
            norm_pose_cut = norm_pose[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 17], :]

            input_norm[0, frame_idx, :36] = norm_pose_cut.flatten()
            input_norm[0, frame_idx, 36:] = norm_angle

        tip_history = np.squeeze(input_data[0, :, 12:15])

        return input_norm, tip_history

    def compute_ang_from_joint(self, joint):  # joint : (21, 3)
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

        angle_label = np.array(angle, dtype=np.float32)

        return angle_label


class HandTracker_our():
    def __init__(self):
        self.track_hand = HandTracker()

    def run(self, input):
        result_hand = self.track_hand.Process_single_newroi(input)

        return result_hand


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


class ObjTracker():
    def __init__(self):
        self.model = YOLO("./objecttracker/yolo11n.yaml")
        self.model = YOLO("./objecttracker/yolo11n.pt").to('cuda')
        self.idx = 0

    def run(self, img, flag_vis=False): # input : img_cv
        # imgSize = (img.shape[0], img.shape[1])  # (360, 640)

        # results = self.model(img, conf=0.4, device=0)
        results = self.model(img, conf=0.4, device=0, verbose=False, classes=[0, 39, 41, 43, 44, 46, 47, 64, 65, 67])
        result = results[0]

        if flag_vis:
            plots = result.plot()
            cv2.imshow("object tracker results", plots)
            cv2.waitKey(1)

        boxes = result.boxes

        center_dict = {}
        flag_hand = False
        for box in boxes:
            bbox = np.squeeze(box.xyxy.cpu().numpy())
            cls = int(box.cls.cpu().numpy()[0])
            if cls == 0:
                # hand detected
                flag_hand = True
            if cls != 0:
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                center_dict[cls] = [center_x, center_y]

        ## visualize obj centers
        # if flag_vis:
            # debug = np.copy(img)
            # for center in center_list:
            #     cv2.circle(debug, center, 5, color=[255, 255, 0], thickness=-1, lineType=cv2.LINE_AA)
            # cv2.imshow("object centers", debug)
            # cv2.waitKey(1)

        return flag_hand, center_dict


def recog_contact(depth, tip):
    depth_vis = np.copy(depth)
    depth_vis[depth_vis > 0.5] = 0
    depth_vis = (depth_vis / 0.5) * 255

    arr_len = 50
    tip_x = min(int(tip[1]) * 2, 719)
    tip_y = min(int(tip[0]) * 2, 1279)

    tip_based_array = [depth_vis[tip_x:min((tip_x + arr_len), 720), tip_y],
                       np.flip(depth_vis[max((tip_x - arr_len), 0):tip_x, tip_y]),
                       depth_vis[tip_x, tip_y:min((tip_y + arr_len), 1280)],
                       np.flip(depth_vis[tip_x, max((tip_y - arr_len), 0):tip_y])]

    flag_contact = []
    for arr_idx, array in enumerate(tip_based_array):
        if len(array) < 1:
            continue
        array_ = np.asarray(np.copy(array), dtype=np.int8)
        for ele_idx, ele in enumerate(array_):
            if ele_idx == 0:
                array_[ele_idx] -= array_[0]
            else:
                array_[ele_idx] -= array[ele_idx - 1]

        array_[array_ > 200] = 0
        array_ *= 10
        array_ = np.abs(array_)
        tip_based_array[arr_idx] = array_

        if len(np.where(array_ > 25)[0]) > 0:
            flag_contact.append(False)
        else:
            flag_contact.append(True)
    if sum(flag_contact) > 1:
        # print("contact, ", sum(flag_contact))
        flag_contact = True
    else:
        # print("no contact, ", sum(flag_contact))
        flag_contact = False

    return flag_contact


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

