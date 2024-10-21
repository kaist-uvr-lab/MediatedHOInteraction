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
from gestureclassifier.point_history_classifier.point_history_classifier import PointHistoryClassifier

from handtracker.module_SARTE import HandTracker


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class GestureClass(IntEnum):
    Stop = 0
    Clockwise = 1
    CounterClockwise = 2
    Neutral = 3
    Up = 4
    Down = 5
    Right = 6
    Left = 7
    Doubletap = 8


class GestureClassfier():
    def __init__(self, ckpt=None, len_tip_history=8):
        self.classifier = PointHistoryClassifier(ckpt=ckpt)
        self.tip_history = deque(maxlen=len_tip_history)
        self.len_tip_history = len_tip_history

        ## args
        self.obj_hand_dist_threshold = 80
        self.gesture_cnt_threshold = 4

        self.activate_gesture = True
        self.max_activate = 3
        self.count_activate = 0

        self.prev_gesture_idx = 0
        self.gesture_cnt = 1


    def run(self, result_hand, obj_center_list):
        mean_hand = np.mean(result_hand, axis=0)
        min_gap = np.inf
        for obj_center in obj_center_list:
            gap = [obj_center[0] - mean_hand[0], obj_center[1] - mean_hand[1]]
            gap = np.linalg.norm(gap)
            if min_gap > gap:
                min_gap = gap
        # print("min gap : ", min_gap)

        if min_gap < self.obj_hand_dist_threshold:
            self.activate_gesture = True
            self.count_activate = 0

        ## count activation, until max_activation
        if self.activate_gesture:
            self.count_activate += 1

        if self.count_activate > self.max_activate:
            self.activate_gesture = False
            self.tip_history.clear()

        result_index = None
        result_index_cml = None

        if self.activate_gesture:
            thumb_uv = result_hand[4, :2]
            index_uv = result_hand[8, :2]

            self.tip_history.append(np.copy(thumb_uv))

            if len(self.tip_history) == self.len_tip_history:
                norm_history = copy.deepcopy(self.tip_history)
                norm_history -= norm_history[0]
                classifier_input = []

                if self.len_tip_history == 8:
                    norm_ratio = 80.
                    for xy_idx in range(len(norm_history)):
                        xy = norm_history[xy_idx]
                        classifier_input.append(xy[0] / norm_ratio)
                        classifier_input.append(-xy[1] / norm_ratio)
                elif self.len_tip_history == 4:
                    norm_ratio = 40.
                    for xy_idx in range(len(norm_history)):
                        xy = norm_history[xy_idx]
                        if xy_idx == len(norm_history) - 1:
                            xy_prev = xy = norm_history[xy_idx - 1]
                            xy_next = 2 * xy - xy_prev
                        else:
                            xy_next = norm_history[xy_idx + 1]

                        xy_inter = (xy + xy_next) / 2.0

                        classifier_input.append(xy[0] / norm_ratio)
                        classifier_input.append(-xy[1] / norm_ratio)

                        classifier_input.append(xy_inter[0] / norm_ratio)
                        classifier_input.append(-xy_inter[1] / norm_ratio)

                result_index, confidence_score = self.classifier(classifier_input)

                ## cumulative gesture results
                if self.prev_gesture_idx == result_index:
                    self.gesture_cnt += 1
                else:
                    self.gesture_cnt = 1
                self.prev_gesture_idx = result_index

                if self.gesture_cnt >= self.gesture_cnt_threshold:
                    result_index_cml = result_index
                else:
                    result_index_cml = None

        return [self.tip_history, result_index, result_index_cml]


class HandTracker_our():
    def __init__(self):
        self.track_hand = HandTracker()

    def run(self, input, img_width, img_height):
        result_hand = self.track_hand.Process_single(input)

        return result_hand


class HandTracker_mp():
    def __init__(self, ckpt=None):
        print("init hand tracker")
        torch.backends.cudnn.benchmark = True
        self.mediahand = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

    def run(self, input, img_width, img_height):
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
        self.model = YOLO("./objecttracker/yolov8n.pt")
        self.idx = 0

    def process_simple(self, img): # input : img_cv
        # imgSize = (img.shape[0], img.shape[1])  # (360, 640)

        # results = self.model(img, conf=0.4, device=0)
        results = self.model(img, conf=0.4, device=0, verbose=False)#, classes=[41, 64, 67])    #
        result = results[0]

        boxes = result.boxes

        center_list = []
        for box in boxes:
            bbox = np.squeeze(box.xyxy.cpu().numpy())
            cls = int(box.cls.cpu().numpy()[0])
            if cls != 0:
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                center = (center_x, center_y)
                center_list.append(center)

        return center_list

    def process(self, img): # input : img_cv
        t0 = time.time()
        if img.shape[-1] == 4:
            img = img[:, :, :-1]
        imgSize = (img.shape[0], img.shape[1])  # (360, 640)

        # results = self.model(img, conf=0.4, device=0)
        results = self.model(img, conf=0.4, device=0, verbose=False, classes=[0, 41, 64, 67])    #
        result = results[0]

        boxes = result.boxes

        bbox_list = []
        center_list = []
        hand_list = []
        for box in boxes:
            bbox = np.squeeze(box.xyxy.cpu().numpy())
            bbox_list.append(bbox)

            cls = int(box.cls.cpu().numpy()[0])
            if cls == 0:
                corner = (bbox[0], bbox[1])
                hand_list.append(corner)
            else:
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                center = (center_x, center_y)
                center_list.append(center)

                # debug
                cv2.circle(img, center, 5, (255, 255, 0), -1, cv2.LINE_AA)

        # filename = "C:/Woojin/research/wiseui_base/Integration/WiseUIServer/save_3_center/" + str(self.idx) + ".png"
        # cv2.imwrite(filename, img)
        # result.save(filename=filename)
        # self.idx += 1

        return hand_list, center_list, img

"""
{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 
15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 
25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 
40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 
45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 
50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 
55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 
60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 
65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 
70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 
75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
"""

