import os
import sys
import cv2
import time
import numpy as np
import torch
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from enum import Enum, IntEnum
import copy
from collections import deque
from ultralytics import YOLO

from handtracker.module_SARTE import HandTracker
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


class ObjTracker():
    def __init__(self, det_cooltime=10):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        YOLO_obj_path = os.path.join(curr_dir, 'pretrained_models', 'yolo11m.pt')
        self.detector_obj = YOLO(YOLO_obj_path)

        self.detector_obj.to(self.device)

        testImg = cv2.imread(os.path.join(curr_dir, './handtracker_wilor/demo_img/test1.jpg'))
        testImg = cv2.resize(testImg, (640, 360))
        _ = self.detector_obj(testImg, verbose=False)

        self.det_cooltime = det_cooltime
        self.obj_cnt = 0
        self.flag_detected = False

    def detect_objs(self, img, depth_image_float, d_wrist, check_cooltime=True):

        # 쿨타임 체크 로직
        if check_cooltime:
            self.obj_cnt += 1
            if self.obj_cnt < self.det_cooltime:
                self.flag_detected = False
                return []
            self.obj_cnt = 0
            self.flag_detected = True
        else:
            self.flag_detected = True

        mask = (depth_image_float > 0) & (depth_image_float - d_wrist <= 0.1)
        mask = mask.astype(np.uint8) * 255

        # YOLO 실행
        results = self.detector_obj(img, verbose=False)

        obj_bb_nearby = []
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = result.names[cls]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if label == 'person':
                    continue

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Check if the object center is within the depth mask
                if mask[int(cy), int(cx)] == False:
                    continue

                obj_bb_nearby.append([x1, y1, x2, y2, label])

        return obj_bb_nearby


class GestureClassfier():
    def __init__(self, ckpt="./gestureclassifier/checkpoints/checkpoint.tar", seq_len=16, model_opt=1):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        if model_opt == 0 or model_opt >= 2 and model_opt < 6:
            num_feature = 78
            self.flag_partial = False
        else:
            num_feature = 60
            self.flag_partial = True

        self.model_gesture = create_model(num_features=num_feature, num_classes=15, model_opt=model_opt)

        checkpoint = torch.load(ckpt, map_location=self.device)  # map_location 추가
        state_dict = checkpoint['model_state_dict']

        # "module." prefix가 있는지 확인
        has_module_prefix = any(k.startswith("module.") for k in state_dict.keys())
        # prefix 제거
        if has_module_prefix:
            new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        else:
            new_state_dict = state_dict

        self.model_gesture.load_state_dict(new_state_dict)
        # --- [수정] DataParallel 제거 및 .to(self.device)로 교체 ---
        self.model_gesture.to(self.device)
        # self.model_gesture = torch.nn.DataParallel(self.model_gesture).cuda() # 기존 코드 제거

        self.model_gesture.eval()

        # default args
        self.seq_len = seq_len
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
        # input : queue (self.seq_len, 63+15)  -> ndarray (self.seq_len, 78)
        input = np.array(input).reshape(self.seq_len, -1)

        input = self._normalize(input)
        if self.flag_partial:
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

