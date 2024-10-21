


import os
import cv2
import numpy as np
import websockets
import struct

from stopwatch import StopWatch
from SocketServer.type_definitions import DataFormat, SensorType, HoloLens2PVImageData, HoloLens2DepthImageData, \
    HoloLens2PointCloudData
import json
import threading
import torch
from natsort import natsorted
import time

from collections import deque
from handtracker.utils.visualize import draw_2d_skeleton
import copy
from PIL import Image, ImageDraw, ImageFont
from modules import HandTracker_mp, ObjTracker, GestureClassfier, GestureClass, HandTracker_our


## args
record_path = "./records/save_4"
record_start_idx = 30

hand_ckpt = "SAR_AGCN_all_v3_loweraug_extraTrue_resnet34_s0_Epochs50"
gesture_ckpt = "point_history_classifier_4.tflite"

obj_hand_dist_threshold = 80
gesture_cnt_threshold = 4

len_tip_history = 8

flag_interpolation = False
flag_gesture = True

## init
stopwatch = StopWatch()
stopwatch.start()


track_obj = ObjTracker()
track_hand = HandTracker_mp()
# track_hand = HandTracker_our()
recog_gesture = GestureClassfier(ckpt=gesture_ckpt, len_tip_history=8)


fontObj = ImageFont.truetype(font='C:/Windows/Fonts/ARIAL.ttf', size=30)


def uvdtoxyz(uvd, K):
    device = uvd.device
    depth = uvd[:, 2] #[21]
    depth = torch.reshape(depth, [21,1]) #[21,1]
    # depth = torch.tile(depth, [1,3]) #[21,3]
    depth = depth.repeat(1,3)
    one = torch.ones(21).reshape(21,1).to(device)
    # uvd_nodepth = torch.concat((uvd[:,:-1], one), axis=1)
    uvd_nodepth = torch.cat((uvd[:,:-1], one), axis=1)
    uvd_scaled = uvd_nodepth * depth
    # xyz = torch.linalg.matmul(torch.linalg.inv(K), uvd_scaled.T) #[3, 21]
    xyz = torch.matmul(torch.inverse(K), uvd_scaled.T) #[3, 21]

    return xyz.T #[21, 3]

def mano3DToCam3D(xyz3D, ext):
    device = xyz3D.device
    xyz3D = torch.squeeze(xyz3D)
    ones = torch.ones((xyz3D.shape[0], 1), device=device)

    xyz4Dcam = torch.cat([xyz3D, ones], axis=1)
    # world to target cam
    xyz3Dcam2 = xyz4Dcam @ ext.T  # [:, :3]

    return xyz3Dcam2

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    uv = torch.matmul(K, xyz.transpose(2, 1)).transpose(2, 1)
    return uv[:, :, :2] / uv[:, :, -1:]


def image_handler_record(baseDir):
    print("Start with recorded set")
    img_list = natsorted(os.listdir(baseDir))

    ## start track when the hand detected.
    ## if detected, activate the flag with {num_activate} frames
    flag_hand_found = False

    activate_gesture = True
    max_activate = 3
    count_activate = 0

    tip_history = deque(maxlen=len_tip_history)
    prev_result_hand = None
    prev_gesture_idx = 0
    gesture_cnt = 1

    result_index_cml = None

    for img_idx, img_name in enumerate(img_list):
        if img_idx < record_start_idx:
            continue
        img_path = os.path.join(baseDir, img_name)

        input = cv2.imread(img_path)
        cv2.imshow("input in records", input)
        cv2.waitKey(1)

        img_height = input.shape[0]
        img_width = input.shape[1]

        obj_center_list = track_obj.process_simple(input)

        debug = np.copy(input)
        for center in obj_center_list:
            cv2.circle(debug, center, 5, color=[255, 255, 0], thickness=-1, lineType=cv2.LINE_AA)

        cv2.imshow("output object", debug)
        cv2.waitKey(1)

        result_hand = track_hand.run(np.copy(input), img_width, img_height)
        if not isinstance(result_hand, np.ndarray):
            continue

        ## visualize hand output
        img_cv = draw_2d_skeleton(input, result_hand)
        cv2.imshow("output hand", img_cv)
        cv2.waitKey(1)

        ## activate gesture only if object is close, with n frame duration
        if flag_gesture:
            gesture_results = recog_gesture.run(result_hand, obj_center_list)

            tip_history, result_index, result_index_cml = gesture_results

            output = np.copy(input)
            ## visualize tip history
            for idx, xy in enumerate(tip_history):
                if idx == 0:
                    continue
                joint = xy[0].astype('int32'), xy[1].astype('int32')
                cv2.circle(output, joint, radius=1 + idx, color=[255.0, 0.0, 0.0], thickness=-1,
                           lineType=cv2.LINE_AA)

            if result_index != None:
                output = Image.fromarray(output)
                I1 = ImageDraw.Draw(output)
                I1.text((10, 200), str(GestureClass(result_index)), fill=(0, 255, 0), font=fontObj)

                if result_index_cml != None:
                    I1.text((10, 300), str(GestureClass(result_index_cml)), fill=(0, 0, 255), font=fontObj)
                output = np.asarray(output)

            ## visualize gesture output
            cv2.imshow("gesture output", output)
            # cv2.imwrite(f"./output/{img_idx}.png", output)
            cv2.waitKey(0)



def main():
    image_handler_record(record_path)


if __name__ == '__main__':
    main()
