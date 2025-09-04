import sys, os
import asyncio
import time
from collections import deque
import cv2
import numpy as np
import websockets
import struct
import json
from PIL import Image, ImageDraw, ImageFont

from modules import HandTracker_our_v2 # GestureClassfier
from utils.visualize import draw_2d_skeleton

sys.path.append("./hl2ss_")
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import hl2ss_utilities
import socket
import multiprocessing as mp
import queue
import keyboard
import datetime
import random
from modules import HandTracker_our_v2
from utils.visualize import draw_2d_skeleton


#### args ####
"""
subject 별로 gesture를 랜덤하게 이어서 수집. 

물체 최소 2종, Finger 2개 별도로 수집. (최소 반복횟수 4회)

일시정지 없음. 실수한 gesture idx기억해서 저장 후 삭제. 많이 누적되면 1회 더 수행하는 것으로 충당.

한번 시행 당 2분 가량 소모.

(주의)
초록때 준비 끝내고, 빨간 원 나온후에 동작시작하도록 
가급적 동작 준비할때는 물체에서 손가락을 뗴고 이동.



"""

# Recording args
subject = 0

trial = 1       # 2 object or more
finger_idx = 1  # 2 type


# fixed part
fingers = ['thumb', 'index']
actions = ['Up', 'Down', 'Left', 'Right', 'Tap', 'Clock', 'CClock']


# Set HoloLens2 Wi-Fi address
host = '192.168.50.31'


#### fixed args ####

# Calibration path (must exist but can be empty)
calibration_path = 'calibration'

# Front RGB camera parameters
pv_width = 640
pv_height = 360
pv_fps = 30

# Buffer length in seconds
buffer_size = 10

# Process depth image per n frame
num_depth_count = 10

prev_label, prev = "Init", time.time()


record_duration_per_gesture = 1.5
pause_duration_per_gesture = 2.5
num_per_action = 3
actions_to_collect = []

for i in range(num_per_action):
    for act in actions:
        actions_to_collect.append(act)

random.shuffle(actions_to_collect)


def get_dummy_frame():
    return 255 * np.ones((pv_height, pv_width, 3), dtype=np.uint8), 255 * np.ones((pv_height, pv_width), dtype=np.uint8)


def main():
    global actions, action_idx, fingers, finger_idx, pv_height, pv_width, actions_to_collect, record_duration_per_gesture, pause_duration_per_gesture, trial, subject


    ###################### init comm. with hololens2 ######################
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    init_variables, max_depth, producer = init_hl2()

    cv2.namedWindow('Prompt')
    cv2.resizeWindow(winname='Prompt', width=640, height=360)
    cv2.moveWindow(winname='Prompt', x=2140, y=920)

    cv2.namedWindow('RGB')
    cv2.resizeWindow(winname='RGB', width=640, height=360)
    cv2.moveWindow(winname='RGB', x=1500, y=920)

    idx_depth = 0

    current_action_index = 0
    record_start_time = None
    record_end_time = None
    total_len = len(actions_to_collect)
    flag_recording = False
    flag_init = False
    flag_once = True

    target_finger = fingers[finger_idx]
    frame_idx = 0

    color_frames = []
    depth_frames = []
    label_log = {}

    track_hand = HandTracker_our_v2()

    try:
        while True:
            start_t = time.time()
            # intermittently receive depth image
            idx_depth += 1
            if idx_depth == num_depth_count:
                idx_depth = 0
                flag_depth = True
            else:
                flag_depth = False

            ###################### receive input ######################
            result = receive_images(init_variables, flag_depth)
            # result = get_dummy_frame()

            if result == None:
                continue

            color, depth = result

            ### save buffer ###
            if flag_init:
                color_frames.append((current_action_index, frame_idx, color.copy()))
                if flag_depth:
                    depth_frames.append((current_action_index, frame_idx, depth.copy()))
                frame_idx += 1

            ### Display RGB ###
            cv2.imshow('RGB', color)
            cv2.waitKey(1)


            ###################### receive keyboard input ######################
            if keyboard.is_pressed('space') and flag_once:
                print("start")
                flag_once = False

                flag_init = True
                flag_recording = False
                record_start_time = time.time()
                record_end_time = time.time()

            if keyboard.is_pressed('esc'):
                print("exiting ...")
                break


            ###################### run tracker ######################

            outs = track_hand.run(color)
            if not outs:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] No hand visible")
                continue
            else:
                all_right, all_uvds, _, _ = outs
                indices = np.where(np.asarray(all_right) == 1)[0]  ### check. 0: left, 1: right
                if len(indices) != 0:
                    uvd_right = np.squeeze(np.asarray(all_uvds)[indices[0]])
                    color = draw_2d_skeleton(color, uvd_right)

            ###################### show prompt ######################


            if flag_init:
                if flag_recording:
                    elapsed = time.time() - record_start_time

                    action_text = actions_to_collect[current_action_index]
                    # save label for current frame
                    label_log[frame_idx] = action_text

                    # 오른쪽 위 빨간 원
                    cv2.circle(color, (pv_width - 30, 30), 15, (0, 0, 255), -1)

                    # 왼쪽 위 텍스트
                    cv2.putText(color, f"{target_finger} - {action_text} ({current_action_index+1}/{total_len})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0) , 2)

                    # 오른쪽 빨간 바 (시간에 따라 줄어듦)
                    bar_height = int((1 - elapsed / record_duration_per_gesture) * pv_height)
                    bar_height = max(bar_height, 0)
                    cv2.rectangle(color, (pv_width - 10, pv_height - bar_height), (pv_width, pv_height), (0, 0, 255), -1)

                    # 녹화 종료 조건
                    if elapsed >= record_duration_per_gesture:
                        flag_recording = False
                        current_action_index += 1
                        record_end_time = time.time()
                else:
                    elapsed = time.time() - record_end_time
                    label_log[frame_idx] = "Natural"
                    # 오른쪽 위 초록 원
                    cv2.circle(color, (pv_width - 30, 30), 15, (0, 255, 0), -1)

                    # 왼쪽 위 텍스트: 대기중...(다음 action)
                    if current_action_index < len(actions_to_collect):
                        next_action = actions_to_collect[current_action_index]
                        wait_text = f"Ready for ...{target_finger} - {next_action}"
                    else:
                        wait_text = "all done"
                    cv2.putText(color, wait_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                    # 오른쪽 초록 바 (시간에 따라 줄어듦)
                    bar_height = int((1 - elapsed / pause_duration_per_gesture) * pv_height)
                    bar_height = max(bar_height, 0)
                    cv2.rectangle(color, (pv_width - 10, pv_height - bar_height), (pv_width, pv_height), (0, 255, 0), -1)

                    if elapsed >= pause_duration_per_gesture:
                        flag_recording = True
                        record_start_time = time.time()

                    # 화면 출력
            else:
                label_log[frame_idx] = "Natural"

            cv2.imshow("Prompt", color)

            if current_action_index == total_len:
                print("done")
                break

            end_t = time.time()

            ## fix to 15 fps
            latency = end_t - start_t
            # print(f"fps : { 1/latency}, ms : {latency*1000}")
            if latency < 0.056:     # for 15 FPS. need manual adjustment
                delay = 0.056 - latency
                time.sleep(delay)

            latency = time.time() - start_t
            print(f"added fps : { 1/latency}, ms : {latency*1000}")

        ### save images ###
        print("saving images in buffer ...")

        color_image_dir = f"dataset/subject_{subject}/trial_{trial}/{target_finger}/rgb"  # 저장할 디렉토리

        if os.path.exists(color_image_dir):
            print("----------------folder already exist. save to trial -1 -----------------------")
            trial = -1
            color_image_dir = f"dataset/subject_{subject}/trial_{trial}/{target_finger}/rgb"  # 저장할 디렉토리

        os.makedirs(color_image_dir, exist_ok=True)
        for action_index, frame_idx, color in color_frames:
            more_path = os.path.join(color_image_dir, f"{action_index+1}_{actions_to_collect[action_index]}")
            os.makedirs(more_path, exist_ok=True)

            out_path = os.path.join(more_path, f"frame_{frame_idx:06d}.png")
            cv2.imwrite(out_path, color)

        depth_image_dir = f"dataset/subject_{subject}/trial_{trial}/{target_finger}/depth"  # 저장할 디렉토리
        os.makedirs(depth_image_dir, exist_ok=True)
        for action_index, frame_idx, depth in depth_frames:
            more_path = os.path.join(depth_image_dir, f"{action_index+1}_{actions_to_collect[action_index]}")
            os.makedirs(more_path, exist_ok=True)

            out_path = os.path.join(more_path, f"depth_{frame_idx:06d}.npy")
            np.save(out_path, depth.astype(np.float32))  # float32 그대로 저장

        with open(f"dataset/subject_{subject}/trial_{trial}/{target_finger}/label.txt", 'w', encoding='utf-8') as f:
            for key, value in label_log.items():
                f.write(f'{key} {value}\n')

        print("done")
    finally:
        sock.close()

        # Stop PV and RM Depth AHAT streams ---------------------------------------
        sink_ht, sink_pv = init_variables[0], init_variables[1]
        sink_pv.detach()
        sink_ht.detach()
        producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
        producer.stop(hl2ss.StreamPort.RM_DEPTH_AHAT)

        # Stop PV subsystem -------------------------------------------------------
        hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

        cv2.destroyAllWindows()


def init_hl2():
    # Start PV Subsystem ------------------------------------------------------
    hl2ss_lnm.start_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    # Get RM Depth AHAT calibration -------------------------------------------
    # Calibration data will be downloaded if it's not in the calibration folder
    calibration_ht = hl2ss_3dcv.get_calibration_rm(host, hl2ss.StreamPort.RM_DEPTH_AHAT, calibration_path)

    uv2xy = calibration_ht.uv2xy  # hl2ss_3dcv.compute_uv2xy(calibration_ht.intrinsics, hl2ss.Parameters_RM_DEPTH_AHAT.WIDTH, hl2ss.Parameters_RM_DEPTH_AHAT.HEIGHT)
    xy1, scale = hl2ss_3dcv.rm_depth_compute_rays(uv2xy, calibration_ht.scale)
    max_depth = calibration_ht.alias / calibration_ht.scale

    xy1_o = hl2ss_3dcv.block_to_list(xy1[:-1, :-1, :])
    xy1_d = hl2ss_3dcv.block_to_list(xy1[1:, 1:, :])

    # Start PV and RM Depth AHAT streams --------------------------------------
    producer = hl2ss_mp.producer()
    producer.configure(hl2ss.StreamPort.PERSONAL_VIDEO,
                       hl2ss_lnm.rx_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO, width=pv_width, height=pv_height,
                                       framerate=pv_fps))
    producer.configure(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss_lnm.rx_rm_depth_ahat(host, hl2ss.StreamPort.RM_DEPTH_AHAT))
    producer.initialize(hl2ss.StreamPort.PERSONAL_VIDEO, pv_fps * buffer_size)
    producer.initialize(hl2ss.StreamPort.RM_DEPTH_AHAT, hl2ss.Parameters_RM_DEPTH_AHAT.FPS * buffer_size)
    producer.start(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.start(hl2ss.StreamPort.RM_DEPTH_AHAT)

    consumer = hl2ss_mp.consumer()
    manager = mp.Manager()
    sink_pv = consumer.create_sink(producer, hl2ss.StreamPort.PERSONAL_VIDEO, manager, None)
    sink_ht = consumer.create_sink(producer, hl2ss.StreamPort.RM_DEPTH_AHAT, manager, None)

    sink_pv.get_attach_response()
    sink_ht.get_attach_response()

    # Initialize PV intrinsics and extrinsics ---------------------------------
    pv_intrinsics = hl2ss.create_pv_intrinsics_placeholder()
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)

    return [sink_ht, sink_pv, pv_intrinsics, pv_extrinsics, xy1_o, xy1_d, scale, calibration_ht], max_depth, producer


def receive_images(init_variables, flag_depth):

    sink_ht, sink_pv, pv_intrinsics, pv_extrinsics, xy1_o, xy1_d, scale, calibration_ht = init_variables

    # Get RM Depth AHAT frame and nearest (in time) PV frame --------------
    _, data_ht = sink_ht.get_most_recent_frame()
    if ((data_ht is None) or (not hl2ss.is_valid_pose(data_ht.pose))):
        return None
    _, data_pv = sink_pv.get_nearest(data_ht.timestamp)
    if ((data_pv is None) or (not hl2ss.is_valid_pose(data_pv.pose))):
        return None

    # Preprocess frames ---------------------------------------------------
    color = data_pv.payload.image
    pv_z = None
    if flag_depth:
        depth = data_ht.payload.depth  # hl2ss_3dcv.rm_depth_undistort(data_ht.payload.depth, calibration_ht.undistort_map)
        z = hl2ss_3dcv.rm_depth_normalize(depth, scale)

    # Update PV intrinsics ------------------------------------------------
    # PV intrinsics may change between frames due to autofocus
    pv_intrinsics = hl2ss.update_pv_intrinsics(pv_intrinsics, data_pv.payload.focal_length,
                                               data_pv.payload.principal_point)
    color_intrinsics, color_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

    # Generate depth map for PV image -------------------------------------
    if flag_depth:
        mask = (depth[:-1, :-1].reshape((-1,)) > 0)
        zv = hl2ss_3dcv.block_to_list(z[:-1, :-1, :])[mask, :]

        ht_to_pv_image = hl2ss_3dcv.camera_to_rignode(calibration_ht.extrinsics) @ hl2ss_3dcv.reference_to_world(
            data_ht.pose) @ hl2ss_3dcv.world_to_reference(data_pv.pose) @ hl2ss_3dcv.rignode_to_camera(
            color_extrinsics) @ hl2ss_3dcv.camera_to_image(color_intrinsics)

        ht_points_o = hl2ss_3dcv.rm_depth_to_points(xy1_o[mask, :], zv)
        pv_uv_o_h = hl2ss_3dcv.transform(ht_points_o, ht_to_pv_image)
        pv_list_depth = pv_uv_o_h[:, 2:]

        ht_points_d = hl2ss_3dcv.rm_depth_to_points(xy1_d[mask, :], zv)
        pv_uv_d_h = hl2ss_3dcv.transform(ht_points_d, ht_to_pv_image)
        pv_d_depth = pv_uv_d_h[:, 2:]

        mask = (pv_list_depth[:, 0] > 0) & (pv_d_depth[:, 0] > 0)

        pv_list_depth = pv_list_depth[mask, :]
        pv_d_depth = pv_d_depth[mask, :]

        pv_list_o = pv_uv_o_h[mask, 0:2] / pv_list_depth
        pv_list_d = pv_uv_d_h[mask, 0:2] / pv_d_depth

        pv_list = np.hstack((pv_list_o, pv_list_d + 1)).astype(np.int32)
        pv_z = np.zeros((pv_height, pv_width), dtype=np.float32)

        u0 = pv_list[:, 0]
        v0 = pv_list[:, 1]
        u1 = pv_list[:, 2]
        v1 = pv_list[:, 3]

        mask0 = (u0 >= 0) & (u0 < pv_width) & (v0 >= 0) & (v0 < pv_height)
        mask1 = (u1 > 0) & (u1 <= pv_width) & (v1 > 0) & (v1 <= pv_height)
        maskf = mask0 & mask1

        pv_list = pv_list[maskf, :]
        pv_list_depth = pv_list_depth[maskf, 0]

        for n in range(0, pv_list.shape[0]):
            u0 = pv_list[n, 0]
            v0 = pv_list[n, 1]
            u1 = pv_list[n, 2]
            v1 = pv_list[n, 3]

            pv_z[v0:v1, u0:u1] = pv_list_depth[n]

    return color, pv_z


def log_event(label):
    global prev_label, prev

    now = time.time()
    # timestamps.append(now)
    # labels.append(label)
    print(f"{prev_label} ~ {label}: {now - prev:.3f}")
    prev_label = label
    prev = now


def compute_ang_from_joint(joint):  # joint : (21, 3)
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


if __name__ == '__main__':
    main()