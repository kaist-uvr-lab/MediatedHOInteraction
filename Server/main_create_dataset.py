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

from modules import HandTracker_our_v2, identify_interacting_finger # GestureClassfier
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


#### args ####

"""
Note
- skip 'Up~Right'-'Index' if Grasp type.
- Natural 클래스는 thumb/index 관계없이 두 손가락은 고정한채로 나머지 손가락 및 손 전체 자세를 자유롭게 움직이는 느낌
- index로 circling하는건 자연스럽게 가능한 선에서만.

"""


# Recording args
trial = 2

# 0: ['Up', 'Down', 'Left', 'Right', 'Tap']         1: ['Clock', 'CClock', 'Natural']
# 0: 1 sec recording                                1: 2 sec recording
action_idx = 0

# Grasp  NonGrasp
state = 'Grasp'

# fingers = ['thumb', 'index']

# Set HoloLens2 Wi-Fi address
host = '192.168.50.31'

"""
[데이터 수집 class]

short clip : 
    - Swipe(Up, Down, Left, Right) * finger(Thumb, Index)

    - Tap * finger(Thumb, Index)
        > Total 10 class

long clip : 
    - Clock/CClock * finger(Thumb, Index)
    - Natural
        > Total 5 class


[수집 condition]

- cam~hand view direction : 정면, 대각 4종
- 물체 최소 3종 (사이즈, 형태 다르게)
- on-object/on-plane
- Gesture range(크게 작게)

- 전체 반복 5회해서 FOLD로 활용.


- 거리는 관계없음(crop), 2d 위치도 관계없음(normalize)
- space 입력으로 record 시작/끝

- augment 방식
    - 10 Frame을 입력으로 샘플링. 2프레임 단위로 간격?
    - rotation변화준다면 아주 약간만. 5도 단위?
    - normalize후 scale 조정? 0.8 ~ 1.2
    -


[수집 params]


"""

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


fingers = ['thumb', 'index']

if action_idx == 0:
    actions = ['Up', 'Down', 'Left', 'Right', 'Tap']
    duration = 1.0  # short : 1 sec -> 15 frame, long : 2 sec -> 30
else:
    actions = ['Clock', 'CClock', 'Natural']
    duration = 2.0

for act in actions:
    for finger in fingers:
        save_dir = f'dataset/trial{trial}/{act}_{finger}_{state}'
        os.makedirs(save_dir, exist_ok=True)


def main():
    global actions, fingers, duration, save_dir, pv_height, pv_width

    ###################### init models ######################

    track_hand = HandTracker_our_v2()

    ###################### init comm. with hololens2 ######################
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    init_variables, max_depth, producer = init_hl2()

    cv2.namedWindow('Prompt')
    cv2.resizeWindow(winname='Prompt', width=640, height=360)
    cv2.moveWindow(winname='Prompt', x=2140, y=920)

    cv2.namedWindow('RGB')
    cv2.resizeWindow(winname='RGB', width=640, height=360)
    cv2.moveWindow(winname='RGB', x=1500, y=920)

    cv2.namedWindow('DEPTH')
    cv2.resizeWindow(winname='DEPTH', width=640, height=360)
    cv2.moveWindow(winname='DEPTH', x=1500, y=560)

    idx_depth = 0

    flag_recording = False
    flag_saved = True

    data_list = []
    bar_width = 20
    bar_color = (0, 0, 255)

    finger_idx = 0
    act_idx = 0
    action = actions[act_idx]
    save_dir = f'dataset/trial{trial}/{action}_{fingers[finger_idx]}_{state}'

    last_pressed_0 = 0
    last_pressed_1 = 0
    cooldown = 0.5

    try:
        while True:
            t1 = time.time()
            # log_event("check server latency")

            # intermittently receive depth image
            idx_depth += 1
            if idx_depth == num_depth_count:
                idx_depth = 0
                flag_depth = True
            else:
                flag_depth = False

            ###################### receive input ######################
            result = receive_images(init_variables, flag_depth)

            if result == None:
                if flag_recording:
                    flag_recording = False
                    flag_saved = True
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] camera error occured. Retry recording")
                continue

            color, depth = result

            ### Display RGBD pair ###
            cv2.imshow('RGB', color)
            if flag_depth:
                cv2.imshow('DEPTH', depth / max_depth)  # scale for visibility
            cv2.waitKey(1)

            ###################### process hand ######################

            outs = track_hand.run(color)
            if not outs:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] No hand visible")
                if flag_recording:
                    flag_recording = False
                    flag_saved = True
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] camera error occured. Retry recording")
                continue

            all_right, all_uvds, _, _ = outs


            ###################### receive keyboard input ######################

            if not flag_recording and keyboard.is_pressed('space'):
                flag_recording = True
                start_time = time.time()

            if keyboard.is_pressed('esc'):
                print("exiting ...")
                break

            if keyboard.is_pressed('a') and time.time() - last_pressed_0 > cooldown:
                last_pressed_0 = time.time()
                act_idx += 1
                print("next actions")
                if act_idx == len(actions):
                    print("end. reset to starting idx. press s to change finger")
                    act_idx = 0
                action = actions[act_idx]
                save_dir = f'dataset/trial{trial}/{action}_{fingers[finger_idx]}_{state}'


            if keyboard.is_pressed('s') and time.time() - last_pressed_1 > cooldown:
                last_pressed_1 = time.time()

                finger_idx += 1
                if finger_idx == 2:
                    print("exiting ...")
                    break

                save_dir = f'dataset/trial{trial}/{action}_{fingers[finger_idx]}_{state}'
                print("next finger")

            ###################### process gesture ######################
            ## process only right hand gesture
            indices = np.where(np.asarray(all_right) == 1)[0]  ### check. 0: left, 1: right

            if len(indices) == 0:
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] No [right] hand visible")
                if flag_recording:
                    flag_recording = False
                    flag_saved = True
                    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] camera error occured. Retry recording")
                continue

            uvd_right = np.squeeze(np.asarray(all_uvds)[indices[0]])
            angle_label = compute_ang_from_joint(uvd_right)

            color_vis = draw_2d_skeleton(color, uvd_right)

            if flag_recording:
                flag_saved = False

                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Recording ...")

                now = time.time()
                elapsed = now - start_time

                d = np.concatenate([uvd_right.flatten(), angle_label])
                data_list.append(d)

                # draw
                ratio = 1 - (elapsed / duration)
                bar_height = int(pv_height * ratio)
                x1 = pv_width - bar_width
                y1 = pv_height - bar_height
                x2 = pv_width
                y2 = pv_height
                cv2.rectangle(color_vis, (x1, y1), (x2, y2), bar_color, -1)

                fps = str(1.0 / (time.time() - t1))
                cv2.putText(color_vis, f'Collecting {action}_{fingers[finger_idx]} ... FPS : {fps}', org=(10, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 255), thickness=2)
                cv2.imshow("Prompt", color_vis)
                cv2.waitKey(1)

                if elapsed > duration:
                    flag_recording = False
            else:
                cv2.putText(color_vis, f'Ready for {action}_{fingers[finger_idx]} ... press space', org=(10, 30),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness=2)
                cv2.imshow("Prompt", color_vis)
                cv2.waitKey(1)


            if not flag_recording and not flag_saved:
                flag_saved = True

                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] end recording...saving data len : {len(data_list)}")
                created_time = int(time.time())
                np.save(os.path.join(save_dir, f'{created_time}'), np.array(data_list))
                data_list = []


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