import sys
import time
from collections import deque
import cv2
import numpy as np

from modules import HandTracker_mp, ObjTracker, GestureClassfier, HandTracker_our, recog_contact

from handtracker.utils.visualize import draw_2d_skeleton

sys.path.append("./hl2ss_")
import hl2ss
import hl2ss_lnm
import hl2ss_mp
import hl2ss_3dcv
import hl2ss_utilities
import socket
import multiprocessing as mp

import random
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont
from threading import Thread
import pickle


## args ##
flag_mediapipe = False
flag_gesture = False
flag_vis_obj = False

## test options ##
flag_natural = True
SUBJECT = 3

# HoloLens address
host = '192.168.1.31'

# Calibration path (must exist but can be empty)
calibration_path = 'calibration'

# Front RGB camera parameters
pv_width = 1280
pv_height = 720
pv_fps = 30

# Buffer length in seconds
buffer_size = 2

# process depth image per n frame
num_depth_count = 5     # 0 for only rgb

contact_que = deque([], maxlen=5)

## GUI
ISPLAYING = False
CURRCLASS = -1
Count_gesture = np.zeros(8)

flag_init = True
show_init = {"path": "./assets/NATURAL.png", "description":"Starting experiment", "class":-1}

# for 7 distinct actions
if not flag_natural:
    # image_info = [{"path":"./assets/LEFT.png", "description":"move LEFT"}, {"path":"./assets/RIGHT.png", "description":"move RIGHT"},
    #               {"path":"./assets/CLOCKWISE.png", "description":"Rotate Clockwise"}, {"path":"./assets/COUNTER-CLOCKWISE.png", "description":"Rotate Counter-clockwise"}
    #               ]
    image_info = [{"path": "./assets/UP.png", "description":"move UP"}, {"path":"./assets/DOWN.png", "description":"move DOWN"},
                  {"path":"./assets/LEFT.png", "description":"move LEFT"}, {"path":"./assets/RIGHT.png", "description":"move RIGHT"},
                  {"path":"./assets/CLOCKWISE.png", "description":"Rotate Clockwise"}, {"path":"./assets/COUNTER-CLOCKWISE.png", "description":"Rotate Counter-clockwise"},
                  {"path":"./assets/TAP.png", "description":"TAPPING"}]  # Replace with your image paths


    TIMER_SETTING = 2.0
    PREPARE_DELAY = 3
    WAIT_DELAY = 1

    # Generate target random gesture list
    REPEAT_NUM = 5
    show_list = []

    # image_info = [{"path":"./assets/DOWN.png", "description":"move DOWN"}, {"path":"./assets/RIGHT.png", "description":"move RIGHT"},
    #               {"path":"./assets/CLOCKWISE.png", "description":"Rotate Clockwise"},{"path":"./assets/COUNTER-CLOCKWISE.png", "description":"Rotate Counter-clockwise"},
    #               {"path":"./assets/COUNTER-CLOCKWISE.png", "description":"Rotate Counter-clockwise"},{"path":"./assets/TAP.png", "description":"TAPPING"}]
    # REPEAT_NUM = 1

    # subject_0_app_0 이거 마저하고, 이름 확인해서 수정후 넣기.(덮어씌워지지않게 주의)
    # 이후 natural 촬영
    for idx in range(REPEAT_NUM):
        for ord, data in enumerate(image_info):
            new_data = data.copy()
            new_data["class"] = ord
            show_list.append(new_data)

# for natural action
if flag_natural:
    image_info = [{"path": "./assets/NATURAL.png", "description":"Free"}]  # Replace with your image paths

    TIMER_SETTING = 5
    PREPARE_DELAY = 4
    WAIT_DELAY = 1

    REPEAT_NUM = 6
    show_list = []
    for idx in range(REPEAT_NUM):
        for ord, data in enumerate(image_info):
            new_data = data.copy()
            new_data["class"] = 7
            show_list.append(new_data)


total_cnt = len(show_list)
done_cnt = 0

class TimerThread(QThread):
    update_timer = pyqtSignal(float)  # Signal to update countdown timer
    timer_end = pyqtSignal()  # Signal for timer end to load the next image

    def run(self):
        global ISPLAYING
        remaining_time = TIMER_SETTING
        while remaining_time > 0:
            self.update_timer.emit(remaining_time)  # Update the timer display
            self.msleep(10)  # Wait 0.1 seconds
            remaining_time -= 0.01
        ISPLAYING = False
        self.update_timer.emit(0)  # Final update to display "0.00"
        self.timer_end.emit()  # Emit signal for timer end

class ImageWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Random Image Display")
        self.image_info = show_list.copy()  # Copy to preserve original list

        # Main layout
        self.layout = QVBoxLayout()
        self.layout.setSpacing(3)  # Set small spacing between elements
        self.layout.setContentsMargins(3, 3, 3, 3)  # Reduce overall margins

        # Dot and timer layout, positioned at top center
        self.dot_timer_layout = QHBoxLayout()
        self.dot_timer_layout.setSpacing(3)
        self.dot_timer_layout.setAlignment(Qt.AlignCenter)  # Center align the dot and timer

        # Placeholder label for red dot
        self.dot_label = QLabel()
        self.dot_label.setFixedSize(20, 20)  # Dot size
        self.dot_timer_layout.addWidget(self.dot_label)

        # Timer display label next to the red dot
        self.timer_label = QLabel(str(TIMER_SETTING))
        self.timer_label.setFont(QFont("Arial", 50))
        self.timer_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.dot_timer_layout.addWidget(self.timer_label)

        # Add the dot and timer layout to the main layout
        self.layout.addLayout(self.dot_timer_layout)


        self.prepare_label = QLabel()
        self.prepare_label.setFont(QFont("Arial", 50))
        self.prepare_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.prepare_label)

        # Description label, positioned between the red dot/timer and the image
        self.description_label = QLabel()
        self.description_label.setFont(QFont("Arial", 50))
        self.description_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.description_label)

        # Image label for showing random images
        self.image_label = QLabel()
        self.layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        self.setLayout(self.layout)

        # Start displaying images
        self.start_new_image_cycle()

    def start_new_image_cycle(self):
        # Check if there are remaining images
        if not self.image_info:
            self.timer_label.setText("No more images!")
            return

        # Select and display a new random image with its description
        self.show_random_image()

        # Hide the red dot when the image is changed
        self.hide_red_dot()

        self.prepare_label.setText("prepare...")

        # Schedule the red dot to appear and the timer to start after 1 second
        QTimer.singleShot(PREPARE_DELAY*1000, self.start_timer)

    def show_random_image(self):
        global CURRCLASS, flag_init, done_cnt, total_cnt

        # Choose and display a random image with its description
        if flag_init:
            image_data = show_init
            flag_init = False
        else:
            image_data = random.choice(self.image_info)
            # Remove the displayed image from the list to avoid repetition
            self.image_info.remove(image_data)
            done_cnt += 1

        pixmap = QPixmap(image_data["path"])
        self.image_label.setPixmap(pixmap.scaled(500, 400, Qt.KeepAspectRatio))

        # Set the description
        self.description_label.setText(image_data["description"]+f" ({done_cnt}/{total_cnt})")

        # Set class info
        CURRCLASS = image_data["class"]


    def start_timer(self):
        global ISPLAYING
        # Show the red dot when the 2-second timer starts
        self.show_red_dot()
        self.prepare_label.setText(" ")

        # Start the timer thread to control the countdown timer
        self.timer_thread = TimerThread()
        self.timer_thread.update_timer.connect(self.update_timer_display)
        self.timer_thread.timer_end.connect(self.schedule_next_image)
        self.timer_thread.start()
        ISPLAYING = True

    def show_red_dot(self):
        # Draw a red dot
        pixmap = QPixmap(20, 20)
        pixmap.fill(Qt.transparent)  # Transparent background for red dot
        painter = QPainter(pixmap)
        painter.setBrush(QColor('red'))
        painter.drawEllipse(-0.5,0,17,17)  # Position red dot in the center
        painter.end()
        self.dot_label.setPixmap(pixmap)  # Update dot label with the red dot

    def hide_red_dot(self):
        # Clear the red dot by clearing the QLabel's pixmap
        self.dot_label.clear()

    def update_timer_display(self, time_left):
        # Update the timer display to show countdown in "X.XX" format
        self.timer_label.setText(f"{time_left:.2f}")

    def schedule_next_image(self):
        # Schedule the next image to appear after 1 second
        QTimer.singleShot(WAIT_DELAY*1000, self.start_new_image_cycle)


def check_playing():
    global ISPLAYING
    while True:
        if ISPLAYING:
            print("Playing")
        else:
            print("Not Playing")


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


def main_single():
    global ISPLAYING, CURRCLASS, flag_init

    ###################### init models ######################
    if flag_mediapipe:
        track_hand = HandTracker_mp()
    else:
        track_hand = HandTracker_our()

    track_obj = ObjTracker()
    track_gesture = GestureClassfier(img_width=640, img_height=360)

    ###################### init comm. with hololens2 ######################
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    init_variables, max_depth, producer = init_hl2()

    idx_depth = 0
    debug_idx = 0

    app = QApplication(sys.argv)
    window = ImageWindow()
    window.resize(400, 500)
    window.show()

    cv2.namedWindow("Output")
    cv2.moveWindow(winname='Output', x=1700, y=500)

    save_list = []

    prev_play = False

    while True:
        idx_depth += 1
        if idx_depth == num_depth_count:
            idx_depth = 0
            flag_depth = True
        else:
            flag_depth = False

        ###################### receive input ######################
        result = receive_images(init_variables, flag_depth)
        if result == None:
            print("no input")
            continue

        color, depth = result

        # Display RGBD pair ---------------------------------------------------
        color = cv2.resize(color, dsize=(640, 360), interpolation=cv2.INTER_AREA)
        # cv2.imshow('RGB', color)
        # if flag_depth:
        #     cv2.imshow('Depth', depth / max_depth)  # scale for visibility
        # cv2.waitKey(1)

        ###################### process object ######################

        # flag_hand, result_obj = track_obj.run(np.copy(color), flag_vis=flag_vis_obj)

        ###################### process hand ######################
        # if flag_hand:
        result_hand = track_hand.run(np.copy(color))
        if not isinstance(result_hand, np.ndarray):
            continue

        ###################### visualize ######################
        img_cv = draw_2d_skeleton(color, result_hand)

        cv2.imshow("Output", img_cv)
        cv2.waitKey(1)

        if not flag_init:
            save_dict = {}
            save_dict["color"] = np.copy(color)
            save_dict["depth"] = np.copy(depth)
            save_dict["joint"] = np.copy(result_hand)
            # save_dict["target_class"] = CURRCLASS
            save_list.append(save_dict)

        ## save data when records end and reset the list
        if prev_play and not ISPLAYING and len(save_list) > 1 and not CURRCLASS == -1:
            with open(f'./pkl_test/Subject_{SUBJECT}_Gesture_{CURRCLASS}_{int(Count_gesture[CURRCLASS])}.pickle', 'wb') as f:
                pickle.dump(save_list, f, pickle.HIGHEST_PROTOCOL)
            save_list = []
            Count_gesture[CURRCLASS] += 1
            print("saved pickle")

        prev_play = ISPLAYING


    sock.close()

    # Stop PV and RM Depth AHAT streams ---------------------------------------
    sink_ht, sink_pv = init_variables[0], init_variables[1]
    sink_pv.detach()
    sink_ht.detach()
    producer.stop(hl2ss.StreamPort.PERSONAL_VIDEO)
    producer.stop(hl2ss.StreamPort.RM_DEPTH_AHAT)

    # Stop PV subsystem -------------------------------------------------------
    hl2ss_lnm.stop_subsystem_pv(host, hl2ss.StreamPort.PERSONAL_VIDEO)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main_single()