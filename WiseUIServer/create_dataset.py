import cv2
import mediapipe as mp
import numpy as np
import time, os
from handtracker.module_SARTE import HandTracker
from handtracker.utils.visualize import draw_2d_skeleton
from handtracker_wilor.module_WILOR import HandTracker_wilor


def compute_ang_from_joint(joint, idx):  # joint : (21, 3)
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

    angle_label = np.array([angle], dtype=np.float32)
    angle_label = np.append(angle_label, idx)

    return angle_label

track_hand = HandTracker()

"""
데이터 수집 리스트

Short clip : Up , Down, Left, Right ~ 6회씩 하고 부족한 클래스 추가
Long clip : Clock, C-Clock, Tap, Natural ~ 3회씩, Natural은 5회씩하고 부족한 클래스 추가

Object
Apple, Cup, Key : All action
Pen : Up, Down, Tap


Note
- Natural에는 각 action전 정지 상태도 넣고 쥔채로 이리저리 움직이는것도 포함

- Pen tap은 손등말고 손바닥쪽이 보이는 대각 뷰로 안내해야함.
- Apple circling. index는 다른 손으로 바닥 지지. thumb은 제외.


"""
short_actions = ['Up', 'Down', 'Left', 'Right']
long_actions = ['Clock', 'CClock', 'Tap', 'Natural']


status_speed = 0   # 0: slow, 1: mid, 2:fast
status_dist = 0    # 0: far, 1: mid
status_trial = 9

status_str = str(status_speed)+str(status_dist)+str(status_trial)

actions = []
action = 'Natural'

if action in short_actions:
    flag_short = True
    num_clip = 100
else:
    flag_short = False
    num_clip = 15

for i in range(num_clip):
    actions.append(action)

seq_length = 13 ## need update

if flag_short:
    secs_for_action = 3.5 #long : 15, short : 3.5
    skip_init_sec = 2
else:
    secs_for_action = 15
    skip_init_sec = 2

cap = cv2.VideoCapture(0)

save_dir = 'dataset/' + str(action) + '_' + status_str
os.makedirs(save_dir, exist_ok=True)


while cap.isOpened():
    for idx, action in enumerate(actions):
        created_time = int(time.time())
        data_our = []

        ret, img = cap.read()

        start_time = time.time()
        while time.time() - start_time < secs_for_action:
            if time.time() - start_time < skip_init_sec:
                flag_save = False
            else:
                flag_save = True

            ## delay for realistic
            cv2.waitKey(30)

            ret, img = cap.read()

            image_rows, image_cols, _ = img.shape   # 640 480

            ## our tracker process
            t1 = time.time()
            all_right, all_uvds, _, _ = track_hand.run(np.copy(img))

            indices = np.where(np.asarray(all_right) == 0)[0]  ### check. 0: left, 1: right

            if len(indices) == 0:
                continue

            uvd_right = np.squeeze(np.asarray(all_uvds)[indices[0]])

            angle_label = compute_ang_from_joint(uvd_right, idx)
            d = np.concatenate([uvd_right.flatten(), angle_label])
            if flag_save:
                data_our.append(d)
                cv2.circle(img, (20, 20), 20, (0, 0, 255), 10)
            # print("ours : ", angle_label)

            img = draw_2d_skeleton(img, uvd_right)
            cv2.putText(img, f'Collecting {action.upper()} action... {idx}', org=(10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            cv2.imshow("result", img)
            cv2.waitKey(1)


            if len(data_our) < 2:
                continue

            if len(data_our) < seq_length:
                curr_len = len(data_our)
            else:
                curr_len = seq_length
            curr_seq = np.asarray(data_our[-curr_len:])

            ## visualize input tip history (thumb)
            tip_history = np.squeeze(curr_seq[:, 12:15])
            for tip_idx in range(tip_history.shape[0]):
                tip = tip_history[tip_idx, :2]
                cv2.circle(img, (int(tip[0]), int(tip[1])), int(tip_idx), (255, 255, 0), -1, cv2.LINE_AA)
            tip_history = np.squeeze(curr_seq[:, 24:27])
            for tip_idx in range(tip_history.shape[0]):
                tip = tip_history[tip_idx, :2]
                cv2.circle(img, (int(tip[0]), int(tip[1])), int(tip_idx), (255, 0, 0), -1, cv2.LINE_AA)

            cv2.imshow("result", img)
            cv2.waitKey(1)

        data_our = np.array(data_our)
        print("raw our : ", action, data_our.shape)
        np.save(os.path.join(save_dir, f'raw_our_{action}_{created_time}'), data_our)

    break
