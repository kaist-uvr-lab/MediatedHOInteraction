import os, sys
import cv2
import numpy as np
import time

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "handtracker_wilor"))

from handtracker.utils.visualize import draw_2d_skeleton
from handtracker_wilor.module_WILOR import HandTracker_wilor
import keyboard



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


os.chdir("./handtracker_wilor")
print("Before:", os.getcwd())

track_hand = HandTracker_wilor()

os.chdir("../")
print("After:", os.getcwd())

"""
[데이터 수집 class]

short clip : 
    - swipe/flicking ~ thumb ~ 4 direction 
    - swipe/flicking ~ index ~ 4 direction 
    
    - double tap/knock ~ thumb 
    - double tap/knock ~ index
    
long clip : 
    - circling ~ thumb ~ 2 direction  
    - natural
        각 action전 정지 상태도 넣고 쥔채로 이리저리 움직이는것도 포함
 
[수집 condition]

- cam~hand view direction : 정면, 대각 4종
- 물체 최소 3종 (사이즈, 형태 다르게)
- on-object/on-plane
- Gesture speed

- 전체 반복 5회해서 FOLD로 활용.


- 거리는 관계없음(crop)
- 특정 키 입력 들어올때 record 시작/끝


"""

## options ##
fingers = ['thumb', 'index']
short_actions = ['Up', 'Down', 'Left', 'Right', 'Tap']
long_actions = ['Clock', 'CClock', 'Natural']


## current stat ##
trial = 0
action = 'Up'
finger_idx = 0       # ['thumb', 'index']

duration = 1.5

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)


save_dir = f'dataset/trial{trial}/{action}_{fingers[finger_idx]}'
os.makedirs(save_dir, exist_ok=True)

flag_recording = False
flag_saved = True

data_list = []
bar_width = 20
bar_color = (0, 0, 255)
height, width = 480, 640

while cap.isOpened():
    t1 = time.time()
    ret, img = cap.read()

    if keyboard.is_pressed('space'):
        time.sleep(0.1)
        flag_recording = not flag_recording

        if flag_recording:
            start_time = time.time()

    if keyboard.is_pressed('esc'):
        print("exit")
        break

    image_rows, image_cols, _ = img.shape  # 640 480

    outs = track_hand.run(np.copy(img))
    if not outs:
        cv2.imshow("rgb", img)
        cv2.waitKey(1)
        continue

    all_right, all_uvds, _, _ = outs

    indices = np.where(np.asarray(all_right) == 1)[0]  ### check. 0: left, 1: right
    if len(indices) == 0:
        continue

    uvd_right = np.squeeze(np.asarray(all_uvds)[indices[0]])
    angle_label = compute_ang_from_joint(uvd_right)

    img = draw_2d_skeleton(img, uvd_right)

    if flag_recording:
        print("recording...")
        now = time.time()
        elapsed = now - start_time

        flag_saved = False

        d = np.concatenate([uvd_right.flatten(), angle_label])
        data_list.append(d)


        # draw
        ratio = 1 - (elapsed / duration)
        bar_height = int(height * ratio)
        x1 = width - bar_width
        y1 = height - bar_height
        x2 = width
        y2 = height
        cv2.rectangle(img, (x1, y1), (x2, y2), bar_color, -1)

        fps = str(1.0 / (time.time() - t1))
        cv2.putText(img, f'Collecting {action}_{fingers[finger_idx]} ... FPS : {fps}', org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow("rgb", img)
        cv2.waitKey(1)

        if elapsed > duration:
            flag_recording = not flag_recording

    elif not flag_recording and not flag_saved:
        print("end recording...saving data len : ", len(data_list))
        flag_saved = True
        created_time = int(time.time())
        np.save(os.path.join(save_dir, f'{created_time}'), np.array(data_list))
        data_list = []
    else:
        cv2.imshow("rgb", img)
        cv2.waitKey(1)



cap.release()
cv2.destroyAllWindows()
