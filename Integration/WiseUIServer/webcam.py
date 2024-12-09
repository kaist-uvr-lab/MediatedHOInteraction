import cv2
import mediapipe as mp
import numpy as np
import time, os


cap = cv2.VideoCapture(0)

save_dir = 'dataset/' + str(action) + '_' + status_str
os.makedirs(save_dir, exist_ok=True)


while cap.isOpened():

    ret, img = cap.read()

    for idx, action in enumerate(actions):
        created_time = int(time.time())
        data_our = []

        start_time = time.time()
        t_prev = time.time()
        while time.time() - start_time < secs_for_action:
            if time.time() - start_time < skip_init_sec:
                flag_save = False
            else:
                flag_save = True

            ## delay for realistic
            cv2.waitKey(30)

            ret, img = cap.read()
            t_recv = time.time()
            # print("t diff : ", t_recv - t_prev)
            t_prev = t_recv

            image_rows, image_cols, _ = img.shape   # 640 480

            ## our tracker process
            t1 = time.time()
            joint = track_hand.Process_single_newroi(np.copy(img))
            # print("track t : ", time.time() - t1)
            angle_label = compute_ang_from_joint(joint, idx)
            d = np.concatenate([joint.flatten(), angle_label])
            if flag_save:
                data_our.append(d)
                cv2.circle(img, (20, 20), 20, (0, 0, 255), 10)
            # print("ours : ", angle_label)

            img = draw_2d_skeleton(img, joint)
            cv2.putText(img, f'Collecting {action.upper()} action... {idx}', org=(10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
            cv2.imshow("result", img)
            cv2.waitKey(1)


            ## mediapipe process
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # result = hands.process(img)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # if result.multi_hand_landmarks is not None:
            #     for res in result.multi_hand_landmarks:
            #         joint = np.zeros((21, 3))
            #         for j, lm in enumerate(res.landmark):
            #             # joint[j] = [lm.x, lm.y, lm.z]#, lm.visibility]
            #
            #             lm_px = mp_drawing._normalized_to_pixel_coordinates_float(lm.x, lm.y, image_cols, image_rows)
            #             joint[j] = [lm_px[0], lm_px[1], lm.z]
            #
            #         angle_label = compute_ang_from_joint(joint)
            #         d = np.concatenate([joint.flatten(), angle_label])
            #         if flag_save:
            #             data_mp.append(d)
            #         # print("mp : ", angle_label)
            #
            #         # vis mp results
            #         mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
            #
            # if flag_save:
            #     cv2.circle(img, (20, 20), 20, (0, 0, 255), 10)
            # cv2.imshow("mp result", img)
            # if cv2.waitKey(1) == ord('q'):
            #     break

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
        # data_mp = np.array(data_mp)
        print("raw our : ", action, data_our.shape)
        # print("raw mp : ", action, data_mp.shape)
        np.save(os.path.join(save_dir, f'raw_our_{action}_{created_time}'), data_our)
        # np.save(os.path.join('dataset', f'raw_mp_{action}_{created_time}'), data_mp)

    break
