import socket
import time
import numpy as np

# Set the server address and port
# udp_ip = "192.168.1.50"  # Use the local address or adjust if needed
udp_ip = "127.0.0.1"  # Use the local address or adjust if needed
udp_port = 5005

# Create the UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    input_data = input("Enter data to send: ") # up, down, left, right, counter_clock, clock, tap
    print(input_data)
    input_bytes = input_data.encode('utf-8')
    sock.sendto(input_bytes, (udp_ip, 5005))

    # input_data = input("Enter int data to send: ")  # up, down, left, right, counter_clock, clock, tap
    # dummy = np.asarray([0.0, float(input_data)], dtype=np.float64)
    # dummy_bytes = dummy.tobytes()
    # sock.sendto(dummy_bytes, (udp_ip, 5005))

    print("sent input data")
    time.sleep(1)

#
#
# import numpy as np
# from tensorflow.keras.models import load_model
# from collections import deque
#
#
# class GestureClassfier():
#     def __init__(self, img_width=640, img_height=360):
#         self.actions = ['Up', 'Down', 'Left', 'Right', 'Clock', 'CClock', 'Tap', 'Natural']
#         self.seq_length = 13
#         self.model = load_model('./gestureclassifier/model.h5')
#
#         self.seq = deque([], maxlen=15)
#         self.action_list = []
#
#         self.norm_ratio_x = img_width / 2.0
#         self.norm_ratio_y = img_height / 2.0
#         self.tip_history = np.zeros(1)
#
#     def init_que(self):
#         self.seq = deque([], maxlen=15)
#         self.action_list = []
#         self.tip_history = np.zeros(1)
#
#     def run(self, joint):
#         pred_idx = -1
#         if joint[0, 0] == joint[1, 0] and joint[0, 1] == joint[1, 1]:
#             return pred_idx, None
#
#         angle_label = self.compute_ang_from_joint(joint)
#         d = np.concatenate([joint.flatten(), angle_label])
#         self.seq.append(d)
#         if len(self.seq) < self.seq_length:
#             return pred_idx, None
#
#         input_norm, self.tip_history = self.preprocess(self.seq)
#
#         pred = self.model.predict(input_norm, verbose=0).squeeze()
#         pred_idx = int(np.argmax(pred))
#         conf = pred[pred_idx]
#
#         if conf < 0.8:
#             this_action = "Natural"
#             pred_idx = 7
#         else:
#             this_action = self.actions[pred_idx]
#             # print(this_action)
#
#         # print(f"{this_action} with {conf}")
#
#         if len(self.action_list) == 0:
#             self.action_list.append(this_action)
#         elif self.action_list[-1] == this_action:
#             self.action_list.append(this_action)
#         else:
#             self.action_list = []
#
#         if len(self.action_list) > 4:
#             return pred_idx, this_action
#         else:
#             return -1, None
#
#     def preprocess(self, seq):
#         seq_list = [*seq]
#         input_data = np.expand_dims(np.array(seq_list[-self.seq_length:], dtype=np.float32), axis=0)  # (1, 10, 78)
#         input_norm = np.zeros((input_data.shape[0], input_data.shape[1], 51))
#         for frame_idx in range(input_data.shape[1]):
#             target_pose = input_data[0, frame_idx, :63].reshape(21, 3)
#             target_angle = input_data[0, frame_idx, 63:]
#
#             # norm 2d pose
#             if frame_idx == 0:
#                 root_pose = target_pose[0, :]
#             norm_pose = target_pose - root_pose
#             norm_pose[:, 0] = norm_pose[:, 0] / self.norm_ratio_x
#             norm_pose[:, 1] = norm_pose[:, 1] / self.norm_ratio_y
#
#             norm_angle = target_angle / 180.0
#
#             nan_indice = np.argwhere(np.isnan(norm_angle))
#             for nan_idx in nan_indice:
#                 norm_angle[nan_idx[0]] = 0.0
#
#             # update pose and angle
#             norm_pose_cut = norm_pose[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 17], :]
#
#             input_norm[0, frame_idx, :36] = norm_pose_cut.flatten()
#             input_norm[0, frame_idx, 36:] = norm_angle
#
#         tip_history = np.squeeze(input_data[0, :, 12:15])
#
#         return input_norm, tip_history
#
#     def compute_ang_from_joint(self, joint):  # joint : (21, 3)
#         # Compute angles between joints
#         v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
#         v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
#         v = v2 - v1  # [20, 3]
#         # Normalize v
#         v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
#
#         # Get angle using arcos of dot product
#         angle = np.arccos(np.einsum('nt,nt->n',
#                                     v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
#                                     v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]
#
#         angle = np.degrees(angle)  # Convert radian to degree
#
#         angle_label = np.array(angle, dtype=np.float32)
#
#         return angle_label
#
#
# track_gesture = GestureClassfier(img_width=640, img_height=360)
# result_hand = np.ones((21, 3))
#
# while True:
#     gesture_idx, gesture = track_gesture.run(result_hand)