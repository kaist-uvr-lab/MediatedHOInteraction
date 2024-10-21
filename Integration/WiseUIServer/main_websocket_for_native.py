import asyncio
import time
from collections import deque

import cv2
import numpy as np
import websockets
import struct
import base64

import stopwatch
from stopwatch import StopWatch
from SocketServer.type_definitions import DataFormat, SensorType, HoloLens2PVImageData, HoloLens2DepthImageData, \
    HoloLens2PointCloudData
import json
import torch
from handtracker.module_SARTE import HandTracker


# stopwatch = StopWatch()
# stopwatch.start()

track_hand = HandTracker()
connected = {}

Ms = torch.FloatTensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])


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

async def client_handler(websocket, path):
    i=0
    client_id = await websocket.recv()
    if (isinstance(client_id, bytes) == True):
        client_id = client_id.decode('utf-8')
    connected[client_id] = websocket

    print(f"Client {client_id} connected")

    t0_list = []
    t1_list = []
    while True:
        try:
            async for message in websocket:
                t1 = time.time()
                data = json.loads(message)
                client_id = data.get('client_id')
                buffer = data.get('buffer')
                if client_id == 'camera':
                    image_buffer = base64.b64decode(buffer)
                    np_img = cv2.imdecode(np.frombuffer(image_buffer, np.uint8), cv2.IMREAD_UNCHANGED)
                    info = data.get('cameraInfo')
                    timestamp = data.get('timeStamp')

                    # cv2.imshow("input in server", np_img)  # 360, 640
                    # cv2.waitKey(1)
                    t2 = time.time()

                    # print(info)
                    # result_hand = track_hand.Process_single_nomp(np_img)
                    _, _, _, _, fx, fy, cx, cy = info
                    fx, fy, cx, cy = 493.31238, 493.2309, 314.9145, 170.60936


                    # Ks = torch.FloatTensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    # print("info : ", info)
                    result_hand = track_hand.Process_single_nomp(np_img)


                    dummy_data = dict()
                    dummy_data['client_id'] = 'camera'
                    dummy_data['frameInfo'] = dict()
                    dummy_data['frameInfo']['frameID'] = i
                    dummy_data['frameInfo']['timestamp_sentFromClient'] = timestamp
                    dummy_data['objectDataPackage'] = dict()

                    ### dummy data
                    dummy_data['handDataPackage'] = dict()
                    # dummy_data['handDataPackage']['joints'] = list()
                    # for id in range(21):
                    #     joint = dict()
                    #     joint['id'] = int(id)
                    #     joint['u'] = float(0)
                    #     joint['v'] = float(0)
                    #     joint['d'] = float(0)
                    #     dummy_data['handDataPackage']['joints'].append(joint)

                    dummy_data['handDataPackage'] = encode_hand_data(result_hand)

                    dummy_data['camInfo'] = dict()
                    dummy_data['camInfo']['fx'] = float(fx)
                    dummy_data['camInfo']['fy'] = float(fy)
                    dummy_data['camInfo']['cx'] = float(cx)
                    dummy_data['camInfo']['cy'] = float(cy)
                    dummy_data['frameInfo']['timestamp_sentFromServer'] = time.time()
                    dummy_data['frameInfo']['delayClientServer'] = dummy_data['frameInfo']['timestamp_sentFromServer'] - \
                                                                   dummy_data['frameInfo']['timestamp_sentFromClient']
                    send_bytes = json.dumps(dummy_data).encode('utf-8')
                    await connected['hololens'].send(send_bytes)
                    i += 1

                    time_clientToServer = t1 - dummy_data['frameInfo']['timestamp_sentFromClient']
                    time_ServerProcess = dummy_data['frameInfo']['timestamp_sentFromServer'] - t1
                    t0_list.append(time_clientToServer)
                    t1_list.append(time_ServerProcess)

                    # if len(t0_list) > 10:
                    #     time_clientToServer = np.average(np.asarray(t0_list))
                    #     print("avg time_clientToServer : ", time_clientToServer)
                    #     t0_list = []
                    # if len(t1_list) > 10:
                    #     time_ServerProcess = np.average(np.asarray(t1_list))
                    #     print("avg time_ServerProcess : ", time_ServerProcess)
                    #     t1_list = []

                elif client_id == "watch":
                    print(buffer)
                    socket_data = dict()
                    if buffer == "batpon":
                        socket_data['client_id'] = 'watch_btap'
                        socket_data['watch_input'] = buffer
                    elif buffer == "btson":
                        socket_data['client_id'] = 'watch_bts'
                        socket_data['watch_input'] = buffer
                    elif buffer == "stbon":
                        socket_data['client_id'] = 'watch_stb'
                        socket_data['watch_input'] = buffer
                    
                    send_bytes = json.dumps(socket_data).encode('utf-8')
                    # await connected['hololens'].send(send_bytes)



        except websockets.exceptions.ConnectionClosed:
            print("Connection closed.")
            break


def encode_hand_data(hand_result):
    """ Encode hand data to json format """

    """ Example """
    """
    handDataPackage['joints_0']
    handDataPackage['joints_1']
    if the hand is not detected, returns zero value joints 

    currently consider single hand
    """
    handDataPackage = dict()
    joints = list()
    num_joints = 21

    # for joint_uvd in hand_result:
    for id in range(num_joints):
        joint = dict()
        joint['id'] = int(id)
        joint['u'] = float(hand_result[id, 0])
        joint['v'] = float(hand_result[id, 1])
        joint['d'] = float(hand_result[id, 2])
        joints.append(joint)
    handDataPackage['joints'] = joints

    # print(joints)

    return handDataPackage


def encode_object_data(object_result):
    """ Example """
    num_obj = 3
    objectDataPackage = dict()

    objects = list()
    for obj_id in range(num_obj):
        objectInfo = dict()
        keyPoints = list()
        for kpt_id in range(8):
            keyPoint = dict()
            keyPoint['id'] = kpt_id
            keyPoint['x'] = 0.123
            keyPoint['y'] = 0.456
            keyPoint['z'] = 0.789
            keyPoints.append(keyPoint)

        objectInfo['keypoints'] = keyPoints
        objectInfo['id'] = obj_id
        objects.append(objectInfo)

    objectDataPackage['objects'] = objects

    return objectDataPackage


# async def main():
#     start_image_server = websockets.serve(image_handler, None, 9091)
#     start_watch_server = websockets.serve(watch_handler, None, 9092)

#     await asyncio.gather(start_image_server, start_watch_server)

# asyncio.get_event_loop().run_until_complete(main())
# asyncio.get_event_loop().run_forever(main())

async def main():
    async with websockets.serve(client_handler, None, 9091):
     print("Test")
     await asyncio.Future()

asyncio.run(main())