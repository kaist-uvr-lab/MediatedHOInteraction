import asyncio
import time
from collections import deque

import cv2
import numpy as np
import websockets
import struct

import stopwatch
from stopwatch import StopWatch
from SocketServer.type_definitions import DataFormat, SensorType, HoloLens2PVImageData, HoloLens2DepthImageData, \
    HoloLens2PointCloudData
import json
import threading
import torch
from handtracker.module_SARTE import HandTracker


stopwatch = StopWatch()
stopwatch.start()

track_hand = HandTracker()


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



async def image_handler(websocket, path):
    print("Boot Image Handler")
    while True:
        try:
            recvData = await websocket.recv()
            recvData = recvData[4:]
            stopwatch.stop('receive_data')

            if recvData == b"#Disconnect#":
                print("Disconnected")
                break
            # print('receive : ' + str(recvData[0:50]))

            start_time = time.time()
            # print('Time to receive data : {}, {} fps'.format(time_to_receive, 1 / (time_to_receive + np.finfo(float).eps)))

            header_size = struct.unpack("<i", recvData[0:4])[0]
            bHeader = recvData[4:4 + header_size]
            header = json.loads(bHeader.decode())
            contents_size = struct.unpack("<i", recvData[4 + header_size: 4 + header_size + 4])[0]

            timestamp_sentfromClient = header['timestamp_sentfromClient']
            # time_to_total = (time.time() - timestamp_sentfromClient) + np.finfo(float).eps
            # print('Time to receive : {}, {} fps'.format(time_to_total, 1 / time_to_total))

            contents = recvData[4 + header_size + 4: 4 + header_size + 4 + contents_size]

            sensorType = header['sensorType']

            if sensorType == SensorType.PV:
                instance = HoloLens2PVImageData(header, contents)
            elif sensorType == SensorType.Depth:
                instance = HoloLens2DepthImageData(header, contents)
            elif sensorType == SensorType.PC:
                instance = HoloLens2PointCloudData(header, contents)
            elif sensorType == SensorType.IMU:
                pass

            # time_to_depack = (time.time() - start_time) + np.finfo(float).eps
            # print('Time to depack : {}, {} fps'.format(time_to_depack, 1 / time_to_depack))

            #time.sleep(0.03)
            input = instance.data
            # cv2.imshow("input in server", input)
            # cv2.waitKey(1)

            result_object = None  # track_object.Process(input)
            result_hand = track_hand.Process_single_nomp(input)

            fx, fy, cx, cy = 493.31238, 493.2309, 314.9145, 170.60936

            """ Packing data for sending to hololens """
            resultData = dict()
            resultData['client_id'] = 'camera'
            resultData['frameInfo'] = dict()
            resultData['frameInfo']['frameID'] = 0
            resultData['frameInfo']['timestamp_sentFromClient'] = 0.0
            resultData['objectDataPackage'] = dict()

            resultData['handDataPackage'] = dict()
            resultData['handDataPackage'] = encode_hand_data(result_hand)
            resultData['camInfo'] = dict()
            resultData['camInfo']['fx'] = float(fx)
            resultData['camInfo']['fy'] = float(fy)
            resultData['camInfo']['cx'] = float(cx)
            resultData['camInfo']['cy'] = float(cy)
            resultData['frameInfo']['timestamp_sentFromServer'] = time.time()
            resultData['frameInfo']['delayClientServer'] = resultData['frameInfo']['timestamp_sentFromServer'] - \
                                                           resultData['frameInfo']['timestamp_sentFromClient']

            # resultData['frameInfo'] = instance.encode_frame_info()
            # resultData['objectDataPackage'] = encode_object_data(result_object)

            """ Send data """
            resultBytes = json.dumps(resultData).encode('utf-8')

            await websocket.send(resultBytes)

            # time_to_process = (time.time() - start_time) + np.finfo(float).eps
            # print('Time to process : {}, {} fps'.format(time_to_process, 1 / time_to_process))
            stopwatch.start()

        except websockets.exceptions.ConnectionClosed:
            print("Connection closed.")
            break

async def watch_handler(websocket, path):
    print("Boot Watch Handler")
    while True:
        try:
            recvData = await websocket.recv()

            print(recvData)
            stopwatch.stop('receive_data')

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

async def main():
    start_image_server = websockets.serve(image_handler, None, 9091)
    start_watch_server = websockets.serve(watch_handler, None, 9092)

    await asyncio.gather(start_image_server, start_watch_server)

asyncio.get_event_loop().run_until_complete(main())
asyncio.get_event_loop().run_forever()

# start_image_server = websockets.serve(image_handler, None, 9091)
# asyncio.get_event_loop().run_until_complete(start_image_server)
# asyncio.get_event_loop().run_forever()

# start_watch_server = websockets.serve(watch_handler, None, 9092)
# asyncio.get_event_loop().run_until_complete(start_watch_server)
# asyncio.get_event_loop().run_forever()