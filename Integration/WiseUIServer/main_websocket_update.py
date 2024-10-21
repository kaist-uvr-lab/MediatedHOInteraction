import asyncio
import time
from collections import deque

import cv2
import numpy as np
import websockets
import struct

from SocketServer.type_definitions import DataFormat, SensorType, HoloLens2PVImageData, HoloLens2DepthImageData, \
    HoloLens2PointCloudData
import json
from PIL import Image, ImageDraw, ImageFont

from modules import HandTracker_mp, ObjTracker, GestureClassfier, GestureClass
from handtracker.utils.visualize import draw_2d_skeleton


## args ##
flag_gesture = True
gesture_ckpt = "point_history_classifier_4.tflite"
fontObj = ImageFont.truetype(font='C:/Windows/Fonts/ARIAL.ttf', size=30)

fx, fy, cx, cy = 493.31238, 493.2309, 314.9145, 170.60936


track_obj = ObjTracker()
track_hand = HandTracker_mp()
recog_gesture = GestureClassfier(ckpt=gesture_ckpt, len_tip_history=8)

debug_idx = 0

def __decode_data__(recvData):
    recvData = recvData[4:]
    if recvData == b"#Disconnect#":
        print("Disconnected")
        return None
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
        print("IMU input, not defined. Break.")
        instance = None

    # time_to_depack = (time.time() - start_time) + np.finfo(float).eps
    # print('Time to depack : {}, {} fps'.format(time_to_depack, 1 / time_to_depack))

    return instance.data


def __encode_data__(result_hand):

    global debug_idx
    debug_idx += 1
    print("debug_idx : ", debug_idx)
    """ Packing data for sending to hololens """
    resultData = dict()
    # resultData['client_id'] = 'camera'
    resultData['frameInfo'] = dict()
    resultData['frameInfo']['frameID'] = debug_idx
    resultData['frameInfo']['timestamp_sentFromClient'] = 0.0
    # resultData['objectDataPackage'] = dict()

    resultData['handDataPackage'] = dict()
    resultData['handDataPackage'] = encode_hand_data(result_hand)
    # resultData['camInfo'] = dict()
    # resultData['camInfo']['fx'] = float(fx)
    # resultData['camInfo']['fy'] = float(fy)
    # resultData['camInfo']['cx'] = float(cx)
    # resultData['camInfo']['cy'] = float(cy)
    # resultData['frameInfo']['timestamp_sentFromServer'] = time.time()
    # resultData['frameInfo']['delayClientServer'] = 0.0 # resultData['frameInfo']['timestamp_sentFromServer'] -  resultData['frameInfo']['timestamp_sentFromClient']

    # resultData['frameInfo'] = instance.encode_frame_info()
    # resultData['objectDataPackage'] = encode_object_data(result_object)

    # print(resultData)
    """ Send data """
    resultBytes = json.dumps(resultData).encode('utf-8')

    return resultBytes


async def image_handler(websocket, path):
    print("Boot Image Handler")
    while True:
        try:
            print("receiving")
            ## receive data from HMD
            recvData = await websocket.recv()
            print("received")
            ## decode data to img
            input = __decode_data__(recvData)
            if not isinstance(input, np.ndarray):
                break

            cv2.imshow("input in server", input)
            cv2.waitKey(1)

            if input.shape[-1] == 4:
                input = input[:, :, :-1]
            img_height = input.shape[0]
            img_width = input.shape[1]

            ## run object tracker

            obj_center_list = track_obj.process_simple(input)

            ## visualize obj centers
            debug = np.copy(input)
            for center in obj_center_list:
                cv2.circle(debug, center, 5, color=[255, 255, 0], thickness=-1, lineType=cv2.LINE_AA)

            cv2.imshow("output object", debug)
            cv2.waitKey(1)

            ## run hand tracker
            result_hand = track_hand.run(np.copy(input), img_width, img_height)
            if not isinstance(result_hand, np.ndarray):
                continue

            ## visualize hand output
            img_cv = draw_2d_skeleton(input, result_hand)
            cv2.imshow("output hand", img_cv)
            cv2.waitKey(1)
            #
            ## run gesture recognizer if hand is close to object
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
                cv2.imshow("output", output)
                # cv2.imwrite(f"./output/{img_idx}.png", output)
                cv2.waitKey(1)

            ## send data to HMD
            resultBytes = __encode_data__(result_hand)

            print("sending")
            await websocket.send(resultBytes)       ## ... stucks after few seconds.
            del resultBytes
            print("sended")

            # time_to_process = (time.time() - start_time) + np.finfo(float).eps
            # print('Time to process : {}, {} fps'.format(time_to_process, 1 / time_to_process))
        except websockets.exceptions.ConnectionClosed:
            print("Connection closed.")
            break


async def watch_handler(websocket, path):
    print("Boot Watch Handler")
    while True:
        try:
            recvData = await websocket.recv()
            print("here")
            print(recvData)

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
        joint['u'] = round(float(hand_result[id, 0]), 2)
        joint['v'] = round(float(hand_result[id, 1]), 2)
        joint['d'] = float(hand_result[id, 2])
        joints.append(joint)
    handDataPackage['joints'] = joints

    print(joints)

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
    print(debug_idx)
    start_image_server = websockets.serve(image_handler, None, 9091)
    start_watch_server = websockets.serve(watch_handler, None, 9092)

    await asyncio.gather(start_image_server)#, start_watch_server)

asyncio.get_event_loop().run_until_complete(main())
asyncio.get_event_loop().run_forever()

