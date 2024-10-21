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


async def image_handler(websocket, path):
    while True:
        recvData = await websocket.recv()
        print('len:', len(recvData))
        recvData = recvData[4:]
        if recvData == b"#Disconnect#":
            print("Disconnected")
            break

        time_to_receive = time.time()
        # datetime.datetime.now()

        # print('Time to receive data : {}, {} fps'.format(time_to_receive, 1 / (time_to_receive + np.finfo(float).eps)))

        header_size = struct.unpack("<i", recvData[0:4])[0]
        bHeader = recvData[4:4 + header_size]
        header = json.loads(bHeader.decode())
        contents_size = struct.unpack("<i", recvData[4 + header_size: 4 + header_size + 4])[0]

        timestamp_sentfromClient = header['timestamp_sentfromClient']
        time_to_total = (time.time() - timestamp_sentfromClient) + np.finfo(float).eps
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

        input = instance.data

        #cv2.imshow("pv", input)
        #cv2.waitKey(1)

        result_object = None  # track_object.Process(input)
        result_hand = None  # track_hand.Process(input)

        ### set sample hand pose ###
        result_hand = []
        sample_uvd = np.ones((21, 3), dtype=np.float32)
        result_hand.append(sample_uvd)

        """ Packing data for sending to hololens """
        resultData = dict()
        resultData['frameInfo'] = instance.encode_frame_info()
        # resultData['objectDataPackage'] = encode_object_data(result_object)
        # resultData['handDataPackage'] = encode_hand_data(result_hand)

        # with open("C:/Woojin/research/sample.json", "w") as json_file:
        #     json.dump(resultData, json_file)
        """ Send data """
        resultBytes = json.dumps(resultData).encode('utf-8')

        time_to_depack = (time.time() - time_to_receive) + np.finfo(float).eps
        # print('Time to depack : {}, {} fps'.format(time_to_depack, 1 / time_to_depack))

        time_to_total = (time.time() - timestamp_sentfromClient) + np.finfo(float).eps
        # print('Time to total : {}, {} fps'.format(time_to_total, 1 / time_to_total))

        """ echo test 용 """
        await websocket.send(resultBytes)
        # await websocket.send("test")


def save_image(image_data):
    image_name = "received_image.png"
    with open(image_name, "wb") as image_file:
        image_file.write(image_data)


start_server = websockets.serve(image_handler, "localhost", 9091)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
