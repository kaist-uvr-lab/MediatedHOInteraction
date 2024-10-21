from datetime import time
import time
import cv2
import numpy as np
import io
from PIL import Image
from stopwatch import StopWatch


class SensorType:
    PV = 1
    Depth = 2
    PC = 3
    IMU = 4

class ImageCompression:
    None_ = 0
    JPEG = 1

class DataFormat:
    RGBA = 1
    BGRA = 2
    ARGB = 3
    RGB = 4

class DataType:
    UINT8 = 1
    UINT16 = 2
    UINT32 = 3
    UINT64 = 4
    FLOAT32 = 5
    FLOAT64 = 6

class HoloLens2SensorData:
    def __init__(self, header):
        self.frameID = header['frameID']
        self.timestamp_sentfromClient = header['timestamp_sentfromClient']
        self.sensorType = header['sensorType']
        self.dataFormat = header['dataFormat']
        self.dataType = get_type(header['dataType'])

    def encode_frame_info(self):
        frame_info = dict()
        frame_info['frameID'] = self.frameID
        frame_info['timestamp_sentFromClient'] = self.timestamp_sentfromClient
        frame_info['timestamp_sentFromServer'] = float(time.time())  # 서버에서 홀로렌즈로 처리 결과를 보낸 시간

        return frame_info

class HoloLens2PVImageData(HoloLens2SensorData):
    def __init__(self, header, raw_data):
        super().__init__(header)
        width = header['width']
        height = header['height']
        dataCompressionType = header['dataCompressionType']
        dataDimension = get_dimension(header['dataFormat'])

        #imageQuality = header['imageQuality']
        # decompress jpg to numpy array

        # stopwatch.start()
        # with Image.open(io.BytesIO(raw_data)) as img:
        #     np_img = np.array(img)
        # stopwatch.stop('pillow')

        stopwatch = StopWatch()
        #stopwatch.start()
        if dataCompressionType == ImageCompression.JPEG:
            np_img = cv2.imdecode(np.frombuffer(raw_data, self.dataType), cv2.IMREAD_UNCHANGED)
            np_img = np.flip(np_img, axis=0)  # 주의 : np_img는 RGB24로 가정함.(4채널 이미지라도 JPG 압축되면 3채널로 변환되기 때문에.)

        else:
            np_img = cv2.imdecode(np.frombuffer(raw_data, self.dataType), cv2.IMREAD_UNCHANGED)
            #np_img = np.frombuffer(raw_data, self.dataType).reshape((height, width, dataDimension))
            np_img = np.flip(np_img, axis=0)  # 주의 : np_img는 RGB24로 가정함.(4채널 이미지라도 JPG 압축되면 3채널로 변환되기 때문에.)

        #stopwatch.stop('imdecode')

        self.intrinsic = np.zeros((3, 3))
        self.extrinsic = np.zeros((4, 4))
        self.data = np_img

        # cv2.imshow("pvimage", np_img)
        # cv2.waitKey(1)
    def encode_frame_info(self):
        return super().encode_frame_info()

class HoloLens2DepthImageData(HoloLens2SensorData):
    def __init__(self, header):
        super().__init__(header)
        self.width = header['width']
        self.height = header['height']

    def encode_frame_info(self):
        return super().encode_frame_info()

class HoloLens2PointCloudData(HoloLens2SensorData):
    def __init__(self, header):
        super().__init__(header)


    def encode_frame_info(self):
        return super().encode_frame_info()

def get_dimension(dataFormat: DataFormat):
    if dataFormat == DataFormat.RGBA or dataFormat == DataFormat.BGRA \
            or dataFormat == DataFormat.ARGB:
        return 4
    elif dataFormat == DataFormat.RGB:
        return 3
    elif dataFormat == DataFormat.U16:
        return 2
    elif dataFormat == DataFormat.U8:
        return 1
    else:
        raise (Exception("Invalid DataFormat Error."))
def get_type(dataType: DataType):
    if dataType == DataType.UINT8:
        return np.uint8
    elif dataType == DataType.UINT16:
        return np.uint16
    elif dataType == DataType.UINT32:
        return np.uint32
    elif dataType == DataType.UINT64:
        return np.uint64
    elif dataType == DataType.FLOAT32:
        return np.float32
    elif dataType == DataType.FLOAT64:
        return np.float64
    else:
        raise (Exception("Invalid DataType Error."))

