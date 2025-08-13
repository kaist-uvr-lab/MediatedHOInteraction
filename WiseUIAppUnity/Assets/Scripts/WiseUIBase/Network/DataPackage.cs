using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using UnityEngine;

[System.Serializable]
public enum DataFormat
{
    RGBA = 1,
    BGRA = 2, //proper to numpy format.
    ARGB = 3,
    RGB = 4,
}
[System.Serializable]
public enum DataType
{
    UINT8 = 1,
    UINT16 = 2,
    UINT32 = 3,
    UINT64 = 4,
    FLOAT32 = 5,
    FLOAT64 = 6,
}
[System.Serializable]
public enum SensorType
{
    PV = 1,
    Depth = 2,
    PointCloud = 3,
    IMU = 4,
}
[System.Serializable]
public enum ImageCompression
{
    None = 0,
    JPEG = 1,
}

[System.Serializable]
public class FrameHeader
{
    public long frameID;
    public double timestamp_sentfromClient;
    public SensorType sensorType;
    public DataType dataType;
    public long dataLenth;

    public FrameHeader(long frameID, SensorType sensorType, DataType dataType)
    {
        var now = DateTime.Now.ToLocalTime();
        var span = now - new DateTime(1970, 1, 1, 0, 0, 0, 0).ToLocalTime();
        timestamp_sentfromClient = span.TotalSeconds;

        this.frameID = frameID;
        this.sensorType = sensorType;
        this.dataType = dataType;
    }

}

[System.Serializable]
public class RGBImageHeader : FrameHeader
{
    public DataFormat dataFormat;
    public ImageCompression dataCompressionType;
    public int imageQuality;
    public int width;
    public int height;

    public RGBImageHeader(long frameID, DataType dataType,
        DataFormat dataFormat, ImageCompression dataCompressionType, int imageQuality, int width, int height) : base(frameID, SensorType.PV, dataType)
    {
        this.dataFormat = dataFormat;
        this.dataCompressionType = dataCompressionType;
        this.imageQuality = imageQuality;
        this.width = width;
        this.height = height;
    }
}

[System.Serializable]
public class DataPackage
{
    protected byte[] Encode(string sHeader, byte[] contents)
    {
        byte[] bHeader = Encoding.ASCII.GetBytes(sHeader);
        byte[] bHeaderSize = BitConverter.GetBytes(bHeader.Length);

        byte[] bContentsSize = BitConverter.GetBytes(contents.Length);

        int totalSize = 4 + bHeader.Length + 4 + contents.Length; // header size(4 byte) + header + contents size(4 byte) + contents
        byte[] bTotalSizeData = BitConverter.GetBytes(totalSize);

        int finalSize = 4 + totalSize;
        byte[] bFinal = new byte[finalSize]; // totalSize(4 byte) + header size(4 byte) + header + contents size(4 byte) + contents
        System.Buffer.BlockCopy(bTotalSizeData, 0, bFinal, 0, 4);
        System.Buffer.BlockCopy(bHeaderSize, 0, bFinal, 4, 4);
        System.Buffer.BlockCopy(bHeader, 0, bFinal, 4 + 4, bHeader.Length);
        System.Buffer.BlockCopy(bContentsSize, 0, bFinal, 4 + 4 + bHeader.Length, 4);
        System.Buffer.BlockCopy(contents, 0, bFinal, 4 + 4 + bHeader.Length + 4, contents.Length);

        return bFinal;
    }
}
[System.Serializable]
public class RGBImageDataPackage : DataPackage
{
    RGBImageHeader header;
    byte[] contents;
    public RGBImageDataPackage(long frameID, Texture2D texture, ImageCompression imageCompression = ImageCompression.None, int jpgQuality = 100)
    {
        var imageFormat = ConvertTextureFormat2ImageFormat(texture.format);
        byte[] bImage;

        var start_time = DateTime.Now.Millisecond;
        if (imageCompression == ImageCompression.JPEG)
            bImage = texture.EncodeToJPG(jpgQuality);
        else
        {
            bImage = texture.EncodeToPNG();
            //bImage = texture.GetRawTextureData();
        }


        var spentTime = DateTime.Now.Millisecond - start_time;
        //Debug.Log("Time to comp. imge : " + spentTime.ToString());

        //DataType도 자동으로 변환하여 넣어주도록 수정할 것.
        header = new RGBImageHeader(frameID, DataType.UINT8, imageFormat, imageCompression, jpgQuality, texture.width, texture.height);
        contents = bImage;
    }

    public byte[] Encode()
    {
        string sHeader = JsonUtility.ToJson(header);
        return Encode(sHeader, contents);
    }
    
    DataFormat ConvertTextureFormat2ImageFormat(TextureFormat textureFormat)
    {
        switch (textureFormat)
        {
            case TextureFormat.RGBA32:
                return DataFormat.RGBA;
            case TextureFormat.BGRA32:
                return DataFormat.BGRA;
            case TextureFormat.ARGB32:
                return DataFormat.ARGB;
        }
        return DataFormat.RGB;
    }

}
[System.Serializable]
public class PointCloudDataPackage : DataPackage
{
    FrameHeader header;
    byte[] contents;

    public PointCloudDataPackage(long frameID, Vector3[] pointCloud)
    {
        //convert Vector3[] to byte[]
        byte[] bPointCloud = new byte[pointCloud.Length * 3 * 4];
        for (int i = 0; i < pointCloud.Length; i++)
        {
            byte[] bX = BitConverter.GetBytes(pointCloud[i].x);
            byte[] bY = BitConverter.GetBytes(pointCloud[i].y);
            byte[] bZ = BitConverter.GetBytes(pointCloud[i].z);

            System.Buffer.BlockCopy(bX, 0, bPointCloud, i * 3 * 4, 4);
            System.Buffer.BlockCopy(bY, 0, bPointCloud, i * 3 * 4 + 4, 4);
            System.Buffer.BlockCopy(bZ, 0, bPointCloud, i * 3 * 4 + 8, 4);
        }
        header = new FrameHeader(frameID, SensorType.PointCloud, DataType.FLOAT32);
        contents = bPointCloud;
    }
    
    public byte[] Encode()
    {
        string sHeader = JsonUtility.ToJson(header);
        return Encode(sHeader, contents);
    }
}



// 서버에서부터 받은 데이터를 처리하는 클래스
[System.Serializable]
public class ResultDataPackage
{
    public FrameInfo frameInfo = new FrameInfo();
    // public ObjectDataPackage objectDataPackage = new ObjectDataPackage();
    public HandDataPackage handDataPackage = new HandDataPackage();
    // public CamInfo camInfo = new CamInfo();

    public double total_delay;
}

[System.Serializable]
public class CamInfo
{
    public float fx, fy, cx, cy;
}

[System.Serializable]
public class FrameInfo
{
    public long frameID;
    /// <summary>
    /// //홀로렌즈에서 이미지를 서버로 보낸 시점
    /// </summary>
    public double timestamp_sentFromClient;

    /// <summary>
    ///  //서버에서 처리 결과값을 홀로렌즈로 보낸 시점\
    /// </summary>
    public double timestamp_sentFromServer;

    /// <summary>
    ///  // 홀로렌즈에서 서버 처리 결과를 받은 시점.
    /// </summary>
    public double timestamp_receivedFromServer;

    public double GetTotalDelay()
    {
        return timestamp_receivedFromServer - timestamp_sentFromClient;
    }
 
}
[System.Serializable]
public class Joint
{
    public int id;
    public float u, v, d;
    //public float q1, q2, q3, q4;
}
[System.Serializable]
public class HandDataPackage
{
    //필요한 거 정의
    public List<Joint> joints = new List<Joint>();

}

// [System.Serializable]
// public class HandDataPrediction
// {
//     //필요한 거 정의
//     public List<Keypoint> keypoints = new List<Keypoint>();

// }

[System.Serializable]
public class Keypoint
{
    public int id;
    public float x, y, z;
}


[System.Serializable]
public class ObjectInfo
{
    public List<Keypoint> keypoints = new List<Keypoint>();
    public int id;
}


[System.Serializable]
public class ObjectDataPackage
{
    public List<ObjectInfo> objects = new List<ObjectInfo>();
}
