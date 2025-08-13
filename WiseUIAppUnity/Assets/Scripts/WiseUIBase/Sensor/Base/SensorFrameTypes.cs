using PointCloudExporter;
using System;
using System.Linq;
using System.Text;
using UnityEngine;
using UnityEngine.Assertions;
using static Unity.Burst.Intrinsics.X86.Avx;

namespace WiseUI.Base
{
    static class CoordinateUtil
    {
        /// <summary>
        /// HoloLens2 ��ǥ�迡�� Unity ��ǥ��� ��ȯ.
        /// ���ν��� ����ȭ�� �ʿ��ϴ�.
        /// </summary>
        /// <param name="src"></param>
        /// <returns></returns>
        public static Matrix4x4 ToUnityCoordinateSystem(this Matrix4x4 src)
        {
            Vector3 pos = src.GetColumn(3);
            Vector3 eulerAngle = src.rotation.eulerAngles;

            eulerAngle.y *= -1.0f;
            eulerAngle.z *= -1.0f;

            Vector3 eulerAngleInRad;
            eulerAngleInRad.x = Mathf.PI * eulerAngle.x / 180.0f;
            eulerAngleInRad.y = Mathf.PI * eulerAngle.y / 180.0f;
            eulerAngleInRad.z = Mathf.PI * eulerAngle.z / 180.0f;

            Matrix4x4 vX;
            vX.m00 = 1.0f; vX.m01 = 0.0f; vX.m02 = 0.0f; vX.m03 = 0.0f;
            vX.m10 = 0.0f; vX.m11 = Mathf.Cos(eulerAngleInRad.x); vX.m12 = -Mathf.Sin(eulerAngleInRad.x); vX.m13 = 0.0f;
            vX.m20 = 0.0f; vX.m21 = Mathf.Sin(eulerAngleInRad.x); vX.m22 = Mathf.Cos(eulerAngleInRad.x); vX.m23 = 0.0f;
            vX.m30 = 0.0f; vX.m31 = 0.0f; vX.m32 = 0.0f; vX.m33 = 1.0f;

            Matrix4x4 vY;
            vY.m00 = Mathf.Cos(eulerAngleInRad.y); vY.m01 = 0.0f; vY.m02 = Mathf.Sin(eulerAngleInRad.y); vY.m03 = 0.0f;
            vY.m10 = 0.0f; vY.m11 = 1.0f; vY.m12 = 0.0f; vY.m13 = 0.0f;
            vY.m20 = -Mathf.Sin(eulerAngleInRad.y); vY.m21 = 0.0f; vY.m22 = Mathf.Cos(eulerAngleInRad.y); vY.m23 = 0.0f;
            vY.m30 = 0.0f; vY.m31 = 0.0f; vY.m32 = 0.0f; vY.m33 = 1.0f;

            Matrix4x4 vZ;
            vZ.m00 = Mathf.Cos(eulerAngleInRad.z); vZ.m01 = -Mathf.Sin(eulerAngleInRad.z); vZ.m02 = 0.0f; vZ.m03 = 0.0f;
            vZ.m10 = Mathf.Sin(eulerAngleInRad.z); vZ.m11 = Mathf.Cos(eulerAngleInRad.z); vZ.m12 = 0.0f; vZ.m13 = 0.0f;
            vZ.m20 = 0.0f; vZ.m21 = 0.0f; vZ.m22 = 1.0f; vZ.m23 = 0.0f;
            vZ.m30 = 0.0f; vZ.m31 = 0.0f; vZ.m32 = 0.0f; vZ.m33 = 1.0f;

            Matrix4x4 vR3 = vY * vX * vZ;
            Vector4 vP = new Vector4(-pos.x, pos.y, pos.z, 1);

            Matrix4x4 dst = vR3;
            dst.SetColumn(3, vP);

            return dst;
        }
    }

    [Serializable]
    public class HoloLensSensorFrame
    {
        public HoloLens2SensorType sensorType;

        public long timestamp;
    }

    [Serializable]
    public class CameraFrame : HoloLensSensorFrame
    {
        [SerializeField]
        protected Matrix4x4 cam2rig;
        [SerializeField]
        protected Matrix4x4 rig2world;
        [SerializeField]
        protected Matrix4x4 cam2world;

        public Matrix4x4 Cam2World_MSCoordinate()
        {
            return cam2world;
        }
    }

    [Serializable]
    public class CameraIntrinsic
    {
        public Vector2 focalLength;
        public Vector2 principalPoint;
        public int imageWidth;
        public int imageHeight;

        public CameraIntrinsic(float fx, float fy, float ppx, float ppy, int width, int height)
        {
            focalLength = new Vector2(fx, fy);
            principalPoint = new Vector2(ppx, ppy);
            imageWidth = width;
            imageHeight = height;
        }
        public Matrix4x4 ToMatrix()
        {
            //Set intrinsic.
            Matrix4x4 intrinsic = Matrix4x4.identity;
            intrinsic.m00 = focalLength.x;
            intrinsic.m11 = focalLength.y;
            intrinsic.m02 = principalPoint.x;
            intrinsic.m12 = principalPoint.y;

            return intrinsic;
        }
    }

    [Serializable]
    public class PVFrame : CameraFrame
    {
        string texturePath;

        Texture2D _texture;

        [SerializeField]
        public CameraIntrinsic cameraIntrinsic;

        public PVFrame(long timestamp, CameraIntrinsic cameraIntrinsic, Matrix4x4 cam2world, string texturePath)
        {
            base.timestamp = timestamp;
            base.cam2world = cam2world;

            sensorType = HoloLens2SensorType.PVCamera;
            this.cameraIntrinsic = cameraIntrinsic;
            this.texturePath = texturePath;
        }

        public Texture2D Texture
        {
            get
            {
                if (_texture == null)
                    _texture = TextureIO.LoadTexture(texturePath);

                return _texture;
            }
        }

        public string TexturePath
        {
            get
            {
                return texturePath;
            }
        }
        public Matrix4x4 Cam2World()
        {
            var result = cam2world.ToUnityCoordinateSystem() * Matrix4x4.Rotate(Quaternion.Euler(0, 180, 0));
            return result;
        }

        public byte[] Encode2DataPacakge(ImageCompression imageCompression = ImageCompression.None, int jpgQuality = 100)
        {
            var dataPackage = new RGBImageDataPackage(timestamp, Texture, imageCompression, jpgQuality);
            return dataPackage.Encode();
        }

        //public byte[] EncodeToBytes(bool compress = false, int jpgQuality = 75)
        //{
        //    var imageFormat = ConvertTextureFormat2ImageFormat(texture.format);
        //    byte[] bImage;
        //    if (compress)
        //        bImage = texture.EncodeToJPG(jpgQuality);
        //    else
        //        bImage = texture.GetRawTextureData();

        //    var header = new RGBImageHeader(timestamp, bImage.Length, DataType.UINT8, imageFormat, comp, jpgQuality, texture.width, texture.height);

        //    byte[] bHeader = Encoding.ASCII.GetBytes(sHeader);
        //    byte[] bHeaderSize = BitConverter.GetBytes(bHeader.Length);


        //    return null;
        //}
    }


    [Serializable]
    public class DepthFrame : CameraFrame
    {
        [SerializeField]
        public MeshInfos _pointCloudMeshInfo;
        public GameObject _pointCloud_GameObject;

        string ply_path; // point cloud�� ���� ply ���� ���. (vertex �Ӹ� �ƴ϶� color, normal ���� ������ �����ϰ� ����)
        string pgm_path; // point cloud�� ���� raw data. (peudo depth ���� ��ϵ� raw data, ���� depth���� ��ȯ�� HoloLens2ForCV�� �����ؾ���.)

        public DepthFrame(long timestamp, Matrix4x4 cam2rig, Matrix4x4 rig2world, string ply_path, string pgm_path, bool shortThrow)
        {
            if (shortThrow)
                sensorType = HoloLens2SensorType.DepthCamera_ShortThrow;
            else
                sensorType = HoloLens2SensorType.DepthCamera_LongThrow;

            base.timestamp = timestamp;
            base.cam2rig = cam2rig;
            base.rig2world = rig2world;
            this.pgm_path = pgm_path;
            this.ply_path = ply_path;

            cam2world = rig2world * cam2rig; // * ���⿡ camera ��ǥ�迡���� point�� ������.
        }


        /// <summary>
        /// ���� depth frame�� �̿��Ͽ� ������ MeshInfo.
        /// </summary>
        public MeshInfos pointCloudMeshInfo
        {
            get
            {
                if (_pointCloudMeshInfo == null)
                    _pointCloudMeshInfo = PointCloudGenerator.LoadPointCloud(ply_path);

                return _pointCloudMeshInfo;
            }
        }

        /// <summary>
        /// ���� depth frame�� depth�����͸� �̿��Ͽ� point cloud�� gameobject�� ����.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="pointSize"></param>
        /// <returns></returns>
        public GameObject GetPointCloudGameObject(string name, float pointSize)
        {
            if (_pointCloud_GameObject == null)
            {
                _pointCloud_GameObject = pointCloudMeshInfo.ToGameObject(name, pointSize);
            }

            return _pointCloud_GameObject;
        }

        /// <summary>
        /// Cam2World Matrix (����Ƽ ��ǥ�迡��). �� extrinsic parameter�� inverse��.
        /// </summary>
        /// <returns></returns>
        public Matrix4x4 Cam2World()
        {
            var result = cam2world.ToUnityCoordinateSystem() * Matrix4x4.Rotate(Quaternion.Euler(0, 0, 180));
            return result;
        }


        /// <summary>
        /// Depth image�� PVFrame�� intrinsic�� extrinsic�� �̿��Ͽ� projection �Ѵ�.
        /// note : 1. project�� �ҿ�Ǵ� �ð��� �� ū��, �� �� shader�� �̿��Ͽ� ó���ϴ� ������� ������ �ʿ䰡 ����.
        /// note : 2. .ply������ �ε��ϸ鼭 vertex���� �Ӹ� �ƴ϶�, normal, color� �Բ� �ε��ϱ� ������ ��ȿ������. ���߿� ������ �ʿ䰡 ����.
        /// </summary>
        /// <param name="frame_pv"></param>
        /// <returns>
        /// �Է� "frame_pv"�� ���� �ȼ� �� ���� �� (0�� hull ��, ��ǥ�� ������ data[x,y]�� (x�� ����, y�� ����)).
        /// </returns>
        public float[,] Project2PVFrame(PVFrame frame_pv)
        {
            var pointCloud = pointCloudMeshInfo.vertices;

            Assert.AreNotEqual(0, pointCloud.Length);
            //Debug.Log(string.Format("Time to gen pc. :{0} ms", (DateTime.Now.Millisecond - start_time)));

            //start_time = DateTime.Now.Millisecond;
            var pointCloud_homo = pointCloud.Select(i => new Vector4(i.x, i.y, i.z, 1));
            var pointCloud_mv = pointCloud_homo.Select(i =>frame_pv.Cam2World().inverse * i).ToArray();
            var pointCloud_projected= pointCloud_mv.Select(i => (frame_pv.cameraIntrinsic.ToMatrix() * i)).ToArray();
            var pointCloud_projected_uv = pointCloud_projected.Select(i => new Vector2(i.x / i.z, i.y / i.z)).ToArray();
            Assert.AreNotEqual(0, pointCloud_projected.Count(i => i.x >= 0 && i.x < frame_pv.cameraIntrinsic.imageWidth && i.y >= 0 && i.y < frame_pv.cameraIntrinsic.imageHeight));
            //Debug.Log(string.Format("Time to project pc. : {0} ms", (DateTime.Now.Millisecond - start_time)));

            float[,] depthData = new float[frame_pv.cameraIntrinsic.imageWidth, frame_pv.cameraIntrinsic.imageHeight];

            // initialize depthData to 0.
            for (int i = 0; i < frame_pv.cameraIntrinsic.imageWidth; i++)
                for (int j = 0; j < frame_pv.cameraIntrinsic.imageHeight; j++)
                    depthData[i, j] = 0;

            for (int i = 0; i < pointCloud_projected_uv.Length; i++)
            {
                var p = pointCloud_projected_uv[i];
                if (p.x >= 0 && p.x < frame_pv.cameraIntrinsic.imageWidth && p.y >= 0 && p.y < frame_pv.cameraIntrinsic.imageHeight)
                    depthData[(int)p.x, (int)p.y] = pointCloud_mv[i].z;
            }
            Assert.AreEqual(depthData.Length, frame_pv.cameraIntrinsic.imageWidth * frame_pv.cameraIntrinsic.imageHeight);

            return depthData;
        }

    }

    [Serializable]
    public class EyeGazeFrame : HoloLensSensorFrame
    {
        //public Matrix4x4 imageToWorldtransform;
    }

    [Serializable]
    public class IMUFrame : HoloLensSensorFrame
    {

    }

    [Serializable]
    public class HLHandFrame : HoloLensSensorFrame
    {

    }

}

