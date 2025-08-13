using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace WiseUI.Base
{
    [Serializable]
    public class HoloLens2PVFileReader : BaseFileReader
    {
        public int imageWidth, imageHeight;
        public Vector2 principalPoint;

        public HoloLens2PVFileReader()
        {
            _sensorType = HoloLens2SensorType.PVCamera;
        }

        public void StartPVCamera(string dataset_path)
        {
            sensorFrames.Clear();
            base.dataset_path = dataset_path;
            DirectoryInfo dinfo = new DirectoryInfo(dataset_path);

            ////Needs to exception handling.
            //string filename_pvtxt = string.Format(@"{0}/{1}_pv.txt", dataset_path, dinfo.Name);

            //get filepath of _pv.txt from dataset_path with linq
            var pvtxt = Directory.GetFiles(dataset_path, "*_pv.txt", SearchOption.AllDirectories).FirstOrDefault();

            List<string> lines_pvtxt = System.IO.File.ReadAllLines(pvtxt).ToList();

            ////Get principal points, image size
            var firstLine = lines_pvtxt[0].Split(',').ToList();
            float[] pp = firstLine.GetRange(0, 2).Select(i => float.Parse(i)).ToArray();
            //principalPoint = new Vector2(pp[0], pp[1]);
            var imageSize = firstLine.GetRange(2, 2).Select(i => int.Parse(i)).ToArray();
            //imageWidth = imageSize[0];
            //imageHeight = imageSize[1];

            var filePathList = lines_pvtxt.GetRange(1, lines_pvtxt.Count - 1);

            // Get principal points, image size from first line.
            imageWidth = imageSize[0];
            imageHeight = imageSize[1];
            principalPoint = new Vector2(pp[0], pp[1]);

            timestamp_start = long.Parse(filePathList[0].Split(',')[0]);
            timestamp_end = long.Parse(filePathList[filePathList.Count - 1].Split(',')[0]);
            
            for (int i = 0; i < filePathList.Count; i++)
            {
                var parts = filePathList[i].Split(',').ToList();
                var timestamp = long.Parse(parts[0]);
                float fx = float.Parse(parts[1]);
                float fy = float.Parse(parts[2]);


                CameraIntrinsic cameraIntrinsic = new CameraIntrinsic(fx, fy, principalPoint.x, principalPoint.y, imageWidth, imageHeight);
                var cam2WorldArray = parts.GetRange(3, parts.Count - 3).Select(i => float.Parse(i)).ToArray();

                var camToWorld = Matrix4x4.identity;
                camToWorld.SetRow(0, new Vector4(cam2WorldArray[0], cam2WorldArray[1], cam2WorldArray[2], cam2WorldArray[3]));
                camToWorld.SetRow(1, new Vector4(cam2WorldArray[4], cam2WorldArray[5], cam2WorldArray[6], cam2WorldArray[7]));
                camToWorld.SetRow(2, new Vector4(cam2WorldArray[8], cam2WorldArray[9], cam2WorldArray[10], cam2WorldArray[11]));
                camToWorld.SetRow(3, new Vector4(cam2WorldArray[12], cam2WorldArray[13], cam2WorldArray[14], cam2WorldArray[15]));

                string texturePath = string.Format("{0}/PV/{1}.png", dataset_path, timestamp);
                var item = new PVFrame(timestamp, cameraIntrinsic, camToWorld, texturePath);

                sensorFrames.Add(timestamp, item);
            }

        }

        public PVFrame GetFrame()
        {
            if(timestamp == 0)
            {
                throw new Exception("timestamp_closest is not set.");
            }

            return sensorFrames[timestamp] as PVFrame;
        }

    }

}


