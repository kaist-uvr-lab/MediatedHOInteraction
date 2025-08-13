using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace WiseUI.Base
{
    [Serializable]
    public class HoloLens2DepthFileReader : BaseFileReader
    {
        bool isShortThrow;
        string depth_data_folder_name;

        public HoloLens2DepthFileReader(bool isShortThrow = false)
        {
            if(isShortThrow)
                _sensorType = HoloLens2SensorType.DepthCamera_ShortThrow;
            else
                _sensorType = HoloLens2SensorType.DepthCamera_LongThrow;

            this.isShortThrow = isShortThrow;

            if (isShortThrow)
                depth_data_folder_name = "Depth AHaT";
            else
                depth_data_folder_name = "Depth Long Throw";
        }

        public void StartDepthCamera(string dataset_path)
        {
            sensorFrames.Clear();
            base.dataset_path = dataset_path;

            string rig2cam_txt = string.Format(@"{0}/{1}_extrinsics.txt", dataset_path, depth_data_folder_name);
            List<string> lines_rig2cam_txt = System.IO.File.ReadAllLines(rig2cam_txt).ToList();

            if(lines_rig2cam_txt.Count != 1)
            {
                var msg = string.Format("The number of lines in {0} is not 1.", rig2cam_txt);
                throw new Exception(msg);
            }

            var rig2camArray = lines_rig2cam_txt[0].Split(',').Select(i => float.Parse(i)).ToArray();
            var rig2cam = Matrix4x4.identity;
            rig2cam.SetRow(0, new Vector4(rig2camArray[0], rig2camArray[1], rig2camArray[2], rig2camArray[3]));
            rig2cam.SetRow(1, new Vector4(rig2camArray[4], rig2camArray[5], rig2camArray[6], rig2camArray[7]));
            rig2cam.SetRow(2, new Vector4(rig2camArray[8], rig2camArray[9], rig2camArray[10], rig2camArray[11]));
            rig2cam.SetRow(3, new Vector4(rig2camArray[12], rig2camArray[13], rig2camArray[14], rig2camArray[15]));


            string rig2world_txt = string.Format(@"{0}/{1}_rig2world.txt", dataset_path, depth_data_folder_name);
            var lines_rig2world_txt = System.IO.File.ReadAllLines(rig2world_txt).ToList();


            // Get timestamp with path.
            timestamp_start = long.Parse(lines_rig2world_txt[0].Split(',')[0]);
            timestamp_end = long.Parse(lines_rig2world_txt[lines_rig2world_txt.Count - 1].Split(',')[0]);
            //timestamp = timestamp_start;

            for (int i = 0; i < lines_rig2world_txt.Count; i++)
            {
                var parts = lines_rig2world_txt[i].Split(',').ToList(); ;
                if(parts.Count != 17)
                {
                    var msg = string.Format("The number of lines in {0} is not 1.", rig2cam_txt);
                    throw new Exception(msg);
                }

                var timestamp = long.Parse(parts[0]);
                var rig2worldArray = parts.GetRange(1, parts.Count - 1).Select(i => float.Parse(i)).ToArray();

                var rig2world = Matrix4x4.identity;
                rig2world.SetRow(0, new Vector4(rig2worldArray[0], rig2worldArray[1], rig2worldArray[2], rig2worldArray[3]));
                rig2world.SetRow(1, new Vector4(rig2worldArray[4], rig2worldArray[5], rig2worldArray[6], rig2worldArray[7]));
                rig2world.SetRow(2, new Vector4(rig2worldArray[8], rig2worldArray[9], rig2worldArray[10], rig2worldArray[11]));
                rig2world.SetRow(3, new Vector4(rig2worldArray[12], rig2worldArray[13], rig2worldArray[14], rig2worldArray[15]));

                var ply_path = string.Format(@"{0}/{1}/{2}.ply", dataset_path, depth_data_folder_name, timestamp);
                var pgm_path = string.Format(@"{0}/{1}/{2}.pgm", dataset_path, depth_data_folder_name, timestamp);

                var item = new DepthFrame(timestamp, rig2cam.inverse, rig2world, ply_path, pgm_path, isShortThrow);
                sensorFrames.Add(timestamp, item);
            }

        }
        public DepthFrame GetFrame()
        {
            if (timestamp == 0)
            {
                throw new Exception("The frame is not exists");
            }
            return (DepthFrame)sensorFrames[timestamp];
        }

    }

}


