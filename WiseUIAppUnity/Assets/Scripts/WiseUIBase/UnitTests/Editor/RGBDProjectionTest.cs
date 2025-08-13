using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace WiseUI.Base
{
    public class RGBDProjectionTest
    {
        string dataset_path;
        string dataset_path_v2;
        long time_interval = 100000; // tick time. = 10 ms.

        [SetUp]
        public void Setup()
        {
            dataset_path = System.IO.Path.Combine(Application.dataPath, "Scripts/WiseUIBase/UnitTests/Editor/Data/SampleDataset/");
            dataset_path_v2 = System.IO.Path.Combine(Application.dataPath, "Scripts/WiseUIBase/UnitTests/Editor/Data/SampleDataset_v2/");
        }

        [UnityTest]
        public IEnumerator Project2PVFrameTest()
        {
            var pvCamera = HoloLens2FileStreamManager.Instance.PVCamera;
            pvCamera.StartPVCamera(dataset_path_v2);
            Assert.AreEqual(5, pvCamera.TotalFrameCount);

            var depthCamera = HoloLens2FileStreamManager.Instance.DepthCamera_ShortThrow;
            depthCamera.StartDepthCamera(dataset_path_v2);

            long current_system_time = pvCamera.Timestamp_Start;

            while (current_system_time <= pvCamera.Timestamp_End + time_interval)
            {
                if (pvCamera.IsNewFrame(current_system_time))
                {
                    try
                    {
                        pvCamera.UpdateLatestTexture(current_system_time);
                        depthCamera.UpdateLatestTexture(current_system_time);
                    }
                    catch (System.Exception e)
                    {
                        current_system_time += time_interval;
                        continue;
                        // 만약 둘다 로드 되지 않으면 다음으로 넘어감.
                        // (첫 프레임의 경우 센서의 캡쳐 시간 차에 의해, 둘 중 하나는 아직 캡쳐되지 않은 경우도 있음.)
                    }
                    Debug.Log(string.Format("Loaded image file : {0}", depthCamera.Timestamp));
                    //int start_time = DateTime.Now.Millisecond;
                    var frame_pv = pvCamera.GetFrame();
                    var frame_depth = depthCamera.GetFrame();

                    float[,] projected_pointcloud = frame_depth.Project2PVFrame(frame_pv);
                   
                    //만약 결과를 보고 싶다면, 아래 주석 해제.(asset 폴더 내에 저장됨.)
                    //WriteDepthDataToImage(projected_pointcloud_texture, frame_pv.texture, frame_depth.timestamp.ToString());
                }
                current_system_time += time_interval;
                yield return null;
            }
            
            void WriteDepthDataToImage(float[,] projected_pointcloud, Texture2D texture_to_overay, string filename)
            {
                var texture_cpy = new Texture2D(texture_to_overay.width, texture_to_overay.height, TextureFormat.RGBA32, false);
                texture_cpy.SetPixels(texture_to_overay.GetPixels());
                texture_cpy.Apply();

                // Render data to pv frame.
                for (int i = 0; i < texture_cpy.width; i++)
                {
                    for (int j = 0; j < texture_cpy.height; j++)
                    {
                        var depth = projected_pointcloud[i, j];
                        if (depth != 0)
                        {
                            texture_cpy.SetPixel(i, j, new Color(depth, depth, depth));
                            //Debug.Log(string.Format("depth : {0}", depth));
                            //Assert.Fail();
                        }
                    }
                }
                texture_cpy.Apply();

                // Write to file.
                var bytes = texture_cpy.EncodeToPNG();
                System.IO.File.WriteAllBytes(System.IO.Path.Combine(Application.dataPath, filename +".png"), bytes);
            }

        }
    }
}
