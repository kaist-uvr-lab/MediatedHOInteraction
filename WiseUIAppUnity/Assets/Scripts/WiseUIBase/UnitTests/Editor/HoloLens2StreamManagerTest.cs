using System.Collections;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using static TMPro.SpriteAssetUtilities.TexturePacker_JsonArray;

namespace WiseUI.Base
{
    public class HoloLens2StreamManagerTest
    {
        string dataset_path;
        string dataset_path_v2;
        long time_interval = 100000; // tick time. = 10 ms.


        [SetUp]
        public void Setup()
        {
            //Total 10 samples (pv, long throw depth, ply, pgm)
            dataset_path = System.IO.Path.Combine(Application.dataPath, "Scripts/WiseUIBase/UnitTests/Editor/Data/SampleDataset/");

            //Total 5 samples (pv, long throw depth, ply, pgm) + (short throw depth, imu, eye gaze, hand)
            dataset_path_v2 = System.IO.Path.Combine(Application.dataPath, "Scripts/WiseUIBase/UnitTests/Editor/Data/SampleDataset_v2/");
        }

        /// <summary>
        /// Test if the HoloLens2FileStreamManager can read the data from the dataset.
        /// </summary>
        /// <returns></returns>
        [UnityTest]
        public IEnumerator HoloLens2PVFileReaderTest()
        {
            var reader = HoloLens2FileStreamManager.Instance.PVCamera;
            reader.StartPVCamera(dataset_path_v2);
            Assert.AreEqual(5, reader.TotalFrameCount);

            //long start_system_time = System.DateTime.Now.Ticks;
            long current_system_time = reader.Timestamp_Start;
            //Debug.Log(string.Format("origin timestamp of recording files : {0} ticks", reader.Timestamp_Start));

            while (current_system_time <= reader.Timestamp_End + time_interval)
            {
                if (reader.IsNewFrame(current_system_time))
                {
                    reader.UpdateLatestTexture(current_system_time);    // reader의 latestTexture를 업데이트 한다.
                    Debug.Log(string.Format("Loaded image file : {0}", reader.Timestamp));
                    PVFrame frame = reader.GetFrame();
                    Color color = frame.Texture.GetPixel(10, 10);     // 만약 카메라로부터 영상이 들어왔으면 픽셀값이 empty값인 0,0,0이 아닐 것이다.(확률적으로)
                    Assert.AreNotEqual(0, frame.timestamp);
                    Assert.AreEqual(760, frame.cameraIntrinsic.imageWidth);
                    Assert.AreNotEqual(color, Color.black);
                }
                current_system_time += time_interval;
                yield return null;
            }
        }


        /// <summary>
        /// Test if the HoloLens2FileStreamManager can read the data from the dataset.
        /// </summary>
        /// <returns></returns>
        [UnityTest]
        public IEnumerator HoloLens2DepthFileReaderTest_LongThrow()
        {
            var reader = HoloLens2FileStreamManager.Instance.DepthCamera_LongThrow;
            reader.StartDepthCamera(dataset_path_v2);
            Assert.AreEqual(5, reader.TotalFrameCount);

            yield return TestHoloLens2DepthFileReader(reader);
        }

        /// <summary>
        /// Test if the HoloLens2FileStreamManager can read the data from the dataset.
        /// </summary>
        /// <returns></returns>
        [UnityTest]
        public IEnumerator HoloLens2DepthFileReaderTest_ShortThrow()
        {
            var reader = HoloLens2FileStreamManager.Instance.DepthCamera_ShortThrow;
            reader.StartDepthCamera(dataset_path_v2);
            Assert.AreEqual(5, reader.TotalFrameCount);

            yield return TestHoloLens2DepthFileReader(reader);
        }

        IEnumerator TestHoloLens2DepthFileReader(HoloLens2DepthFileReader reader)
        {
            long current_system_time = reader.Timestamp_Start;
            //Debug.Log(string.Format("origin timestamp of recording files : {0} ticks", reader.Timestamp_Start));
            while (current_system_time <= reader.Timestamp_End + time_interval)
            {
                if (reader.IsNewFrame(current_system_time))
                {
                    reader.UpdateLatestTexture(current_system_time);    // reader의 latestTexture를 업데이트 한다.
                    Debug.Log(string.Format("Loaded image file : {0}", reader.Timestamp));
                    var frame = reader.GetFrame();

                    Assert.AreNotEqual(0, frame.timestamp);
                    Assert.NotNull(frame.pointCloudMeshInfo);
                    Assert.NotNull(frame.pointCloudMeshInfo.vertices);
                    Assert.AreNotEqual(0, frame.pointCloudMeshInfo.vertices);
                    var go = frame.GetPointCloudGameObject(frame.timestamp.ToString(), 0.1f);
                    Assert.NotNull(go);
                }
                current_system_time += time_interval;
                //Debug.Log(string.Format("after {0} ticks", time_interval));
                yield return null;
            }
        }
        /// <summary>
        /// Test if the HoloLens2FileStreamManager can read the data from the WebCam(Realtime).
        /// </summary>
        /// <returns></returns>
        [UnityTest]
        public IEnumerator HoloLens2PVSensorReaderTest()
        {
            var reader = HoloLens2SensorStreamManager.Instance.PVCamera;
            reader.StartPVCamera(PVCameraType.r640x360xf30, TextureFormat.RGBA32);
            Assert.IsTrue(reader.IsPlaying);

            yield return null; // WebCamTexture는 동작 중이지만, 다음 프레임부터 Webcam으로부터 텍스쳐가 제대로 받아와짐.(이유는 모름)

            long test_time_tick = 10000000; // = 1 sec = 1*E+9 nano sec = 1*E+7 ticks.
            long start_time = System.DateTime.Now.Ticks;

            long spent = 0;
            while (spent < test_time_tick)
            {
                //새로운 프레임이 들어왔는지 체크한다. (내부적으로 캡쳐 쓰레드가 분리되어있기 때문.)
                if (reader.IsNewFrame)
                {
                    //Debug.Log(spent);
                    reader.UpdateLatestTexture();               // reader의 latestTexture를 업데이트 한다.
                    var texture = reader.GetLastestTexture();   // update된 latestTexture를 가져온다. (Update..와 Get..을 나눈 이유가 있으므로 의아하더라도 일단 이 방식대로 사용할 것.)
                    Color color = texture.GetPixel(10, 10);     // 만약 카메라로부터 영상이 들어왔으면 픽셀값이 empty값인 0,0,0이 아닐 것이다.(확률적으로)

                    //Debug.Log(reader.Timestamp);
                    Assert.AreNotEqual(color, Color.black);
                }
                spent = System.DateTime.Now.Ticks - start_time;
                yield return null;
            }
            reader.StopPVCamera();
            Assert.IsFalse(reader.IsPlaying);
        }
        [UnityTest]
        public IEnumerator HoloLens2EyeGazeFileReaderTest()
        {
            var reader = HoloLens2FileStreamManager.Instance.DepthCamera_LongThrow;
            reader.StartDepthCamera(dataset_path_v2);
            Assert.AreEqual(5, reader.TotalFrameCount);

            yield return TestHoloLens2DepthFileReader(reader);
        }
        [UnityTest]
        public IEnumerator HoloLensIMUFileReaderTest()
        {
            var reader = HoloLens2FileStreamManager.Instance.DepthCamera_LongThrow;
            reader.StartDepthCamera(dataset_path_v2);
            Assert.AreEqual(5, reader.TotalFrameCount);

            yield return TestHoloLens2DepthFileReader(reader);
        }
        [UnityTest]
        public IEnumerator HoloLensHandSkeletonFileReaderTest()
        {
            var reader = HoloLens2FileStreamManager.Instance.DepthCamera_LongThrow;
            reader.StartDepthCamera(dataset_path_v2);
            Assert.AreEqual(5, reader.TotalFrameCount);

            yield return TestHoloLens2DepthFileReader(reader);
        }
        // 테스트 실패 시로 WebCam이 종료 되지 않은 경우를 방지.
        [TearDown]
        public void Teardown()
        {
            var reader = HoloLens2SensorStreamManager.Instance.PVCamera;
            if (reader.IsPlaying)
                reader.StopPVCamera();
        }
    }


}
