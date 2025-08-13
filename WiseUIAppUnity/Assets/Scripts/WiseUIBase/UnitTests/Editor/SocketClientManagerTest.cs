using NUnit.Framework;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEngine;
using UnityEngine.TestTools;
using static Unity.Burst.Intrinsics.X86.Avx;



namespace WiseUI.Base
{
    /// <summary>
    /// server, client를 한 프로세스에서 함께 열어서 테스트 할 예정이었으나,
    /// 어떤 방법을 써도 한 프로세스에서 connect, send, receive등이 make sense하게 발생하지 않아서 클라이언트 테스트 코드만 작성함. 서버는 python으로 열어야함.
    /// </summary>
    public class SocketClientManagerTest
    {
        //string host = "143.248.102.68"; // 우진 컴.
        //string host = "143.248.95.25"; // 익범 컴.
        string host = "127.0.0.1";

        int port = 9091;
        int test_count = 100;
        int timeout = 10000 ; //ms
        static readonly double time_benchmark_tcp_only = 0.075;
        static readonly double time_benchmark_websocket = 0.003;
        
        System.Diagnostics.Stopwatch stopwatch = new System.Diagnostics.Stopwatch();
        
        [TearDown]
        public void TearDown()
        {
            SocketClientManager.Instance.Disconnect();
        }

        [Test]
        public void JsonDecodeTest()
        {
            string t1 = "{\"frameInfo\": {\"frameID\": 395, \"timestamp_receive\": 1668847899.4188118, \"timestamp_send\": 1668847899.4287872}, \"objectDataPackage\": {\"objects\": [{\"keypoints\": [{\"id\": 0, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 1, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 2, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 3, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 4, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 5, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 6, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 7, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}], \"id\": 0}, {\"keypoints\": [{\"id\": 0, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 1, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 2, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 3, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 4, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 5, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 6, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 7, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}], \"id\": 1}, {\"keypoints\": [{\"id\": 0, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 1, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 2, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 3, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 4, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 5, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 6, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 7, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}], \"id\": 2}]}, \"handDataPackage\": {\"joints\": [{\"id\": 0, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 1, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 2, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 3, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 4, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 5, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 6, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 7, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 8, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 9, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 10, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 11, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 12, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 13, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 14, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 15, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 16, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 17, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 18, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 19, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}, {\"id\": 20, \"x\": 0.123, \"y\": 0.456, \"z\": 0.789}]}}";
            //Debug.Log(JsonUtility.ToJson(data));
            ResultDataPackage data = JsonUtility.FromJson<ResultDataPackage>(t1);

            //Validate Contents.
            Assert.AreEqual(395, data.frameInfo.frameID);
        }

        [UnityTest]
        public IEnumerator SendPVImageTest()
        {
            SocketClientManager.Instance.Connect(host, port);
            while(!SocketClientManager.Instance.IsConnected())
                yield return null; // wait for 1 frame to make sure the connection is established.

            Assert.True(SocketClientManager.Instance.IsConnected());
            var texture = new Texture2D(640, 360, TextureFormat.RGB24, false);
            stopwatch.Start();
            
            for (int ti = 0; ti < test_count; ti++) // total test_count 만큼 테스트 한다.
            {
                var frameID = DateTime.Now.Ticks;
                //var dataPackage = new RGBImageDataPackage(ti, texture);
                var dataPackage = new RGBImageDataPackage(ti, texture);
                var data = dataPackage.Encode();
                
                var task = SocketClientManager.Instance.SendAsync(data);
                // Wait until the send task is complete
                yield return new WaitUntil(() => task.IsCompleted);

                //Get current time as miliseconds.
                var start = DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond;
                while (true)
                {
                    try
                    {
                        // 1. 가장 최신 데이터를 가져오는 방법.(Queue에서 마지막 데이터를 가져온다. 누락된 frame 생길 수 있지만 delay 없음.)
                        var frameData = SocketClientManager.Instance.GetLatestReceivedData(); // 만약 새로운 frameData가 도착하지 않았다면 NoDataReceivedExecption이 이 함수 내부에서 발생한다.
                        break;
                    }
                    catch (NoDataReceivedExecption e)
                    {
                        //Debug.Log(e.Message);
                        var spentTime = (DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond) - start;

                        if (spentTime > timeout) //무한 루프 방지를 위해 응답이 timeout동안 없으면 실패로 간주한다.
                            Assert.Fail("Time out");
                    }
                }
            }

            int receive_count = SocketClientManager.Instance.GetQueueCount();

            //Debug.LogFormat("Send data count : {0}, Received data count : {1}, ", test_count, receive_count);

            Assert.AreEqual(test_count, receive_count);

            //  calculate average delay time.
            var time = SocketClientManager.Instance.GetAllReceivedData().Select(x => x.frameInfo.GetTotalDelay()).Average();
            Assert.Less(time, time_benchmark_websocket);
             Debug.LogFormat("benchmark_t1: {0}, test: {1}", time_benchmark_websocket, time);

            SocketClientManager.Instance.Disconnect();
        }

        [UnityTest]
        public IEnumerator SendPVImageTest_JPEG()
        {
            SocketClientManager.Instance.Connect(host, port);
            yield return null; // wait for 1 frame to make sure the connection is established.

            Assert.True(SocketClientManager.Instance.IsConnected());

            for (int ti = 0; ti < test_count; ti++) // total test_count 만큼 테스트 한다.
            {
                var frameID = DateTime.Now.Ticks;
                var texture = new Texture2D(640, 360, TextureFormat.RGB24, false);

                var dataPackage = new RGBImageDataPackage(frameID, texture, ImageCompression.JPEG, 75);
                var data = dataPackage.Encode();
                SocketClientManager.Instance.Send(data);
                var start = DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond;

                while (true)
                {
                    try
                    {
                        // 1. 가장 최신 데이터를 가져오는 방법.(Queue에서 마지막 데이터를 가져온다. 누락된 frame 생길 수 있지만 delay 없음.)
                        ResultDataPackage frameData = SocketClientManager.Instance.GetLatestReceivedData(); // 만약 새로운 frameData가 도착하지 않았다면 NoDataReceivedExecption이 이 함수 내부에서 발생한다.
                        ValidateContents(frameData);

                        break;
                    }
                    catch (NoDataReceivedExecption e)
                    {
                        //Debug.Log(e.Message);
                        start = DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond;
                        var spentTime = (DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond) - start;

                        if (spentTime > timeout) //무한 루프 방지를 위해 응답이 timeout동안 없으면 실패로 간주한다.
                            Assert.Fail("Time out");
                    }

                }

            }
            var receive_count = SocketClientManager.Instance.GetQueueCount();
            //Debug.LogFormat("Send data count : {0}, Received data count : {1}, ", test_count, receive_count);
            Assert.AreEqual(test_count, receive_count);

            //  calculate average delay time.
            var time = SocketClientManager.Instance.GetAllReceivedData().Select(x => x.frameInfo.GetTotalDelay()).Average();
            Debug.LogFormat("benchmark_t1: {0}, test: {1}", time_benchmark_websocket, time);
            Assert.Less(time, time_benchmark_websocket);

            SocketClientManager.Instance.Disconnect();
        }

        [UnityTest]
        public IEnumerator SendPointCloudTest()
        {
            SocketClientManager.Instance.Connect(host, port);
            yield return null; // wait for 1 frame to make sure the connection is established.

            //Assert.True(SocketClientManager.Instance.IsConnected());

            for (int ti = 0; ti < test_count; ti++) // total test_count 만큼 테스트 한다.
            {
                var frameID = DateTime.Now.Ticks;
                Vector3[] pointCloud = new Vector3[10000];
                var dataPackage = new PointCloudDataPackage(frameID, pointCloud);
                var data = dataPackage.Encode();
                SocketClientManager.Instance.Send(data);
                // Get current time as miliseconds.
                var start = DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond;

                //while (true)
                //{
                //    try
                //    {
                //        // 1. 가장 최신 데이터를 가져오는 방법.(Queue에서 마지막 데이터를 가져온다. 누락된 frame 생길 수 있지만 delay 없음.)
                //        ResultDataPackage frameData;
                //        SoketClientManager.Instance.GetLatestReceivedData(out frameData); // 만약 새로운 frameData가 도착하지 않았다면 NoDataReceivedExecption이 이 함수 내부에서 발생한다.

                //        // 2. 도착한 순서대로 데이터를 가져오는 방법. (Queue에서 첫번재 데이터를 가져온다. delay 생길 수 있지만 누락된 frame 없음.)
                //        //SocketClientManager.Instance.GetOldestReceivedData(out frameData);

                //        ValidateContents(frameData);
                //        list_received_data.Add(frameData);

                //        break;
                //    }
                //    catch (NoDataReceivedExecption e)
                //    {
                //        //Debug.Log(e.Message);
                //        start = DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond;
                //        var spentTime = (DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond) - start;

                //        if (spentTime > timeout) //무한 루프 방지를 위해 응답이 timeout동안 없으면 실패로 간주한다.
                //            Assert.Fail("Time out");
                //    }

                //}

            }
            //var remain_count = SocketClientManager.Instance.GetQueueCount();
            //Assert.AreEqual(test_count, remain_count);

            ////  calculate average delay time.
            //var time = list_received_data.Select(x => x.frameInfo.GetTotalDelay()).Average();
            //Debug.LogFormat("benchmark_t1: {0}, test: {1}", time_benchmark_tcp_only, time);

            SocketClientManager.Instance.Disconnect();
        }

        void ValidateContents(ResultDataPackage package)
        {
            ///
            /// 여기에 아래와 같이 ResultDataPackage의 내용을 확인하는 코드를 작성하세요.
            ///
            Assert.IsNotNull(package.handDataPackage);
            // Assert.IsNotNull(package.objectDataPackage);
            // Assert.AreEqual(21, package.handDataPackage.joints.Count);

            // var frameInfo = frameData.frameInfo;
            // var handData = frameData.handDataPackage;

            // Debug.LogFormat("frameID : {0}, handdata : {1},", frameData.frameInfo, frameData.handDataPackage.joints.GetType());

            //HandDataPackage handData = frameData.handDataPackage;

            //Debug.LogFormat("handdata joint_0 len : {0},", frameData.handDataPackage.joints.Count);
            //Debug.LogFormat("handdata 0 x : {0},", frameData.handDataPackage.joints[0].u);
            // string str = handData.joints[0].x.ToString();
            // Console.WriteLine(str);
            // var joints_0 = handData["joints_0"];
            // // joints_0[0~20]['u', 'v', 'd']

            // if(handData.Count > 1){
            //     var joints_1 = handData["joints_1"];
            // }
        }
    }
}