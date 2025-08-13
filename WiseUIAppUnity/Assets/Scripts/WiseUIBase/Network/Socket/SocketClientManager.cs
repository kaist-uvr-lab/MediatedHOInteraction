using System;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace WiseUI.Base
{
    public class NoDataReceivedExecption : Exception
    {
        public NoDataReceivedExecption(string message) : base(message)
        {

        }

    }
    [Serializable]
    public class SocketClientManager : Singleton<SocketClientManager>
    {
        Client client;
        
        List<ResultDataPackage> queueResultData = new List<ResultDataPackage>();


        readonly object lock_object = new object();

        bool isNewHandDataReceived;
        
        bool isNewObjectDataReceived;
        
        public bool IsNewHandDataReceived
        {
            get
            {
                lock (lock_object)
                {
                    return isNewHandDataReceived;
                }
            }
            set
            {
                lock (lock_object)
                {
                    isNewHandDataReceived = value;
                }
            }
        }
        
        public bool IsNewObjectDataReceived
        {
            get
            {
                lock (lock_object)
                {
                    return isNewObjectDataReceived;
                }
            }
            set
            {
                lock (lock_object)
                {
                    isNewObjectDataReceived = value;
                }
            }
        }

        public void SetHandDataReceived(bool value)
        {
            lock (lock_object)
            {
                isNewHandDataReceived = value;
            }
        }

        public bool IsConnected()
        {
           if(client != null)
                return client.IsConnected();
            else
                return false;
        }

        /* maxBufferSize
        Currently, it is required for 4096 on one hand 8192 on 2 hands. 
        */
        public void Connect(string serverIP, int serverPort, bool isWebSocket = true)
        {
            IsNewHandDataReceived = false;
            IsNewObjectDataReceived = false;
            queueResultData.Clear();

            if (isWebSocket)
                client = new WebSocketClient();
            else
                client = new TCPClient();
            
            client.Connect(serverIP, serverPort);
            client.BeginReceive(OnReceiveData);
        }

        public void Disconnect()
        {
            IsNewHandDataReceived = false;
            IsNewObjectDataReceived = false;
            queueResultData.Clear();

            SendEncodedData("#Disconnect#");
            Thread.Sleep(100);

            if (client != null)
                client.Disconnect();
        }
        
        public void Send(byte[] buffer)
        {
            if (client != null)
                client.Send(buffer);
        }
        public async Task SendAsync(byte[] buffer)
        {
            if (client != null)
                await client.SendAsync(buffer);
        }
 
        public int GetQueueCount()
        {
            lock (lock_object)
            {
                return queueResultData.Count;
            }
        }
        /*
         * ���� : �Ʒ� �Լ��� main thread���� ȣ��Ǵ� �Լ��� �ƴ�.
         */
        
        public void OnReceiveData(byte[] buffer)
        {
            string receivedDataString = Encoding.ASCII.GetString(buffer, 0, buffer.Length);
            var resultData = JsonUtility.FromJson<ResultDataPackage>(receivedDataString);
            var now = DateTime.Now.ToLocalTime();
            var span = now - new DateTime(1970, 1, 1, 0, 0, 0, 0).ToLocalTime();
            resultData.frameInfo.timestamp_receivedFromServer = span.TotalSeconds;
            //Debug.LogFormat("Total dealy: {0}", latestResultData.frameInfo.GetTotalDelay());
            // Debug.Log("Received Data : " + latestResultData.frameInfo.frameID);

            lock (lock_object)
            {
                isNewHandDataReceived = true;
                isNewObjectDataReceived = true;
                queueResultData.Add(resultData);
            }

            //GetComponent<TrackHand>().ReceiveHandData(resultData.handData);

            //Debug.LogFormat("{0} {1} {2} {3} {4} {5}",
            //    resultData.objectDataPackage.objects.Count,
            //    resultData.objectDataPackage.objects[1].id,
            //    resultData.objectDataPackage.objects[1].keypoints[0].id,
            //    resultData.objectDataPackage.objects[1].keypoints[0].x,
            //    resultData.objectDataPackage.objects[1].keypoints[0].y,
            //    resultData.objectDataPackage.objects[1].keypoints[0].z
            //    );

        }

        /// <summary>
        /// //warning. it's not thread safe!
        /// </summary>
        /// <returns></returns>
        public List<ResultDataPackage> GetAllReceivedData()
        {
            return queueResultData;
        }

        public ResultDataPackage GetLatestReceivedData()
        {
            bool localFlag = false;
            
            lock (lock_object)
            {
                localFlag = isNewHandDataReceived;
                if (localFlag)
                { 
                    isNewHandDataReceived = false;
                    //deep copy
                    var latest = queueResultData[queueResultData.Count - 1];
                    return JsonUtility.FromJson<ResultDataPackage>(JsonUtility.ToJson(latest));
                }
                else
                {
                    throw new NoDataReceivedExecption("No new data received.");
                }
            }
        }

        public ResultDataPackage GetOldestReceivedData()
        {
            if (queueResultData.Count > 0)
            {
                lock (lock_object)
                {
                    var resultDataPackage = queueResultData[0];
                    queueResultData.RemoveAt(0);

                    //deep copy
                    var data = JsonUtility.FromJson<ResultDataPackage>(JsonUtility.ToJson(resultDataPackage));
                    return data;
                }
            }
            else
            {
                throw new NoDataReceivedExecption("No new data received.");
            }
        }
        
        public void SendEncodedData(string message)
        {
            var content = Encoding.UTF8.GetBytes(message);
            var contentSize = BitConverter.GetBytes(content.Length);

            int totalSize = 4 + content.Length;
            byte[] buffer = new byte[totalSize];
            System.Buffer.BlockCopy(contentSize, 0, buffer, 0, 4);
            System.Buffer.BlockCopy(content, 0, buffer, 4, content.Length);
            Send(buffer);

            //byte[] buffer = new byte[content.Length];
            //System.Buffer.BlockCopy(content, 0, buffer, 0, content.Length);
            //Send(contentSize);
            //Send(buffer);
        }

        private void OnDestroy()
        {
            Disconnect();
        }
    }

}

