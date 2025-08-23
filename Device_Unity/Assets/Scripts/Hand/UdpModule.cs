using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.Experimental.UI;
using System;
using System.Collections.Generic;
using System.Linq;


public class UdpModule : MonoBehaviour
{
    private UdpClient udpClient;
    private Thread receiveThread;
    private int port = 5005;
    public HandModuleTemplate handTemplate;
    string debugText;

    void Start()
    {
        udpClient = new UdpClient(port);
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    private void ReceiveData()
    {
        while (true)
        {
            try
            {
                IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, port);
                byte[] data = udpClient.Receive(ref remoteEndPoint);

                //// (old)if data is string data
                // string input_message = Encoding.UTF8.GetString(data);
                // demoManager.GetInputMessage(input_message);
                // Debug.Log("Received input: " + input_message);             

                //// 250227. receive full hand data. need to debug
                // double[] dataArray = new double[data.Length / 8];
                // Buffer.BlockCopy(data, 0, dataArray, 0, data.Length);

                // Debug.Log("Received data: " + string.Join(", ", dataArray));
                // debugText = string.Join(", ", dataArray);

                // // from python server : dataArray = np.asarray([result_hand.flatten(), float(gesture_idx)], dtype=np.float64)
                // double[] handPose3D = dataArray.Take(63).ToArray();
                // int gestureIdx = (int)Math.Ceiling(dataArray[dataArray.Length - 1]);
                // handTemplate.GetInputMessage(handPose3D, gestureIdx);
            }
            catch (SocketException ex)
            {
                Debug.Log("SocketException: " + ex.Message);
            }
        }
    }

    void OnApplicationQuit()
    {
        if (receiveThread != null)
        {
            receiveThread.Abort();
        }
        udpClient.Close();
    }
}