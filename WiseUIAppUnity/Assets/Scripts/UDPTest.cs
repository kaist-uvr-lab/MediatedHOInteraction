using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.Experimental.UI;
using System;
using System.Collections.Generic;



public class UDPTest : MonoBehaviour
{
    private UdpClient udpClient;
    private Thread receiveThread;
    public int port = 5005;
    string debugText;
    public DemoManager demoManager;

    List<string> Gestures = new List<string>(new string[]{"up","down","left","right","clock","cclock","tap"});

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
                //data is string data
                string input_message = Encoding.UTF8.GetString(data);
                demoManager.GetInputMessage(input_message);
                Debug.Log("Received input: " + input_message);             



                // // float list input
                // double[] dataArray = new double[data.Length / 8];
                // Buffer.BlockCopy(data, 0, dataArray, 0, data.Length);
                // Debug.Log("Received data: " + string.Join(", ", dataArray));
                // debugText = string.Join(", ", dataArray);

                // //assume dataArray = [gesture class(float), ...]
                // int gesture = (int)Math.Ceiling(dataArray[1]);
                // demoManager.GetInputMessage(Gestures[gesture]);
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