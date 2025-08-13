#if UNITY_EDITOR
using Codice.Client.BaseCommands;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using UnityEditor.Experimental.GraphView;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.TestTools;
using WebSocketSharp;
//using System.Net.WebSockets;
using WiseUI.Base;
using Stopwatch = System.Diagnostics.Stopwatch;
public class UnityWebRequestTest
{
    const int test_count = 100;
    string uploadURL = "http://localhost:9090/upload";
    string webSocketURL = "ws://localhost:9091";
    int count_recieved = 0;
    Texture2D texture = new Texture2D(640, 360, TextureFormat.RGB24, false);
    Stopwatch stopwatch = new Stopwatch();
    List<ResultDataPackage> resultDataPackages = new List<ResultDataPackage>();
    DateTime lastRecivedTime = DateTime.Now;
    
    [UnityTest]
    public IEnumerator UnityWebRequestSimpleTest()
    {
        stopwatch.Start();
        for (int i = 0; i < test_count; i++)
        {
            byte[] textureBytes = texture.EncodeToPNG();

            List<IMultipartFormSection> formData = new List<IMultipartFormSection>();
            formData.Add(new MultipartFormFileSection("image", textureBytes, "image.png", "image/png"));


            UnityWebRequest request = UnityWebRequest.Post(uploadURL, formData);
            yield return request.SendWebRequest();


            if (request.result == UnityWebRequest.Result.ConnectionError || request.result == UnityWebRequest.Result.ProtocolError)
            {
                Debug.LogError("Error: " + request.error);
            }
            else
            {
                //Debug.Log("Image sent! Server response: " + request.downloadHandler.text);
            }
        }
        stopwatch.Stop();
        double averageTime = stopwatch.Elapsed.TotalMilliseconds / test_count;
        Debug.Log("Average time per image: " + averageTime + " ms");
    }

    //[UnityTest]
    //public IEnumerator WebSocketClientTest()
    //{
    //    var webSocket = new WebSocketSharp.WebSocket(webSocketURL);
    //    webSocket.OnMessage += OnMessage;
    //    webSocket.Connect();
        
    //    while (!webSocket.IsAlive)
    //    {
    //        yield return null;
    //    }
    //    stopwatch.Start();
    //    for (int i = 0; i < test_count; i++)
    //    {
    //        var dataPackage = new RGBImageDataPackage(i, texture);
    //        webSocket.Send(dataPackage.Encode());
    //    }
    //    stopwatch.Stop();
    //    double averageTime = stopwatch.Elapsed.TotalMilliseconds / test_count;
    //    Debug.Log("Time to send : " + averageTime + " ms");

    //    Thread.Sleep(10000);
    //    Debug.LogFormat("Send data count : {0}, Received data count : {1}, ", test_count, resultDataPackages.Count);
        
    //    var time = resultDataPackages.Select(x => x.frameInfo.GetTotalDelay()).Average();
    //    Debug.LogFormat("benchmark_t1: {0}, test: {1}", 0, time);
        
    //    //double averageTime = stopwatch.Elapsed.TotalMilliseconds / test_count;
    //    //Debug.Log("Average time per image: " + averageTime + " ms");

    //    if (webSocket != null)
    //    {
    //        webSocket.Close();
    //    }
    //}

    private void OnMessage(object sender, MessageEventArgs e)
    {
        var buffer = e.RawData;
        string receivedDataString = Encoding.ASCII.GetString(buffer, 0, buffer.Length);
        var latestResultData = JsonUtility.FromJson<ResultDataPackage>(receivedDataString);
        var now = DateTime.Now.ToLocalTime();
        var span = now - new DateTime(1970, 1, 1, 0, 0, 0, 0).ToLocalTime();
        latestResultData.frameInfo.timestamp_receivedFromServer = span.TotalSeconds;
        Debug.LogFormat("Total dealy: {0}", latestResultData.frameInfo.GetTotalDelay());
        
        var span2 = now - lastRecivedTime;
        //Debug.LogFormat("Total dealy: {0}", span2.TotalMilliseconds);
        lastRecivedTime = now;
        
        resultDataPackages.Add(latestResultData);
        

    }

}
#endif