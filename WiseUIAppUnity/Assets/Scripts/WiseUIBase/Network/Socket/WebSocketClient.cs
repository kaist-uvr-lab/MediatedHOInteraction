using System;
using System.Collections;
using System.Collections.Generic;
using System.Net.Sockets;
using System.Net;
using UnityEngine;
using System.Threading;
using System.Net.WebSockets;
using System.Text;
using System.Threading.Tasks;
using UnityEngine.Assertions;
using System.Linq;
using Stopwatch = System.Diagnostics.Stopwatch;

public class WebSocketClient : Client
{
    private const int bufferSize = 4096;
    private ClientWebSocket webSocket = new ClientWebSocket();
    ReceiveCallback receiveCallback;
    //private CancellationTokenSource cancellationTokenSource = new CancellationTokenSource();
    private Thread receiveThread;
    Stopwatch stopwatch = new Stopwatch();

    public override void Connect(string serverIP, int serverPort)
    {
        string webSocketURL = string.Format("ws://{0}:{1}", serverIP, serverPort);
        Uri serverUri = new Uri(webSocketURL); // replace with your server URI
        
        webSocket.ConnectAsync(serverUri, CancellationToken.None).Wait();
    }

    public override bool IsConnected()
    {
        if (webSocket != null)
            return webSocket.State == WebSocketState.Open;
        else
            return false;
    }

    public override void Disconnect()
    {
        if (IsConnected())
        {
            if (receiveThread != null)
                receiveThread.Interrupt();

            //cancellationTokenSource.Cancel();
            webSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", CancellationToken.None).Wait();
        }
    }

    public override void BeginReceive(ReceiveCallback receiveCallback)
    {
        receiveThread = new Thread(new ThreadStart(WebSocketReceive));
        this.receiveCallback = receiveCallback;
        receiveThread.Start();
    }

    private void WebSocketReceive()
    {
        ArraySegment<byte> incomingData = new ArraySegment<byte>(new byte[bufferSize]);
        try
        {
            while (webSocket.State == WebSocketState.Open)
            {

                var task = webSocket.ReceiveAsync(incomingData, CancellationToken.None);
                task.Wait();

                if (task.Result.Count > 0)
                {
                    var data = new byte[task.Result.Count];
                    Array.Copy(incomingData.Array, data, task.Result.Count);
                    receiveCallback(data);
                }
            }
        }
        catch (ThreadInterruptedException)
        {
            return;
        }

    }


    public override void Send(byte[] buffer)
    {
        if (IsConnected())
        {
            stopwatch.Restart();
            webSocket.SendAsync(new ArraySegment<byte>(buffer), WebSocketMessageType.Binary, true, CancellationToken.None).Wait();
            stopwatch.Stop();
            DebugText.Instance.lines["Time to send"] = stopwatch.ElapsedMilliseconds.ToString();
        }

    }

    public override async Task SendAsync(byte[] buffer)
    {
        var segment = new ArraySegment<byte>(buffer);

        if (webSocket.State == WebSocketState.Open)
        {
            try
            {
                await webSocket.SendAsync(segment, WebSocketMessageType.Binary, true, CancellationToken.None);
                //Debug.Log($"Message sent: {message}");
            }
            catch (Exception ex)
            {
                DebugText.Instance.lines["WebSocket"] = $"WebSocket send error: {ex.Message}";
            }
        }
        else
        {
            DebugText.Instance.lines["WebSocket"] = "WebSocket is not connected. Current state: " + webSocket.State.ToString();
            // 웹소켓이 연결되지 않은 경우 추가적인 처리를 수행합니다.
        }
    }

}
