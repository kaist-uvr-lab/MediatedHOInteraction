using System;
using System.Collections;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

public class DNetWebSocketTest : MonoBehaviour
{
    private ClientWebSocket clientWebSocket = new ClientWebSocket();
    private Uri serverUri = new Uri("ws://localhost:9091/"); // replace with your server URI

    private async void Start()
    {
        await ConnectWebSocket();
    }

    private async Task ConnectWebSocket()
    {
        await clientWebSocket.ConnectAsync(serverUri, CancellationToken.None);
        StartCoroutine(WebSocketReceive());
        await WebSocketSend("Hello, WebSocket server");
    }

    private async Task WebSocketSend(string message)
    {
        var bytes = Encoding.UTF8.GetBytes(message);
        await clientWebSocket.SendAsync(new ArraySegment<byte>(bytes), WebSocketMessageType.Text, true, CancellationToken.None);
    }

    private IEnumerator WebSocketReceive()
    {
        ArraySegment<byte> buffer = new ArraySegment<byte>(new byte[1024]);

        while (true)
        {
            WebSocketReceiveResult result = null;

            // Use Task.Run to avoid blocking the main thread
            yield return Task.Run(async () =>
            {
                result = await clientWebSocket.ReceiveAsync(buffer, CancellationToken.None);
            });

            var receivedMessage = Encoding.UTF8.GetString(buffer.Array, 0, result.Count);
            Debug.Log($"Received: {receivedMessage}");
        }
    }

    private async void OnApplicationQuit()
    {
        await clientWebSocket.CloseAsync(WebSocketCloseStatus.NormalClosure, "", CancellationToken.None);
    }
}