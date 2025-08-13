using System;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading.Tasks;
using UnityEngine;

public class TCPClient : Client
{
    readonly int maxBufferSize = 8192;
    Socket socket;
    byte[] receiveBuffer;
    EndPoint remoteEP;
    ReceiveCallback receiveCallback;
    
    public override void Connect(string serverIP, int serverPort)
    {
        socket = new Socket(AddressFamily.InterNetwork, SocketType.Stream, ProtocolType.Tcp);
        socket.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.SendTimeout, 1000);

        remoteEP = new IPEndPoint(IPAddress.Parse(serverIP), serverPort);

        socket.BeginConnect(remoteEP, new AsyncCallback(ConnectCallback), null);
    }
  
    public override bool IsConnected()
    {
        if (socket == null)
            return false;

        return socket.Connected;
    }

    private void ConnectCallback(IAsyncResult ar)
    {
        try
        {
            socket.EndConnect(ar);
        }
        catch (SocketException e)
        {
            Debug.LogError(e.Message);
        }
    }

    public override void Send(byte[] buffer)
    {
        if (socket != null)
        {
            try
            {
                socket.BeginSend(buffer, 0, buffer.Length, 0, new AsyncCallback(SendCallback), null);

                //socket.Send(buffer);
            }
            catch (SocketException e)
            {
                Debug.LogException(e);
            }
            catch (ObjectDisposedException e)
            {
                //Debug.LogException(e);
            }
        }
    }

    private void SendCallback(IAsyncResult ar)
    {
        try
        {
            // Complete sending the data to the remote device.
            int bytesSent = socket.EndSend(ar);
        }
        catch (Exception e)
        {
            Debug.LogException(e);
        }
    }

    public override void BeginReceive(ReceiveCallback callback)
    {
        receiveBuffer = new byte[maxBufferSize];
        socket.BeginReceiveFrom(receiveBuffer, 0, maxBufferSize, SocketFlags.None, ref remoteEP, OnDataReceive, null);
        receiveCallback = callback;
    }
    
    void OnDataReceive(IAsyncResult ar)
    {
        try
        {
            int bytesRead = socket.EndReceive(ar);
            if (bytesRead > 0)
            {
                byte[] buffer = new byte[bytesRead];
                Array.Copy(receiveBuffer, buffer, bytesRead);
                receiveCallback(buffer);
            }
            socket.BeginReceiveFrom(receiveBuffer, 0, receiveBuffer.Length, SocketFlags.None, ref remoteEP, OnDataReceive, null);
        }
        catch (SocketException e)
        {
            //Debug.LogException(e);
        }
        catch (ObjectDisposedException e)
        {

        }

    }

    public override void Disconnect()
    {
        if (socket != null)
        {
            socket.Close();
            socket.Dispose();
        }
    }

    public override Task SendAsync(byte[] buffer)
    {
        throw new NotImplementedException();
    }
}
