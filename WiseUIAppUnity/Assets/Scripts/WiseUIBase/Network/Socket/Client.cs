using ARRC.Framework;
using System.Collections;
using System.Collections.Generic;
using System.Net.WebSockets;
using System.Threading.Tasks;
using UnityEngine;

public abstract class Client
{
    public delegate void ReceiveCallback(byte[] buffer);
    
    public abstract void Connect(string serverIP, int serverPort);
    public abstract void Disconnect();
    public abstract void Send(byte[] buffer);

    public abstract Task SendAsync(byte[] buffer);
    
    public abstract void BeginReceive(ReceiveCallback receiveCallback);
    public abstract bool IsConnected();
    

}
