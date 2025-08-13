using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DeviceManager : MonoBehaviour
{
    public enum DeviceMode
    {
        hmd,
        hybrid,
    }
    [SerializeField]
    [Header("Device Mode")]
    private DeviceMode currDevice = DeviceMode.hmd;
    public DeviceMode CurrDevice
    {
        get { return currDevice; }
        set { currDevice = value; }
    }
}
