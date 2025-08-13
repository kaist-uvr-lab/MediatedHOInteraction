using Microsoft.MixedReality.Toolkit.Utilities;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Photon.Pun;

public class InteractionManager : MonoBehaviour
{
    //SSJ ���� �������뿡 ������ ������ ���� EventRegister���� �ٰŸ� ���Ÿ� �Է� �б⸦ �ٽ� ­��, Interaction Manager�� ���� ���� �ʿ�
    [Header("Scene Objects")]
    private GameObject prevObj;
    private MeshOutline objOutline;
    private bool isObjSelected = false;

    private DeviceManager deviceManager;
    [SerializeField]
    private PhotonView PV;

    // Start is called before the first frame update
    void Awake()
    {
        deviceManager = GetComponent<DeviceManager>();
        PV = GameObject.FindGameObjectWithTag("RPCFunction").GetComponent<PhotonView>();
    }

    public void IsNearInteraction(bool isNearInteraction)
    {
        if (isNearInteraction)
            deviceManager.CurrDevice = DeviceManager.DeviceMode.hybrid;
        else
            deviceManager.CurrDevice = DeviceManager.DeviceMode.hmd;
    }

    public void OnObjSelected(GameObject obj)
    {
        Debug.Log("On OBJ Seleteced");
        if (objOutline != null)
            objOutline.enabled = false;

        if (!isObjSelected)
            isObjSelected = true;
        else
        {
            if (obj == prevObj)
                isObjSelected = false;
        }

        objOutline = obj.GetComponent<MeshOutline>();
        objOutline.enabled = isObjSelected;

        UpdateWebView(isObjSelected, obj.name);

        prevObj = obj;
    }


    private void UpdateWebView(bool val, string str)
    {
        //string url = "https://en.wikipedia.org/wiki/" + str;

        switch (deviceManager.CurrDevice)
        {
            case DeviceManager.DeviceMode.hmd:
                //LaunchWeb(val, url);
                PV.RPC("RPC_ObjClickForWebview", RpcTarget.All, str);
                break;
            case DeviceManager.DeviceMode.hybrid:
                //PV.RPC("ShowOnPhone", RpcTarget.Others, val, url);
                PV.RPC("RPC_ObjClickForWebview", RpcTarget.All, str);
                break;
        }
    }

    private void LaunchWeb(bool val, string str)
    {
        if (val)
            Launch(str);
    }

    private void Launch(string uri)
    {
#if UNITY_WSA
            UnityEngine.WSA.Launcher.LaunchUri(uri, false);
#else
        //Application.OpenURL(uri);
#endif
    }
}
