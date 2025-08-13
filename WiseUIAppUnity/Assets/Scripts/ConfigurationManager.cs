
using UnityEngine;

namespace WiseUI.Base
{
    public class ConfigurationManager : Singleton<ConfigurationManager>
    {
        public string hostIP;
        public int port;

        public PVCameraType pvCameraType;

        string default_hostIP = "192.168.1.147";
        int default_port = 9093;

        PVCameraType defaultPVCameraType = PVCameraType.r640x360xf30;

        public void Load()
        {
            if (PlayerPrefs.HasKey("hostIP"))
                hostIP = PlayerPrefs.GetString("hostIP");
            else
                hostIP = default_hostIP;

            if (PlayerPrefs.HasKey("port"))
                port = PlayerPrefs.GetInt("port");
            else
                port = default_port;

            if (PlayerPrefs.HasKey("pvCameraType"))
                pvCameraType = (PVCameraType)PlayerPrefs.GetInt("pvCameraType");

            else
                pvCameraType = defaultPVCameraType;
        }

        public void Reset()
        {
            PlayerPrefs.DeleteKey("hostIP");
            PlayerPrefs.DeleteKey("port");
            PlayerPrefs.DeleteKey("pvCameraType");

        }
        public void Save(string hostIP, int port, PVCameraType cameraType)
        {
            PlayerPrefs.SetString("hostIP", hostIP);
            PlayerPrefs.SetInt("port", port);
            PlayerPrefs.SetInt("pvCameraType", (int)cameraType);

            PlayerPrefs.Save();
        }
    }


}
