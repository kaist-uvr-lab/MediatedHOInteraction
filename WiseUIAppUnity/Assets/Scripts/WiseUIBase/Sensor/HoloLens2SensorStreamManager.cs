using Microsoft.MixedReality.Toolkit;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace WiseUI.Base
{

    public class HoloLens2SensorStreamManager : Singleton<HoloLens2SensorStreamManager>
    {
        [SerializeField]
        public PVCameraType pvCameraType = PVCameraType.r640x360xf30;
        public TextureFormat textureFormat = TextureFormat.BGRA32;

        [SerializeField]
        HoloLens2PVCameraReader _pvCamera;
        [SerializeField]
        BaseSensorReader _depthCamera;
        [SerializeField]
        BaseSensorReader _imu;
        [SerializeField]
        BaseSensorReader _gaze;

        [SerializeField]
        public HoloLens2PVCameraReader PVCamera
        {
            get
            {
                return _pvCamera;
            }
        }
        [SerializeField]
        public BaseSensorReader DepthCamera
        {
            get
            {
                return _depthCamera;
            }
        }
        [SerializeField]
        public BaseSensorReader IMU
        {
            get
            {
                return _imu;
            }
        }
        [SerializeField]
        public BaseSensorReader Gaze
        {
            get
            {
                return _gaze;
            }
        }


        public HoloLens2SensorStreamManager()
        {
            //this.pvCameraType = pvCameraType;
            //this.textureFormat = textureFormat;
            _pvCamera = new HoloLens2PVCameraReader();
        }
      
    }
}
