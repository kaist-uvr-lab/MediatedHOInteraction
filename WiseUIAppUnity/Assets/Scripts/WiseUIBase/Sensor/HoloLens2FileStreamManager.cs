using Microsoft.MixedReality.Toolkit;
using Microsoft.MixedReality.Toolkit.Utilities;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace WiseUI.Base
{
    [Serializable]
    public class HoloLens2FileStreamManager : Singleton<HoloLens2FileStreamManager>
    {
        [SerializeReference]
        HoloLens2PVFileReader _pvCamera;

        [SerializeReference]
        HoloLens2DepthFileReader _depthCamera_longThrow, _depthCamera_shortThrow;

        [SerializeReference]
        BaseSensorReader _imu;

        [SerializeReference]
        BaseSensorReader _gaze;

        public bool enablePVCamera = true;
        public bool enableDepthCamera = true;
        public bool enableIMU;
        public bool enableGaze;



        public HoloLens2PVFileReader PVCamera
        {
            get
            {
                return _pvCamera;
            }
        }
        public HoloLens2DepthFileReader DepthCamera_LongThrow
        {
            get
            {
                return _depthCamera_longThrow;
            }
        }
        public HoloLens2DepthFileReader DepthCamera_ShortThrow
        {
            get
            {
                return _depthCamera_shortThrow;
            }
        }
        public BaseSensorReader IMU
        {
            get
            {
                return _imu;
            }
        }
        public BaseSensorReader Gaze
        {
            get
            {
                return _gaze;
            }
        }
        public HoloLens2FileStreamManager()
        {
            _pvCamera = new HoloLens2PVFileReader();

            _depthCamera_longThrow = new HoloLens2DepthFileReader();
            _depthCamera_shortThrow = new HoloLens2DepthFileReader(true);
        }

        //public void StartEnabledSensors(string dataset_path)
        //{
        //    if(enablePVCamera)
        //    {

        //        _pvCamera.StartPVCamera(dataset_path);
        //    }
        //    if (enableDepthCamera)
        //    {
        //        //_depthCamera = new HoloLens2DepthFileReader();
        //        //_depthCamera.StartDepthCamera(dataset_path);
        //    }

        //}
        //public PVFrame GetFrame(long timestamp)
        //{
        //    if (enablePVCamera)
        //    {
        //        _pvCamera.UpdateLatestTexture(timestamp);
        //        return _pvCamera.GetFrame();
        //    }
        //    else
        //    {
        //        throw new Exception("PVCamera is not enabled.");
        //    }
        //}


    }



}
