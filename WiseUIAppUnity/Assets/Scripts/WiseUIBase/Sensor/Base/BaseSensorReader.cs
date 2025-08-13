using System;
using UnityEngine;


namespace WiseUI.Base
{
    [Serializable]
    public enum HoloLens2SensorType { PVCamera = 0, DepthCamera_LongThrow = 1, DepthCamera_ShortThrow =2, IMU = 3, Hand = 4, EyeGaze = 5 };

    [Serializable]
    public class BaseSensorReader
    {
        [SerializeField]
        protected HoloLens2SensorType _sensorType;
        public HoloLens2SensorType HoloLens2SensorType
        {
            get
            {
                return _sensorType;
            }
        }

        [SerializeField]
        protected long timestamp=0;
        public long Timestamp
        {
            get { return timestamp; }
        }

    }
}
