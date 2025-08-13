using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace WiseUI.Base
{
    [Serializable]
    public abstract class BaseFileReader : BaseSensorReader
    {
        [SerializeField]
        protected long timestamp_start, timestamp_end;

        [SerializeField]
        protected string dataset_path;

        [SerializeField]
        protected SerializableDictionary<long, HoloLensSensorFrame> sensorFrames = new SerializableDictionary<long, HoloLensSensorFrame>();

        public long Timestamp_Start
        {
            get { return timestamp_start; }
        }
        public long Timestamp_End
        {
            get { return timestamp_end; }
        }

        public int TotalFrameCount
        {
            get { return sensorFrames.Count; }
        }
  

        public bool IsNewFrame
        {
            get
            {
                return true;
            }
        }
        public void UpdateLatestTexture(long target_timestamp)
        {
            if (target_timestamp < timestamp_start || target_timestamp > timestamp_end)
            {
                throw new Exception("target_timestamp is out of range.");
            }

            var registed_timestamps = sensorFrames.Keys.ToList();

            //find cloest timestamp,  where the timestamp is smaller than current timestamp.
            long cloestTimestamp = 0;
            for (int i = 0; i < registed_timestamps.Count; i++)
            {
                if (registed_timestamps[i] <= target_timestamp && registed_timestamps[i] > cloestTimestamp)
                    cloestTimestamp = registed_timestamps[i];
                else
                    break;
            }

            timestamp = cloestTimestamp;
        }

    }

}
