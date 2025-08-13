using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace WiseUI.Base
{

    [Serializable]
    public class SerializableHoloLensSensorFrame : Dictionary<long, HoloLensSensorFrame>, ISerializationCallbackReceiver
    {
        [SerializeField]
        private List<long> keys = new List<long>();

        [SerializeReference]
        private List<HoloLensSensorFrame> values = new List<HoloLensSensorFrame>();

        // save the dictionary to lists
        public void OnBeforeSerialize()
        {
            keys.Clear();
            values.Clear();
            foreach (KeyValuePair<long, HoloLensSensorFrame> pair in this)
            {
                keys.Add(pair.Key);
                values.Add(pair.Value);
            }
        }

        // load dictionary from lists
        public void OnAfterDeserialize()
        {
            this.Clear();

            if (keys.Count != values.Count)
                throw new System.Exception(string.Format("there are {0} keys and {1} values after deserialization. Make sure that both key and value types are serializable."));

            for (int i = 0; i < keys.Count; i++)
                this.Add(keys[i], values[i]);
        }
    }

    [Serializable]
    public class BaseFileReader : BaseSensorReader
    {
        protected long timestamp_start, timestamp_end;

        protected string dataset_path;


        [SerializeReference]
        protected SerializableHoloLensSensorFrame  sensorFrames = new SerializableHoloLensSensorFrame();

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
        
        public bool IsNewFrame(long target_timestamp)
        {
            if (target_timestamp < timestamp_start)
                return false;
            
            var closest_timestamp = GetCloestTimestamp(target_timestamp);
            
            if (timestamp != closest_timestamp)
                return true;
            
            else
                return false;
       
        }

        public void UpdateLatestTexture(long target_timestamp)
        {
            var closest_timestamp = GetCloestTimestamp(target_timestamp);
            
            timestamp = closest_timestamp;
        }

        public void UpdateLatestTexture()
        {
            //get next timestamp
            var keys = sensorFrames.Keys.ToList();
            var index = keys.IndexOf(timestamp);
            if (index < keys.Count - 1)
                timestamp = keys[index + 1];
            else
                throw new Exception("target_timestamp is out of range");

        }
        
        long GetCloestTimestamp(long target_timestamp)
        {
            if (target_timestamp < timestamp_start)
            {
                throw new Exception("target_timestamp is out of range");
            }
            var registed_timestamps = sensorFrames.Keys.ToList();
            
            //find cloest timestamp,  where the timestamp is smaller than current timestamp.
            long cloestTimestamp = timestamp_start;
            for (int i = 0; i < registed_timestamps.Count; i++)
            {
                if (registed_timestamps[i] <= target_timestamp && registed_timestamps[i] >= cloestTimestamp)
                    cloestTimestamp = registed_timestamps[i];
                else
                    break;
            }

            return cloestTimestamp;
        }
    }

}
