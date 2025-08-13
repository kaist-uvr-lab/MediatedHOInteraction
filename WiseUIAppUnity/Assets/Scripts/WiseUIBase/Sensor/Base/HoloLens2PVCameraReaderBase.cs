using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;


namespace WiseUI.Base
{
    [Serializable]
    public class HoloLens2PVCameraReaderBase : BaseSensorReader
    {
        public virtual void StartPVCamera() { }
        public virtual void StartPVCamera(PVCameraType pVCameraType, TextureFormat textureFormat = TextureFormat.BGRA32) { }
        public virtual void StopPVCamera() { }
        public virtual void UpdateLatestTexture() { }
        public virtual void UpdateLatestTexture(int targetFrameID) { }
        public virtual void UpdateLatestTexture(long targetFrameID) { }
        public virtual PVFrame GetFrame() { return null; }
        public virtual Texture2D GetLastestTexture() { return null; }
        public virtual bool IsNewFrame { get; }
        public virtual int TotalFrameCount { get; }
    }

}
