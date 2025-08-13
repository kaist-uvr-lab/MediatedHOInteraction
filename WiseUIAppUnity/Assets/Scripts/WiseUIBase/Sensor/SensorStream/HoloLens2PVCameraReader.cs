using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using UnityEngine;
using UnityEngine.UI;


#if ENABLE_WINMD_SUPPORT
using HoloLens2Stream;
#endif

namespace WiseUI.Base
{
    public enum PVCameraType { r640x360xf30 = 0, r760x428xf30 = 1/*, r1280x720xf30 = 2 */};
    //#define ENABLE_WINMD_SUPPORT
    public class HoloLens2PVCameraReader : BaseSensorReader
    {
        [SerializeField]
        public PVCameraType pvCameraType = PVCameraType.r640x360xf30;
        
        [SerializeField]
        Texture2D latestTexture;

#if ENABLE_WINMD_SUPPORT
        DefaultStream pvCameraStream;
#elif USE_OPENCV
        WebcamOpenCVStream pvCameraStream; //It's psuedo hololens camera for testing.
#else
        WebcamStream pvCameraStream; //It's psuedo hololens camera for testing.
#endif
   
        public void StartPVCamera(PVCameraType pVCameraType, TextureFormat textureFormat = TextureFormat.BGRA32)
        {
            //frameID = 0;

            int width, height;
            if (pvCameraType == PVCameraType.r640x360xf30)
            {
                width = 640;
                height = 360;
            }


            else if (pvCameraType == PVCameraType.r760x428xf30)
            {
                width = 760;
                height = 428;
            }
            else
                throw (new Exception("not supported resolution."));


            DebugText.Instance.lines["Init PV camera"] = "Ready.";

#if ENABLE_WINMD_SUPPORT
                pvCameraStream = new DefaultStream();
                latestTexture = new Texture2D(width, height, TextureFormat.BGRA32, false);
                _ = pvCameraStream.StartPVCamera(width);
#elif USE_OPENCV
                pvCameraStream = new WebcamOpenCVStream();
                latestTexture = new Texture2D(width, height, TextureFormat.RGB24, false);
                pvCameraStream.StartPVCamera(width, height);
#else
                pvCameraStream = new WebcamStream();
                latestTexture = new Texture2D(width, height, TextureFormat.BGRA32, false);
                pvCameraStream.StartPVCamera(pVCameraType, TextureFormat.BGRA32);
#endif
            DebugText.Instance.lines["Init PV camera"] = "Started.";

        }
        public bool IsPlaying
        {
            get
            {
                if (pvCameraStream == null)
                {
                    return false;
                }
                return pvCameraStream.IsPlaying;
            }
        }
        public bool IsNewFrame
        {
            get
            {
                return pvCameraStream.IsNewFrame;
            }

        }


        public int UpdateLatestTexture()
        {
            //frameID++;
            timestamp = System.DateTime.Now.Ticks;
            byte[] frameBuffer = pvCameraStream.GetPVCameraBuffer();
            //DebugText.Instance.lines["frameTexture.Length"] = frameBuffer.Length.ToString();
            latestTexture.LoadRawTextureData(frameBuffer);
            latestTexture.Apply();
            return frameBuffer.Length;
        }

        public Texture2D GetLastestTexture()
        {
            return latestTexture;
        }

        public void StopPVCamera()
        {
            if (pvCameraStream != null)
            {
#if ENABLE_WINMD_SUPPORT
            _ = pvCameraStream.StopPVCamera();
#else
                pvCameraStream.StopPVCamera();
#endif
            }
        }

        public void StopSensorsEvent()
        {
            if (pvCameraStream != null)
            {

            }
        }
        private void OnApplicationFocus(bool focus)
        {
            if (!focus) StopSensorsEvent();
        }

        private void OnDestroy()
        {

            StopPVCamera();

            //if (socket != null && socket.isConnected)
            //    socket.Disconnect();
        }


    }
}

