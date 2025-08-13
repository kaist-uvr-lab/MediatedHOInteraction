using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace WiseUI.Base
{
    public class WebcamStream
    {
        WebCamTexture webCamTexture;
        Texture2D targetTexture;

        //long _timestamp;           
        //public long Timestamp
        //{
        //    get { return _timestamp; }
        //}
        public bool IsPlaying
        {
            get
            {
                if (webCamTexture == null)
                {
                    return false;
                }
                return webCamTexture.isPlaying;
            }
        }

        public void StartPVCamera(PVCameraType pVCameraType, TextureFormat textureFormat = TextureFormat.BGRA32)
        {
            int target_width, target_heigth, target_fps;

            if (pVCameraType == PVCameraType.r640x360xf30)
            {
                target_width = 640;
                target_heigth = 360;
                target_fps = 30;
            }
            else //if (pVCameraType == PVCameraType.r760x428xf30)
            {
                target_width = 768;
                target_heigth = 428;
                target_fps = 30;
            }

            webCamTexture = new WebCamTexture(WebCamTexture.devices.First<WebCamDevice>().name, target_width, target_heigth, target_fps);
            webCamTexture.Play();
            targetTexture = new Texture2D(webCamTexture.width, webCamTexture.height, textureFormat, false);

        }
        public bool IsNewFrame
        {
            get
            {
            
                return webCamTexture.didUpdateThisFrame;
            }
        }

        public void StopPVCamera()
        {
            if (webCamTexture != null && webCamTexture.isPlaying)
            {
                webCamTexture.Stop();
                webCamTexture = null;
            }
        }

        //public Dictionary<string, Resolution[]> GetAvailableResolutions()
        //{
        //    Dictionary<string, Resolution[]> resolutions = new Dictionary<string, Resolution[]>();

        //    foreach (var dev in WebCamTexture.devices)
        //        resolutions.Add(dev.name, WebCamTexture.devices[0].availableResolutions);

        //    return resolutions;
        //}

        public byte[] GetPVCameraBuffer(/*compression*/)
        {
            int width = webCamTexture.width;
            int height = webCamTexture.height;
            //save the active render texture
            RenderTexture temp = RenderTexture.active;
            
            RenderTexture copiedRenderTexture = new RenderTexture(width, height, 0);
            Graphics.Blit(webCamTexture, copiedRenderTexture, new Material(Shader.Find("FlipShader")));

            RenderTexture.active = copiedRenderTexture;
            //copy to texture 2d
            targetTexture.ReadPixels(new Rect(0, 0, width, height), 0, 0);
            targetTexture.Apply();

            RenderTexture.active = temp;

            //Color[] cdata = webCamTexture.GetPixels();
            //targetTexture.SetPixels(cdata);
            //targetTexture.Apply();

            //get nano seconds.
            //_timestamp = System.DateTime.Now.Ticks;

            return targetTexture.GetRawTextureData();

        }

   
    }

}
