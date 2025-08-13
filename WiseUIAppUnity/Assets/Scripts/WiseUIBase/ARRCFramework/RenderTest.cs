using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using WiseUI.Base;

public class RenderTest : MonoBehaviour
{

    public void Rendering()
    {
        
        var pvFrame = HoloLens2FileStreamManager.Instance.PVCamera.GetFrame();
        Matrix4x4 cameraMa = pvFrame.cameraIntrinsic.ToMatrix();
        //material.SetMatrix("_CameraMatrix", cameraMa);

        //create render texture
        RenderTexture rt = new RenderTexture(pvFrame.cameraIntrinsic.imageWidth, pvFrame.cameraIntrinsic.imageHeight, 24);
        rt.enableRandomWrite = true;
        //set render texture
        GetComponent<Camera>().targetTexture = rt;
        //GetComponent<Camera>().RenderWithShader(Shader.Find("Unlit/NewUnlitShader"), "RenderType");
        //GetComponent<Camera>().RenderWithShader(Shader.Find("Custom/NewSurfaceShader"), "RenderType");
        //render with unlit shader
        //Shader replaceShader = Shader.Find("Custom/NewSurfaceShader");
        Shader replaceShader = Shader.Find("Custom/DrawSimple");
        GetComponent<Camera>().SetReplacementShader(replaceShader, null);
        GetComponent<Camera>().Render();
        //GetComponent<Camera>().RenderWithShader(Shader.Find("Unlit/NewUnlitShader"), "RenderType");
        GetComponent<Camera>().targetTexture = null;

        //RenderTexture rt2 = new RenderTexture(pvFrame.cameraIntrinsic.imageWidth, pvFrame.cameraIntrinsic.imageHeight, 24);
        //rt2.enableRandomWrite = true;
        //var material = new Material(Shader.Find("Unlit/Simple"));
        //Graphics.Blit(rt, rt2, material);


        //// /save render texture to png
        RenderTexture.active = rt;
        Texture2D tex = new Texture2D(pvFrame.cameraIntrinsic.imageWidth, pvFrame.cameraIntrinsic.imageHeight, TextureFormat.RGB24, false);
        tex.ReadPixels(new Rect(0, 0, pvFrame.cameraIntrinsic.imageWidth, pvFrame.cameraIntrinsic.imageHeight), 0, 0);
        tex.Apply();
        byte[] bytes = tex.EncodeToPNG();
        System.IO.File.WriteAllBytes(Application.dataPath + "/../" + "test.png", bytes);
        RenderTexture.active = null;
        DestroyImmediate(tex);
        //DestroyImmediate(rt);

    }

}
