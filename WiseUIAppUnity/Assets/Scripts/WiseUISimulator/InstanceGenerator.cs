using Microsoft.MixedReality.Toolkit;
using PointCloudExporter;
using System;
using Unity.XR.CoreUtils;
using UnityEngine;
using UnityEngine.Rendering;
using WiseUI.Base;
using static TMPro.SpriteAssetUtilities.TexturePacker_JsonArray;

public static class InstanceGenerator
{
    static float scale_intrinsic = 0.0001f; //HoloLens2 좌표계와 Unity 좌표계의 스케일 단위 변환.
    static string name_env = "Spatial Environment";
    static string pc_tsdf = "tsdf-pc.ply";
    static string pv_camera_name = "PV Camera";
    static string long_depth_camera_name = "Depth long throw";
    static string short_depth_camera_name = "Depth short throw";

    public static GameObject CreateEnvironment(string dataset_path, string name_ply)
    {
        var environment = new GameObject(name_env);

        var filepath_pointcloud = string.Format(@"{0}/{1}/{2}", dataset_path, "pinhole_projection", pc_tsdf);
        var meshInfo = PointCloudGenerator.LoadPointCloud(filepath_pointcloud);
        //var pc = PointCloudGenerator.Generate(name_ply, meshInfo, 0.1f);
        var pc = meshInfo.ToGameObject(name_ply, 0.1f);

        pc.transform.parent = environment.transform;

        return environment;
    }

    public static GameObject CreatePVFrameObject(PVFrame pvFrame, Transform parent, float transparency)
    {
        GameObject go_camera = CreateCameraObject(pvFrame.cameraIntrinsic);
        //GameObject go_image = CreateImageObject(pvFrame.texture, transparency);
        go_camera.transform.parent = parent;
        //go_image.transform.parent = go_camera.transform;

        ChangePVFrameObjectProperty(pvFrame, go_camera, transparency);

        return go_camera;
    }

    public static GameObject CreateDepthFrameObject(DepthFrame frame, Transform parent, bool load_pointcloud, float size_point)
    {
        bool isShortThrow;
        if (frame.sensorType == HoloLens2SensorType.DepthCamera_ShortThrow)
            isShortThrow = true;
        else if (frame.sensorType == HoloLens2SensorType.DepthCamera_LongThrow)
            isShortThrow = false;
        else
            throw new Exception("Invalid sensor type.");

        GameObject go_camera = CreateDepthObject(isShortThrow);

        go_camera.transform.parent = parent;

        ChangeDepthFrameObjectProperty(frame, go_camera, load_pointcloud, size_point);


        return go_camera;
    }

    public static void ChangePVFrameObjectProperty(PVFrame pvFrame, GameObject go_camera, float transparency)
    {
        if (pvFrame == null)
        {
            Debug.LogError("PVFrame is null.");
        }
        //Convert transform.
        var rh = pvFrame.Cam2World();
        Vector3 position = rh.GetColumn(3);
        Quaternion rotation = rh.rotation;
        //float aspectRatio = (float)pvFrame.cameraIntrinsic.imageHeight / pvFrame.cameraIntrinsic.imageWidth;


        //go_PV.GetComponent<MeshRenderer>().enabled = false;
        go_camera.transform.position = position;
        go_camera.transform.rotation = rotation;

        go_camera.GetComponent<BlendDuringRender>().SetBlendedTexture(pvFrame.Texture);
        go_camera.GetComponent<BlendDuringRender>().SetTransparency(transparency);

        //RenderTexture renderTexture = new RenderTexture(pvFrame.cameraIntrinsic.imageWidth, pvFrame.cameraIntrinsic.imageHeight, 24);
        //RenderTexture.active = renderTexture;
        //go_camera.GetComponent<Camera>().Render();
        //RenderTexture.active = null;

        //mat_pv.SetTexture("_SecondTex", pvFrame.Texture);
        //Graphics.Blit(renderTexture, mat_pv);
        ////Create Camera Object.
        //// note : unity camera의 intrinsic parameter와 실제 image의 intrinsic parameter가 다르다.
        //// focal length_x 만 적용해서 image와 camera사이의 간격을 띄우고, fov만 적용한 상태이다.
        //// 실제 intrinsic parameter를 unity camera에 적용하는 방법은 이슈로 남겨둔다.
        //GameObject go_image = go_camera.transform.Find("Image").gameObject;
        //var texture = pvFrame.texture;
        //Renderer rend = go_image.GetComponent<Renderer>();
        //rend.sharedMaterial.mainTexture = texture;
        //rend.sharedMaterial.SetColor("_color", new Color(1f, 1f, 1f, transparency));

        //float fx = pvFrame.cameraIntrinsic.focalLength.x;
        //float fy = pvFrame.cameraIntrinsic.focalLength.y;
        ////Debug.Log(string.Format("{0}, {1}", fx, fy));

        //go_image.transform.localPosition = Vector3.zero;
        //go_image.transform.localRotation = Quaternion.identity;
        //go_image.transform.Rotate(90, 180, 0);
        //go_image.transform.Translate(Vector3.down * fx * scale_intrinsic, Space.Self);
        //go_image.transform.localScale = new Vector3(pvFrame.cameraIntrinsic.imageWidth * scale_intrinsic * 0.1f, 1, pvFrame.cameraIntrinsic.imageHeight * scale_intrinsic * 0.1f);
        //go_image.transform.Translate(Vector3.right * fy * 2 / 1000f, Space.Self);
    }

    public static void ChangeDepthFrameObjectProperty(DepthFrame frame, GameObject go_camera, bool load_pointcloud, float size_point)
    {
        //Convert transform.
        var rh = frame.Cam2World();
        Vector3 position = rh.GetColumn(3);
        Quaternion rotation = rh.rotation;
        go_camera.transform.position = position;
        go_camera.transform.rotation = rotation;

        string name = "PointCloud";
        var go_pointCloud = go_camera.transform.Find(name);

        if (go_pointCloud != null)
            GameObject.DestroyImmediate(go_pointCloud.gameObject);

        if (load_pointcloud)
        {
            var pointCloud = frame.GetPointCloudGameObject(name, size_point);
            pointCloud.transform.parent = go_camera.transform;
        }
    }

    static GameObject CreateImageObject(Texture2D texture, float transparency)
    {
        GameObject imgGO = GameObject.CreatePrimitive(PrimitiveType.Plane);

        var collider = imgGO.GetComponent<MeshCollider>();
        UnityEngine.Object.DestroyImmediate(collider);

        imgGO.name = "Image";
        Renderer rend = imgGO.GetComponent<Renderer>();
        rend.material = new Material(Shader.Find("Somian/Unlit/Transparent"));
        rend.sharedMaterial.mainTexture = texture;
        rend.sharedMaterial.SetColor("_Color", new Color(1f, 1f, 1f, transparency));

        return imgGO;
    }

    static GameObject CreateCameraObject(CameraIntrinsic intrinsic)
    {
        //Create cameraObject.
        GameObject camGO = new GameObject(pv_camera_name);
        var s = camGO.AddComponent<ImageControllerMono>();
        s.sensorType = HoloLens2SensorType.PVCamera;
        Camera cam = camGO.AddComponent<Camera>();
        cam.usePhysicalProperties= true;

        // How to  adjust intrinsic parameter to unity camera.
        // note, https://answers.unity.com/questions/814701/how-to-simulate-unity-pinhole-camera-from-its-intr.html
        float f = intrinsic.focalLength.x * 0.1f; // f can be arbitrary, as long as sensor_size is resized to to make ax,ay consistient
        float sizeX = f * intrinsic.imageWidth / intrinsic.focalLength.x;
        float sizeY = f * intrinsic.imageHeight / intrinsic.focalLength.y;

        float shiftX = -(intrinsic.principalPoint.x - intrinsic.imageWidth / 2.0f) / intrinsic.imageWidth; 
        float shiftY = (intrinsic.principalPoint.y - intrinsic.imageHeight / 2.0f) / intrinsic.imageHeight;

        cam.focalLength = f;
        cam.sensorSize = new Vector2(sizeX, sizeY);
        cam.lensShift = new Vector2(shiftX, shiftY);

        cam.clearFlags = CameraClearFlags.Depth;
        cam.nearClipPlane = 0.03f;
        cam.farClipPlane = 100f;

        var s2 = camGO.AddComponent<BlendDuringRender>();
        s2.Init();
        return camGO;
    }

    static GameObject CreateDepthObject(bool shortThrow)
    {
        string camera_name;
        if (shortThrow)
            camera_name = short_depth_camera_name;
        else
            camera_name = long_depth_camera_name;

        //Create cameraObject.
        GameObject camGO = new GameObject(camera_name);
        var s = camGO.AddComponent<ImageControllerMono>();

        if(shortThrow)
            s.sensorType = HoloLens2SensorType.DepthCamera_ShortThrow;
        else
            s.sensorType = HoloLens2SensorType.DepthCamera_LongThrow;

        Camera cam = camGO.AddComponent<Camera>();
        cam.enabled = false;
        cam.clearFlags = CameraClearFlags.Depth;
        cam.nearClipPlane = 0.03f;
        cam.farClipPlane = 100f;

        return camGO;
    }


}


