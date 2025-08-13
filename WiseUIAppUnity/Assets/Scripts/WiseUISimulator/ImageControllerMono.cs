using System.Collections.Generic;
using UnityEngine;
using WiseUI.Base;

#if UNITY_EDITOR
using UnityEditor;
#endif


#if UNITY_EDITOR
[ExecuteInEditMode]
#endif

public class ImageControllerMono : MonoBehaviour
{
    public float scale_gizmo = 0.01f;
    Color[] color = {Color.white, Color.blue, Color.yellow};
    public HoloLens2SensorType sensorType;
    bool showPointCloud = true;

    [HideInInspector]
    //public bool isSelected;
    public int selState = 0; //0 ; unsel, 1 ; only, 2; additionally
    Camera cam;
    void Awake()
    {
        //image = transform.Find("Image").gameObject;
        //if (image == null)
        //    Debug.LogError("This gameobject must include \"Image\" as child object.");

    }

    //선택, zoom level에 따른 Gizmo 색상 결정.
    void OnDrawGizmos()
    {
        switch (sensorType)
        {
            case HoloLens2SensorType.PVCamera:
                Gizmos.color = Color.white;
                break;

            case HoloLens2SensorType.DepthCamera_LongThrow:
                Gizmos.color = Color.blue;
                break;

            case HoloLens2SensorType.DepthCamera_ShortThrow:
                Gizmos.color = Color.yellow;
                break;

        }

        cam = GetComponent<Camera>();
        if (cam == null)
            Debug.LogError("This gameobject must include \"Image\" as child object.");

        Gizmos.DrawSphere(transform.position, transform.lossyScale.x * scale_gizmo);

        //Gizmos.color = new Color(1f, 0f, 1f, 1f);
        //Gizmos.DrawLine(transform.position, transform.position + transform.forward);
        //var pvFrame = HoloLens2FileStreamManager.Instance.PVCamera.GetFrame();
        Gizmos.matrix = Matrix4x4.TRS(transform.position, transform.rotation, new Vector3(cam.aspect, 1.0f, 1.0f));
        //Gizmos.DrawFrustum(Vector3.zero, cam.fieldOfView, cam.farClipPlane, cam.nearClipPlane, 1.0f);
        Gizmos.DrawFrustum(Vector3.zero, cam.fieldOfView, 1, cam.nearClipPlane, 1.0f);

        //Gizmos.DrawFrustum(transform.position, cam.fieldOfView, cam.farClipPlane, cam.nearClipPlane, cam.aspect);

    }

    public void OnUnselected()
    {
        selState = 0;
        //image.GetComponent<MeshRenderer>().enabled = false;
        //gameObject.SetActive(false);
    }
    public void OnSelectedOnly()
    {
        selState = 1;
        //image.GetComponent<MeshRenderer>().enabled = true;
        //gameObject.SetActive(true);
    }
    public void OnSelectedAdditionaly()
    {
        selState = 2;
        //image.GetComponent<MeshRenderer>().enabled = false;
        //gameObject.SetActive(false);
    }

    void OnGUI()
    {
        if (Event.current.type == EventType.Layout || Event.current.type == EventType.Repaint)
        {
#if UNITY_EDITOR
            EditorUtility.SetDirty(this); // this is important, if omitted, "Mouse down" will not be display
#endif
        }

        if (selState == 1)
        {
            if (Event.current.type == EventType.MouseDown && Event.current.button == 1) //right mouse down
            {
                if (showPointCloud)
                {
                    gameObject.GetComponent<Camera>().cullingMask = 1;
                }
                else
                {
                    gameObject.GetComponent<Camera>().cullingMask |= 1 << LayerMask.NameToLayer("PointCloud");
                }
                showPointCloud = !showPointCloud;
            }
        }
    }


}
