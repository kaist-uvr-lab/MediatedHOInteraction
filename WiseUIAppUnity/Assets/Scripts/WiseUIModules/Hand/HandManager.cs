using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using UnityEngine;
using UnityEngine.TestTools;
using WiseUI.Base;
using WiseUI.Modules;
using Microsoft.MixedReality.Toolkit.Utilities;

public class HandManager : MonoBehaviour
{

    #region Singleton
    /// <summary>
    /// Creates instance of SkeletonManager
    /// </summary>
    public static HandManager instance;
        
    #endregion

    // for hand mesh    
    public GameObject handJointMesh;
    public GameObject targetPVCamera;

    GameObject hand;
    GameObject[] handJoints = new GameObject[21];
    Vector3[] joints3DWorld = new Vector3[21];
    float defaultDepth = 0.370f;


    // [debug] sample data on unity play scene
    Vector3[] sampleHand = new Vector3[21];   
    Matrix4x4 CamIntrinsic = Matrix4x4.identity;
    Matrix4x4 CamExtrinsic = Matrix4x4.identity;
        

    public void Start()
    {      
        // Set public handJointMesh from prefab, targetPVCamera from child of WiseUIAgent
        handJointMesh = Resources.Load("Hand/HandSkeleton") as GameObject;

        GameObject wiseUI_Agent = GameObject.Find("WiseUI Agent");        
        targetPVCamera = wiseUI_Agent.transform.Find("PV Camera").gameObject;

        // Transform the joints3DWorld to WiseUIAgent.PVCamera coordinate
        // Set handJoints array to control
        Vector3 initVector = new Vector3(0, 0, 0);        

        hand = Instantiate(handJointMesh, initVector, Quaternion.identity) as GameObject;
        hand.transform.parent = wiseUI_Agent.transform;
        //hand.transform.position = targetPVCamera.transform.position;
        //hand.transform.rotation = targetPVCamera.transform.rotation;

        for (int i = 0; i < 21; i++)
        {
            handJoints[i] = hand.transform.GetChild(i).gameObject;
        }
        
        //hand.transform.position = targetPVCamera.transform.position;
        //hand.transform.rotation = targetPVCamera.transform.rotation;

        // set sample poses. uvd coordinate in server output.
        /*
        sampleHand[0] = new Vector3(473.5f,303.9f, 0.0f);
        sampleHand[1] = new Vector3(478.4f,254.3f,-0.0129f);
        sampleHand[2] = new Vector3(480.0f,210.7f,-0.0306f);
        sampleHand[3] = new Vector3(474.4f,176.5f,-0.0447f);
        sampleHand[4] = new Vector3(443.5f,173.0f,-0.0696f);
        sampleHand[5] = new Vector3(428.2f,151.2f,0.00875f);
        sampleHand[6] = new Vector3(403.2f,99.4f,0.0170f);
        sampleHand[7] = new Vector3(380.2f,68.3f,0.0142f);
        sampleHand[8] = new Vector3(358.6f,31.2f,0.0132f);
        sampleHand[9] = new Vector3(391.8f,166.2f,0.0168f);
        sampleHand[10] = new Vector3(350.7f,120.3f,0.0141f);
        sampleHand[11] = new Vector3(320.7f,91.1f,0.00638f);
        sampleHand[12] = new Vector3(288.1f,56.6f,0.00332f);
        sampleHand[13] = new Vector3(367.9f,208.8f,0.0182f);
        sampleHand[14] = new Vector3(335.8f,163.1f,0.0161f);
        sampleHand[15] = new Vector3(301.5f,131.4f,0.0104f);
        sampleHand[16] = new Vector3(270.5f,94.5f,0.0073f);
        sampleHand[17] = new Vector3(351.4f,244.3f,0.0154f);
        sampleHand[18] = new Vector3(322.4f,221.8f,0.0165f);
        sampleHand[19] = new Vector3(292.7f,199.1f,0.0147f);
        sampleHand[20] = new Vector3(262.3f,170.3f,0.0152f);

        CamIntrinsic.m00 = 586.931f;
        CamIntrinsic.m01 = 0.0f;
        CamIntrinsic.m02 = 373.957f;
        CamIntrinsic.m03 = 0.0f;
        CamIntrinsic.m10 = 0.0f;
        CamIntrinsic.m11 = 586.538f;
        CamIntrinsic.m12 = 202.832f;
        CamIntrinsic.m13 = 0.0f;
        CamIntrinsic.m20 = 0.0f;
        CamIntrinsic.m21 = 0.0f;
        CamIntrinsic.m22 = 1.0f;
        CamIntrinsic.m23 = 0.0f;
        CamIntrinsic.m30 = 0.0f;
        CamIntrinsic.m31 = 0.0f;
        CamIntrinsic.m32 = 0.0f;
        CamIntrinsic.m33 = 1.0f;

        CamExtrinsic.m00 = -0.9474882f;
        CamExtrinsic.m01 = -0.1431952f;
        CamExtrinsic.m02 = -0.2859391f;
        CamExtrinsic.m03 = 0.260081f;
        CamExtrinsic.m10 = 0.07067199f;
        CamExtrinsic.m11 = 0.7782769f;
        CamExtrinsic.m12 = -0.6239315f;
        CamExtrinsic.m13 = -0.0119431f;
        CamExtrinsic.m20 = 0.3118838f;
        CamExtrinsic.m21 = -0.6113756f;
        CamExtrinsic.m22 = -0.7272883f;
        CamExtrinsic.m23 = -0.735249f;
        CamExtrinsic.m30 = 0.0f;
        CamExtrinsic.m31 = 0.0f;
        CamExtrinsic.m32 = 0.0f;
        CamExtrinsic.m33 = 1.0f;

        Debug.Log("Set sample data, start");
        */
        
    }

    public void OnReceivedResultData(PVFrame pvFrame, DepthFrame depthFrame, ResultDataPackage frameData)
    {        
        // Unpack ResultDataPackage
        HandDataPackage handData = frameData.handDataPackage;

        Matrix4x4 CamIntrinsic = pvFrame.cameraIntrinsic.ToMatrix();
        Matrix4x4 CamExtrinsic = pvFrame.Cam2World().inverse;
        var projection = CamIntrinsic * CamExtrinsic;       // matrix multiplication check
        Matrix4x4 inv = projection.inverse;

        int imgH = pvFrame.cameraIntrinsic.imageHeight;
        int imgW = pvFrame.cameraIntrinsic.imageWidth;

        var rootDepth = 0.0f;
        //float[,] projected_pointcloud_texture = depthFrame.Project2PVFrame(pvFrame);  // (760, 428)   value range ~ 0.470
        
        try{
            
            // manual root depth (~6/8)
            rootDepth = defaultDepth;
            // update to real depth data (TBD)
            //rootDepth = extractRootDepth(projected_pointcloud_texture, handData, pvFrame.cameraIntrinsic);
        }
        catch(Exception e)
        {
            Debug.Log(e);
        }

        
        // Compute hand joint on world coordinate
        for (int i = 0; i < 21; i++)
        {
            // pixel 단위. u starts from left, v start from bottom.
            Vector3 joint3DImage = new Vector3(handData.joints[i].u, imgH - handData.joints[i].v, 1);
            joint3DImage *= (rootDepth + handData.joints[i].d);

            Vector3 joint3DWorld = inv.MultiplyPoint3x4(joint3DImage);
           
            
            joints3DWorld[i] = new Vector3(joint3DWorld.x, joint3DWorld.y, joint3DWorld.z);   
        }
        
        for (int i = 0; i < 21; i++)
        {
            handJoints[i].transform.position = joints3DWorld[i];            
        }
        //hand.transform.rotation = targetPVCamera.transform.rotation;


    }

    float extractRootDepth(float[,] projected_pointcloud_texture, HandDataPackage handData, CameraIntrinsic camInfo)
    {
        float rootDepth = 0.0f;

        int wrist_u = (int)handData.joints[0].u;
        int wrist_v = camInfo.imageHeight - (int)handData.joints[0].v;

        int scan_u = 0;
        int scan_v = 0;
        int scan_range = 5;
        
        while (scan_u < scan_range)
        {
            if (wrist_u + scan_u < camInfo.imageWidth){
                rootDepth = projected_pointcloud_texture[wrist_u + scan_u, wrist_v + scan_v];
                if (rootDepth != 0.0f) { break; }
            }
            if (wrist_u - scan_u > -1){
                rootDepth = projected_pointcloud_texture[wrist_u - scan_u, wrist_v + scan_v];
                if (rootDepth != 0.0f) { break; }              
            }                  
            scan_u += 1;
        }
        if (rootDepth == 0.0f)
        {
            Debug.Log("wrist of depth is not captured");
            rootDepth = defaultDepth;
        }

        return rootDepth;
    }
}