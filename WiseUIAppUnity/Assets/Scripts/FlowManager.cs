using Microsoft.MixedReality.Toolkit.Experimental.UI;
using Microsoft.MixedReality.Toolkit.UI;
using System.Collections;
using TMPro;
using UnityEngine;
using System;
using WiseUI.Base;
using WiseUI.Modules;
using System.Threading;
using System.Text;
using System.Collections.Generic;
using Unity.XR.CoreUtils;
using System.Linq;
using Microsoft.MixedReality.Toolkit.Utilities;
using UnityEditor;
using Microsoft.MixedReality.Toolkit.Input;

public class FlowManager : MonoBehaviour
{
    // Modules
    //public HoloLens2PVCameraReader pvCameraReader;
    //public SocketClientManager socketClient;
    public ARRCObjectronDetector objectDetector;

    public Interactable confButton;

    // Title UI
    public TextMeshPro stateMessage;
    public Interactable closeButton;


    // TCP UI
    public Interactable connectButton;
    public MRTKTMPInputField hostIPField, portField;

    public GameObject images;

    // Capture UI
    public Interactable startCaptureButton;
    public InteractableToggleCollection pvCamToggles;
    public GameObject pvImagePlane;
    Coroutine cameraTextureUpdateHandle;

    // Detection UI
    public Interactable startDetectionButton;
    public InteractableToggleCollection detectionToggles;
    public GameObject detectedImagePlane;
    Coroutine detectionUpdateHandle;

    // Hand
    private GameObject handJointMesh, handSkeletonMesh;
    private GameObject rightWrist;
    private GameObject leftWrist;
    private MixedRealityPose rightPose;
    private MixedRealityPose leftPose;
    Vector3 initVector = new Vector3(0, 0, 0);
    GameObject hand, handSkeleton;
    GameObject[] handJoints = new GameObject[21];
    GameObject[] handBars = new GameObject[20];
    Vector3[] joints3DWorld = new Vector3[21];
    float imgH = 360.0f; //pvFrame.cameraIntrinsic.imageHeight;
    float imgW = 640.0f; //pvFrame.cameraIntrinsic.imageWidth;
    GameObject mainCam, handMesh;
    int reinit_count = 0;    
    bool flag_right_hand = true;
    int gesture_class = 0;
    Vector3[] sample_handpose = new Vector3[21];

    //Camera Image planes

    //WebSocketClient webSocketClient;
    System.Diagnostics.Stopwatch stopwatch = new System.Diagnostics.Stopwatch();

    List<double> delays = new List<double>();

    private void setInitHand()
    {

        sample_handpose[0] = new Vector3(0.458f, 0.012f, 0.257f);
        sample_handpose[1] = new Vector3(0.442f, 0.017f, 0.268f);
        sample_handpose[2] = new Vector3(0.438f, 0.022f, 0.280f);
        sample_handpose[3] = new Vector3(0.434f, 0.030f, 0.288f);
        sample_handpose[4] = new Vector3(0.436f, 0.037f, 0.306f);
        sample_handpose[5] = new Vector3(0.431f, 0.045f, 0.270f);
        sample_handpose[6] = new Vector3(0.432f, 0.058f, 0.279f);
        sample_handpose[7] = new Vector3(0.441f, 0.064f, 0.292f);
        sample_handpose[8] = new Vector3(0.447f, 0.071f, 0.306f);
        sample_handpose[9] = new Vector3(0.448f, 0.052f, 0.271f);
        sample_handpose[10] = new Vector3(0.454f, 0.066f, 0.281f);
        sample_handpose[11] = new Vector3(0.464f, 0.072f, 0.295f);
        sample_handpose[12] = new Vector3(0.474f, 0.079f, 0.310f);
        sample_handpose[13] = new Vector3(0.471f, 0.051f, 0.273f);
        sample_handpose[14] = new Vector3(0.473f, 0.063f, 0.282f);
        sample_handpose[15] = new Vector3(0.484f, 0.069f, 0.297f);
        sample_handpose[16] = new Vector3(0.491f, 0.074f, 0.312f);
        sample_handpose[17] = new Vector3(0.491f, 0.047f, 0.275f);
        sample_handpose[18] = new Vector3(0.495f, 0.055f, 0.283f);
        sample_handpose[19] = new Vector3(0.501f, 0.058f, 0.294f);
        sample_handpose[20] = new Vector3(0.505f, 0.063f, 0.306f);

    }

    private void Awake()
    {
        delays.EnsureCapacity(100);
        
        //pvCameraReader = GameObject.Find("WiseUI Agent").GetComponent<HoloLens2PVCameraReader>();
        //socketClient = GameObject.Find("WiseUI Agent").GetComponent<SocketClientManager>();
        //objectDetector = GameObject.Find("WiseUI Agent").GetComponent<ARRCObjectronDetector>();

        confButton = transform.Find("Setting").GetComponent<Interactable>();
        confButton.OnClick.AddListener(OnConfigurationButtonClick);
        images = transform.Find("Images").gameObject;
        pvImagePlane = transform.Find("Images/PVImagePlane").gameObject;
        detectedImagePlane = transform.Find("Images/DetectedImagePlane").gameObject;
        //transform.Find("Pannel").gameObject.SetActive(false);

        //Title state
        stateMessage = transform.Find("Pannel/TitleBar/Title").GetComponent<TextMeshPro>();
        closeButton = transform.Find("Pannel/TitleBar/TitleButton/Close").GetComponent<Interactable>();
        closeButton.OnClick.AddListener(CloseButtonClick);

        //TCP Connect
        hostIPField = transform.Find("Pannel/HostAddress/Host IP").GetComponent<MRTKTMPInputField>();
        portField = transform.Find("Pannel/HostAddress/Port").GetComponent<MRTKTMPInputField>();
        connectButton = transform.Find("Pannel/Connect").GetComponent<Interactable>();
        connectButton.OnClick.AddListener(ConnectButtonClick);

        //PV sensor resolution.
        pvCamToggles = transform.Find("Pannel/PVSensorGroup").GetComponent<InteractableToggleCollection>();
        startCaptureButton = transform.Find("Pannel/StartCapture").GetComponent<Interactable>();
        startCaptureButton.OnClick.AddListener(OnStartCaptureButtonClick);

        //Detection target.
        detectionToggles = transform.Find("Pannel/ModelGroup").GetComponent<InteractableToggleCollection>();
        startDetectionButton = transform.Find("Pannel/StartDetection").GetComponent<Interactable>();
        startDetectionButton.OnClick.AddListener(OnStartDetectionButtonClick);

        //For Hand
        mainCam = GameObject.Find("Main Camera");
        handMesh = GameObject.Find("R_Hand_MRTK_Rig");
        rightWrist = Resources.Load("Hand/Joint_wrist") as GameObject;
        rightWrist = Instantiate(rightWrist, initVector, Quaternion.identity) as GameObject;
        leftWrist = Resources.Load("Hand/Joint_wrist") as GameObject;
        leftWrist = Instantiate(leftWrist, initVector, Quaternion.identity) as GameObject;
        handJointMesh = Resources.Load("Hand/HandJoints") as GameObject;
        hand = Instantiate(handJointMesh, mainCam.transform) as GameObject;
        handSkeletonMesh = Resources.Load("Hand/HandSkeleton") as GameObject;
        handSkeleton = Instantiate(handSkeletonMesh, mainCam.transform) as GameObject;
    }
    
    private void Start()
    {
        LoadUIContents();
        //images.SetActive(false);
        pvImagePlane.SetActive(false);
        // detectedImagePlane.SetActive(false);

        setInitHand();
        for (int i = 0; i < 21; i++)
        {
            handJoints[i] = hand.transform.GetChild(i).gameObject;
            if (i != 20)
            {
                handBars[i] = handSkeleton.transform.GetChild(i).gameObject;
            }
        }
    }

    public void Update()
    {
        if (HandJointUtils.FindHand(handedness: Handedness.Right) != null)
        {
            HandJointUtils.TryGetJointPose(TrackedHandJoint.Wrist, Handedness.Right, out rightPose);
            rightWrist.transform.position = rightPose.Position;
            rightWrist.transform.rotation = rightPose.Rotation;
            flag_right_hand = true;
        }

        if (HandJointUtils.FindHand(handedness: Handedness.Left) != null)
        {
            HandJointUtils.TryGetJointPose(TrackedHandJoint.Wrist, Handedness.Left, out leftPose);
            leftWrist.transform.position = leftPose.Position;
            leftWrist.transform.rotation = leftPose.Rotation;
            flag_right_hand = false;
        }
    }

    void LoadUIContents()
    {
        ConfigurationManager.Instance.Load();
        hostIPField.text = ConfigurationManager.Instance.hostIP;
        portField.text = ConfigurationManager.Instance.port.ToString();
        pvCamToggles.SetSelection((int)ConfigurationManager.Instance.pvCameraType);
    }
    
    void SaveUIContents()
    {
        string ip = hostIPField.text;
        int port = int.Parse(portField.text);
        PVCameraType pVCameraType = (PVCameraType)pvCamToggles.CurrentIndex;
        ConfigurationManager.Instance.Save(ip, port, pVCameraType);
        Debug.Log(string.Format("{0}, {1},{2}", ip, port, pVCameraType.ToString()));
    }
    
    void OnConfigurationButtonClick()
    {
        //LoadUIContents();
    }
    
    void CloseButtonClick()
    {
        SaveUIContents();
    }
    
    private void OnDestroy()
    {
        //SaveUIContents();
    }
    
    void ConnectButtonClick()
    {
        //Debug.Log(hostIPField.text);

        if (!connectButton.IsToggled)
        {
            SocketClientManager.Instance.Disconnect();

            //var content = Encoding.UTF8.GetBytes("#Disconnect#");
            //var contentSize = BitConverter.GetBytes(content.Length);

            //int totalSize = 4 + content.Length;
            //byte[] buffer = new byte[totalSize];
            //System.Buffer.BlockCopy(contentSize, 0, buffer, 0, 4);
            //System.Buffer.BlockCopy(content, 0, buffer, 4, content.Length);
            //webSocketClient.Send(buffer);
            //Thread.Sleep(100);
            
            //if (webSocketClient != null)
            //    webSocketClient.Disconnect();
            stateMessage.text = string.Format("Success to disconnect");
        }

        try
        {
            string ip = hostIPField.text;
            int port = int.Parse(portField.text);

            SocketClientManager.Instance.Connect(ip, port);
            //webSocketClient = new WebSocketClient();
            //webSocketClient.Connect(ip, port);

            stateMessage.color = Color.white;
            stateMessage.text = string.Format("Success to connect : {0}:{1}", ip, port);
            StartCoroutine(UpdateConnection());
        }
        catch(Exception e)
        {
            stateMessage.text = string.Format("Fail to connect : {0}", e.Message);
            stateMessage.color = Color.red;
            connectButton.IsToggled = false;
            StopCoroutine(UpdateConnection());
        }
    }

    void OnStartCaptureButtonClick()
    {
        try
        {
            if (!startCaptureButton.IsToggled)
            {
                HoloLens2SensorStreamManager.Instance.PVCamera.StopPVCamera();
                //pvImagePlane.SetActive(false);

                if (cameraTextureUpdateHandle != null)
                    StopCoroutine(cameraTextureUpdateHandle);

                // if (detectionUpdateHandle != null)
                //     StopCoroutine(detectionUpdateHandle);

                return;
            }

            int idx = pvCamToggles.CurrentIndex;
            //pvImagePlane.SetActive(true);
            HoloLens2SensorStreamManager.Instance.PVCamera.StartPVCamera((PVCameraType)idx);
            cameraTextureUpdateHandle = StartCoroutine(UpdateCameraTexutre());
            // detectionUpdateHandle = StartCoroutine(UpdateDetection());
        }
        catch(System.Exception e)
        {
            stateMessage.text = string.Format("Fail : {0}", e.Message);
            
            stateMessage.color = Color.red;
            startCaptureButton.IsToggled = false;
        }

    }

    void OnStartDetectionButtonClick()
    {
        try
        {
            if (!startDetectionButton.IsToggled)
            {
                // detectedImagePlane.SetActive(false);

                if (detectionUpdateHandle != null)
                    StopCoroutine(detectionUpdateHandle);
                return;
            }

            // detectedImagePlane.SetActive(true);
            // int idx = detectionToggles.CurrentIndex;
            detectionUpdateHandle = StartCoroutine(UpdateDetection());

            //objectDetector.LoadModel((ModelType)idx);
            //stateMessage.text = string.Format("Load Model OK.");
        }
        catch (System.Exception e)
        {
            stateMessage.text = string.Format("Fail : {0}", e.Message);
            stateMessage.color = Color.red;
            startDetectionButton.IsToggled = false;
        }
    }
    
    IEnumerator UpdateCameraTexutre()
    {
        while (true)
        {
            if (HoloLens2SensorStreamManager.Instance.PVCamera.IsNewFrame)
            {
                HoloLens2SensorStreamManager.Instance.PVCamera.UpdateLatestTexture();
                Texture2D latestTexture = HoloLens2SensorStreamManager.Instance.PVCamera.GetLastestTexture();
                pvImagePlane.GetComponent<MeshRenderer>().material.mainTexture = latestTexture;
                //Texture2D latestTexture = pvImagePlane.GetComponent<MeshRenderer>().material.mainTexture as Texture2D;

                //stopwatch.Reset();
                //latestTexture.EncodeToJPG(75);
                //stopwatch.Stop();
                //DebugText.Instance.lines["comp time"]=stopwatch.ElapsedMilliseconds.ToString();
                
                
                //var dataPackage = new RGBImageDataPackage(HoloLens2SensorStreamManager.Instance.PVCamera.Timestamp, latestTexture);

                if (connectButton.IsToggled)
                {
                    var dataPackage = new RGBImageDataPackage(HoloLens2SensorStreamManager.Instance.PVCamera.Timestamp, latestTexture, ImageCompression.JPEG, 75);
                    var data = dataPackage.Encode();
                    
                    //SocketClientManager.Instance.Send(data);
                    var task = SocketClientManager.Instance.SendAsync(data);
                    //var task = webSocketClient.SendAsync(data);
                    //Wait until the send task is complete
                    yield return new WaitUntil(() => task.IsCompleted);
                }

                //if (startDetectionButton.IsToggled)
                //{
                //    objectDetector.Run(latestTexture);
                //}

                //socketClient.SendRGBImage(pvCameraReader.FrameID, latestTexture, ImageCompression.None);
                //float time_to_send = Time.time - start_time;
                //DebugText.Instance.lines["Time_to_send"] = time_to_send.ToString();
                stateMessage.color = Color.yellow;
                DebugText.Instance.lines["IsNewFrame"] = "True";
            }
            else
            {
                stateMessage.color = Color.red;
                DebugText.Instance.lines["IsNewFrame"] = "False";
            }

            yield return new WaitForEndOfFrame();

        }
    }

    IEnumerator UpdateConnection()
    {
        while (true)
        {
            // 18�� �̺κ� ��ġ��.
            // receiveData�� �ϳ��� �����忡�� �񵿱��Լ��� ó������.
            if (SocketClientManager.Instance.IsNewHandDataReceived)
            {
                var frameData = SocketClientManager.Instance.GetLatestReceivedData(); // ���� ���ο� frameData�� �������� �ʾҴٸ� NoDataReceivedExecption�� �� �Լ� ���ο��� �߻��Ѵ�.
                
                string debug_idx = frameData.frameInfo.frameID.ToString();
                stateMessage.text = string.Format("receive idx : {0}", debug_idx);

                // DebugText.Instance.lines["frame_id"] = frameData.frameInfo.frameID.ToString();
                // DebugText.Instance.lines["delay_comm"] = frameData.frameInfo.GetTotalDelay().ToString();
                // delays.Add(frameData.frameInfo.GetTotalDelay());
                // var avg = delays.Average();
                // DebugText.Instance.lines["delay_avg"] = avg.ToString();


            }

            yield return null;
        }

    }

    IEnumerator UpdateDetection()
    {
        while (true)
        {
            if (SocketClientManager.Instance.IsNewHandDataReceived)
            {
                DebugText.Instance.lines["IsNewHand"] = "True";

                var frameData = SocketClientManager.Instance.GetLatestReceivedData();

                var handData = frameData.handDataPackage.joints;
                DebugText.Instance.lines["handData"] = handData.Count.ToString();
                
                Matrix4x4 intrinsic = Matrix4x4.identity;
                // var camData = frameData.camInfo;
                // intrinsic.m00 = camData.fx;
                // intrinsic.m11 = camData.fy;
                // intrinsic.m02 = camData.cx;
                // intrinsic.m12 = camData.cy;
                
                // 493.31238, 493.2309, 314.9145, 170.60936
                intrinsic.m00 = 493.31f;
                intrinsic.m11 = 493.23f;
                intrinsic.m02 = 314.91f;
                intrinsic.m12 = 170.61f;

                Vector3 maincam_pos = mainCam.transform.localPosition;
                Quaternion maincam_rot = mainCam.transform.localRotation;
                Vector3 maincam_scale = new Vector3(1, 1, 1);

                Matrix4x4 extrinsic = Matrix4x4.TRS(maincam_pos, maincam_rot, maincam_scale);

                var projection = intrinsic * extrinsic.inverse;
                Matrix4x4 inv = projection.inverse;

                var rightWristPos_rel = rightWrist.transform.position - mainCam.transform.position;
                var wrist_gap = rightWrist.transform.position;


                // update joints
                for (int i = 0; i < 21; i++)
                {
                    // pixel u starts from left, v start from bottom.
                    // handData[i].d : negative z
                    Vector3 joint3DImage = new Vector3(handData[i].u, imgH - handData[i].v, 1);

                    joint3DImage *= rightWristPos_rel.z + handData[i].d;
                    Vector3 joint3DWorld = inv.MultiplyPoint3x4(joint3DImage);

                    joints3DWorld[i] = new Vector3(joint3DWorld.x, joint3DWorld.y, joint3DWorld.z);

                    // adjust gap between hololens wrist - prediction
                    if (i == 0)
                    {
                        wrist_gap = rightWrist.transform.position - joints3DWorld[0];
                    }
                    Vector3 newLocation = joints3DWorld[i] + wrist_gap;

                    if (reinit_count > 25)
                    {
                        handJoints[i].transform.position = Vector3.MoveTowards(handJoints[i].transform.position, newLocation, 0.2f * Time.deltaTime);
                        reinit_count = 0;
                    }
                    else
                    {
                        handJoints[i].transform.position = newLocation;
                        // Debug.Log(i.ToString() + " : " + newLocation.ToString("F3"));
                    }
                }
                
                // update bars
                for (int finger = 0; finger < 5; finger++)
                {
                    for (int idx = 0; idx < 4; idx++)
                    {
                        var A = handJoints[idx + finger * 4].transform.position;
                        var B = handJoints[idx + finger * 4 + 1].transform.position;
                        if (idx == 0)
                        {
                            A = handJoints[0].transform.position;
                        }
                        Vector3 diff = A - B;
                        var targetBar = handBars[idx + finger * 4];

                        targetBar.transform.rotation = Quaternion.FromToRotation(Vector3.up, diff.normalized);
                        targetBar.transform.position = (A + B) / 2;
                        var scaleCylinder = targetBar.transform.localScale;
                        scaleCylinder.y = diff.magnitude / 2f;
                        targetBar.transform.localScale = scaleCylinder;
                    }
                }
                reinit_count += 1;
            }
            else
            {
                stateMessage.color = Color.red;
                DebugText.Instance.lines["IsNewHand"] = "False";
            }
            yield return null;
        }
    }

    ////// public functions to utilize hands ///////
    // return single hand pose currently tracking (TBD : cover both hands)
    public GameObject[] Get_hand_pose()
    {      
        return handJoints;
    }

    // return hand side of currently tracking
    public bool Get_hand_side(){
        return flag_right_hand;
    }
    // public void Get_hand_gesture(){
    //     return gesture_class;
    // }

}

