using ARRC.ARRCTexture;
using Codice.Utils;
using OpenCVForUnity.VideoModule;
using System;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using UnityEditor;
using UnityEngine;
using UnityEngine.Playables;
using WiseUI.Base;
using WiseUI.Modules;
using static TMPro.SpriteAssetUtilities.TexturePacker_JsonArray;

namespace WiseUI.Simulator
{
    public class SimulatorWindow : EditorWindow
    {
        [SerializeField]
        string dataset_path;

        [SerializeField]
        string hostIP = "127.0.0.1";

        [SerializeField]
        int port = 9091;

        bool showObjectVizualizationMenu = true;
        bool showNetworkMenu = true;
        bool showFrameSelectionMenu = true;
        bool showArUcoMenu = true;

        Vector2 scrollPos = Vector2.zero;
        public long current_timestamp, prev_timestamp, end_timestamp;
        bool send2server = true;
        public bool isLoaded = false;
        public bool isPlaying = false;


        float transparency = 0.5f;
        long time_interval_ms = 33;
        float time_progress = 0;
        bool jpgCompression = true;
        int jpgQuality = 75;

        bool show_pointcloud_tsdf = true;
        bool show_pointcloud_longThrow = true;
        bool show_pointcloud_shortThrow = false;

        [SerializeField]
        public FlowManager flowManager;
        [SerializeField]
        public GameObject wiseUI_Agent;
        [SerializeField]
        public GameObject environment;
        [SerializeField]
        public GameObject go_frame_pv;
        [SerializeField]
        public Material mat_pv;
        [SerializeField]
        public GameObject go_frame_depth_longThrow;
        [SerializeField]
        public GameObject go_frame_depth_shortThrow;
        //[SerializeField]
        //public GameObject tagDetector;

        const string name_ply_tsdf = "PointCloud_TSDF";

        //모듈 제어
        bool detectArUcoMarker = false;
        bool detectMobilityState = true;
        bool detectHandTracker = false;
        
        public void StartAll()
        {
            //포인트 클라우드 환경 생성
            if (environment != null)
                DestroyImmediate(environment);
            environment = InstanceGenerator.CreateEnvironment(dataset_path, name_ply_tsdf);
            current_timestamp = 0;
            time_progress = 0;

            //WiseUI Agent 생성
            wiseUI_Agent = GameObject.Find("WiseUI Agent");
            if (wiseUI_Agent != null)
                DestroyImmediate(wiseUI_Agent);
            wiseUI_Agent = new GameObject("WiseUI Agent");

            // 홀로렌즈 스트림 활성화.
            HoloLens2FileStreamManager.Instance.PVCamera.StartPVCamera(dataset_path);
            var start_timestamp = HoloLens2FileStreamManager.Instance.PVCamera.Timestamp_Start;
            var requested_timestamp = start_timestamp;
            HoloLens2FileStreamManager.Instance.PVCamera.UpdateLatestTexture(requested_timestamp);

            var span_time = HoloLens2FileStreamManager.Instance.PVCamera.Timestamp_End - HoloLens2FileStreamManager.Instance.PVCamera.Timestamp_Start;
            end_timestamp = span_time / TimeSpan.TicksPerMillisecond;
            var pvFrame = HoloLens2FileStreamManager.Instance.PVCamera.GetFrame();
            go_frame_pv = InstanceGenerator.CreatePVFrameObject(pvFrame, wiseUI_Agent.transform, transparency);
            go_frame_pv.tag = "PVCamera";

            var new_cam = Instantiate(go_frame_pv);
            new_cam.name = "ObjectCamera";
            new_cam.transform.parent = go_frame_pv.transform;
            new_cam.transform.localPosition = Vector3.zero;
            new_cam.transform.localRotation = Quaternion.identity;
            new_cam.transform.localScale = Vector3.one;
            new_cam.tag = "ObjectCamera";
            DestroyImmediate(new_cam.GetComponent<BlendDuringRender>());
            DestroyImmediate(new_cam.GetComponent<ImageControllerMono>());


            try
            {
                HoloLens2FileStreamManager.Instance.DepthCamera_LongThrow.StartDepthCamera(dataset_path);
                HoloLens2FileStreamManager.Instance.DepthCamera_LongThrow.UpdateLatestTexture(HoloLens2FileStreamManager.Instance.DepthCamera_LongThrow.Timestamp_Start);
                //span_time = HoloLens2FileStreamManager.Instance.DepthCamera.Timestamp_End - HoloLens2FileStreamManager.Instance.DepthCamera.Timestamp_Start;
                var depthFrame_long = HoloLens2FileStreamManager.Instance.DepthCamera_LongThrow.GetFrame();
                go_frame_depth_longThrow = InstanceGenerator.CreateDepthFrameObject(depthFrame_long, wiseUI_Agent.transform, show_pointcloud_longThrow, 0.01f);

                HoloLens2FileStreamManager.Instance.DepthCamera_ShortThrow.StartDepthCamera(dataset_path);
                HoloLens2FileStreamManager.Instance.DepthCamera_ShortThrow.UpdateLatestTexture(HoloLens2FileStreamManager.Instance.DepthCamera_ShortThrow.Timestamp_Start);
                //span_time = HoloLens2FileStreamManager.Instance.DepthCamera.Timestamp_End - HoloLens2FileStreamManager.Instance.DepthCamera.Timestamp_Start;
                var depthFrame_short = HoloLens2FileStreamManager.Instance.DepthCamera_ShortThrow.GetFrame();
                go_frame_depth_shortThrow = InstanceGenerator.CreateDepthFrameObject(depthFrame_short, wiseUI_Agent.transform, show_pointcloud_shortThrow, 0.001f);
            }
            catch(FileNotFoundException e)
            {
                Debug.Log(e.Message);
            }

   
            // WiseUIAgent에 모듈 추가, 및 Start호출.
            var script = wiseUI_Agent.AddComponent<VelocityBasedMobilityClassifier>();
            script.Start();
            if (detectMobilityState)
            {
                wiseUI_Agent.GetComponent<VelocityBasedMobilityClassifier>().DetectMobilityState(current_timestamp / 1000f);
                wiseUI_Agent.GetComponent<VelocityBasedMobilityClassifier>().SetActiveMobileStatePlate(true);
            }
            else
            {
                wiseUI_Agent.GetComponent<VelocityBasedMobilityClassifier>().SetActiveMobileStatePlate(false);
            }


            // HandTracker 추가            
            var handManager = wiseUI_Agent.AddComponent<HandManager>();
            handManager.Start();


        }

        public void DestroyAll()
        {
            if (wiseUI_Agent != null)
                DestroyImmediate(wiseUI_Agent);

            if (environment != null)
                DestroyImmediate(environment);

            HoloLens2FileStreamManager.Instance.RemoveInstance();
            SocketClientManager.Instance.RemoveInstance();
        }


        void OnGUI()
        {
            scrollPos = EditorGUILayout.BeginScrollView(scrollPos, false, false);

            EditorGUILayout.BeginVertical(GUI.skin.box);
            if (GUILayout.Button("Select Recording Path"))
            {
                var path = EditorUtility.OpenFolderPanel("Select dataset Path", dataset_path, "");

                if(!string.IsNullOrEmpty(path))
                {
                    dataset_path = path;
                    SavePreperence();
                }
                    
            }

            EditorGUILayout.LabelField(dataset_path, EditorStyles.boldLabel);
            EditorGUILayout.EndVertical();

            if (!string.IsNullOrEmpty(dataset_path))
            {
                EditorGUILayout.BeginHorizontal();
                var playButtonContent = EditorGUIUtility.IconContent("PlayButton On");
                var stopButtonContent = EditorGUIUtility.IconContent("d_PreMatQuad");
                var pauseButtonContent = EditorGUIUtility.IconContent("PauseButton");
                if (!isLoaded && GUILayout.Button(playButtonContent))
                {
                    
                    StartAll();
                    isLoaded = true;
                    isPlaying = false;
                }

                else if(isLoaded && GUILayout.Button(stopButtonContent))
                {
                    DestroyAll();
                    isLoaded = false;
                    isPlaying = false;
                }
                
                if (isPlaying)
                {
                    if (GUILayout.Button(pauseButtonContent))
                    {
                        //just stop
                        isPlaying = !isPlaying;
                    }
                }
                else 
                {
                    var backup = GUI.backgroundColor;
                    GUI.backgroundColor = Color.cyan;
                    if (GUILayout.Button(pauseButtonContent))
                    {
                        //just resume
                        isPlaying = !isPlaying;
                    }
                    GUI.backgroundColor = backup;
                }
                EditorGUILayout.EndHorizontal();
            }
            if (isLoaded)
            {

                EditorGUILayout.BeginVertical(GUI.skin.box);
                showFrameSelectionMenu = EditorGUILayout.Foldout(showFrameSelectionMenu, "Frame selection", EditorStyles.boldLabel);
                if (showFrameSelectionMenu) FrameSelectionMenu();
                EditorGUILayout.EndVertical();

                EditorGUILayout.BeginVertical(GUI.skin.box);
                showObjectVizualizationMenu = EditorGUILayout.Foldout(showObjectVizualizationMenu, "Vizualization", EditorStyles.boldLabel);
                if (showObjectVizualizationMenu) ObjectVisulizationMenu();
                EditorGUILayout.EndVertical();

                EditorGUILayout.BeginVertical(GUI.skin.box);
                showNetworkMenu = EditorGUILayout.Foldout(showNetworkMenu, "Network", EditorStyles.boldLabel);
                if (showNetworkMenu) NetworkMenu();
                EditorGUILayout.EndVertical();

                EditorGUILayout.BeginVertical(GUI.skin.box);
                showArUcoMenu = EditorGUILayout.Foldout(showArUcoMenu, "Modules", EditorStyles.boldLabel);
                if (showArUcoMenu) ModulesOption();
                EditorGUILayout.EndVertical();

                //var texture = HoloLens2FileStreamManager.Instance.PVCamera.GetFrame().Texture;
                //render texture to current screen.
                //var rt = new RenderTexture(texture.width, texture.height, 0);
                //rt.enableRandomWrite = true;
                //Graphics.Blit(texture, go_camera.GetComponent<Camera>().targetTexture);
                //        go_camera.GetComponent<Camera>().targetTexture = rt;
                //GUI.DrawTexture(new Rect(0, 0, texture.width, texture.height), texture);

            }
            EditorGUILayout.EndScrollView();
        }
        void ObjectVisulizationMenu()
        {
            SerializedObject serializedObject = new SerializedObject(this);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("environment"), true);
            EditorGUILayout.PropertyField(serializedObject.FindProperty("wiseUI_Agent"), true);
            serializedObject.ApplyModifiedProperties(); // Remember to apply modified properties
            serializedObject.Update();

            GUILayout.BeginHorizontal();
            GUILayout.Label("Transparency");
            var new_transparency = EditorGUILayout.Slider(transparency, 0, 1f);
            if(new_transparency != transparency)
            {
                go_frame_pv.GetComponent<BlendDuringRender>().SetTransparency(transparency);
            }
            transparency = new_transparency;
                

            GUILayout.EndHorizontal();


            GUILayout.BeginHorizontal();
            GUILayout.Label("PointCloud");
            var backup = EditorGUIUtility.labelWidth;
            show_pointcloud_tsdf = OptimziedToggleField("Merged", show_pointcloud_tsdf);
            if (environment != null)
                environment.transform.Find(name_ply_tsdf).gameObject.SetActive(show_pointcloud_tsdf);

            if (go_frame_depth_longThrow != null)
            {
                show_pointcloud_longThrow = OptimziedToggleField("Long", show_pointcloud_longThrow);
                go_frame_depth_longThrow.SetActive(show_pointcloud_longThrow);
                var pointcloud_long = go_frame_depth_longThrow.transform.Find("PointCloud");
                if (pointcloud_long != null)
                    pointcloud_long.gameObject.SetActive(show_pointcloud_longThrow);
            }

            if (go_frame_depth_shortThrow != null)
            {
                show_pointcloud_shortThrow = OptimziedToggleField("Short", show_pointcloud_shortThrow);
                var pointcloud_short = go_frame_depth_shortThrow.transform.Find("PointCloud");
                go_frame_depth_shortThrow.SetActive(show_pointcloud_shortThrow);
                if (pointcloud_short != null)
                    pointcloud_short.gameObject.SetActive(show_pointcloud_shortThrow);
            }
            EditorGUIUtility.labelWidth = backup;
            GUILayout.EndHorizontal();
        }

        void NetworkMenu()
        {
            hostIP = EditorGUILayout.TextField("Host IP", hostIP);
            port = EditorGUILayout.IntField("Port", port);

            if ((!SocketClientManager.Instance.IsConnected() && GUILayout.Button("Connect")))
            {
                if (isLoaded)
                {
                    try
                    {
                        SocketClientManager.Instance.Connect(hostIP, port);
                        SavePreperence();
                    }
                    catch (SocketException e)
                    {
                        Debug.LogError(e);
                    }
                    
                }
                else
                {
                    Debug.Log("Load dataset first.");
                }
            }
            if ((SocketClientManager.Instance.IsConnected()))
            {
                send2server = EditorGUILayout.Toggle("Send data to server", send2server);
                jpgCompression = EditorGUILayout.Toggle("JPG compression", jpgCompression);
                if (jpgCompression)
                    jpgQuality = EditorGUILayout.IntSlider("JPG quality", jpgQuality, 0, 100);
                if (GUILayout.Button("Disconnect"))
                {
                    try
                    {
                        SocketClientManager.Instance.Disconnect();

                        Debug.Log("Disconnect.");
                        //isConnected = false;
                    }
                    catch (SocketException e)
                    {
                        Debug.LogError(e);
                    }
                }
            }


        }


        void FrameSelectionMenu()
        {
            OptimziedLabelField("Total PV frames : " + HoloLens2FileStreamManager.Instance.PVCamera.TotalFrameCount);
            OptimziedLabelField("Total depth frames(long) : " + HoloLens2FileStreamManager.Instance.DepthCamera_LongThrow.TotalFrameCount);
            OptimziedLabelField("Total depth frames(short) : " + HoloLens2FileStreamManager.Instance.DepthCamera_ShortThrow.TotalFrameCount);

            var new_time_progress = EditorGUILayout.Slider(time_progress, 0, end_timestamp / 1000f);
            time_interval_ms = EditorGUILayout.LongField("Frame interval (ms)", time_interval_ms);
            if (new_time_progress != time_progress)
                current_timestamp = (long)(new_time_progress * 1000f);
            else
                new_time_progress = current_timestamp / 1000f;
            time_progress = new_time_progress;

            EditorGUILayout.BeginHorizontal();
            //convert ms to hours minutes seconds
            TimeSpan span_time = TimeSpan.FromMilliseconds(current_timestamp);
            string current_time_str = string.Format("{0:D2}:{1:D2}.{2:D3}",
                span_time.Minutes,
                span_time.Seconds,
                span_time.Milliseconds);

            EditorGUILayout.LabelField("Current time : " + current_time_str);

            //convert millisecond to tick time.

            //Debug.Log(frame_interval_tick);
            if ((Event.current.keyCode == KeyCode.A || GUILayout.Button("< (A)")) && (current_timestamp - time_interval_ms) >= 0)
            {
                current_timestamp -= time_interval_ms;
                Repaint();
            }

            if ((Event.current.keyCode == KeyCode.D || GUILayout.Button("(D) >")) && (current_timestamp + time_interval_ms) <= end_timestamp)
            {
                current_timestamp += time_interval_ms;
                Repaint();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();
            //end_timestamp = EditorGUILayout.LongField("End framd id", end_timestamp);
            //convert ms to hours minutes seconds
            span_time = TimeSpan.FromMilliseconds(end_timestamp);
            string end_time_str = string.Format("{0:D2}:{1:D2}.{2:D3}",
                span_time.Minutes,
                span_time.Seconds,
                span_time.Milliseconds);

            EditorGUILayout.LabelField("End time : " + end_time_str);
            EditorGUILayout.EndHorizontal();


        }

        void ModulesOption()
        {
            detectArUcoMarker = EditorGUILayout.Toggle("ArUco marker", detectArUcoMarker);
            detectMobilityState = EditorGUILayout.Toggle("Mobility state", detectMobilityState);
            detectHandTracker = EditorGUILayout.Toggle("Hand tracker", detectHandTracker);

        }

        void OnTimeChanged()
        {
            if (!isLoaded)
            {
                Debug.Log("Load dataset first.");
                return;
            }

            var start_timestamp = HoloLens2FileStreamManager.Instance.PVCamera.Timestamp_Start;
            var frame_interval_tick = current_timestamp * TimeSpan.TicksPerMillisecond;
            var requested_timestamp = start_timestamp + frame_interval_tick;
            bool isNewFrame = HoloLens2FileStreamManager.Instance.PVCamera.IsNewFrame(requested_timestamp);

            if (isNewFrame)
            {
                HoloLens2FileStreamManager.Instance.PVCamera.UpdateLatestTexture(requested_timestamp);
                var pvFrame = HoloLens2FileStreamManager.Instance.PVCamera.GetFrame();

                if (show_pointcloud_longThrow)
                {
                    try
                    {
                        HoloLens2FileStreamManager.Instance.DepthCamera_LongThrow.UpdateLatestTexture(requested_timestamp);
                        var depthFrame_long = HoloLens2FileStreamManager.Instance.DepthCamera_LongThrow.GetFrame();
                        InstanceGenerator.ChangeDepthFrameObjectProperty(depthFrame_long, go_frame_depth_longThrow, show_pointcloud_longThrow, 0.01f);
                    }
                    catch (Exception e)
                    {
                        // 요청한 timestamp에 해당하는 depth frame이 없을 경우.
                    }
                }

                if (show_pointcloud_shortThrow)
                {
                    try
                    {
                        HoloLens2FileStreamManager.Instance.DepthCamera_ShortThrow.UpdateLatestTexture(requested_timestamp);
                        var depthFrame_short = HoloLens2FileStreamManager.Instance.DepthCamera_ShortThrow.GetFrame();
                        InstanceGenerator.ChangeDepthFrameObjectProperty(depthFrame_short, go_frame_depth_shortThrow, show_pointcloud_shortThrow, 0.001f);
                    }
                    catch (Exception e)
                    {
                        // 요청한 timestamp에 해당하는 depth frame이 없을 경우.
                    }
                }


                if (SocketClientManager.Instance.IsConnected() && send2server)
                {
                    byte[] data;
                    if (jpgCompression)
                        data = pvFrame.Encode2DataPacakge(ImageCompression.JPEG, jpgQuality);
                    else
                        data = pvFrame.Encode2DataPacakge();

                    SocketClientManager.Instance.Send(data);

                    ResultDataPackage frameData;
                    // Get current time as miliseconds.
                    var start = DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond;
                    int timeout = 100; //ms
                    while (true)
                    {
                        try
                        {
                            // 1. 가장 최신 데이터를 가져오는 방법.(Queue에서 마지막 데이터를 가져온다. 누락된 frame 생길 수 있지만 delay 없음.)
                            frameData = SocketClientManager.Instance.GetLatestReceivedData(); // 만약 새로운 frameData가 도착하지 않았다면 NoDataReceivedExecption이 이 함수 내부에서 발생한다.
                            Debug.LogFormat("Delay: {0}, frameID:{1}", frameData.frameInfo.GetTotalDelay(), frameData.frameInfo.frameID);
                            break;
                        }
                        catch (NoDataReceivedExecption e)
                        {
                            //Debug.Log(e.Message);
                            var spentTime = (DateTime.Now.Ticks / TimeSpan.TicksPerMillisecond) - start;

                            if (spentTime > timeout) //무한 루프 방지를 위해 응답이 timeout동안 없으면 실패로 간주한다.
                                Debug.LogError("network timeout error.");
                        }
                    }

                    // 여기서 서버로부터 결과가 왔을 때 해당 모듈의 OnReceivedResultData함수 호출.
                    wiseUI_Agent.GetComponent<ExampleModule>().OnReceivedResultData(frameData);

                    if (detectHandTracker)    // handtracker use toggle 가능한 UI패널 추가 필요.
                    {
                        try
                        {
                            HoloLens2FileStreamManager.Instance.DepthCamera_ShortThrow.UpdateLatestTexture(requested_timestamp);
                            var depthFrame_short = HoloLens2FileStreamManager.Instance.DepthCamera_ShortThrow.GetFrame();
                            wiseUI_Agent.GetComponent<HandManager>().OnReceivedResultData(pvFrame, depthFrame_short, frameData);
                        }
                        catch (Exception e)
                        {
                            // 요청한 timestamp에 해당하는 depth frame이 없을 경우.
                        }    
                    }
                    

                }
                if (detectMobilityState)
                {
                    wiseUI_Agent.GetComponent<VelocityBasedMobilityClassifier>().DetectMobilityState(current_timestamp / 1000f);
                    wiseUI_Agent.GetComponent<VelocityBasedMobilityClassifier>().SetActiveMobileStatePlate(true);
                }
                else
                {
                    wiseUI_Agent.GetComponent<VelocityBasedMobilityClassifier>().SetActiveMobileStatePlate(false);
                }

                //if (detectArUcoMarker)
                //{
                //    TagDetector td = tagDetector.GetComponent<TagDetector>();
                //    td.Initialize(pvFrame);
                //}

                //화면에 표시될 pvFrame을 렌더링한다.
                InstanceGenerator.ChangePVFrameObjectProperty(pvFrame, go_frame_pv, transparency);
            }

            prev_timestamp = current_timestamp;
        }

        void Update()
        {
            if (!isLoaded)
                return;

            //check current_frame_id is changed.
            if (current_timestamp != prev_timestamp && current_timestamp >= 0 && current_timestamp <= end_timestamp)
                OnTimeChanged();

            if (isPlaying)
            {
                if (current_timestamp + time_interval_ms >= 0 && current_timestamp + time_interval_ms <= end_timestamp)
                    current_timestamp += time_interval_ms;

                else
                    isPlaying = false;

                Repaint();
            }
        }

        private void OnDestroy()
        {
            DestroyAll();
            SavePreperence();
        }

        public void LoadPreperence()
        {
            if (PlayerPrefs.HasKey("hostIP"))
                hostIP = PlayerPrefs.GetString("hostIP");
            if (PlayerPrefs.HasKey("port"))
                port = PlayerPrefs.GetInt("port");
            if (PlayerPrefs.HasKey("dataset_path"))
            {
                dataset_path = PlayerPrefs.GetString("dataset_path");
            }
        }

        public void SavePreperence()
        {
            //Debug.Log("SavePreperence");
            PlayerPrefs.SetString("hostIP", hostIP);
            PlayerPrefs.SetInt("port", port);
            PlayerPrefs.SetString("dataset_path", dataset_path);
        }

        bool OptimziedToggleField(string label, bool toggle)
        {
            var backup = EditorGUIUtility.labelWidth;
            var textDimensions = GUI.skin.label.CalcSize(new GUIContent(label));
            EditorGUIUtility.labelWidth = textDimensions.x;
            toggle = EditorGUILayout.Toggle(label, toggle);
            EditorGUIUtility.labelWidth = backup;
            return toggle;
        }

        void OptimziedLabelField(string label, GUIStyle style = null)
        {
            var backup = EditorGUIUtility.labelWidth;
            var textDimensions = GUI.skin.label.CalcSize(new GUIContent(label));
            EditorGUIUtility.labelWidth = textDimensions.x;
            if (style != null)
                EditorGUILayout.LabelField(label, style);
            else
                EditorGUILayout.LabelField(label);
            EditorGUIUtility.labelWidth = backup;
        }

        [MenuItem("WiseUI/Simulator")]
        public static SimulatorWindow OpenWindow()
        {
            return Instance;
        }


        static SimulatorWindow instance;

        public static SimulatorWindow Instance
        {
            get
            {
                if (instance)
                    return instance;
                else
                {
                    instance = GetWindow<SimulatorWindow>(false, "WiseUI Simulator");
                    instance.LoadPreperence();

                    return instance;
                }
            }
        }
    }


}
