using ARRC.Framework;
using Codice.Utils;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Unity.EditorCoroutines.Editor;
using UnityEditor;
using UnityEditor.Experimental.GraphView;
using UnityEditorInternal;
using UnityEngine;
using UnityEngine.Experimental.XR.Interaction;
using WiseUI.Base;
using WiseUI.Simulator;
using static Codice.Client.Common.Servers.RecentlyUsedServers;
using static UnityEditor.Progress;


[System.Serializable]
public class AnnoConfig
{
    public List<AnnoItem> items = new List<AnnoItem>();
    public long startTimestamp;
    public long endTimestamp;
    public string dataset_name;
    public string dataset_split;
    public string dataset_split_type;
    public int scene_number;
}
[Serializable]
public class AnnoItem
{
    public int id;
    public string name;
    public Vector3 position;
    public Quaternion rotation;
    public Vector3 scale;
}
public class ObjectAnnotationWindow : EditorWindow
{
    Vector3 scrollPosition;

    [SerializeField] List<GameObject> bboxes = new List<GameObject>();

    [SerializeField]
    string conf_path;
    [SerializeField]
    string result_path;

    [SerializeField] GameObject environment;
    [SerializeField] GameObject bbox_parent;
    const string name_ply_tsdf = "PointCloud_TSDF";
    const string name_config_file = "anno_config.json";
    const string path_bbox_prefab = "Assets/Scripts/WiseUIModules/Object/DatasetAnnotation/BBox.prefab";

    [SerializeField]
    bool isLoaded = false;


    [SerializeField]
    GameObject wiseUI_Agent;
    [SerializeField]
    public GameObject go_frame_pv;
    [SerializeField]
    public GameObject go_frame_depth_longThrow;
    [SerializeField]
    public GameObject go_frame_depth_shortThrow;

    long startTimestamp = 0;
    long endTimestamp = 0;

    [SerializeField]
    string dataset_name;
    [SerializeField]
    string dataset_split;
    [SerializeField]
    string dataset_split_type;
    [SerializeField]
    int scene_number;

    bool generate = false;
    EditorCoroutine coroutine_generate;

    private void OnGUI()
    {

        if (GUILayout.Button("New config file"))
        {
            var path = EditorUtility.SaveFilePanel("Select configuration file", null, "anno_config", "json");

            if (!string.IsNullOrEmpty(path))
            {
                conf_path = path;
                bboxes.Clear();

                if (bbox_parent != null)
                    DestroyImmediate(bbox_parent);

                bbox_parent = new GameObject("BBoxes");

                LoadConfiguration(conf_path);
                isLoaded = true;
                SavePreperence();
            }
        }

        if (GUILayout.Button("Load config file"))
        {
            var path = EditorUtility.OpenFilePanel("Select config file", conf_path, "json");

            if (!string.IsNullOrEmpty(path))
            {
                conf_path = path;
                bboxes.Clear();

                if (bbox_parent != null)
                    DestroyImmediate(bbox_parent);

                bbox_parent = new GameObject("BBoxes");

                LoadConfiguration(conf_path);
                isLoaded = true;
                SavePreperence();
            }
        }

        SerializedObject serializedObject = new SerializedObject(this);
        EditorGUILayout.PropertyField(serializedObject.FindProperty("result_path"), true);
        EditorGUILayout.PropertyField(serializedObject.FindProperty("dataset_split"), true);
        EditorGUILayout.PropertyField(serializedObject.FindProperty("dataset_split_type"), true);
        EditorGUILayout.PropertyField(serializedObject.FindProperty("scene_number"), true);
        serializedObject.ApplyModifiedProperties();
        serializedObject.Update();

        if (isLoaded)
        {
            AnnotationMenu();

            if (!generate)
            {
                EditorGUILayout.BeginHorizontal();
                GUILayout.Label(startTimestamp.ToString());
                if (GUILayout.Button("Pick_start"))
                {
                    var simulatorWindow = GetWindow<SimulatorWindow>(false, "WiseUI Simulator");
                    startTimestamp = simulatorWindow.current_timestamp;

                }
                GUILayout.Label(endTimestamp.ToString());
                if (GUILayout.Button("Pick_end"))
                {
                    var simulatorWindow = GetWindow<SimulatorWindow>(false, "WiseUI Simulator");
                    endTimestamp = simulatorWindow.current_timestamp;
                }
                EditorGUILayout.EndHorizontal();

                if (GUILayout.Button("Generate dataset"))
                {
                    var path = EditorUtility.OpenFolderPanel("Select result folder", result_path, "");

                    if (!string.IsNullOrEmpty(path))
                    {
                        result_path = path;
                        //GenerateDataset();
                        generate = true;

                        var start_timestamp = HoloLens2FileStreamManager.Instance.PVCamera.Timestamp_Start;
                        var frame_interval_tick = startTimestamp * TimeSpan.TicksPerMillisecond;
                        var requested_timestamp = start_timestamp + frame_interval_tick;

                        HoloLens2FileStreamManager.Instance.PVCamera.UpdateLatestTexture(requested_timestamp);

                        coroutine_generate = EditorCoroutineUtility.StartCoroutine(GenerateDataset(), this);
                        SavePreperence();
                    }
                }
            }
            else
            {
                if (GUILayout.Button("Cancel"))
                {
                    EditorCoroutineUtility.StopCoroutine(coroutine_generate);
                    generate = false;
                    Debug.Log("Cancel generating dataset");
                }
            }
        }

    }

    IEnumerator GenerateDataset()
    {
        Debug.Log("Start generating dataset");
        
        var pvCamera = HoloLens2FileStreamManager.Instance.PVCamera;
        var renderCameraObject = GameObject.FindGameObjectWithTag("PVCamera");
        var targetObjectList = GameObject.FindGameObjectsWithTag("OBJECT").ToList();

        var obj_size_list = bboxes.Select(i => i.transform.lossyScale * 1000.0f).ToList();
        BOPHelper bopHelper = new BOPHelper(result_path, dataset_name, dataset_split, obj_size_list);
        bopHelper.BeginRecordingScene(scene_number);

        var backup_renderMode = renderCameraObject.GetComponent<BlendDuringRender>().renderMode;
        renderCameraObject.GetComponent<BlendDuringRender>().renderMode = RenderMode.Normal;
        while (true)
        {
            try
            {
                pvCamera.UpdateLatestTexture();
                long im_id = pvCamera.Timestamp;
                Debug.LogFormat("Timestamp: {0}", pvCamera.Timestamp);

                var pvFrame = pvCamera.GetFrame();

                var rh = pvFrame.Cam2World();

                bopHelper.AddJsonLine(im_id, rh, bboxes.Select(i => i.transform.localToWorldMatrix).ToList());
                string rgbFilePath = bopHelper.GetRGBFilePath(im_id);
                int cullingMask = (1 << LayerMask.NameToLayer("OBJECT")) | (1 << LayerMask.NameToLayer("BACKGROUND"));

                //copy file
                File.Copy(pvFrame.TexturePath, rgbFilePath, true);
                // SaveMaskImage(bopHelper, im_id, camCompo);
                // SaveVisibleMaskImage(bopHelper, im_id, renderCameraObject.GetComponent<Camera>()
                //    , targetObjectList, pvFrame.cameraIntrinsic.imageWidth, pvFrame.cameraIntrinsic.imageWidth);
                // SaveImage(rgbFilePath, camCompo, cullingMask, false);

            }
            catch (Exception e)
            {
                generate = false;
                break;
            }
            var start_timestamp = pvCamera.Timestamp_Start;
            var frame_interval_tick = endTimestamp * TimeSpan.TicksPerMillisecond;
            var requested_timestamp = start_timestamp + frame_interval_tick;

            if (pvCamera.Timestamp > requested_timestamp)
            {
                generate = false;
                break;
            }

       
            yield return null;


        }
        bopHelper.EndRecordingScene();

        /*
            int total_count = num_imgs_for_each_scene * (sceneCount_train + sceneCount_val + sceneCount_test);
            int sampleCount = 0;
            progress = 0;
        */
    }

    void SaveVisibleMaskImage(BOPHelper bopHelper, long im_id, Camera camCompo, List<GameObject> targetObjectList, int width, int height)
    {
        var black_mat = new Material(Shader.Find("Unlit/Color"));
        black_mat.color = new Color(0, 0, 0, 0);

        var white_mat = new Material(Shader.Find("Unlit/Color"));
        white_mat.color = new Color(1, 1, 1, 1);


        var backup_materials_target = targetObjectList
          .Select(i => i.GetComponent<MeshRenderer>().sharedMaterial).ToList();


        for (int gt_id = 0; gt_id < targetObjectList.Count; gt_id++)
        {
            var target_obj = targetObjectList[gt_id];

            for (int ti = 0; ti < targetObjectList.Count; ti++)
                targetObjectList[ti].GetComponent<MeshRenderer>().sharedMaterial = black_mat;

            target_obj.GetComponent<MeshRenderer>().sharedMaterial = white_mat;

            string visibleMaskFilePath = bopHelper.GetVisibleMaskFilePath(im_id, gt_id);

            int cullingMask = 1 << LayerMask.NameToLayer("OBJECT");
            WriteMaskImage(visibleMaskFilePath, width, height, camCompo, cullingMask);

        }
        for (int gt_id = 0; gt_id < targetObjectList.Count; gt_id++)
            targetObjectList[gt_id].GetComponent<MeshRenderer>().sharedMaterial = backup_materials_target[gt_id];
        
    }
    //void SaveMaskImage(BOPHelper bopHelper, int im_id, Camera camCompo)
    //{
    //    var black_mat = new Material(Shader.Find("Unlit/Color"));
    //    black_mat.color = new Color(0, 0, 0, 0);

    //    var white_mat = new Material(Shader.Find("Unlit/Color"));
    //    white_mat.color = new Color(1, 1, 1, 1);

    //    var backup_materials_target = targetObjectList
    //       .Select(i => i.GetComponent<MeshRenderer>().sharedMaterial).ToList();

    //    for (int gt_id = 0; gt_id < dummyObjectList.Count; gt_id++)
    //        dummyObjectList[gt_id].layer = LayerMask.NameToLayer("BACKGROUND");

    //    for (int gt_id = 0; gt_id < targetObjectList.Count; gt_id++)
    //    {
    //        for (int oid = 0; oid < targetObjectList.Count; oid++)
    //            targetObjectList[oid].layer = LayerMask.NameToLayer("BACKGROUND");

    //        var obj = targetObjectList[gt_id];
    //        obj.layer = LayerMask.NameToLayer("OBJECT");
    //        obj.GetComponent<MeshRenderer>().sharedMaterial = white_mat;

    //        string maskFilePath = bopHelper.GetMaskFilePath(im_id, gt_id);

    //        int cullingMask = 1 << LayerMask.NameToLayer("OBJECT");
    //        SaveImage(maskFilePath, camCompo, cullingMask, true); //render only target

    //    }
    //    for (int gt_id = 0; gt_id < dummyObjectList.Count; gt_id++)
    //        dummyObjectList[gt_id].layer = LayerMask.NameToLayer("OBJECT");

    //    for (int gt_id = 0; gt_id < targetObjectList.Count; gt_id++)
    //    {
    //        targetObjectList[gt_id].layer = LayerMask.NameToLayer("OBJECT");
    //        targetObjectList[gt_id].GetComponent<MeshRenderer>().sharedMaterial = backup_materials_target[gt_id];
    //    }

    //}

    void WriteMaskImage(string filepath, int width, int height, Camera cam, int cullingMask)
    {
        RenderTexture tempRT = new RenderTexture(width, height, 24);
        Texture2D tex;
        cam.cullingMask = cullingMask; // (1 << LayerMask.NameToLayer("OBJECT")) | (1 << LayerMask.NameToLayer("BACKGROUND"));
      
        tempRT.format = RenderTextureFormat.R8;
        cam.clearFlags = CameraClearFlags.SolidColor;
        cam.backgroundColor = Color.black;
        cam.targetTexture = tempRT;
        //cam.RenderWithShader(Shader.Find("Unlit/Texture"), null);
        cam.Render();
        tex = new Texture2D(width, height, TextureFormat.R8, false);
       

        RenderTexture.active = tempRT;
        tex.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        RenderTexture.active = null;
        cam.targetTexture = null;

        byte[] bytes = tex.EncodeToPNG();
        System.IO.File.WriteAllBytes(filepath, bytes);
    }

    void AnnotationMenu()
    {
        GUIStyle headerStyle = new GUIStyle(EditorStyles.label);
        headerStyle.alignment = TextAnchor.MiddleCenter;

        EditorGUILayout.BeginHorizontal(EditorStyles.toolbar);
        GUILayout.Label("id", headerStyle, GUILayout.Width(40));
        GUILayout.Label("name", headerStyle);
        GUILayout.Box("", GUIStyle.none, GUILayout.Width(50));
        EditorGUILayout.EndHorizontal();

        scrollPosition = EditorGUILayout.BeginScrollView(scrollPosition);
        GUIContent useContent = new GUIContent(">", "Restore");
        GUIContent AddContent = new GUIContent("+", "add");
        GUIContent deleteContent = new GUIContent("-", "Remove");

        int addIndex = -1;
        int removeIdx = -1;
        for (int i = 0; i < bboxes.Count; i++)
        {
            GameObject item = bboxes[i];
            EditorGUILayout.BeginHorizontal();
            GUILayout.Label(i.ToString(), GUILayout.Width(40));
            //GUILayout.Label(((DTConfigurationItem)item).title);
            GUILayout.Label(item.gameObject.name, GUILayout.Width(120));
            item.transform.parent = bbox_parent.transform;

            //DateTime time = new DateTime(item.timestamp);

            //GUILayout.Label(time.ToString("yyyy-MM-dd HH:mm"), GUILayout.Width(120));

            if (GUILayout.Button(useContent, GUILayout.Width(20)))
            {
                Selection.activeGameObject = bboxes[i];
                SceneView.lastActiveSceneView.FrameSelected();
            }
            if (GUILayout.Button(AddContent, GUILayout.Width(20)))
            {
                addIndex = i;
            }
            if (GUILayout.Button(deleteContent, GUILayout.Width(20)))
            {
                removeIdx = i;
            }
            EditorGUILayout.EndHorizontal();
        }
        if (removeIdx != -1)
        {
            DestroyImmediate(bboxes[removeIdx]);
            bboxes.RemoveAt(removeIdx);
        }
        if (addIndex != -1)
        {
            //Find prefab in project.
            GameObject bbox_obj = Instantiate(AssetDatabase.LoadAssetAtPath<GameObject>(path_bbox_prefab));
            bbox_obj.name = "obj_" + (addIndex + 1).ToString();
            bbox_obj.transform.parent = bbox_parent.transform;
            if (bboxes.Count > 0)
            {
                bbox_obj.transform.position = bboxes[addIndex].transform.position;
                bbox_obj.transform.rotation = bboxes[addIndex].transform.rotation;
                bbox_obj.transform.localScale = bboxes[addIndex].transform.localScale;
            }
            bboxes.Insert(addIndex + 1, bbox_obj);
            //SaveConfiguration(string.Format("{0}/{1}", dataset_path, name_config_file));
        }
        if (bboxes.Count == 0 && GUILayout.Button("+"))
        {
            //Find prefab in project.
            GameObject bbox_obj = Instantiate(AssetDatabase.LoadAssetAtPath<GameObject>(path_bbox_prefab));
            bbox_obj.name = "obj_" + bboxes.Count.ToString();
            bbox_obj.transform.parent = bbox_parent.transform;
            bboxes.Add(bbox_obj);
        }

        if (GUILayout.Button("Save this configuration"))
        {
            var dir = Path.GetDirectoryName(conf_path);
            var filename = Path.GetFileNameWithoutExtension(conf_path);
            var path = EditorUtility.SaveFilePanel("Save configuration", dir, filename, "json");

            if (!string.IsNullOrEmpty(path))
            {
                conf_path = path;
                SaveConfiguration(conf_path);
            }

        }
        EditorGUILayout.EndScrollView();
    }

    public void LoadConfiguration(string path)
    {
        if (System.IO.File.Exists(path))
        {
            string json = System.IO.File.ReadAllText(path);
            AnnoConfig list = JsonUtility.FromJson<AnnoConfig>(json);

            foreach (var bbox in bboxes)
            {
                DestroyImmediate(bbox);
            }
            bboxes.Clear();

            foreach (var item in list.items)
            {
                GameObject new_bbox = Instantiate(AssetDatabase.LoadAssetAtPath<GameObject>(path_bbox_prefab));
                new_bbox.name = item.name;
                new_bbox.transform.position = item.position;
                new_bbox.transform.rotation = item.rotation;
                new_bbox.transform.localScale = item.scale;
                new_bbox.transform.parent = bbox_parent.transform;
                new_bbox.tag = "BBox";
                bboxes.Add(new_bbox);

            }
            startTimestamp = list.startTimestamp;
            endTimestamp = list.endTimestamp;
            dataset_name = list.dataset_name;
            dataset_split = list.dataset_split;
            dataset_split_type = list.dataset_split_type;
            scene_number = list.scene_number;
        }
        else
        {
            Debug.Log("No configuration file");
        }
    }
    public void SaveConfiguration(string path)
    {
        // write position, rotation, scale for each bbox to json.
        AnnoConfig anno = new AnnoConfig();
        for (int i = 0; i < bboxes.Count; i++)
        {
            var bbox = bboxes[i];

            var item = new AnnoItem();
            item.id = i;
            item.name = bbox.name;
            item.position = bbox.transform.position;
            item.rotation = bbox.transform.rotation;
            item.scale = bbox.transform.localScale;
            anno.items.Add(item);
        }
        anno.startTimestamp = startTimestamp;
        anno.endTimestamp = endTimestamp;
        anno.dataset_name = dataset_name;
        anno.dataset_split = dataset_split;
        anno.dataset_split_type = dataset_split_type;
        anno.scene_number = scene_number;

        string json = JsonUtility.ToJson(anno, true);
        System.IO.File.WriteAllText(path, json);
        Debug.Log("Save configuration to " + path);
    }
    public void OnDestroy()
    {
        SavePreperence();
        if (bbox_parent)
            DestroyImmediate(bbox_parent);
    }
    public void LoadPreperence()
    {
        if (PlayerPrefs.HasKey("anno_conf_path"))
        {
            conf_path = PlayerPrefs.GetString("anno_conf_path");
        }

        if (PlayerPrefs.HasKey("anno_result_path"))
        {
            result_path = PlayerPrefs.GetString("anno_result_path");
        }
    }

    public void SavePreperence()
    {
        //Debug.Log("SavePreperence");
        PlayerPrefs.SetString("anno_conf_path", conf_path);
        PlayerPrefs.SetString("anno_result_path", result_path);
    }

    [MenuItem("Tools/Annotation Helper")]
    public static ObjectAnnotationWindow OpenWindow()
    {
        return Instance;
    }
    static ObjectAnnotationWindow instance;


    public static ObjectAnnotationWindow Instance
    {
        get
        {
            if (instance)
                return instance;
            else
            {
                instance = GetWindow<ObjectAnnotationWindow>(false, "Annotation Helper");

                instance.LoadPreperence();
                return instance;
            }
        }
    }
}

