using Microsoft.Azure.SpatialAnchors;
using Microsoft.Azure.SpatialAnchors.Unity;
using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.XR.ARFoundation;

#if WINDOWS_UWP
using Windows.Storage;
#endif

[RequireComponent(typeof(ARAnchorManager))]
[RequireComponent(typeof(SpatialAnchorManager))]
public class SpatialAnchorController : MonoBehaviour
{
    private bool[] m_wasHandAirTapping = { false, false };

    [SerializeField]
    [Tooltip("Spatial Anchor로 관리할 객체를 등록.")]
    private GameObject[] anchors;
    private Vector3[] initialPositions;
    private Quaternion[] initialRotation;

    /// <summary>
    /// Used to manage Azure Spatial Anchor sessions and queries in a Unity scene
    /// </summary>
    private SpatialAnchorManager m_cloudSpatialAnchorManager;

    /// <summary>
    /// The SpatialAnchorManager must have its session created after the Start() method.
    /// Instead, we create it if it has not been created while starting the session.
    /// </summary>
    private bool m_cloudSpatialAnchorManagerSessionCreated = false;

    /// <summary>
    /// All anchor GameObjects found by or manually created in this demo. Used to ensure these are all
    /// destroyed when the session is stopped.
    /// </summary>
    private List<GameObject> m_foundOrCreatedAnchorObjects;

    /// <summary>
    /// If a new cloud session is started after we have previously saved cloud anchors, this spatial
    /// anchor watcher will rediscover them.
    /// </summary>
    private CloudSpatialAnchorWatcher m_cloudSpatialAnchorWatcher;

    /// <summary>
    /// The Ids of all cloud spatial anchors created during this demo. Used to when creating a watcher
    /// to re-discover these anchors after stopping and starting the cloud session.
    /// SSJ: id로 관리하기위해 기존 List에서 Dictionary로 변경 
    /// </summary>
    private Dictionary<string, string> m_cloudSpatialAnchorIDs = new Dictionary<string, string>();

    public async void OnStartSessionButtonPressed() => await StartSessionAsync();

    public void OnStopSessionButtonPressed() => StopAndCleanupSession();
    public void OnUpdateAnchorButtonPressed() => UpdateSpatialAnchors();

    /// <summary>
    /// Setup references to other components on this GameObject.
    /// </summary>
    private void Awake()
    {
        m_cloudSpatialAnchorManager = GetComponent<SpatialAnchorManager>();
        m_foundOrCreatedAnchorObjects = new List<GameObject>();
        // Anchor를 받아 올 시 초기 위치 저장을 위해 만들어둠
        initialPositions = new Vector3[anchors.Length];
        initialRotation = new Quaternion[anchors.Length];
        for (int i = 0; i < anchors.Length; i++)
        {
            initialPositions[i] = Vector3.zero;
            initialRotation[i] = Quaternion.identity;
        }
    }

    /// <summary>
    /// Ensure this sample scene is properly configured, then create the spatial anchor manager session.
    /// </summary>
    private void Start()
    {
        // Ensure the ARAnchorManager is properly setup for this sample
        ARAnchorManager arAnchorManager = GetComponent<ARAnchorManager>();
        if (!arAnchorManager.enabled || arAnchorManager.subsystem == null)
        {
            Debug.LogError($"ARAnchorManager not enabled or available; sample anchor functionality will not be enabled.");
            return;
        }

        // Ensure anchor prefabs are properly setup for this sample
        if (arAnchorManager.anchorPrefab != null)
        {
            // When using ASA, ARAnchors are managed internally and should not be instantiated with custom behaviors
            Debug.LogError("The anchor prefab for ARAnchorManager must be set to null.");
            return;
        }


        // Ensure the SpatialAnchorManager is properly setup for this sample
        if (string.IsNullOrWhiteSpace(m_cloudSpatialAnchorManager.SpatialAnchorsAccountId) ||
            string.IsNullOrWhiteSpace(m_cloudSpatialAnchorManager.SpatialAnchorsAccountKey) ||
            string.IsNullOrWhiteSpace(m_cloudSpatialAnchorManager.SpatialAnchorsAccountDomain))
        {
            Debug.LogError($"{nameof(SpatialAnchorManager.SpatialAnchorsAccountId)}, {nameof(SpatialAnchorManager.SpatialAnchorsAccountKey)} and {nameof(SpatialAnchorManager.SpatialAnchorsAccountDomain)} must be set on {nameof(SpatialAnchorManager)}");
            return;
        }

        m_cloudSpatialAnchorManager.LogDebug += (sender, args) => Debug.Log($"Debug: {args.Message}");
        m_cloudSpatialAnchorManager.Error += (sender, args) => Debug.LogError($"Error: {args.ErrorMessage}");
        m_cloudSpatialAnchorManager.AnchorLocated += SpatialAnchorManagerAnchorLocated;
        m_cloudSpatialAnchorManager.LocateAnchorsCompleted += (sender, args) => Debug.Log("Locate anchors completed!");

        Debug.Log($"Tap the button below to create and start the session.");

        //SSJ: code for save & load Spatial Anchor
        GetAzureAnchorIdsFromDisk();
        Debug.Log("Start(): m_cloudSpatialAnchorIDs");
        foreach (var kvp in m_cloudSpatialAnchorIDs)
        {
            Debug.Log($"Key: {kvp.Key}, Value: {kvp.Value}");
        }
    }

    private void OnDestroy() => StopAndCleanupSession(); // When this sample is destroyed, ensure the session is cleaned up.

    /// <summary>
    /// Start the session and begin searching for anchors previously saved.
    /// </summary>
    private async Task StartSessionAsync()
    {
        // CreateSessionAsync cannot be called during Start(), since the SpatialAnchorManager may not have Start()ed yet itself.
        // Instead, we ensure the session is created before we start it.
        if (!m_cloudSpatialAnchorManagerSessionCreated)
        {
            await m_cloudSpatialAnchorManager.CreateSessionAsync();
            m_cloudSpatialAnchorManagerSessionCreated = true;
        }

        if (m_cloudSpatialAnchorManager.IsSessionStarted)
        {
            Debug.LogWarning("Cannot start session; session already started.");
            return;
        }

        // Start the session
        Debug.Log("Starting session...");
        await m_cloudSpatialAnchorManager.StartSessionAsync();
        Debug.Log($"Session started!");

        // And create the watcher to look for any created anchors
        if (m_cloudSpatialAnchorIDs.Count > 0)
        {
            Debug.Log($"Creating watcher to look for {m_cloudSpatialAnchorIDs.Count} spatial anchors");
            AnchorLocateCriteria anchorLocateCriteria = new AnchorLocateCriteria();
            anchorLocateCriteria.Identifiers = m_cloudSpatialAnchorIDs.Values.ToArray();
            m_cloudSpatialAnchorWatcher = m_cloudSpatialAnchorManager.Session.CreateWatcher(anchorLocateCriteria);
            Debug.Log($"Watcher created!");

            //SSJ: 본 클래스에 있는 m_cloudSpatialAnchorManager.AnchorLocated += SpatialAnchorManagerAnchorLocated; 에서 불러온 watcher에 따라 해당 객체 재배치진행
        }
    }

    /// <summary>
    /// For each anchor located by the spatial anchor manager, instantiate and setup a corresponding GameObject.
    /// SSJ: Cloud 에서 받아온 Anchor들이 생성되는 부분
    /// </summary>
    private void SpatialAnchorManagerAnchorLocated(object sender, AnchorLocatedEventArgs args)
    {
        //JSS: 새로 앵커를 만들고 해당 코드를 로컬에 저장하는것까지는 되는데, 해당 코드로 cloud에 있는걸 못가져 오는 듯 한?
        Debug.Log($"Anchor recognized as a possible anchor {args.Identifier} {args.Status}");

        if (args.Status == LocateAnchorStatus.Located)
        {
            UnityDispatcher.InvokeOnAppThread(() =>
            {
                CloudSpatialAnchor cloudSpatialAnchor = args.Anchor;
                Pose anchorPose = cloudSpatialAnchor.GetPose();

                // The method CloudToNative below adds a new ARAnchor component to the GameObject,
                // replacing any existing ARAnchor component. Scripts on prefabs intended for use
                // with ASA should wait until Start() to obtain references to ARAnchor components.
                // JSS: 기존 코드는 새로운 객체를 생성하고 아래의 스크립트를 등록, 하단의 기존 객체를 찾아서 <CloudNativeAnchor>스크립트 등록으로 변경
                /*
                GameObject anchorGameObject = Instantiate(m_sampleSpatialAnchorPrefab, anchorPose.position, anchorPose.rotation);
                anchorGameObject.AddComponent<CloudNativeAnchor>().CloudToNative(cloudSpatialAnchor);
                anchorGameObject.GetComponent<PersistableAnchorVisuals>().Name = cloudSpatialAnchor.Identifier;
                anchorGameObject.GetComponent<PersistableAnchorVisuals>().Persisted = true;

                m_foundOrCreatedAnchorObjects.Add(anchorGameObject);
                */

                Debug.Log($"AnchorLocated: cloudSpatialAnchor.Identifier - {cloudSpatialAnchor.Identifier}, anchor postion - {anchorPose.position}");
                if (m_cloudSpatialAnchorIDs.ContainsValue(cloudSpatialAnchor.Identifier))
                {
                    foreach (var pair in m_cloudSpatialAnchorIDs)
                    {
                        if (pair.Value == cloudSpatialAnchor.Identifier)
                        {
                            int index = int.Parse(pair.Key);
                            Debug.Log("SpatialAnchorManagerAnchorLocated: formal obj pos - " + anchors[index].transform.position);
                            anchors[index].transform.position = anchorPose.position;
                            anchors[index].transform.rotation = anchorPose.rotation;
                            Debug.Log("SpatialAnchorManagerAnchorLocated: after obj pos - " + anchors[index].transform.position);
                            initialPositions[index] = anchors[index].transform.position;
                            initialRotation[index] = anchors[index].transform.rotation;
                            anchors[index].AddComponent<ARAnchor>();
                            anchors[index].AddComponent<CloudNativeAnchor>().CloudToNative(cloudSpatialAnchor);
                        }
                    }
                }
            });
        }
    }

    private void UpdateSpatialAnchors()
    {
        //SSJ Anchor위치가 처음과 달라진 것이 있다면 Cloud Anchor를 삭제하고 새로 등록함
        for (int i = 0; i < anchors.Length; i++)
        {
            Debug.Log($"UpdateSpatialAnchors - anchors[i].transform.position{anchors[i].transform.position}");
            Debug.Log($"UpdateSpatialAnchors - initialPositions[i]{initialPositions[i]}");
            if (anchors[i].transform.position != initialPositions[i])
            {
                DeleteAnchorFromCloudAsync(anchors[i]);
                //삭제가 끝나고 생성되도록 변경(Delete에 save를 넣었음) -> 이후 DeleteThenSave 함수 만들어야할듯
                //SaveAnchorToCloudAsync(anchors[i]);
            }
        }
    }

    /// <summary>
    /// Stop the session and cleanup any anchors created.
    /// </summary>
    private void StopAndCleanupSession()
    {
        //SSJ 생성했던 CloudeAnchor들의 id를 로컬에 저장
        SaveAzureAnchorIdsToDisk();

        // Stop the session
        if (!m_cloudSpatialAnchorManager.IsSessionStarted)
        {
            Debug.LogWarning("Cannot stop session; session has not started.");
            return;
        }

        Debug.Log("Stopping session...");
        m_cloudSpatialAnchorManager.StopSession();
        m_cloudSpatialAnchorManager.Session.Reset();

        // Stop the watcher, if it exists
        if (m_cloudSpatialAnchorWatcher != null)
        {
            m_cloudSpatialAnchorWatcher.Stop();
            m_cloudSpatialAnchorWatcher = null;
        }
        Debug.Log("Session stopped!");
    }



    /// <summary>
    /// Delete the cloud anchor corresponding to a local ARAnchor.
    /// </summary>
    private async void DeleteAnchorFromCloudAsync(GameObject gameObject)
    {
        Debug.Log($"DeleteAnchorFromCloudAsync - gameObject: {gameObject}");
        CloudNativeAnchor cloudNativeAnchor = gameObject.GetComponent<CloudNativeAnchor>();
        Debug.Log($"cloudNativeAnchor: {cloudNativeAnchor}"); // SSJ: 여기가 계속 null 로 나오네, 기능 고치고 예외처리도 해야 할 듯
        if (cloudNativeAnchor != null)
        {
            CloudSpatialAnchor cloudSpatialAnchor = cloudNativeAnchor.CloudAnchor;
            Debug.Log($"Deleting cloud anchor: {cloudSpatialAnchor.Identifier}");
            await m_cloudSpatialAnchorManager.DeleteAnchorAsync(cloudSpatialAnchor);
            m_cloudSpatialAnchorIDs.Remove(IndexInAnchors(gameObject, anchors).ToString());
            Destroy(cloudNativeAnchor);
        }
        SaveAnchorToCloudAsync(gameObject);
    }


    /// <summary>
    /// Save a local ARAnchor as a cloud anchor.
    /// </summary>
    private async void SaveAnchorToCloudAsync(GameObject gameObject)
    {
        gameObject.AddComponent<ARAnchor>();
        CloudNativeAnchor cloudNativeAnchor = gameObject.AddComponent<CloudNativeAnchor>();

        await cloudNativeAnchor.NativeToCloud();
        CloudSpatialAnchor cloudSpatialAnchor = cloudNativeAnchor.CloudAnchor;

        // In this sample app, the cloud anchors are deleted explicitly.
        // Here, we show how to set an anchor to expire automatically.
        cloudSpatialAnchor.Expiration = DateTimeOffset.Now.AddDays(7);

        while (!m_cloudSpatialAnchorManager.IsReadyForCreate)
        {
            await Task.Delay(1000);
            float createProgress = m_cloudSpatialAnchorManager.SessionStatus.RecommendedForCreateProgress;
            Debug.Log($"Move your device to capture more environment data: {createProgress:0%}");
        }


        // Now that the cloud spatial anchor has been prepared, we can try the actual save here
        Debug.Log($"Saving cloud anchor...");
        await m_cloudSpatialAnchorManager.CreateAnchorAsync(cloudSpatialAnchor);

        bool saveSucceeded = cloudSpatialAnchor != null;
        if (!saveSucceeded)
        {
            Debug.LogError("Failed to save, but no exception was thrown.");
            return;
        }

        m_cloudSpatialAnchorIDs[IndexInAnchors(gameObject, anchors).ToString()] = cloudSpatialAnchor.Identifier;
        Debug.Log($"Save cloud anchor: {cloudSpatialAnchor.Identifier}");
        Debug.Log($"anchor position: {cloudSpatialAnchor.GetPose().position}");
    }

    public void DeleteAnchorToMove(GameObject targetObject)
    {
        ARAnchor arAnchor = targetObject.GetComponent<ARAnchor>();
        if (arAnchor != null)
        {
            Destroy(arAnchor);
            Debug.Log("ARAnchor component deleted from the target object.");
        }

        CloudNativeAnchor cloudNativeAnchor = targetObject.GetComponent<CloudNativeAnchor>();
        if (cloudNativeAnchor != null)
        {
            Destroy(cloudNativeAnchor);
            Debug.Log("CloudNativeAnchor component deleted from the target object.");
        }
    }



    private int IndexInAnchors(GameObject targetObject, GameObject[] array)
    {
        for (int i = 0; i < array.Length; i++)
        {
            if (array[i] == targetObject)
            {
                return i;
            }
        }
        return -1;
    }

    private string GetAnchorIdForUI(string id)
    {
        if (m_cloudSpatialAnchorIDs.ContainsKey(id))
        {
            return m_cloudSpatialAnchorIDs[id];
        }
        return null;
    }

    //SSJ: code for save & load Spatial Anchor
    private void SaveAzureAnchorIdsToDisk()
    {
        Debug.Log("\nAnchorModuleScript.SaveAzureAnchorIdsToDisk()");

        string filename = "SavedAzureAnchorIDs.txt";
        string path = Application.persistentDataPath;

#if WINDOWS_UWP
            StorageFolder storageFolder = ApplicationData.Current.LocalFolder;
            path = storageFolder.Path.Replace('\\', '/') + "/";
#endif

        string filePath = Path.Combine(path, filename);

        string serializedIDs = JsonConvert.SerializeObject(m_cloudSpatialAnchorIDs, Formatting.Indented);

        File.WriteAllText(filePath, serializedIDs);

        Debug.Log($"Current Azure anchor IDs successfully saved to path '{filePath}'");

        Debug.Log("SaveToLocal: serializedIDs - " + serializedIDs);
        Debug.Log("SaveToLocal: m_cloudSpatialAnchorIDs");
        foreach (var kvp in m_cloudSpatialAnchorIDs)
        {
            Debug.Log($"Key: {kvp.Key}, Value: {kvp.Value}");
        }
    }

    private void GetAzureAnchorIdsFromDisk()
    {
        Debug.Log("\nAnchorModuleScript.GetAzureAnchorIdsFromDisk()");

        string filename = "SavedAzureAnchorIDs.txt";
        string path = Application.persistentDataPath;

#if WINDOWS_UWP
            StorageFolder storageFolder = ApplicationData.Current.LocalFolder;
            path = storageFolder.Path.Replace('\\', '/') + "/";
#endif

        string filePath = Path.Combine(path, filename);
        if (File.Exists(filePath))
        {
            string serializedIDs = File.ReadAllText(filePath);

            // Deserialize the string to a list using "|" as a separator
            m_cloudSpatialAnchorIDs = JsonConvert.DeserializeObject<Dictionary<string, string>>(serializedIDs);

            Debug.Log($"Current Azure anchor IDs successfully updated with saved IDs from path '{filePath}'");
            foreach (var pair in m_cloudSpatialAnchorIDs)
            {
                Debug.Log($"Key: {pair.Key}, Value: {pair.Value}");
            }
        }
        else
        {
            Debug.Log($"No saved Azure anchor IDs found at path '{filePath}'");
        }
    }
}
