using System.Net;
using System.Net.Sockets;
using System.Net.Security;
using System.Security.Cryptography.X509Certificates;
using System.Threading;
using UnityEngine;
using System.IO;
using System.Collections;
using TMPro;

public class WebClientDownload : MonoBehaviour
{
    private UdpClient udpClient;
    private Thread receiveThread;
    private int port = 5005;
    
    public TextMeshProUGUI statusText;
    public string serverUrl = "http://192.168.50.247:8000";
    
    private bool shouldDownloadGlb = false;
    private string glbFilePath;
    private GameObject previousGlb;
    private bool downloadComplete = false;
    private bool downloadSuccess = false;
    private string downloadError = "";

    void Start()
    {
        ServicePointManager.ServerCertificateValidationCallback = 
            (sender, certificate, chain, sslPolicyErrors) => true;
        
        UpdateStatus("Waiting for signal...");
        
        glbFilePath = Path.Combine(Application.persistentDataPath, "downloaded.glb");
        
        // UDP 리스너 시작
        udpClient = new UdpClient(port);
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
        
        UpdateStatus("Listening on port " + port);
        void OnApplicationQuit()
    {
        if (receiveThread != null)
        {
            receiveThread.Abort();
        }
        if (udpClient != null)
        {
            udpClient.Close();
        }
    }
}

    private void UpdateStatus(string message)
    {
        Debug.Log("[Status] " + message);
        if (statusText != null)
        {
            UnityMainThreadDispatcher.Instance().Enqueue(() =>
            {
                statusText.text = message;
            });
        }
    }

    private void ReceiveData()
    {
        while (true)
        {
            try
            {
                IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, port);
                byte[] data = udpClient.Receive(ref remoteEndPoint);
                
                string signal = System.Text.Encoding.UTF8.GetString(data);
                Debug.Log("Received signal: '" + signal + "' from " + remoteEndPoint);
                
                if (signal.Trim() == "1")
                {
                    UpdateStatus("Signal received!");
                    System.Threading.Thread.Sleep(500);
                    shouldDownloadGlb = true;
                }
            }
            catch (SocketException ex)
            {
                Debug.LogError("SocketException: " + ex.Message);
            }
            catch (System.Exception ex)
            {
                Debug.LogError("Exception: " + ex.Message);
            }
        }
    }

    void Update()
    {
        if (shouldDownloadGlb)
        {
            shouldDownloadGlb = false;
            StartCoroutine(DownloadAndSpawnGlb());
        }
    }

    IEnumerator DownloadAndSpawnGlb()
    {
        string downloadUrl = serverUrl + "/output.glb";
        UpdateStatus("URL: " + downloadUrl);
        yield return new WaitForSeconds(1);
        
        UpdateStatus("Using WebClient...");
        
        downloadComplete = false;
        downloadSuccess = false;
        downloadError = "";
        
        Thread downloadThread = new Thread(() => DownloadWithWebClient(downloadUrl));
        downloadThread.IsBackground = true;
        downloadThread.Start();
        
        float elapsed = 0f;
        while (!downloadComplete && elapsed < 30f)
        {
            elapsed += Time.deltaTime;
            UpdateStatus("Downloading... " + elapsed.ToString("F0") + "s");
            yield return new WaitForSeconds(1);
        }
        
        if (elapsed >= 30f)
        {
            UpdateStatus("Download TIMEOUT!");
            yield break;
        }
        
        if (downloadSuccess)
        {
            UpdateStatus("Download SUCCESS!");
            yield return new WaitForSeconds(1);
            
            FileInfo fileInfo = new FileInfo(glbFilePath);
            UpdateStatus("Size: " + fileInfo.Length + " bytes");
            yield return new WaitForSeconds(1);
            
            UpdateStatus("Loading GLB...");
            yield return StartCoroutine(LoadAndSpawnGlb());
        }
        else
        {
            UpdateStatus("FAILED: " + downloadError);
            yield return new WaitForSeconds(3);
            UpdateStatus("Ready for next signal...");
        }
    }

    void DownloadWithWebClient(string url)
    {
        try
        {
            using (WebClient client = new WebClient())
            {
                client.DownloadFile(url, glbFilePath);
                downloadSuccess = true;
            }
        }
        catch (System.Exception ex)
        {
            downloadSuccess = false;
            downloadError = ex.Message;
        }
        finally
        {
            downloadComplete = true;
        }
    }

    IEnumerator LoadAndSpawnGlb()
    {
        UpdateStatus("Checking file...");
        yield return new WaitForSeconds(1);
        
        if (!File.Exists(glbFilePath))
        {
            UpdateStatus("File NOT found!");
            yield break;
        }
        
        long fileSize = new FileInfo(glbFilePath).Length;
        UpdateStatus("File size: " + fileSize);
        yield return new WaitForSeconds(2);
        
        UpdateStatus("Creating GltfImport...");
        var gltf = new GLTFast.GltfImport();
        
        UpdateStatus("Calling Load...");
        yield return new WaitForSeconds(1);
        
        var loadTask = gltf.Load(glbFilePath);
        
        float timeout = 30f;
        float elapsed = 0f;
        
        while (!loadTask.IsCompleted && elapsed < timeout)
        {
            elapsed += Time.deltaTime;
            if ((int)elapsed % 2 == 0)
            {
                UpdateStatus("Loading... " + elapsed.ToString("F0") + "s");
            }
            yield return null;
        }
        
        if (elapsed >= timeout)
        {
            UpdateStatus("Loading TIMEOUT!");
            yield break;
        }
        
        yield return loadTask;
        
        UpdateStatus("LoadingDone: " + gltf.LoadingDone);
        yield return new WaitForSeconds(2);
        
        if (gltf.LoadingDone)
        {
            if (previousGlb != null)
            {
                Destroy(previousGlb);
            }
            
            Camera mainCamera = Camera.main;
            if (mainCamera == null)
            {
                UpdateStatus("Camera NOT found!");
                yield break;
            }
            
            Vector3 spawnPosition = mainCamera.transform.position + mainCamera.transform.forward * 1.0f;
            UpdateStatus("Spawn pos: " + spawnPosition);
            yield return new WaitForSeconds(1);
            
            GameObject glbObject = new GameObject("DownloadedGLB");
            glbObject.transform.position = spawnPosition;
            glbObject.transform.rotation = mainCamera.transform.rotation;
            
            bool success = gltf.InstantiateMainScene(glbObject.transform);
            
            if (success)
            {
                UpdateStatus("SUCCESS! GLB spawned!");
                
                // 목표 크기 설정 (예: 0.3m = 30cm)
                float targetSize = 0.2f;
                
                // 현재 월드 크기 계산
                Bounds bounds = CalculateWorldBounds(glbObject);
                float currentSize = Mathf.Max(bounds.size.x, bounds.size.y, bounds.size.z);
                
                Debug.Log("Current size: " + currentSize + "m");
                Debug.Log("Target size: " + targetSize + "m");
                
                // 정규화 스케일 계산
                float scale = targetSize / currentSize;
                glbObject.transform.localScale = Vector3.one * scale;
                
                Debug.Log("Applied scale: " + scale);
                
                previousGlb = glbObject;
                
                FixShaders(glbObject);
                
                AddManipulationComponents(glbObject);
                
                UpdateStatus("Object is interactive!");
            }
            else
            {
                UpdateStatus("Instantiate FAILED!");
                Destroy(glbObject);
            }
        }
        else
        {
            UpdateStatus("Loading FAILED!");
        }
    }
    
    Bounds CalculateWorldBounds(GameObject obj)
    {
        Renderer[] renderers = obj.GetComponentsInChildren<Renderer>();
        if (renderers.Length == 0)
        {
            return new Bounds(obj.transform.position, Vector3.one * 0.1f);
        }
        
        Bounds bounds = renderers[0].bounds;
        foreach (Renderer r in renderers)
        {
            bounds.Encapsulate(r.bounds);
        }
        
        return bounds;
    }
    
    void FixShaders(GameObject obj)
    {
        Debug.Log("=== SHADER DEBUG START ===");
        
        Renderer[] renderers = obj.GetComponentsInChildren<Renderer>();
        Debug.Log("Renderers: " + renderers.Length);
        
        int matNum = 0;
        foreach (Renderer r in renderers)
        {
            Debug.Log("Renderer name: " + r.gameObject.name);
            foreach (Material m in r.materials)
            {
                matNum++;
                Debug.Log("  Mat" + matNum + ": " + m.name);
                Debug.Log("  Shader: " + m.shader.name);
                Debug.Log("  Texture: " + (m.mainTexture != null ? m.mainTexture.name : "NULL"));
                
                if (m.mainTexture != null)
                {
                    // 텍스처 정보 상세 로그
                    Texture2D tex = m.mainTexture as Texture2D;
                    if (tex != null)
                    {
                        Debug.Log("  Texture size: " + tex.width + "x" + tex.height);
                        Debug.Log("  Texture format: " + tex.format);
                    }
                }
            }
        }
        
        // 셰이더 변경하지 말고 원본 그대로 유지
        Debug.Log("Keeping ORIGINAL shaders and textures");
        Debug.Log("=== SHADER DEBUG END ===");
        UpdateStatus("Materials kept original!");
    }
    
    void AddManipulationComponents(GameObject obj)
    {
        MeshFilter[] meshFilters = obj.GetComponentsInChildren<MeshFilter>();
        if (meshFilters.Length > 0)
        {
            foreach (MeshFilter mf in meshFilters)
            {
                if (mf.gameObject.GetComponent<Collider>() == null)
                {
                    MeshCollider meshCollider = mf.gameObject.AddComponent<MeshCollider>();
                    meshCollider.convex = true;
                }
            }
        }
        else
        {
            BoxCollider boxCollider = obj.AddComponent<BoxCollider>();
        }
        
        try
        {
            var manipulator = obj.AddComponent<Microsoft.MixedReality.Toolkit.UI.ObjectManipulator>();
            manipulator.HostTransform = obj.transform;
            manipulator.AllowFarManipulation = true;
            manipulator.OneHandRotationModeNear = Microsoft.MixedReality.Toolkit.UI.ObjectManipulator.RotateInOneHandType.RotateAboutGrabPoint;
            manipulator.OneHandRotationModeFar = Microsoft.MixedReality.Toolkit.UI.ObjectManipulator.RotateInOneHandType.RotateAboutGrabPoint;
            
            obj.AddComponent<Microsoft.MixedReality.Toolkit.Input.NearInteractionGrabbable>();
            
            var constraintManager = obj.AddComponent<Microsoft.MixedReality.Toolkit.UI.ConstraintManager>();
            constraintManager.AutoConstraintSelection = true;
            
            var minMaxScale = obj.AddComponent<Microsoft.MixedReality.Toolkit.UI.MinMaxScaleConstraint>();
            minMaxScale.ScaleMinimum = 0.1f;
            minMaxScale.ScaleMaximum = 5.0f;
            minMaxScale.RelativeToInitialState = true;
            
            Debug.Log("MRTK components added!");
        }
        catch (System.Exception ex)
        {
            Debug.LogError("Manipulation error: " + ex.Message);
        }
        
        Rigidbody rb = obj.AddComponent<Rigidbody>();
        rb.isKinematic = true;
        rb.useGravity = false;
    }
}