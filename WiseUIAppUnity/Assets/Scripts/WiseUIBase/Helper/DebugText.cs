using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.UI;

public class DebugText : MonoBehaviour
{
    [SerializeField]
    public Dictionary<string, string> lines = new Dictionary<string, string>();

    private void Update()
    {
        string content = string.Empty;
        foreach (var line in lines)
        {
            content += string.Format("{0}:{1}\n", line.Key, line.Value);

        }
        GetComponent<Text>().text = content;
    }


    static DebugText _instance;
    public static DebugText Instance
    {
        get
        {
            if (_instance == null)
            {
                GameObject obj = GameObject.Find("DebugText");
                if (obj == null)
                {
                    var main_cam = GameObject.FindGameObjectWithTag("MainCamera");
                    if(main_cam != null)
                    {
                        //Check DebugText prefab file exists in resources folder.
                        obj = Instantiate(Resources.Load("DebugCanvas")) as GameObject;
                        Assert.IsNotNull(obj, "DebugText prefab file does not exist in resources folder.");

                        //obj.transform.parent = main_cam.transform;
                    }
                    else
                    {
                        Assert.IsNotNull(main_cam, "Gameobject with MainCamera Tag does not exist in the scene.");
                    }
                }
                _instance = obj.transform.GetChild(0).gameObject.GetComponent<DebugText>();

                //check unity editor mode.
                if (!Application.isEditor)
                    DontDestroyOnLoad(obj);
            }

            return _instance;
        }
    }


}
