using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EncodeImageTest : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        var texture = gameObject.GetComponent<MeshRenderer>().material.mainTexture as Texture2D;
        byte[] size = texture.EncodeToJPG(75);
        //DebugText.Instance.lines["size"] = size.Length.ToString();


    }
}
