using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Playables;
using UnityEngine.XR;
using WiseUI.Base;

public class ExampleModule : MonoBehaviour
{
    // Start is called before the first frame update
    void Awake()
    {
      
    }
    void Update()
    {
        
    }
    // Update is called once per frame
    public void OnReceivedResultData(ResultDataPackage frameData)
    {
        //Debug.Log("OnReceivedData");
        
        var pvFrame = HoloLens2FileStreamManager.Instance.PVCamera.GetFrame();

        // 예시: 박스 그리기.
        int startX = 10;
        int startY = 10;
        int endX = 100;
        int endY = 100;
        int lineWidth = 3;

        for (int x = startX; x <= endX; x++)
        {
            for (int y = startY; y <= endY; y++)
            {
                if (x < startX + lineWidth || x > endX - lineWidth ||
                    y < startY + lineWidth || y > endY - lineWidth)
                {
                    pvFrame.Texture.SetPixel(x, y, Color.red); 
                }
            }
        }
        pvFrame.Texture.Apply();

    }
}
