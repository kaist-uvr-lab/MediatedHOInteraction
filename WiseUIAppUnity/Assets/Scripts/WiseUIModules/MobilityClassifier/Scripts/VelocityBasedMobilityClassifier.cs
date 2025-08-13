using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.Utilities;
using TMPro;

public class VelocityBasedMobilityClassifier : MonoBehaviour
{
    public GameObject statePlate;
    public MobilityState mobilityState = MobilityState.Stationary;

    [SerializeField]
    private float velocity;
    [SerializeField]
    private float threshold = 0.5f;

    private Vector3 speed;
    private Vector3 prev_position;
    private float prev_timestamp_sec;

    private GameObject mainCamera;
    
    public void Start()
    {
        mainCamera = GameObject.FindGameObjectWithTag("PVCamera");

        if (statePlate == null)
        {
            statePlate = Instantiate(Resources.Load("Notification") as GameObject);
            statePlate.transform.parent = mainCamera.transform;
            statePlate.transform.localPosition = new Vector3(0, 0.01f, 0.05f);
            statePlate.transform.localRotation = Quaternion.Euler(0, 0, 0);
            statePlate.transform.localScale = new Vector3(0.05f, 0.05f, 0.05f);
        }


    }
    public void SetActiveMobileStatePlate(bool flag)
    {
        statePlate.SetActive(flag);
    }

    public void DetectMobilityState(float curretTimestamp_sec)
    {
        speed = (mainCamera.transform.position - prev_position) / (curretTimestamp_sec- prev_timestamp_sec);
        
        velocity = speed.magnitude;

        if (velocity > threshold)
        {
            mobilityState = MobilityState.Walking;
            statePlate.transform.Find("Walking/Text").GetComponent<TextMeshPro>().text = "Physical status recognized\r\n:WALKING";
        }
        else
        {
            mobilityState = MobilityState.Stationary;
            statePlate.transform.Find("Walking/Text").GetComponent<TextMeshPro>().text = "Context recognized:\r\nSTATIONARY";
        }
        prev_position = mainCamera.transform.position;
        prev_timestamp_sec = curretTimestamp_sec;
    }

}
