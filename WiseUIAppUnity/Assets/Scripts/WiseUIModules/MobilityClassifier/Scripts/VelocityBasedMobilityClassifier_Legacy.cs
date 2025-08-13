using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.Utilities;

public class VelocityBasedMobilityClassifier_Legacy : MonoBehaviour
{
    public List<GameObject> displayedObjects = new List<GameObject>();
    public MobilityState mobilityState;

    [SerializeField]
    private float velocity;
    [SerializeField]
    private float threshold = 0.5f;

    private Vector3 speed;
    private Vector3 position;

    void Start()
    {
        foreach(var obj in displayedObjects)
        {
            if (obj != null)
            {
                obj.SetActive(false);
            }
        }
        
        mobilityState = MobilityState.Stationary;
    }

    void Update()
    {
        speed = (CameraCache.Main.transform.position - position) / Time.deltaTime;

        position = CameraCache.Main.transform.position;
        velocity = speed.magnitude;

        if (velocity > threshold)
        {
            if (mobilityState == MobilityState.Stationary)
            {
                mobilityState = MobilityState.Walking;
                SetActiveObjects(true);
            }
        }
        else
        {
            if (mobilityState == MobilityState.Walking)
            {
                mobilityState = MobilityState.Stationary;
                SetActiveObjects(false);
            }
        }

    }

    void SetActiveObjects(bool flag)
    {
        foreach(var obj in displayedObjects)
        {
            obj?.SetActive(flag);
        }
    }
}
