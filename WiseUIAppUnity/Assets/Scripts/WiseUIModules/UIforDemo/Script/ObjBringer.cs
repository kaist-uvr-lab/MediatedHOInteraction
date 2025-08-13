using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ObjBringer : MonoBehaviour
{
    /*
    [SerializeField]
    private GameObject[] demo;
    */
    [SerializeField]
    private GameObject[] route;
    [SerializeField]
    private Transform cameraPos;
    private List<Vector3> positionList;
    private List<Quaternion> rotationList;


    void Awake()
    {
        positionList = new List<Vector3>();
        rotationList = new List<Quaternion>();

        SaveTransformToList();
    }

    private void SaveTransformToList()
    {
        positionList.Clear();
        rotationList.Clear();
        /*
        foreach (GameObject obj in demo)
        {
            positionList.Add(obj.transform.position);
            rotationList.Add(obj.transform.rotation);
        }
        */
        foreach (GameObject obj in route)
        {
            positionList.Add(obj.transform.position);
            rotationList.Add(obj.transform.rotation);
        }
    }

    public void ResetPosition()
    {

        // Check if the number of saved transforms matches the total number of objects in 'demo' and 'route'
        if (positionList.Count == route.Length && rotationList.Count == route.Length)
        {
            int index = 0;

            /*
            foreach (GameObject obj in demo)
            {
                obj.transform.position = positionList[index];
                obj.transform.rotation = rotationList[index];
                index++;
            }
            */

            // Apply transform information from the lists back to 'route' GameObjects
            foreach (GameObject obj in route)
            {
                obj.transform.position = positionList[index];
                obj.transform.rotation = rotationList[index];
                index++;
            }
        }
        else
        {
            Debug.LogWarning("Transform lists do not match the number of objects in 'demo' and 'route'.");
        }
    }

    public void BringObjectToMyPosition(GameObject objToBring)
    {
        Debug.Log("Bring it: " + cameraPos.position);
        if (objToBring != null)
        {
            objToBring.transform.position = cameraPos.position + Vector3.up * 0.5f;
        }
        else
        {
            Debug.LogWarning("The specified object to bring is null.");
        }
    }
}
