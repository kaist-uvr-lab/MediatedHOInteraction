using UnityEngine;
using System.Collections;

public class RotateAround : MonoBehaviour {

    public Vector3 RotationObject;
    public float speed = 1.0f;
	// Use this for initialization
	void Start () {
	
	}
	
	// Update is called once per frame
	void Update () {
        transform.RotateAround(Vector3.zero, Vector3.up, 5 * Time.deltaTime * speed);
	}
}
