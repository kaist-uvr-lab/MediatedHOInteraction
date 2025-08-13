using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using WiseUI.Base;

[CustomEditor(typeof(SetLocationMono))]
public class SetLocationMonoEditor : Editor
{
    SetLocationMono mono;

    public void OnEnable()
    {
        mono = (SetLocationMono)target;
    }
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();
        if (GUILayout.Button("Add pose"))
        {
            mono.AddPose(HoloLens2FileStreamManager.Instance.PVCamera.Timestamp);

        }
     
    }
    

    public void Update()
    {
    }
}
