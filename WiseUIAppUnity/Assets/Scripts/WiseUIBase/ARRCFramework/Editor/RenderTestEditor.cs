using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;


//custom editor of RenderTest
[CustomEditor(typeof(RenderTest))]
public class RenderTestEditor : Editor
{
    RenderTest mono;
    private void OnEnable()
    {
        mono = (RenderTest)target;
    }
    //onInspectorGUI
    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();
        if (GUILayout.Button("Render"))
        {
            mono.Rendering();
        }
    }
}
