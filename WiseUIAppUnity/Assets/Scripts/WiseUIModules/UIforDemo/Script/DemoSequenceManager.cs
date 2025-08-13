using Microsoft.MixedReality.Toolkit.UI;
using Microsoft.MixedReality.Toolkit.Utilities.Solvers;
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class DemoSequenceManager : MonoBehaviour
{
    [SerializeField]
    private GameObject[] guideTarget;
    [SerializeField]
    private GameObject directionalIndicator;
    private DirectionalIndicator directionalIndicatorScript;
    [SerializeField]
    private Transform cameraPos;
    [SerializeField]
    private ButtonConfigHelper changeModeButton;
    [SerializeField]
    private float triggeringDistance = 1.0f;

    private int demoSequence;
    private float distance;
    private Boolean isDemoEnded;
    private Boolean isEditMode;

    public void OnChangeModeButtonPressed() => ModeChange();

    private void Awake()
    {
        directionalIndicator.SetActive(true);
        directionalIndicatorScript = directionalIndicator.GetComponent<DirectionalIndicator>();
        isEditMode = true;
        changeModeButton.MainLabelText = "Cur Mode\n: Edit";
        isDemoEnded = false;
        demoSequence = 0;
        for (int i = 0; i < guideTarget.Length; i++)
        {
            guideTarget[i].SetActive(true);
        }
        directionalIndicator.SetActive(false);
    }

    void Update()
    {
        if(!isDemoEnded && !isEditMode)
        {
            distance = Vector3.Distance(guideTarget[demoSequence].transform.position, cameraPos.position);
            if (distance < triggeringDistance)
            {
                guideTarget[demoSequence++].SetActive(false);
                if (demoSequence < guideTarget.Length)
                {
                    guideTarget[demoSequence].SetActive(true);
                    directionalIndicatorScript.DirectionalTarget = guideTarget[demoSequence].transform;
                }
                else
                {
                    isDemoEnded = true;
                    directionalIndicator.SetActive(false);
                }
            }
        } 
    }

    private void ModeChange()
    {
        if(isEditMode)
        {
            isEditMode = false;
            changeModeButton.MainLabelText = "Cur Mode\n: Play";
            for (int i = 0; i < guideTarget.Length; i++)
            {
                guideTarget[i].SetActive(false);
            }
            guideTarget[demoSequence].SetActive(true);
            directionalIndicator.SetActive(true);
            directionalIndicatorScript.DirectionalTarget = guideTarget[demoSequence].transform;
        }
        else if(!isEditMode)
        {
            isEditMode = true;
            changeModeButton.MainLabelText = "Cur Mode\n: Edit";
            for (int i = 0; i < guideTarget.Length; i++)
            {
                guideTarget[i].SetActive(true);
            }
            directionalIndicator.SetActive(false);
        }
    }
}
