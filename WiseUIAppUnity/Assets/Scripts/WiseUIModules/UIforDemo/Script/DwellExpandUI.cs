using Microsoft.MixedReality.Toolkit.Dwell;
using Microsoft.MixedReality.Toolkit.Input;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class DwellExpandUI : BaseDwellSample
{
    [SerializeField]
    private GameObject UIBefore = null;
    [SerializeField]
    private GameObject UIAfter = null;
    [SerializeField]
    private Transform listItems = null;

    private void Update()
    {
        float value = DwellHandler.DwellProgress;
        dwellVisualImage.transform.localScale = new Vector3(value, 1, 0);
    }

    /// <inheritdoc/>
    public override void DwellCompleted(IMixedRealityPointer pointer)
    {
        dwellVisualImage.transform.localScale = Vector3.zero;
        base.DwellCompleted(pointer);
    }

    /// <inheritdoc/>
    public override void ButtonExecute()
    {
        UIAfter.SetActive(true);
        var textMeshObjects = listItems.GetComponentsInChildren<TextMeshProUGUI>();

        foreach (var textObject in textMeshObjects)
        {
            textObject.text = int.Parse(textObject.text) + 5 + "";
        }
        UIBefore.SetActive(false);
    }
}
