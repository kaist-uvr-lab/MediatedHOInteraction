using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EventRegister : MonoBehaviour, IMixedRealityTouchHandler
{
    private InteractionManager interactionManager;
    private Interactable interactable;  //OnClick 이벤트 처리
    private NearInteractionTouchable touchable; //
    

    private void Start()
    {
        GameObject interactionManagerObject = GameObject.Find("InteractionManager");
        interactable = GetComponent<Interactable>();
        touchable = gameObject.GetComponent<NearInteractionTouchable>();

        if (interactionManagerObject != null)
        {
            interactionManager = interactionManagerObject.GetComponent<InteractionManager>();
        }
        else
        {
            Debug.LogError("Failed to get InteractionManager");
        }
        if (interactable != null)
        {
            interactable.OnClick.AddListener(() => interactionManager.IsNearInteraction(false));
            interactable.OnClick.AddListener(() => interactionManager.OnObjSelected(gameObject));
        }
        else
        {
            Debug.LogError("Failed to get Interactable");
        }
    }

    public void OnTouchStarted(HandTrackingInputEventData eventData)
    {
        Debug.Log("OnTouchStarted: " + gameObject);
        interactionManager.IsNearInteraction(true);
        interactionManager.OnObjSelected(gameObject);
    }

    public void OnTouchCompleted(HandTrackingInputEventData eventData)
    {
    }

    public void OnTouchUpdated(HandTrackingInputEventData eventData)
    {
    }
}
