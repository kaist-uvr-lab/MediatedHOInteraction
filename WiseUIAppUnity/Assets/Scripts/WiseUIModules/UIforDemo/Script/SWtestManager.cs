using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

public class SWtestManager : MonoBehaviour
{
    public async void OnCamPoseTracMapperButtonPressed() => await GetCamPoseTracMapperATE();

    public async void OnObjPoseEstimatorButtonPressed() => await GetObjPoseEstimatorIOU();
    public async void OnGestureInputDetectorButtonPressed() => await GetGestureInputDetectorFalsePositive();

    private async Task GetCamPoseTracMapperATE()
    {

    }   

    private async Task GetObjPoseEstimatorIOU()
    {

    }

    private async Task GetGestureInputDetectorFalsePositive()
    {

    }
}