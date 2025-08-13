using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Barracuda;
using UnityEditor;
using UnityEngine;
using UnityEngine.PlayerLoop;
using static Unity.Burst.Intrinsics.X86;
using static UnityEngine.GraphicsBuffer;

[Serializable]
public class SE3
{
    [SerializeField]
    public Vector3 t;
    [SerializeField]
    public Quaternion q;

    
}

public class SetLocationMono : MonoBehaviour
{
    [SerializeField]
    public SerializableDictionary<long, SE3> pose_list = new SerializableDictionary<long, SE3>();

    [SerializeField]
    public long current_frame_id;
    public int target_frame_id;
    public GameObject camera;
    
    
   
    public void AddPose(long frame_id)
    {
        SE3 se3 = new SE3();
        se3.t = transform.position;
        se3.q = transform.rotation;
     
        if (pose_list.ContainsKey(frame_id))
        {
            pose_list[frame_id] = se3;
        }
        else
        {
            pose_list.Add(frame_id, se3);
        }
    }

    public void SetPose(long frame_id)
    {
        Check();
        // interporation between two frames.
        if (pose_list.ContainsKey(frame_id))
        {
            SE3 se3 = pose_list[frame_id];
            transform.position = se3.t;
            transform.rotation = se3.q;
        }
        else
        {
            //그 다음 frame_id를 찾는다
            //get closest frame_id in pose_list
            var keys = pose_list.Keys.Select(x => x).ToList();
            if (keys.Count != 0)
            {
                var sorted = keys.OrderBy(item => Math.Abs(frame_id - item)).ToList();
                long closest_1 = sorted[0];

                if (keys.Count == 1)
                {
                    //Debug.Log("keys 1");
                    SE3 se3 = pose_list[closest_1];
                    transform.position = se3.t;
                    transform.rotation = se3.q;
                }
                else if (keys.Count > 1)
                {
                    long closest_2 = sorted[1];

                    //Debug.LogFormat("{0}, {1}, {2}", closest_1, closest_2, frame_id);

                    long start_id = Math.Min(closest_1, closest_2);
                    long end_id = Math.Max(closest_1, closest_2);

                    if(frame_id >= start_id && frame_id <end_id)
                    {
                        Vector3 start_t = pose_list[start_id].t;
                        Vector3 end_t = pose_list[end_id].t;
                        Quaternion start_q = pose_list[start_id].q;
                        Quaternion end_q = pose_list[end_id].q;

                        ////interpolation between two se3 poses
                        float alpha = (frame_id - start_id) / (float)(end_id - start_id);
                        var t = (1 - alpha) * start_t + alpha * end_t;
                        var q = Quaternion.Slerp(start_q, end_q, alpha);
                        //Debug.Log(alpha);
                        transform.position = t;
                        transform.rotation = q;
                    }
                    else if(frame_id < start_id)
                    {
                        SE3 se3 = pose_list[start_id];
                        transform.position = se3.t;
                        transform.rotation = se3.q;
                    }
                    else
                    {
                        SE3 se3 = pose_list[end_id];
                        transform.position = se3.t;
                        transform.rotation = se3.q;

                    }

                    ////등록된 frame중 가장 가까운 frame_id를 찾는다
                    //int start_id = 0, end_id = 0;
       
                }
            }



           
            ////pose_list.Keys.
        }

    }

    public void Check()
    {
        camera = GameObject.FindWithTag("PVCamera");

        try
        {
            //Vector3 heading = (transform.position - camera.transform.position);
            var dot = Vector3.Angle(transform.forward, camera.transform.forward);
            var distance = Vector3.Distance(transform.position, camera.transform.position);
            //Debug.LogFormat("{0} dot:{1} dis:{2}",transform.name, dot.ToString(), distance.ToString());


            float threshold;
            string[] names = { "clock", "album", "book", "harnomica" };
            if(names.Contains(transform.name))
                threshold = 1.5f;
      
            else
                threshold = 8;

            if (distance < threshold && dot < 90)
            {
                gameObject.SetActive(true);
            }
            else
            {
                gameObject.SetActive(false);
            }
        }
        catch(Exception ex)
        {

        }
      
    }
}

