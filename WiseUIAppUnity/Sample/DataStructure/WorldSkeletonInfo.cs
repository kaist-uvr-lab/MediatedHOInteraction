using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Scripting;
using System.Runtime.InteropServices;

/// <summary>
/// Contains information about the skeleton joints.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct WorldSkeletonInfo
{
    /// <summary>
    /// Position of the joints.
    /// normalized values between 0 and 1
    /// </summary>
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 21)]
    public Vector3[] joints;

    /// <summary>
    /// Orientation of the joints.
    /// normalized values between 0 and 1
    /// </summary>
    [MarshalAs(UnmanagedType.ByValArray, SizeConst = 21)]
    public Quaternion[] orientation_joints;
}
