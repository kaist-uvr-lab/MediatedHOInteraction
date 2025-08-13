using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// Used for visualization of camera image
/// </summary>
public struct VisualizationInfo
{
    /// <summary>
    /// The Texture 2D information of the input image.
    /// </summary>
    public Texture2D rgb_image;

    /// <summary>
    /// The Texture 2D information of the cutout image.
    /// </summary>
    public Texture2D occlusion_rgb;
}