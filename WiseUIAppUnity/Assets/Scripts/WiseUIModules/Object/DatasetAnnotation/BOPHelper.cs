using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.Assertions;


public class BOPHelper 
{
    string datasetPath;
    public string datasetName;
    public string splitType;

    string exportPath;
    string sceneDir;
    string rgbDir;
    string modelDir;
    string maskDir;
    string maskVisibleDir;

    public List<string> lines_scene_gt = new List<string>();
    public List<string> lines_scene_camera = new List<string>();
    public List<string> lines_scene_info = new List<string>();

    public BOPHelper(string datasetPath, string datasetName, string splitType, List<Vector3> obj_size_list)
    {
        this.datasetPath = datasetPath;
        this.datasetName = datasetName;
        this.splitType = splitType;

        exportPath = Path.Combine(datasetPath, datasetName);
        exportPath = Path.Combine(exportPath, splitType);

        //Ensure directory.
        Directory.CreateDirectory(exportPath);


        modelDir = Path.Combine(exportPath, "models");
        Directory.CreateDirectory(modelDir);

        string model_filepath = Path.Combine(modelDir, "model_info.json");

        write_model_info(model_filepath, obj_size_list);
    }

    public void BeginRecordingScene(int sceneIdx)
    {
        lines_scene_gt.Clear();
        lines_scene_camera.Clear();
        lines_scene_info.Clear();

        sceneDir = Path.Combine(exportPath, sceneIdx.ToString("D6"));
        Directory.CreateDirectory(sceneDir);

        rgbDir = Path.Combine(sceneDir, "rgb");
        Directory.CreateDirectory(rgbDir);

        maskDir = Path.Combine(sceneDir, "mask");
        Directory.CreateDirectory(maskDir);

        maskVisibleDir = Path.Combine(sceneDir, "mask_visib");
        Directory.CreateDirectory(maskVisibleDir);
    }
    
    public void EndRecordingScene()
    {
        string scene_gt_filepath = Path.Combine(sceneDir, "scene_gt.json");
        WriteJsonFile(scene_gt_filepath, lines_scene_gt);

        string scene_camera_filepath = Path.Combine(sceneDir, "scene_camera.json");
        WriteJsonFile(scene_camera_filepath, lines_scene_camera);

        //string scene_gt_info_filepath = Path.Combine(sceneDir, "scene_gt_info.json");
        //WriteJsonFile(scene_gt_info_filepath, lines_scene_info);

        lines_scene_gt.Clear();
    }


    public void AddJsonLine(long im_id, Matrix4x4 local2wrold_cam, List<Matrix4x4> list_local2world_obj)
    {
        //scene_gt
        string line = string.Format("\"{0}\": ", im_id);
        line += "[";
        
        for (int cnt = 0; cnt < list_local2world_obj.Count; cnt++)
        {
            var local2world_obj = list_local2world_obj[cnt];
            int obj_id = cnt;
            line += make_scene_gt_token(local2wrold_cam, local2world_obj, obj_id);
            if (cnt < list_local2world_obj.Count - 1)
                line += ", ";
        }

        line += "],";
        lines_scene_gt.Add(line);

        //camera_gt
        string kString = string.Format("\"cam_K\": [{0:F6},{1:F6},{2:F6},{3:F6},{4:F6},{5:F6},{6:F6},{7:F6},{8:F6}]", 417.0321350097656, 0.0, 320.0, 0.0, 417.0321350097656, 240.0, 0.0, 0.0, 1.0);
        string cameraLine = "\"" + im_id.ToString() + "\": {" + string.Format("{0},", kString) + " \"depth_scale\": 1.0},";
        lines_scene_camera.Add(cameraLine);

    }


    public string GetRGBFilePath<T>(T im_id) where T : IFormattable
    {
        string filename = im_id.ToString("D6", CultureInfo.InvariantCulture) + ".png";
        return Path.Combine(rgbDir, filename);
    }
    public string GetMaskFilePath<T1, T2>(T1 im_id, T2 gt_id) where T1 : IFormattable where T2 : IFormattable
    {
        string filename = string.Format(CultureInfo.InvariantCulture, "{0:D6}_{1:D6}.png", im_id, gt_id);
        return Path.Combine(maskDir, filename);
    }

    public string GetVisibleMaskFilePath<T1, T2>(T1 im_id, T2 gt_id) where T1 : IFormattable where T2 : IFormattable
    {
        string filename = string.Format(CultureInfo.InvariantCulture, "{0:D6}_{1:D6}.png", im_id, gt_id);
        return Path.Combine(maskVisibleDir, filename);
    }

    //scene
    public string make_scene_gt_token(Matrix4x4 localToWorldMatrix, Matrix4x4 targetObjectTransform, int obj_id)
    {
        //// Get object target pose relative to camera
        var m2c = localToWorldMatrix.inverse * targetObjectTransform;

        //// Inverse Y axis for screen coordinate.
        m2c.SetRow(1, -m2c.GetRow(1));

        string rotationString = string.Format("\"cam_R_m2c\": [{0:F6},{1:F6},{2:F6},{3:F6},{4:F6},{5:F6},{6:F6},{7:F6},{8:F6}]", m2c.m00, m2c.m01, m2c.m02, m2c.m10, m2c.m11, m2c.m12, m2c.m20, m2c.m21, m2c.m22);
        string positionString = string.Format("\"cam_t_m2c\": [{0:F6},{1:F6},{2:F6}]", m2c.m03, m2c.m13, m2c.m23);
        string objIDString = string.Format("\"obj_id\": {0}", obj_id);

        string token = "{" + string.Format("{0}, {1}, {2}", rotationString, positionString, objIDString) + "}";
        return token;
    }
    public void write_model_info(string filepath, List<Vector3> obj_size_list)
    {
        var lines_model_info = new List<string>();
        
        for (int cnt = 0; cnt < obj_size_list.Count; cnt++)
        {
            string line = string.Format("\"{0}\": ", cnt);
            line += make_model_info_token(obj_size_list[cnt]);
            line += ',';
            lines_model_info.Add(line);
        }
        WriteJsonFile(filepath, lines_model_info);

    }
    public string make_model_info_token(Vector3 size)
    {
        string min_x = string.Format("\"min_x\": {0:F6}", -size.x/2);
        string min_y = string.Format("\"min_y\": {0:F6}", -size.x/2);
        string min_z = string.Format("\"min_z\": {0:F6}", -size.x/2);
        
        string size_x = string.Format("\"size_x\": {0:F6}", size.x);
        string size_y = string.Format("\"size_y\": {0:F6}", size.y);
        string size_z = string.Format("\"size_z\": {0:F6}", size.z);

        string token = "{" + string.Format("{0}, {1}, {2}, {3}, {4}, {5}", min_x, min_y, min_z, size_x, size_y, size_z) + "}";
        return token;
    }
  
    void WriteJsonFile(string filepath, List<string> lines)
    {
        //remove last ,
        string lastLine = lines[lines.Count - 1];
        lastLine = lastLine.Substring(0, lastLine.Length - 1);
        lines[lines.Count - 1] = lastLine;
        
        lines.Insert(0, "{");
        lines.Add("}");

        SaveFile(filepath, lines);
    }
    /// <summary>
    /// Saves list of strings to file
    /// </summary>
    /// <param name="path">Path and filename</param>
    private void SaveFile(string fileName, List<string> lines)
    {
        StringBuilder str = new StringBuilder();

        foreach (string line in lines)
            str.AppendLine(line);

        File.WriteAllText(fileName, str.ToString());
    }

}
