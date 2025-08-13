using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEditor;
using UnityEngine.TestTools;
using ARRC.ARRCTexture;

namespace WiseUI.Base
{
    public class UcoMarkerDetectionTest_r2
    {

        string dataset_path;
        string prefab_path;
        [SetUp]
        public void Setup()
        {
            dataset_path = System.IO.Path.Combine(Application.dataPath, "Scripts/WiseUIModules/MarkerDetection/UnitTests/Data/");
            prefab_path = "Assets/Scripts/WiseUIModules/MarkerDetection/Axis3D.prefab";
        }
        [Test]
        public void UcoMarDetectionTestSimplePasses()
        {
            var test_sample_path = System.IO.Path.Combine(dataset_path, "aruco_sample.png");
            
            // Load image sample
            var texture = TextureIO.LoadTexture(test_sample_path);
            Assert.AreEqual(512, texture.width);
            Assert.AreEqual(512, texture.height);

            // Detect uco marker
            TagDetector_ tag_detector = new TagDetector_();
            tag_detector.Init(0.1f, ArUcoDictionary.DICT_6X6_250);
            var output = tag_detector.DetectMarkers(texture);

            Assert.AreEqual(1, output.Count);

            //load prefab and instantiate
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(prefab_path);
            var marker = GameObject.Instantiate(prefab);
            
            //GameObject test_obj = new GameObject();
        }


    }

}


