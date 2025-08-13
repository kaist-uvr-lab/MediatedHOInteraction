using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEditor;
using UnityEngine.TestTools;

namespace WiseUI.Base
{
    public class UcoMarkerDetectionTest
    {
        [Test]
        public void UcoMarDetectionTestSimplePasses()
        {
            string test_sample_dir = "aruco_sample";
            // Load image sample
            var input = LoadTexture2DAsset(test_sample_dir);
            // Detect uco marker
            var go = GameObject.Find("TagDetector");
            //var go = new GameObject("TagDetector");
            //var td = go.AddComponent<TagDetector>();
            if (go == null)
            {
                Assert.Fail();
            }
            TagDetector td = go.GetComponent<TagDetector>();
            bool result = td.Initialize(input);
            Assert.True(result);
            
        }

        public Texture2D LoadTexture2DAsset(string imageFileName)
        {
            string[] allCandidates = AssetDatabase.FindAssets(imageFileName);
            Assert.True(allCandidates.Length > 0);

            var texture =
                AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(allCandidates[0]), typeof(Texture2D)) as
                Texture2D;

            return texture;
        }
    }

}


