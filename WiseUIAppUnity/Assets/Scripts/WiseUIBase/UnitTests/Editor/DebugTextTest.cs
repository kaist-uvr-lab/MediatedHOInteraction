using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace WiseUI.Base
{
    public class DebugTextTest
    {
        // A Test behaves as an ordinary method
        [Test]
        public void DebugTextTestSimplePasses()
        {
            DebugText.Instance.lines.Add("line1", "test");
            DebugText.Instance.lines.Add("line2", "test2");
            DebugText.Instance.lines.Add("line3", "test3");

        }
    }
}

