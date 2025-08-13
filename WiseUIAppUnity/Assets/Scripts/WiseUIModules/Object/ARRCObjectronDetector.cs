using System;
using System.Collections;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine;


namespace WiseUI.Modules
{

    public enum ModelType { MobilePoseBase_virnect, MobilePoseShape_chair, MobilePoseShape_shose };

    [Serializable]
    public class ARRCObjectronDetector : MonoBehaviour
    {
        [SerializeField]
        protected IWorker engine;
        [SerializeField]
        public NNModel[] nnModels;
        [SerializeField]
        BaseRunner runner;

        public void LoadModel(ModelType modelType)
        {
            DisposeModel();
            var model = ModelLoader.Load(nnModels[(int)modelType]);
            engine = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model, false);
            //engine = WorkerFactory.CreateWorker(model, WorkerFactory.Device.GPU);

            switch (modelType)
            {
                case ModelType.MobilePoseBase_virnect:
                    runner = new MobilePoseBaseRunner(engine);
                    break;
                case ModelType.MobilePoseShape_chair:
                case ModelType.MobilePoseShape_shose:
                    runner = new MobilePoseShapeRunner(engine);
                    break;
            }
        }
        public void Run(Texture2D inputTexture)
        {
            runner.Run(inputTexture);
        }
        void DisposeModel()
        {
            if (engine != null)
                engine.Dispose();
        }
        void OnDestroy()
        {
            DisposeModel();
        }

    }

}
