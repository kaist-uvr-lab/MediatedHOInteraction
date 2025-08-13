using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.Calib3dModule;
using OpenCVForUnity.UnityUtils;
using OpenCVForUnity.ObjdetectModule;

public class TagDetector : MonoBehaviour
{
    /// <summary>
    /// ArUco tag dictionary id
    /// </summary>
    public ArUcoDictionary dictionaryId = ArUcoDictionary.DICT_6X6_250;

    /// <summary>
    /// The length of the markers' side [m]
    /// </summary>  
    public float markerLength = 0.1f;

    /// <summary>
    /// Axis gameobject
    /// </summary>
    public GameObject axis3D;

    /// <summary>
    /// Simulation camera
    /// </summary>
    private Camera HLCamera;

    private GameObject camera_go;

    private Mat rgbMat;

    private Texture2D texture;

    private GameObject pv_go;

    private Mat pvMat;

    public Texture2D testImg;
    // Start is called before the first frame update

    public bool Initialize(Texture2D pvframe)
    {
        rgbMat = new Mat(pvframe.height, pvframe.width, CvType.CV_8UC3);
        texture = new Texture2D(rgbMat.cols(), rgbMat.rows(), TextureFormat.RGBA32, false);
        gameObject.GetComponent<Renderer>().sharedMaterial.mainTexture = texture;
        bool result = DetectMarkers(pvframe);
        return result;
    }

    private void Start()
    {
        pv_go = GameObject.Find("PV Image");
        if(pv_go != null)
        {
            camera_go = pv_go.transform.Find("Camera").gameObject;
            if (camera_go != null)
            {
                HLCamera = camera_go.GetComponent<Camera>();
            }
            else
            {
                HLCamera = null;
            }

            Renderer rend = pv_go.GetComponent<Renderer>();
            var pv_img_texture = rend.sharedMaterial.mainTexture;
            bool result = Initialize((Texture2D)pv_img_texture);
        }
        else
        {
            Debug.Log("Please load dataset at simulation window");
        }
    }

    private void Update()
    {
        if (pv_go != null)
        {
            Renderer rend = pv_go.GetComponent<Renderer>();
            var pv_img_texture = rend.sharedMaterial.mainTexture;
            Initialize((Texture2D)pv_img_texture);
        }
        else
        {
            Debug.Log("Please load dataset at simulation window");
        }
    }

    public bool DetectMarkers(Texture2D imgTexture)
    {
        bool result = false;
        Utils.texture2DToMat(imgTexture, rgbMat);
        gameObject.transform.localScale = new Vector3(imgTexture.width, imgTexture.height, 1);

        float width = rgbMat.width();
        float height = rgbMat.height();
        float imageSizeScale = 1.0f;
        float widthScale = (float)Screen.width / width;
        float heightScale = (float)Screen.height / height;

        // set camera parameters.
        //ToDo : input real camera parameters
        int max_d = (int)Mathf.Max(width, height);
        double fx = max_d;
        double fy = max_d;
        double cx = width / 2.0f;
        double cy = height / 2.0f;
        Mat camMatrix = new Mat(3, 3, CvType.CV_64FC1);
        camMatrix.put(0, 0, fx);
        camMatrix.put(0, 1, 0);
        camMatrix.put(0, 2, cx);
        camMatrix.put(1, 0, 0);
        camMatrix.put(1, 1, fy);
        camMatrix.put(1, 2, cy);
        camMatrix.put(2, 0, 0);
        camMatrix.put(2, 1, 0);
        camMatrix.put(2, 2, 1.0f);
        MatOfDouble distCoeffs = new MatOfDouble(0, 0, 0, 0);

        // calibration camera matrix values.
        Size imageSize = new Size(width * imageSizeScale, height * imageSizeScale);
        double apertureWidth = 0;
        double apertureHeight = 0;
        double[] fovx = new double[1];
        double[] fovy = new double[1];
        double[] focalLength = new double[1];
        Point principalPoint = new Point(0, 0);
        double[] aspectratio = new double[1];

        Calib3d.calibrationMatrixValues(camMatrix, imageSize, apertureWidth, apertureHeight, fovx, fovy, focalLength, principalPoint, aspectratio);

        Mat ids = new Mat();
        List<Mat> corners = new List<Mat>();
        List<Mat> rejectedCorners = new List<Mat>();
        Mat rotMat = new Mat(3, 3, CvType.CV_64FC1);

        MatOfPoint3f objPoints = new MatOfPoint3f(
            new Point3(-markerLength / 2f, markerLength / 2f, 0),
            new Point3(markerLength / 2f, markerLength / 2f, 0),
            new Point3(markerLength / 2f, -markerLength / 2f, 0),
            new Point3(-markerLength / 2f, -markerLength / 2f, 0)
            );

        Dictionary dictionary = Objdetect.getPredefinedDictionary((int)dictionaryId);
        DetectorParameters detectorParams = new DetectorParameters();
        detectorParams.set_useAruco3Detection(true);
        RefineParameters refineParameters = new RefineParameters(10f, 3f, true);
        ArucoDetector arucoDetector = new ArucoDetector(dictionary, detectorParams, refineParameters);
        // undistort image.
        Calib3d.undistort(rgbMat, rgbMat, camMatrix, distCoeffs);
        // detect markers.
        arucoDetector.detectMarkers(rgbMat, corners, ids, rejectedCorners);
        // if at least one marker detected
        if (ids.total() > 0)
        {
            Objdetect.drawDetectedMarkers(rgbMat, corners, ids, new Scalar(0, 255, 0));
            for (int i = 0; i < ids.total(); i++)
            {
                using (Mat rvec = new Mat(1, 1, CvType.CV_64FC3))
                using (Mat tvec = new Mat(1, 1, CvType.CV_64FC3))
                using (Mat corner_4x1 = corners[i].reshape(2, 4)) // 1*4*CV_32FC2 => 4*1*CV_32FC2
                using (MatOfPoint2f imagePoints = new MatOfPoint2f(corner_4x1))
                {
                    // Calculate pose for each marker
                    Calib3d.solvePnP(objPoints, imagePoints, camMatrix, distCoeffs, rvec, tvec);

                    // In this example we are processing with RGB color image, so Axis-color correspondences are X: blue, Y: green, Z: red. (Usually X: red, Y: green, Z: blue)
                    Calib3d.drawFrameAxes(rgbMat, camMatrix, distCoeffs, rvec, tvec, markerLength * 0.5f);

                    if (i == 0)
                    {
                        // Get translation vector
                        double[] tvecArr = new double[3];
                        tvec.get(0, 0, tvecArr);

                        // Get rotation vector
                        Mat rvec_3x1 = rvec.reshape(1, 3);

                        // Convert rotation vector to rotation matrix.
                        Calib3d.Rodrigues(rvec_3x1, rotMat);
                        double[] rotMatArr = new double[rotMat.total()];
                        rotMat.get(0, 0, rotMatArr);

                        // Convert OpenCV camera extrinsic parameters to Unity Matrix4x4.
                        Matrix4x4 transformationM = new Matrix4x4(); // from OpenCV
                        transformationM.SetRow(0, new Vector4((float)rotMatArr[0], (float)rotMatArr[1], (float)rotMatArr[2], (float)tvecArr[0]));
                        transformationM.SetRow(1, new Vector4((float)rotMatArr[3], (float)rotMatArr[4], (float)rotMatArr[5], (float)tvecArr[1]));
                        transformationM.SetRow(2, new Vector4((float)rotMatArr[6], (float)rotMatArr[7], (float)rotMatArr[8], (float)tvecArr[2]));
                        transformationM.SetRow(3, new Vector4(0, 0, 0, 1));
                        Debug.Log("transformationM " + transformationM.ToString());

                        Matrix4x4 invertYM = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(0.1f, -0.1f, 0.1f));
                        Debug.Log("invertYM " + invertYM.ToString());
                        result = true;

                        // right-handed coordinates system (OpenCV) to left-handed one (Unity)
                        // https://stackoverflow.com/questions/30234945/change-handedness-of-a-row-major-4x4-transformation-matrix
                        Matrix4x4 ARM = invertYM * transformationM * invertYM;
                        if(HLCamera != null)
                        {
                            ARM = HLCamera.transform.localToWorldMatrix * ARM;
                            ARUtils.SetTransformFromMatrix(axis3D.transform, ref ARM);
                        }
                    }
                }
            }
        }
        else
        {
            //Debug.Log("No Tag detected");
        }
        Utils.matToTexture2D(rgbMat, texture);
        return result;
    }
}
