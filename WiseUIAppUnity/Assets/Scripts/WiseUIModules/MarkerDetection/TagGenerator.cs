using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using OpenCVForUnity.CoreModule;
using OpenCVForUnity.ImgcodecsModule;
using OpenCVForUnity.ObjdetectModule;
using OpenCVForUnity.UnityUtils;
using System.IO;

public enum MarkerID
{
    MarkerID_0,
    MarkerID_1,
    MarkerID_2,
    MarkerID_3,
    MarkerID_4,
    MarkerID_5,
    MarkerID_6,
    MarkerID_7,
    MarkerID_8,
    MarkerID_9,
}

public enum MarkerType
{
    CanonicalMarker,
    GridBoard,
    ChArUcoBoard,
    //ChArUcoDiamondMarker //Not yet implemented.
}

public enum ArUcoDictionary
{
    DICT_4X4_50 = Objdetect.DICT_4X4_50,
    DICT_4X4_100 = Objdetect.DICT_4X4_100,
    DICT_4X4_250 = Objdetect.DICT_4X4_250,
    DICT_4X4_1000 = Objdetect.DICT_4X4_1000,
    DICT_5X5_50 = Objdetect.DICT_5X5_50,
    DICT_5X5_100 = Objdetect.DICT_5X5_100,
    DICT_5X5_250 = Objdetect.DICT_5X5_250,
    DICT_5X5_1000 = Objdetect.DICT_5X5_1000,
    DICT_6X6_50 = Objdetect.DICT_6X6_50,
    DICT_6X6_100 = Objdetect.DICT_6X6_100,
    DICT_6X6_250 = Objdetect.DICT_6X6_250,
    DICT_6X6_1000 = Objdetect.DICT_6X6_1000,
    DICT_7X7_50 = Objdetect.DICT_7X7_50,
    DICT_7X7_100 = Objdetect.DICT_7X7_100,
    DICT_7X7_250 = Objdetect.DICT_7X7_250,
    DICT_7X7_1000 = Objdetect.DICT_7X7_1000,
    DICT_ARUCO_ORIGINAL = Objdetect.DICT_ARUCO_ORIGINAL,
}

public class TagGenerator : MonoBehaviour
{
    /// <summary>
    ///  The size of the output marker image (px)
    /// </summary>
    [SerializeField]
    private int markerSize = 1000;

    /// <summary>
    /// The marker type
    /// </summary>
    [SerializeField]
    private MarkerType markerType = MarkerType.GridBoard;

    /// <summary>
    /// The dictionary identifier.
    /// </summary>
    [SerializeField]
    private ArUcoDictionary dictionaryId = ArUcoDictionary.DICT_6X6_250;

    /// <summary>
    /// The marker identifier.
    /// </summary>
    [SerializeField]
    private MarkerID markerId = MarkerID.MarkerID_1;

    /// <summary>
    /// Marker image save path
    /// </summary>
    [SerializeField]
    private string path = "./Assets/Scripts/WiseUIModules/MarkerDetection/Images/";

    /// <summary>
    /// The marker image matrix
    /// </summary>
    private Mat markerImg;

    /// <summary>
    /// The marker texture
    /// </summary>
    private Texture2D texture;

    /// <summary>
    /// GridBoard or ChArUcoBoard setting
    /// </summary>
    ///width of the marker borders.
    public int borderBits = 1;
    /// if markerType is GridBoard then number of markers in X/Y direction 
    /// else if markerType is ChArUcoBoard then number of chessboard squares in X/Y direction
    public int boardMarkersX = 6;
    public int boardMarkersY = 9;
    /// if markerType is GridBoard then marker side length[m]
    /// else if markerType is ChArUcoBoard then square side length[m]
    public float boardComponentsLength = 0.04f;
    /// if markerType is GridBoard then separation between two markers[m]
    /// else if markerType is ChArUcoBoard then marker side length[m]
    public float boardComponentsSeparation = 0.01f;
    // minimum margins (in pixels) of the board in the output image
    public int boardMarginSize = 10;

    private void Start()
    {
        markerImg = new Mat(markerSize, markerSize, CvType.CV_8UC3);
        texture = new Texture2D(markerImg.cols(), markerImg.rows(), TextureFormat.RGB24, false);
    }

    private void CreateMarkerImg()
    {
        markerImg.Dispose();
        markerImg = new Mat(markerSize, markerSize, CvType.CV_8UC3);
        texture = new Texture2D(markerImg.cols(), markerImg.rows(), TextureFormat.RGB24, false);
        markerImg.setTo(Scalar.all(255)); //Set marker image to white (blank image)

        gameObject.transform.localScale = new Vector3(markerImg.cols(), markerImg.rows(), 1);

        Dictionary dictionary = Objdetect.getPredefinedDictionary((int)dictionaryId);

        // draw marker.
        switch (markerType)
        {
            default:
            case MarkerType.CanonicalMarker:
                Objdetect.generateImageMarker(dictionary, (int)markerId, markerSize, markerImg, borderBits);
                Debug.Log("draw CanonicalMarker: " + "dictionaryId " + (int)dictionaryId + " markerId " + (int)markerId + " sidePixels " + markerSize + " borderBits " + borderBits);
                break;
            case MarkerType.GridBoard:
                GridBoard gridBoard = new GridBoard(new Size(boardMarkersX, boardMarkersY), boardComponentsLength, boardComponentsSeparation, dictionary);
                gridBoard.generateImage(new Size(markerSize, markerSize), markerImg, boardMarginSize, borderBits);
                gridBoard.Dispose();
                Debug.Log("draw GridBoard: " + "markersX " + boardMarkersX + " markersY " + boardMarkersY + " markerLength " + boardComponentsLength +
                " markerSeparation " + boardComponentsSeparation + "dictionaryId " + (int)dictionaryId + " outSize " + markerSize + " marginSize " + boardMarginSize + " borderBits " + borderBits);
                break;
            case MarkerType.ChArUcoBoard:
                CharucoBoard charucoBoard = new CharucoBoard(new Size(boardMarkersX, boardMarkersY), boardComponentsLength, boardComponentsSeparation, dictionary);
                charucoBoard.generateImage(new Size(markerSize, markerSize), markerImg, boardMarginSize, borderBits);
                charucoBoard.Dispose();
                Debug.Log("draw ChArUcoBoard: " + "markersX " + boardMarkersX + " markersY " + boardMarkersY + " markerLength " + boardComponentsLength +
                " markerSeparation " + boardComponentsSeparation + "dictionaryId " + (int)dictionaryId + " outSize " + markerSize + " marginSize " + boardMarginSize + " borderBits " + borderBits);
                break;
        }

        Utils.matToTexture2D(markerImg, texture, true, 0, true); //Convert Mat markerImg to Texture2D texture
    }

    private void SaveMarkerImg()
    {
        string format = "png";
        string savePath = "";
        MatOfInt compressionParams = new MatOfInt(Imgcodecs.IMWRITE_PNG_COMPRESSION, 0);

        switch (markerType)
        {
            default:
            case MarkerType.CanonicalMarker:
                savePath = Path.Combine(path, "CanonicalMarker-d" + (int)dictionaryId + "-i" + (int)markerId + "-sp" + markerSize + "-bb" + borderBits + "." + format);
                break;
            case MarkerType.GridBoard:
                savePath = Path.Combine(path, "GridBoard-mx" + boardMarkersX + "-my" + boardMarkersY + "-d" + (int)dictionaryId + "-os" + markerSize + "-bb" + borderBits + "." + format);
                break;
            case MarkerType.ChArUcoBoard:
                savePath = Path.Combine(path, "ChArUcoBoard-mx" +boardMarkersX + "-my" + boardMarkersY + "-d" + (int)dictionaryId + "-os" + markerSize + "-bb" + borderBits + "." + format);
                break;
        }

        if (!Directory.Exists(path))
        {
            Directory.CreateDirectory(path);
        }

        Imgcodecs.imwrite(savePath, markerImg, compressionParams);
        Debug.Log("savePath: " + savePath);
    }

    public void OnGenerateButtonPressed()
    {
        CreateMarkerImg();
        SaveMarkerImg();
    }

    void OnDestroy()
    {
        if (markerImg != null)
            markerImg.Dispose();
    }
}
