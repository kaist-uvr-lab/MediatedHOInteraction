using System.IO;
using UnityEngine;
using UnityEngine.Rendering;
using WiseUI.Base;


public enum RenderMode { Normal, OverlayImage };
[ExecuteInEditMode]
public class BlendDuringRender : MonoBehaviour
{
    public Texture2D texture;

    [SerializeField]
    Material blendMaterial, unlitSimpleMaterial;

    [SerializeField]
    CommandBuffer commandBuffer;

    public RenderMode renderMode = RenderMode.OverlayImage;

    private void Start()
    {
        Init();
    }
    public void Init()
    {
        blendMaterial = new Material(Shader.Find("Custom/BlendShader"));

        commandBuffer = new CommandBuffer();
        commandBuffer.name = "ImageBlender";
    }
    public void SetBlendedTexture(Texture2D texture)
    {
        this.texture = texture;
        blendMaterial.SetTexture("_ImageTex", texture);
    }
    public void SetTransparency(float transparency)
    {
        blendMaterial.SetFloat("_Transparency", transparency);
    }

    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        
        if (renderMode == RenderMode.OverlayImage && commandBuffer != null && texture != null)
        {
            commandBuffer.Clear();
            blendMaterial.SetTexture("_MainTex", source);
            commandBuffer.Blit(source, destination, blendMaterial);
            Graphics.ExecuteCommandBuffer(commandBuffer);
        }
        else
        {
            Graphics.Blit(source, destination);
        }
    }
}
