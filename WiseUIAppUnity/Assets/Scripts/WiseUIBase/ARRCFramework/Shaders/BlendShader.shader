Shader "Custom/BlendShader"
{
    Properties
    {
        _ImageTex("ImageTex", 2D) = "white" {}
        _BBoxTex("BBoxTex", 2D) = "white" {}
        _Transparency("Transparency", Range(0, 1)) = 0.5

    }
        SubShader
    {
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                float4 vertex : SV_POSITION;
            };
            sampler2D _MainTex;
            sampler2D _ImageTex;
            sampler2D _BBoxTex;
			float _Transparency;

            v2f vert(appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = v.uv;
                return o;
            }

            fixed4 frag(v2f i) : SV_Target
            {
                fixed4 col1 = tex2D(_MainTex, i.uv);
                fixed4 col2 = tex2D(_ImageTex, i.uv);
                fixed4 blendCol= col1 * (1 - _Transparency) + col2 * _Transparency; // Blend based on alpha

				if(col1[0] == 0 && col1[1] == 0 && col1[2]==0)
					return blendCol;
                else
					return col1;
            }
            ENDCG
        }
    }
}
