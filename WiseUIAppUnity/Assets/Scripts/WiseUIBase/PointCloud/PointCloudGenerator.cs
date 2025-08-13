using System.IO;
using System;
using System.Linq;
using UnityEngine;


namespace PointCloudExporter
{
	public static class PointCloudGenerator
	{
        enum DataType { __Float, __Double };
        const int verticesMax = 128 * 128;

        //[Header("Point Cloud")]
        //public string fileName = "point";

        //[Header("Renderer")]
        //public float size = 0.1f;
        //public Texture sprite;
        //public Shader shader;

        //[Header("Displace")]
        //[Range(0,1)]
        //public float should = 0.5f;
        //public float time = 1f;
        //public float speed = 1f;
        //public float noiseScale = 1f;
        //public float noiseSpeed = 1f;
        //public float targetSpeed = 1f;
        //[Range(0,1)]
        //public float noisy = 0.1f;
        //public Transform targetDisplace;

        //[Header("Baking")]
        //public int details = 32;
        //public int circleRadius = 32;
        //public Shader shaderBaked;

        //private MeshInfos points;

        //private Mesh[] meshArray;
        //private float displaceFiredAt = -1000f;
        //private Texture2D colorMapTexture;

        /*
		void Update ()
		{
			material.SetFloat("_Size", size);
			material.SetTexture("_MainTex", sprite);

			if (displaceFiredAt + time > Time.time) {
				Displace(Time.deltaTime);
			}
		}*/

        public static MeshInfos LoadPointCloud(string filePath, int maximumVertex = 6000000, float fScale = 1.0f)
        {
            MeshInfos data = new MeshInfos();
            int levelOfDetails = 1;
            if (File.Exists(filePath))
            {
                using (BinaryReader reader = new BinaryReader(File.Open(filePath, FileMode.Open)))
                {
                    int cursor = 0;
                    int length = (int)reader.BaseStream.Length;
                    string lineText = "";
                    bool header = true;
                    int vertexCount = 0;
                    int colorDataCount = 3;
                    int index = 0;
                    int step = 0;
                    int normalDataCount = 0;
                    DataType dataType = DataType.__Float;

                    //문제 1.
                    //Data Type에 따른 파씽 처리가 제대로되어 있지 않다.
                    //예를 들어 {x,y,z}, {nx, ny, nz}, 에 대한 type이 파일마다 다를 수 있는데,
                    //현재 헤더에 명시된 property [type] x을 기준으로 x,y,z, nx, ny,nz의 read type을 결정하기 때문에
                    //에러가 발생할 수 있다. 20201008 IbJeon.

                    //문제 2.
                    //PC의 좌표계 정립이 되어 있지 않다. 현재 x y,z를 읽을 때 z에 -1을 곱하도록 되어있다.
                    while (cursor + step < length)
                    {
                        if (header)
                        {
                            char v = reader.ReadChar();
                            if (v == '\n')
                            {
                                if (lineText.Contains("end_header"))
                                {
                                    header = false;
                                }
                                else if (lineText.Contains("element vertex"))
                                {
                                    string[] array = lineText.Split(' ');
                                    if (array.Length > 0)
                                    {
                                        int subtractor = array.Length - 2;
                                        vertexCount = Convert.ToInt32(array[array.Length - subtractor]);

                                        if (vertexCount > maximumVertex)
                                        {
                                            levelOfDetails = 1 + (int)Mathf.Floor(vertexCount / maximumVertex);
                                            vertexCount = maximumVertex;
                                        }
                                        data.vertexCount = vertexCount;
                                        data.vertices = new Vector3[vertexCount];
                                        data.normals = new Vector3[vertexCount];
                                        data.colors = new Color[vertexCount];
                                    }
                                }
                                else if (lineText.Contains("property uchar alpha"))
                                {
                                    colorDataCount = 4;
                                }
                                else if (lineText.Contains("property float n") || lineText.Contains("property double n"))
                                {
                                    normalDataCount += 1;
                                }
                                else if (lineText.Contains("property double x"))
                                { //property double x 만을 보고 버텍스 및 노말 타입을 정하기 때문에 문제의 소지가 있다.
                                    dataType = DataType.__Double;
                                }

                                lineText = "";
                            }
                            else
                            {
                                lineText += v;
                            }
                            step = sizeof(char);
                            cursor += step;
                        }
                        else
                        {
                            if (index < vertexCount)
                            {
                                if (dataType == DataType.__Float)
                                {
                                    float px = -reader.ReadSingle();
                                    float py = reader.ReadSingle();
                                    float pz = reader.ReadSingle();

                                    data.vertices[index] = new Vector3(px, py, pz) * fScale;
                                    if (normalDataCount == 3)
                                    {
                                        float nx = -reader.ReadSingle();
                                        float ny = reader.ReadSingle();
                                        float nz = reader.ReadSingle();

                                        data.normals[index] = new Vector3(nx, ny, nz);
                                    }
                                    else
                                    {
                                        data.normals[index] = new Vector3(1f, 1f, 1f);
                                    }
                                    data.colors[index] = new Color(reader.ReadByte() / 255f, reader.ReadByte() / 255f, reader.ReadByte() / 255f, 1f);

                                    step = sizeof(float) * 6 * levelOfDetails + sizeof(byte) * colorDataCount * levelOfDetails;
                                }
                                else if (dataType == DataType.__Double)
                                {
                                    double px = -reader.ReadDouble();
                                    double py = reader.ReadDouble();
                                    double pz = reader.ReadDouble();

                                    data.vertices[index] = new Vector3((float)px, (float)py, (float)pz) * fScale;

                                    if (normalDataCount == 3)
                                    {
                                        double nx = -reader.ReadDouble();
                                        double ny = reader.ReadDouble();
                                        double nz = reader.ReadDouble();

                                        data.normals[index] = new Vector3((float)nx, (float)ny, (float)nz);
                                    }
                                    else
                                    {
                                        data.normals[index] = new Vector3(1f, 1f, 1f);
                                    }
                                    data.colors[index] = new Color(reader.ReadByte() / 255f, reader.ReadByte() / 255f, reader.ReadByte() / 255f, 1f);

                                    step = sizeof(double) * 6 * levelOfDetails + sizeof(byte) * colorDataCount * levelOfDetails;
                                }

                                cursor += step;

                                //아래의 경우 무시한다.
                                if (colorDataCount > 3)
                                {
                                    reader.ReadByte();
                                }

                                if (levelOfDetails > 1)
                                {
                                    for (int l = 1; l < levelOfDetails; ++l)
                                    {
                                        for (int f = 0; f < 3 + normalDataCount; ++f)
                                        {
                                            reader.ReadSingle();
                                        }
                                        for (int b = 0; b < colorDataCount; ++b)
                                        {
                                            reader.ReadByte();
                                        }
                                    }
                                }

                                ++index;
                            }
                        }
                    }
                }
            }
            return data;
        }

        public static GameObject ToGameObject(this MeshInfos meshInfo, string name, float point_size)
        {
            GameObject pointCloudObj = new GameObject(name);
            Material ptMaterial = new Material(Shader.Find("Unlit/PointCloud"));
            ptMaterial.SetFloat("_Size", point_size);

            Texture2D sprite = (Texture2D)Resources.Load("Textures/Circle");
            //ptMaterial.mainTexture = sprite as Texture;
            ptMaterial.SetTexture("_MainTex", sprite);

            Generate(meshInfo, ptMaterial, MeshTopology.Points, pointCloudObj);

            return pointCloudObj;
        }

        static void Generate(MeshInfos meshInfos, Material materialToApply, MeshTopology topology, GameObject parentObj)
        {
            int vertexCount = meshInfos.vertexCount;
            int meshCount = (int)Mathf.Ceil(vertexCount / (float)verticesMax);

            var meshArray = new Mesh[meshCount];

            int index = 0;
            int meshIndex = 0;
            int vertexIndex = 0;

            int resolution = GetNearestPowerOfTwo(Mathf.Sqrt(vertexCount));

            while (meshIndex < meshCount)
            {

                int count = verticesMax;
                if (vertexCount <= verticesMax)
                {
                    count = vertexCount;
                }
                else if (vertexCount > verticesMax && meshCount == meshIndex + 1)
                {
                    count = vertexCount % verticesMax;
                }

                Vector3[] subVertices = meshInfos.vertices.Skip(meshIndex * verticesMax).Take(count).ToArray();
                Vector3[] subNormals = meshInfos.normals.Skip(meshIndex * verticesMax).Take(count).ToArray();
                Color[] subColors = meshInfos.colors.Skip(meshIndex * verticesMax).Take(count).ToArray();
                int[] subIndices = new int[count];
                for (int i = 0; i < count; ++i)
                {
                    subIndices[i] = i;
                }

                Mesh mesh = new Mesh();
                mesh.bounds = new Bounds(Vector3.zero, Vector3.one * 100f);
                mesh.vertices = subVertices;
                mesh.normals = subNormals;
                mesh.colors = subColors;
                mesh.SetIndices(subIndices, topology, 0);

                Vector2[] uvs2 = new Vector2[mesh.vertices.Length];
                for (int i = 0; i < uvs2.Length; ++i)
                {
                    float x = vertexIndex % resolution;
                    float y = Mathf.Floor(vertexIndex / (float)resolution);
                    uvs2[i] = new Vector2(x, y) / (float)resolution;
                    ++vertexIndex;
                }
                mesh.uv2 = uvs2;

                GameObject go = CreateGameObjectWithMesh(mesh, materialToApply, parentObj.name + "_" + meshIndex, parentObj.transform);

                meshArray[meshIndex] = mesh;
                go.transform.parent = parentObj.transform;

                index += count;
                ++meshIndex;


            }
        }

        static GameObject CreateGameObjectWithMesh(Mesh mesh, Material materialToApply, string name = "GeneratedMesh", Transform parent = null)
        {
            GameObject meshGameObject = GameObject.CreatePrimitive(PrimitiveType.Cube);
            GameObject.DestroyImmediate(meshGameObject.GetComponent<Collider>());
            meshGameObject.GetComponent<MeshFilter>().mesh = mesh;
            meshGameObject.GetComponent<Renderer>().sharedMaterial = materialToApply;
            meshGameObject.name = name;
            meshGameObject.transform.parent = parent;
            meshGameObject.transform.localPosition = Vector3.zero;
            meshGameObject.transform.localRotation = Quaternion.identity;
            meshGameObject.transform.localScale = Vector3.one;
            return meshGameObject;
        }

        /// <summary>
        /// refer. : http://stackoverflow.com/questions/466204/rounding-up-to-nearest-power-of-2
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        static int GetNearestPowerOfTwo(float x)
        {
            return (int)Mathf.Pow(2f, Mathf.Ceil(Mathf.Log(x) / Mathf.Log(2f)));
        }

        //public void Export ()
        //{
        //	MeshInfos triangles = GetTriangles(points, size);
        //	materialBaked = new Material(shaderBaked);
        //	Generate(triangles, materialBaked, MeshTopology.Triangles);
        //	materialBaked.SetTexture("_MainTex", GetBakedColors(triangles));
        //}

        //public void Displace()
        //{
        //	displaceFiredAt = Time.time;
        //}

        //public void Displace(float dt)
        //{
        //	int meshInfosIndex = 0;
        //	for (int meshIndex = 0; meshIndex < meshArray.Length; ++meshIndex)
        //	{
        //		Mesh mesh = meshArray[meshIndex];
        //		Vector3[] vertices = mesh.vertices;
        //		Vector3[] normals = mesh.normals;
        //		Vector3 offsetNoise = new Vector3();
        //		Vector3 offsetTarget = new Vector3();
        //		Matrix4x4 matrixWorld = transform.localToWorldMatrix;
        //		Matrix4x4 matrixLocal = transform.worldToLocalMatrix;
        //		for (int vertexIndex = 0; vertexIndex < vertices.Length; ++vertexIndex)
        //		{
        //			Vector3 position = matrixWorld.MultiplyVector(vertices[vertexIndex]) + transform.position;
        //			Vector3 normal = normals[vertexIndex];

        //			offsetNoise.x = Mathf.PerlinNoise(position.x * noiseScale, position.y * noiseScale);
        //			offsetNoise.y = Mathf.PerlinNoise(position.y * noiseScale, position.z * noiseScale);
        //			offsetNoise.z = Mathf.PerlinNoise(position.z * noiseScale, position.x * noiseScale);
        //			offsetNoise = (offsetNoise * 2f - Vector3.one) * noiseSpeed;

        //			offsetTarget = Vector3.Normalize(position - targetDisplace.position) * targetSpeed;

        //			float noisyFactor = Mathf.Lerp(1f, Random.Range(0f, 1f), noisy);
        //			float shouldMove = Mathf.InverseLerp(1f - should, 1f, Mathf.PerlinNoise(normal.x * 8f, normal.y * 8f));

        //			position += (offsetNoise + offsetTarget) * dt * speed * noisyFactor * shouldMove;
        //			vertices[vertexIndex] = matrixLocal.MultiplyVector(position - transform.position);
        //			++meshInfosIndex;
        //		}
        //		mesh.vertices = vertices;
        //	}
        //}

        //public void Reset ()
        //{
        //	int meshInfosIndex = 0;
        //	for (int meshIndex = 0; meshIndex < meshArray.Length; ++meshIndex) {
        //		Mesh mesh = meshArray[meshIndex];
        //		Vector3[] vertices = mesh.vertices;
        //		for (int vertexIndex = 0; vertexIndex < vertices.Length; ++vertexIndex) {
        //			vertices[vertexIndex] = meshInfo.vertices[meshInfosIndex];
        //			++meshInfosIndex;
        //		}
        //		mesh.vertices = vertices;
        //	}
        //}

        //public MeshInfos GetTriangles (MeshInfos points, float radius)
        //{
        //	MeshInfos triangles = new MeshInfos();
        //	triangles.vertexCount = points.vertexCount * 3;
        //	triangles.vertices = new Vector3[triangles.vertexCount];
        //	triangles.normals = new Vector3[triangles.vertexCount];
        //	triangles.colors = new Color[triangles.vertexCount];
        //	int index = 0;
        //	int meshVertexIndex = 0;
        //	int meshIndex = 0;
        //	Vector3[] vertices = meshArray[meshIndex].vertices;
        //	for (int v = 0; v < triangles.vertexCount; v += 3) {
        //		Vector3 center = vertices[meshVertexIndex];
        //		Vector3 normal = points.normals[index];
        //		Vector3 tangent = Vector3.Normalize(Vector3.Cross(Vector3.up, normal));
        //		Vector3 up = Vector3.Normalize(Vector3.Cross(tangent, normal));

        //		triangles.vertices[v] = center + tangent * -radius / 1.5f;
        //		triangles.vertices[v+1] = center + up * radius;
        //		triangles.vertices[v+2] = center + tangent * radius / 1.5f;

        //		triangles.normals[v] = normal;
        //		triangles.normals[v+1] = normal;
        //		triangles.normals[v+2] = normal;

        //		Color color = points.colors[index];
        //		triangles.colors[v] = color;
        //		triangles.colors[v+1] = color;
        //		triangles.colors[v+2] = color;

        //		++meshVertexIndex;

        //		if (meshVertexIndex >= meshArray[meshIndex].vertices.Length) {
        //			meshVertexIndex = 0;
        //			++meshIndex;
        //			if (meshIndex < meshArray.Length) {
        //				vertices = meshArray[meshIndex].vertices;
        //			}
        //		}

        //		++index;
        //	}
        //	return triangles;
        //}

        //public Texture2D GetBakedColors (MeshInfos triangles)
        //{
        //	List<Color> colorList = new List<Color>();
        //	int[] colorIndexMap = new int[triangles.vertexCount / 3];
        //	int globalIndex = 0;
        //	for (int meshIndex = 0; meshIndex < meshArray.Length; ++meshIndex) {
        //		Mesh mesh = meshArray[meshIndex];
        //		Color[] colors = mesh.colors;
        //		for (int i = 0; i < colors.Length; i += 3) {
        //			Color color = colors[i];
        //			Color colorSimple = new Color(Mathf.Floor(color.r * details) / details, Mathf.Floor(color.g * details) / (float)details, Mathf.Floor(color.b * details) / (float)details);

        //			int colorIndex = colorList.IndexOf(colorSimple);
        //			if (colorIndex == -1) {
        //				colorIndex = colorList.Count;
        //				colorList.Add(colorSimple);
        //			}

        //			colorIndexMap[globalIndex] = colorIndex;
        //			++globalIndex;
        //		}
        //	}

        //	int colorCount = colorList.Count;
        //	int columnCount = GetNearestPowerOfTwo(Mathf.Sqrt(colorCount));
        //	int rowCount = columnCount;//1 + (int)Mathf.Floor(colorCount / (float)columnCount);
        //	int width = circleRadius * columnCount;
        //	int height = circleRadius * rowCount;

        //	colorMapTexture = new Texture2D(width, height, TextureFormat.RGBA32, false);
        //	Color[] colorMapArray = new Color[width * height];
        //	Vector2 pos;
        //	Vector2 target = new Vector2(0.5f, 0.3f);
        //	for (int i = 0; i < colorList.Count; ++i) {
        //		int x = i % columnCount;
        //		int y = (int)Mathf.Floor(i / columnCount);
        //		for (int c = 0; c < circleRadius*circleRadius; ++c) {
        //			int ix = c % circleRadius;
        //			int iy = (int)Mathf.Floor(c / circleRadius);
        //			pos.x = ix / (float)circleRadius;
        //			pos.y = iy / (float)circleRadius;
        //			float dist = Mathf.Clamp01(Vector2.Distance(target, pos));
        //			int colorIndex = x * circleRadius + y * width * circleRadius + ix + iy * width;
        //			float circle = 1f - Mathf.InverseLerp(0.2f, 0.35f, dist);
        //			colorMapArray[colorIndex] = Color.Lerp(Color.clear, colorList[i], circle);
        //		}
        //	}
        //	colorMapTexture.SetPixels(colorMapArray);
        //	colorMapTexture.Apply(false);

        //	Vector2 halfSize = new Vector2(0.5f * circleRadius / (float)width, 0.5f * circleRadius / (float)height);
        //	Vector2 right = Vector2.right * halfSize.x;
        //	Vector2 up = Vector2.up * halfSize.y;
        //	globalIndex = 0;
        //	for (int meshIndex = 0; meshIndex < meshArray.Length; ++meshIndex) {
        //		Mesh mesh = meshArray[meshIndex];
        //		Vector2[] uvs = new Vector2[mesh.vertices.Length];
        //		for (int i = 0; i < uvs.Length; i += 3) {

        //			int colorIndex = colorIndexMap[globalIndex];
        //			float x = ((colorIndex % columnCount) * circleRadius) / (float)width;
        //			float y = (Mathf.Floor(colorIndex / columnCount) * circleRadius) / (float)height;
        //			Vector2 center = new Vector2(x + halfSize.x, y + halfSize.y);

        //			uvs[i] = center + right - up;
        //			uvs[i+1] = center + up;
        //			uvs[i+2] = center - right - up;

        //			++globalIndex;
        //		}
        //		mesh.uv = uvs;
        //	}

        //	return colorMapTexture;
        //}

        //public Texture2D GetBakedMap ()
        //{
        //	return colorMapTexture;
        //}
    }
}
