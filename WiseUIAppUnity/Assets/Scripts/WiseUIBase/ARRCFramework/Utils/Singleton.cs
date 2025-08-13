using UnityEngine;

/// <summary>
/// warning: this class is not thread safe.
/// </summary>
/// <typeparam name="T"></typeparam>
/// 
public class Singleton<T> : MonoBehaviour where T : MonoBehaviour
{
    private static T instance;

    public static T Instance
    {
        get
        {
            if (instance == null)
            {
                GameObject obj;
                obj = GameObject.Find(typeof(T).Name);
                if (obj == null)
                {
                    obj = new GameObject(typeof(T).Name);
                    instance = obj.AddComponent<T>();
                }
                else
                {
                    instance = obj.GetComponent<T>();
                }
            }
            return instance;
        }
    }
   
    public void Awake()
    {
        DontDestroyOnLoad(gameObject);
    }

    public void RemoveInstance()
    {
        DestroyImmediate(this.gameObject);
        instance = null;
    }
}