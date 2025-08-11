using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

[ExecuteAlways]
public class WPMeta : MonoBehaviour
{
    public WaypointScriptableObject waypointData;

    private void OnValidate()
    {
        // Auto-sync GameObject name with ScriptableObject name
        if (waypointData != null && gameObject.name != waypointData.waypointName)
        {
            gameObject.name = waypointData.waypointName;
        }
    }

}