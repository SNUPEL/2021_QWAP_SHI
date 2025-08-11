using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RLWP_Manager : MonoBehaviour
{
    public static RLWP_Manager Instance;

    public Dictionary<string, GameObject> waypoints = new();
    public Dictionary<string, WaypointScriptableObject> waypointData = new();

    private void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Debug.LogWarning("RL_WPManager: Duplicate detected. Destroying extra.");
            Destroy(gameObject);
            return;
        }

        Instance = this;

        GameObject[] found = GameObject.FindGameObjectsWithTag("RLWaypoint");
        foreach (var wp in found)
        {
            string cleanName = wp.name.Trim();
            if (!waypoints.ContainsKey(cleanName))
                waypoints.Add(cleanName, wp);
        }

        WaypointScriptableObject[] foundSOs = Resources.LoadAll<WaypointScriptableObject>("RLWaypoints");
        foreach (var so in foundSOs)
        {
            string cleanName = so.waypointName.Trim();
            if (!waypointData.ContainsKey(cleanName))
                waypointData.Add(cleanName, so);
        }

        Debug.Log($"RL_WPManager: Registered {waypoints.Count} RL waypoints.");
    }

    public GameObject GetWaypoint(string name)
    {
        waypoints.TryGetValue(name.Trim(), out GameObject wp);
        return wp;
    }

    public WaypointScriptableObject GetWaypointSO(string name)
    {
        name = name.Trim();
        if (waypointData.TryGetValue(name, out var so)) return so;

        Debug.LogWarning($"RL_WPManager: No WaypointScriptableObject found for '{name}'");
        return null;
    }
}
