using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using System;

public class WPManager : MonoBehaviour
{
    public static WPManager Instance;
    public Dictionary<string, GameObject> waypoints = new();
    public Dictionary<string, WaypointScriptableObject> waypointData = new();

    public GameObject GetWaypointByName(string name)
    {
        name = name.Trim();
        foreach (var wp in waypoints)
        {
            if (wp.Key.Trim().Equals(name, StringComparison.OrdinalIgnoreCase))
            {
                return wp.Value;
            }
        }
        Debug.LogWarning($"WPManager: Waypoint '{name}' not found.");
        return null;
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

        Debug.LogWarning($"WPManager: No WaypointScriptableObject found for '{name}'");
        return null;
    }

    private void Awake()
    {
        Instance = this;

        GameObject[] found = GameObject.FindGameObjectsWithTag("RL_Waypoint");
        foreach (var wp in found)
        {
            string cleanName = wp.name.Trim();
            if (!waypoints.ContainsKey(cleanName))
                waypoints.Add(cleanName, wp);
        }

        WaypointScriptableObject[] foundSOs = Resources.LoadAll<WaypointScriptableObject>("Waypoints");
        foreach (var so in foundSOs)
        {
            string cleanName = so.waypointName.Trim();
            if (!waypointData.ContainsKey(cleanName))
                waypointData.Add(cleanName, so);
        }

        Debug.Log($"WPManager: Registered {waypoints.Count} waypoints.");
    }
}

//public class WPManager : MonoBehaviour
//{
//    public Dictionary<string, GameObject> waypoints = new();
//    public Dictionary<string, WaypointScriptableObject> waypointData = new();

//    public virtual string WaypointTag => "Waypoint";
//    public virtual string ResourceFolder => "Waypoints";

//    public GameObject GetWaypoint(string name)
//    {
//        waypoints.TryGetValue(name.Trim(), out GameObject wp);
//        return wp;
//    }

//    public WaypointScriptableObject GetWaypointSO(string name)
//    {
//        name = name.Trim();
//        if (waypointData.TryGetValue(name, out var so)) return so;

//        Debug.LogWarning($"{this.GetType().Name}: No WaypointScriptableObject found for '{name}'");
//        return null;
//    }

//    protected virtual void Awake()
//    {
//        GameObject[] found = GameObject.FindGameObjectsWithTag(WaypointTag);
//        foreach (var wp in found)
//        {
//            string cleanName = wp.name.Trim();
//            if (!waypoints.ContainsKey(cleanName))
//                waypoints.Add(cleanName, wp);
//        }

//        WaypointScriptableObject[] foundSOs = Resources.LoadAll<WaypointScriptableObject>(ResourceFolder);
//        foreach (var so in foundSOs)
//        {
//            string cleanName = so.waypointName.Trim();
//            if (!waypointData.ContainsKey(cleanName))
//                waypointData.Add(cleanName, so);
//        }

//        Debug.Log($"{this.GetType().Name}: Registered {waypoints.Count} waypoints.");
//    }
//}

//public class Sim_Manager : WPManager
//{
//    public static Sim_Manager Instance;

//    protected override void Awake()
//    {
//        Instance = this;
//        base.Awake();
//    }

//    public override string WaypointTag => "Waypoint";
//    public override string ResourceFolder => "Waypoints";
//}

//public class RL_WPManager : WPManager
//{
//    public static RL_WPManager Instance;

//    protected override void Awake()
//    {
//        Instance = this;
//        base.Awake();
//    }

//    public override string WaypointTag => "RLWaypoint";
//    public override string ResourceFolder => "RLWaypoints";
//}