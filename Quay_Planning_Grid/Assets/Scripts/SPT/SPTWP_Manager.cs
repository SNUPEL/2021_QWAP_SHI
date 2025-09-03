using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using System;


public class SPTWP_Manager : MonoBehaviour
{
    public static SPTWP_Manager Instance;

    public Dictionary<string, GameObject> waypoints = new();

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
 
    private void Awake()
    {
        Instance = this;

        GameObject[] found = GameObject.FindGameObjectsWithTag("SPT_Waypoint");
        foreach (var wp in found)
        {
            string cleanName = wp.name.Trim();
            if (!waypoints.ContainsKey(cleanName))
                waypoints.Add(cleanName, wp);
        }

        Debug.Log($"WPManager: Registered {waypoints.Count} waypoints.");
    }
}

