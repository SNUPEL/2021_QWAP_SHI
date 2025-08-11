using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
using System;


public class AIController : MonoBehaviour
{
    public string currentTarget = "";
    private string previousTarget = "";

    private QuayVisualizer quayVisualizer;  // drag & drop in Inspector

    private Transform gridOrigin;     // Set this in Inspector near Sink
    public int gridRows = 8;
    public int gridCols = 10;
    public float gridSpacing = 1.0f;

    private static int deliveredCount = 0; // shared across all ships
    private bool hasBeenDelivered = false; // prevent multiple triggers
    public static int DeliveredCount => deliveredCount;

    private void Awake()
    {
        if (quayVisualizer == null)
        {
            quayVisualizer = FindObjectOfType<QuayVisualizer>();
        }
       GameObject sinkObj = GameObject.FindGameObjectWithTag("RL_Waypoint");
        if (sinkObj != null)
        {
            gridOrigin = GameObject.Find("Sink").transform;
        }

    }
    public void MoveTo(string locationName)
    {
        if (string.IsNullOrWhiteSpace(locationName)) return;

        string trimmedLocation = locationName.Trim();

        var target = WPManager.Instance.GetWaypoint(trimmedLocation);
        if (target == null)
        {
            Debug.LogWarning($"{gameObject.name}: No waypoint found named '{trimmedLocation}'");
            return;
        }

        // Update current target
        currentTarget = trimmedLocation;

        // If previous target was a quay wall, mark it free
        if (IsQuayWall(previousTarget))
        {
            int prevIndex = quayVisualizer.quayScoreDB.quayWallNames.IndexOf(previousTarget);
            if (prevIndex >= 0)
            {
                quayVisualizer.SetQuayEngagement(prevIndex, false);
            }
        }

        // If current target is a quay wall, mark it engaged
        if (IsQuayWall(currentTarget))
        {
            int currIndex = quayVisualizer.quayScoreDB.quayWallNames.IndexOf(currentTarget);
            if (currIndex >= 0)
            {
                quayVisualizer.SetQuayEngagement(currIndex, true);
            }
        }

        // Save this target as previous for next time
        previousTarget = currentTarget;

        // Actually move the ship (you currently just teleport)
        transform.position = target.transform.position;

        if (currentTarget == "Sink" && !hasBeenDelivered)
        {
            MoveShipToGrid();
            hasBeenDelivered = true; // prevent re-entering
        }
    }

    bool IsQuayWall(string locationName)
    {
        if (string.IsNullOrEmpty(locationName)) return false;

        // Check if this location name exists in your quayWallNames list
        return quayVisualizer.quayScoreDB.quayWallNames.Contains(locationName);
    }

    void MoveShipToGrid()
    {

        if (gridOrigin == null)
        {
            Debug.LogError("Grid origin not set! Can't position delivered ship.");
            return;
        }

        int row = deliveredCount / gridCols;
        int col = deliveredCount % gridCols;

        Vector3 offset = new Vector3(col * gridSpacing, 0, row * gridSpacing);
        transform.position = gridOrigin.position + offset;

        deliveredCount++;

        // Optional: visually mark the ship as "done"
        //GetComponent<Renderer>().material.color = Color.gray;

        // Disable this controller so it doesn't move again
        this.enabled = false;

        //Color assignement based on the costs
        //float cost = ship.totalCost;
        //Color costColor = GetColorForCost(cost);
        //GetComponent<Renderer>().material.color = costColor;


    }
}
