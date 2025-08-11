using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
using System;


public class AIController : MonoBehaviour
{
    public static AIController Instance { get; private set; }
    //public int currIndex { get; private set; } = -1;  // -1 means not on any quay
    private int moveCount = 0;

    public string currentTarget = "";
    private string previousTarget = "";

    private QuayVisualizer quayVisualizer;  // drag & drop in Inspector

    private Transform gridOrigin;     // Set this in Inspector near Sink
    public int gridRows = 8;
    public int gridCols = 10;
    public float gridSpacing = 1.0f;

    public SimulationClock simulationClock; // assign in inspector or find in Start
    [SerializeField] private float shipSize = 2.5f;  
    [SerializeField] private float gap = 0.05f;       // extra space between ships

    private static int deliveredCount = 0; // shared across all ships
    private bool hasBeenDelivered = false; // prevent multiple triggers
    public static int DeliveredCount => deliveredCount;
    public static int maxdelivery = 80;
    private ShipRuntime shipRuntime;

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
        shipRuntime = GetComponent<ShipRuntime>();


    }
    public void MoveTo(string locationName)
    {
        if (string.IsNullOrWhiteSpace(locationName))
            return;

        if (string.Equals(locationName.Trim(), "Source", StringComparison.OrdinalIgnoreCase))
            return;
        string trimmedLocation = locationName.Trim();
        // Don't move if the target location is the same as currentTarget
        if (trimmedLocation.Equals(currentTarget, StringComparison.OrdinalIgnoreCase))
        {
            // Already at this location, no need to move
            return;
        }
        var target = WPManager.Instance.GetWaypoint(trimmedLocation);
        if (target == null)
        {
            Debug.LogWarning($"{gameObject.name}: No waypoint found named '{trimmedLocation}'");
            return;
        }

        // Remember old target (what we were at before moving)
        string oldTarget = currentTarget;

        // Update current target
        currentTarget = trimmedLocation;
        moveCount++;
        if (shipRuntime != null)
        {
            shipRuntime.IncrementMoveCount();
        }
       

        // If we were on a quay and are leaving it (oldTarget != currentTarget), free the old quay

        if (!string.IsNullOrWhiteSpace(oldTarget) && !oldTarget.Equals("Source", StringComparison.OrdinalIgnoreCase) && QuayInfoPanel.Instance != null)
        {
            quayVisualizer.SetQuayEngagement(oldTarget, false);
            QuayInfoPanel.Instance?.NotifyShipChange(oldTarget, null);

        }

        if (IsQuayWall(currentTarget) && QuayInfoPanel.Instance != null)
        {
            quayVisualizer.SetQuayEngagement(currentTarget, true);
            ShipRuntime ship = GetComponent<ShipRuntime>();
            QuayInfoPanel.Instance?.NotifyShipChange(currentTarget, ship);
   
        }
        // Teleport/move to the waypoint
        transform.position = target.transform.position;

        // If going to Sink, free previous quay (if any), then deliver
        if (currentTarget.Equals("Sink", System.StringComparison.OrdinalIgnoreCase) && !hasBeenDelivered)
        {
            //if (!string.IsNullOrEmpty(oldTarget) && IsQuayWall(oldTarget))
            //{
            //    int prevIdx = quayVisualizer.quayScoreDB.quayWallNames.IndexOf(oldTarget);
            //    if (prevIdx >= 0)
            //        quayVisualizer.SetQuayEngagement(prevIdx, false);
            //}
            quayVisualizer.SetQuayEngagement(oldTarget, false);

            MoveShipToGrid();
            hasBeenDelivered = true;
        }

        // keep previousTarget if you still need it elsewhere
        previousTarget = oldTarget;
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

            float spacing = shipSize + gap;

            int row = deliveredCount / gridCols;
            int col = deliveredCount % gridCols;

            // Offset in local orientation
            Vector3 offset = (gridOrigin.right * col * spacing) +
                             (gridOrigin.forward * row * spacing);

            transform.position = gridOrigin.position + offset;

            deliveredCount++;

            this.enabled = false;

            if (DeliveredCount >= maxdelivery)
            {
                Debug.Log("All ships delivered. Stopping simulation.");
                SimulationClock.Instance.simulationStarted = false;
            }

            // Optional: visually mark the ship as "done"
            //GetComponent<Renderer>().material.color = Color.gray;

            //Color assignement based on the costs
            //float cost = ship.totalCost;
            //Color costColor = GetColorForCost(cost);
            //GetComponent<Renderer>().material.color = costColor;
        }
    public static void ResetDeliveredCount()
    {
        deliveredCount = 0;

    }
    
}
