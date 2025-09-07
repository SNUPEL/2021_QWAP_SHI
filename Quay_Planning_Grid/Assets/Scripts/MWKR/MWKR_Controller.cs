using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class MWKR_Controller : MonoBehaviour
{
    public static MWKR_Controller Instance { get; private set; }
    //public int currIndex { get; private set; } = -1;  // -1 means not on any quay
    private int moveCount = 0;

    public string currentTarget = "";
    private string previousTarget = "";
    [SerializeField] private MWKR_Visualizer mwkrVisualizer; // drag your SPT visualizer here

    private Transform gridOrigin1;     // Set this in Inspector near Sink
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
    private MWKR_Runtime shipRuntime;

    private void Awake()
    {
        mwkrVisualizer = MWKR_Visualizer.Instance;
        if (mwkrVisualizer == null)
        {
            Debug.LogError("QuayVisualizer.Instance is null! Make sure QuayVisualizer exists in the scene before ships are spawned.");
        }

        Transform sinkTransform = null;

        if (MWKR_WPManager.Instance != null)
        {
            sinkTransform = MWKR_WPManager.Instance.GetWaypointByName("Sink")?.transform;
        }

        if (sinkTransform != null)
        {
            gridOrigin1 = sinkTransform;
        }
        else
        {
            Debug.LogWarning("SPT Sink not found!");
        }

        shipRuntime = GetComponent<MWKR_Runtime>();

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
        var target = MWKR_WPManager.Instance.GetWaypoint(trimmedLocation);
        if (target == null)
        {
            Debug.LogWarning($"{gameObject.name}: No waypoint found named '{trimmedLocation}'");
            return;
        }

        // Remember old target (what we were at before moving)
        string oldTarget = currentTarget;

        // Update current target
        currentTarget = trimmedLocation;
        
        // If we were on a quay and are leaving it (oldTarget != currentTarget), free the old quay

        if (!string.IsNullOrWhiteSpace(oldTarget) && !oldTarget.Equals("Source", StringComparison.OrdinalIgnoreCase) && QuayInfoPanel.Instance != null)
        {
            mwkrVisualizer.SetQuayEngagement(oldTarget, false);
            QuayInfoPanel.Instance?.NotifyShipChanges(oldTarget, null);

        }

        if (IsQuayWall(currentTarget) && QuayInfoPanel.Instance != null)
        {
            mwkrVisualizer.SetQuayEngagement(currentTarget, true);
            MWKR_Runtime mship = GetComponent<MWKR_Runtime>();
            //QuayInfoPanel.Instance?.NotifyShipChanges(currentTarget, mship);
            mship.IncrementMoveCount();
        }
        // Teleport/move to the waypoint
        transform.position = target.transform.position;

        // If going to Sink, free previous quay (if any), then deliver
        if (currentTarget.Equals("Sink", System.StringComparison.OrdinalIgnoreCase) && !hasBeenDelivered)
        {

            mwkrVisualizer.SetQuayEngagement(oldTarget, false);

            MoveShipToGrid();
            hasBeenDelivered = true;

            var runtime = GetComponent<MWKR_Runtime>();
            if (runtime != null)
            {
                int sinkDay = SimulationClock.Instance != null ? SimulationClock.Instance.simulationTime : 0;
                runtime.MarkDelivered(sinkDay);
            }
        }

        // keep previousTarget if you still need it elsewhere
        previousTarget = oldTarget;
    }
    bool IsQuayWall(string locationName)
    {
        if (string.IsNullOrEmpty(locationName)) return false;

        // Check if this location name exists in your quayWallNames list
        return mwkrVisualizer != null && mwkrVisualizer.quayScoreDB.quayWallNames.Contains(locationName);
    }
    void MoveShipToGrid()
    {
        if (gridOrigin1 == null)
        {
            Debug.LogError("Grid origin not set! Can't position delivered ship.");
            return;
        }

        float spacing = shipSize + gap;

        int row = deliveredCount / gridCols;
        int col = deliveredCount % gridCols;

        // Offset in local orientation
        Vector3 offset = (gridOrigin1.right * col * spacing) +
                         (gridOrigin1.forward * row * spacing);

        transform.position = gridOrigin1.position + offset;

        deliveredCount++;

        this.enabled = false;

        if (DeliveredCount >= maxdelivery)
        {
            Debug.Log("All ships delivered. Stopping simulation.");
            SimulationClock.Instance.simulationStarted = false;
        }

        MWKR_Runtime runtime = GetComponent<MWKR_Runtime>();
        if (runtime != null)
        {
            float totalCost = runtime.TotalCost;

            // Example: map totalCost to a 0–1 range for gradient
            float minCost = 15000f;      // expected min total cost
            float maxCost = 90000f; // expected max total cost
            float t = Mathf.Clamp01((totalCost - minCost) / (maxCost - minCost));

            // Gradient from white (low cost) → black (high cost)
            Color costColor = Color.Lerp(Color.white, Color.black, t);

            // Assign to renderer
            Renderer rend = GetComponent<Renderer>();
            if (rend != null)
            {
                rend.material.color = costColor;
            }
        }
    }
    public static void ResetDeliveredCount()
    {
        deliveredCount = 0;

    }
}
