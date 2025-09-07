using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MWKR_Builder : MonoBehaviour
{
    public static MWKR_Builder Instance { get; private set; }

    public GameObject shipPrefab; // assign in inspector
    private MWKR_Data simData;
    public float delayBetweenSpawns = 0.5f;

    private List<ShipData> pendingShips = new List<ShipData>();
    private List<GameObject> activeShips = new List<GameObject>();
    public IReadOnlyList<GameObject> ActiveShips => activeShips.AsReadOnly();
    MWKR_Visualizer currentVisualizer;

    private int shipsToSpawn = 80; // limit for testing

    private void Awake()
    {
        if (Instance == null) Instance = this;
        else Destroy(gameObject);
    }

    private void Start()
    {
        Time.timeScale = 30.0f;

    }
    public void InitializeBuilder()
    {
        if (simData == null)
        {
            simData = FindObjectOfType<MWKR_Data>();
        }

        if (simData == null)
        {
            Debug.LogError("MWKR_Builder: No MWKR_Data found in the scene!");
            return;
        }
        simData.LoadSimulationLogs("log-MWKR-MF.csv");

        LoadAndSortShips();
        shipsToSpawn = 80;
        if (SimulationClock.Instance != null)
        {
            SimulationClock.Instance.OnTimeChanged += HandleTimeChanged;
        }
    }
    void LoadAndSortShips()
    {
        pendingShips = new List<ShipData>(Resources.LoadAll<ShipData>("Ships"));
        Debug.Log($"Loaded {pendingShips.Count} ships from Resources/Ships");
        pendingShips.Sort((a, b) => a.Launching_Date.CompareTo(b.Launching_Date));
    }

    public void HandleTimeChanged(int simTime)
    {
        while (pendingShips.Count > 0 && pendingShips[0].Launching_Date <= simTime && shipsToSpawn > 0)
        {
            ShipData shipData = pendingShips[0];
            SpawnShip(shipData);
            pendingShips.RemoveAt(0);
            shipsToSpawn--;
        }
    }

    void SpawnShip(ShipData shipData)
    {
        // Get Source waypoint position from your WPManager singleton
        GameObject sourceWP = MWKR_WPManager.Instance.GetWaypoint("Source");
        if (sourceWP == null)
        {
            Debug.LogError("Source waypoint not found!");
            return;
        }

        Vector3 basePos = sourceWP.transform.position + new Vector3(0, 1.0f, 0);

        // Instantiate ship at spawnPos without NavMesh or movement
        GameObject newShip = Instantiate(shipPrefab, basePos, Quaternion.identity);
        newShip.name = $"Ship_{shipData.Ship_Name}";

        // Get or add ShipRuntime component and assign data + logs
        MWKR_Runtime runtime = newShip.GetComponent<MWKR_Runtime>();
        if (runtime == null)
        {
            runtime = newShip.AddComponent<MWKR_Runtime>();
        }

        runtime.Data = shipData;
        if (simData != null)
        {
            string normalizedShipName = shipData.Ship_Name.Trim().ToUpperInvariant();
            Debug.Log("SimulationData.Instance = " + MWKR_Data.Instance);

            runtime.Logs = simData.GetLogsForShips(shipData.Ship_Name);
            Debug.Log($"[{shipData.Ship_Name}] Assigned {runtime.Logs.Count} logs");
        }
        else
        {
            Debug.LogWarning("Simulation_Data singleton not found!");
        }

        // Register ship for tracking
        activeShips.Add(newShip);

        // Start processing logs immediately
        runtime.StartFrom(SimulationClock.Instance.simulationTime);
    }

    public void ResetBuilder()
    {
        // Debug.LogWarning("[ShipBuilder] Resetting and clearing ships.");

        foreach (var ship in activeShips)
        {
            Destroy(ship);
        }
        activeShips.Clear();
        if (SimulationClock.Instance != null)
            SimulationClock.Instance.OnTimeChanged -= HandleTimeChanged;
        pendingShips.Clear();
        shipsToSpawn = 80;
    }
}
