using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class RL_ShipBuilder : MonoBehaviour
{
    public static RL_ShipBuilder Instance { get; private set; }

    public GameObject rlShipPrefab; // Assign in Inspector
    private RL_Simulation_Data RLData;
    public float delayBetweenSpawns = 2f;
    public Vector3 spawnPosition = Vector3.zero;

    private List<ShipData> pendingShips = new List<ShipData>();
    private List<GameObject> activeShips = new List<GameObject>();
    int shipsToSpawn = 80; // limit for testing

    private void Awake()
    {
        if (Instance == null) Instance = this;
        else Destroy(gameObject);
    }

    void LoadAndSortShips()
    {
        pendingShips = new List<ShipData>(Resources.LoadAll<ShipData>("RLShips")); // <-- RL-specific folder
        Debug.Log($"[RL] Loaded {pendingShips.Count} ships from Resources/RLShips");

        pendingShips.Sort((a, b) => a.Launching_Date.CompareTo(b.Launching_Date));
    }

    public void LoadSchedule(RL_Simulation_Data newData)
    {
        RLData = newData;
        LoadAndSortShips();

        SimulationClock.Instance.OnTimeChanged -= HandleTimeChanged;
        SimulationClock.Instance.OnTimeChanged += HandleTimeChanged;

        HandleTimeChanged(SimulationClock.Instance.simulationTime);
    }

    public void HandleTimeChanged(int simTime)
    {
        Debug.Log($"[RL_Builder] HandleTimeChanged({simTime}). Pending: {pendingShips.Count}");

        while (pendingShips.Count > 0 && pendingShips[0].Launching_Date <= simTime)
        {
            ShipData shipData = pendingShips[0];
            InstantiateShips(shipData);
            pendingShips.RemoveAt(0);
            shipsToSpawn--;
        }
    }

    void InstantiateShips(ShipData shipData)
    {
        var spawnWP = RLWP_Manager.Instance.GetWaypoint("Source");
        if (spawnWP == null)
        {
            Debug.LogError("[RL_ShipBuilder] No Source waypoint found!");
            return;
        }

        Vector2 offset2D = Random.insideUnitCircle * 10f;
        Vector3 offset = new Vector3(offset2D.x, 0, offset2D.y);
        Vector3 spawnPos = spawnWP.transform.position + offset;

        if (!NavMesh.SamplePosition(spawnPos, out NavMeshHit hit, 1f, NavMesh.AllAreas))
        {
            Debug.LogError($"[RL] No NavMesh found near spawn position {spawnPos} for ship {shipData.Ship_Name}");
            return;
        }

        GameObject ship = Instantiate(rlShipPrefab, hit.position, Quaternion.identity);
        ship.name = $"RL_Ship_{shipData.Ship_Name}";
        activeShips.Add(ship);

        var agent = ship.GetComponent<NavMeshAgent>();
        if (agent != null)
        {
            agent.baseOffset = 0.1f;
            agent.Warp(hit.position);
        }

        var runtime = ship.AddComponent<RL_ShipRuntime>();
        runtime.Data = shipData;

        if (RLData == null)
            RLData = FindObjectOfType<RL_Simulation_Data>();

        if (RLData != null)
        {
            runtime.Logs = RLData.GetLogsForShips(shipData.Ship_Name);
        }

        var ai = ship.GetComponent<RL_Controller>();
        if (ai == null)
            ai = ship.AddComponent<RL_Controller>();

        runtime.PrintAllLogs();
        StartCoroutine(StartAfterNavMeshReady(runtime));
    }

    IEnumerator StartAfterNavMeshReady(RL_ShipRuntime runtime)
    {
        //yield return new WaitForEndOfFrame();
        //yield return new WaitUntil(() => runtime.GetComponent<NavMeshAgent>()?.isOnNavMesh == true);

        //runtime.StartFrom(SimulationClock.Instance.simulationTime);
        yield return new WaitForEndOfFrame();

        // Store references safely
        var agent = runtime != null ? runtime.GetComponent<NavMeshAgent>() : null;

        // Protect against null or destroyed objects
        yield return new WaitUntil(() =>
            runtime != null &&
            agent != null &&
            agent.isOnNavMesh
        );

        // Double-check again before continuing
        if (runtime == null || agent == null) yield break;

        runtime.StartFrom(SimulationClock.Instance.simulationTime);
    }

    public void ResetBuilder()
    {
        Debug.Log("[RL_ShipBuilder] Resetting RL ships");

        foreach (var ship in activeShips)
        {
            Destroy(ship);
        }

        activeShips.Clear();

        pendingShips = new List<ShipData>(Resources.LoadAll<ShipData>("RLShips"));
        pendingShips.Sort((a, b) => a.Launching_Date.CompareTo(b.Launching_Date));
        shipsToSpawn = 80;

        Debug.Log($"[RL_ShipBuilder] Reloaded {shipsToSpawn} RL ships");
    }

    void Start()
    {
        LoadAndSortShips();
        SimulationClock.Instance.OnTimeChanged += HandleTimeChanged;
    }
}