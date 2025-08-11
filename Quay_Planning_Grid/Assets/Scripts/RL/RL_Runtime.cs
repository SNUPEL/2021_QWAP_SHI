using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;

public class RL_ShipRuntime : MonoBehaviour
{
    public ShipData Data; // The SO with metadata
    public List<SimulationData> Logs = new List<SimulationData>(); // The logs associated with this ship

    private RL_Controller controller;
    private HashSet<int> triggeredLogIndices = new HashSet<int>();
    private NavMeshAgent agent;

    public int LaunchDay => Data?.Launching_Date ?? -1;

    public void StartFrom(int simTime)
    {
        controller = GetComponent<RL_Controller>();
        agent = GetComponent<NavMeshAgent>();

        Debug.Log($"[RL] {Data?.Ship_Name} initialized. Logs: {Logs.Count}");

        int currentSimTime = SimulationClock.Instance.simulationTime;

        // Step 1: Trigger the latest past-or-now log (if any)
        SimulationData latestLog = null;
        int latestIndex = -1;

        for (int i = 0; i < Logs.Count; i++)
        {
            var log = Logs[i];
            if (log.Time <= currentSimTime)
            {
                latestLog = log;
                latestIndex = i;
            }
        }

        if (latestLog != null)
        {
            string loc = latestLog.Location?.Trim();

            if (!string.IsNullOrWhiteSpace(loc) &&
                !loc.Equals("Source", StringComparison.OrdinalIgnoreCase))
            {
                controller.MoveTo(loc);
            }

            triggeredLogIndices.Add(latestIndex);
        }

        // Step 2: Mark all previous logs as triggered (even if not acted upon)
        for (int i = 0; i < Logs.Count; i++)
        {
            if (Logs[i].Time <= currentSimTime)
            {
                triggeredLogIndices.Add(i);
            }
        }
    }

    void Update()
    {
        if (Logs == null || Logs.Count == 0) return;

        int currentSimTime = SimulationClock.Instance.simulationTime;

        for (int i = 0; i < Logs.Count; i++)
        {
            var log = Logs[i];
            string loc = log.Location?.Trim();

            if (log.Time <= currentSimTime && !triggeredLogIndices.Contains(i))
            {
                Debug.Log($"[RL][T={log.Time}] {Data.Ship_Name}: {log.Operation} at {log.Location} - {log.Log}");

                if (controller != null &&
                    !string.IsNullOrWhiteSpace(log.Location) &&
                    !string.Equals(loc, "Source", StringComparison.OrdinalIgnoreCase) &&
                    !string.Equals(controller.currentTarget, loc, StringComparison.OrdinalIgnoreCase))
                {
                    controller.MoveTo(loc);
                }

                triggeredLogIndices.Add(i);
            }
        }
    }

    public List<SimulationData> GetLogsForShips(string Ship_Index)
    {
        return Logs.FindAll(log => log.Ship_Index == Ship_Index);
    }

    public void PrintAllLogs()
    {
        if (Logs == null || Logs.Count == 0)
        {
            Debug.LogWarning($"[RL] No logs found for ship: {Data?.Ship_Name} ({Data?.Ship_Index})");
            return;
        }
        // Debug.Log($"[RL Logs] for {Data.Ship_Name} ({Data.Ship_Index}):");
    }
}