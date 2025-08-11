using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
//using UnityEngine.AI;

public class ShipRuntime : MonoBehaviour
{
    public ShipData Data;
    public List<SimulationData> Logs = new List<SimulationData>();
    private AIController controller;
    private HashSet<int> triggeredLogIndices = new HashSet<int>();
    public int lossCost;
    public int delayCost;
    public int moveCost;
    public int totalCost;

    public int LaunchDay => Data?.Launching_Date ?? -1;

    public void StartFrom(int simTime)
    {
        controller = GetComponent<AIController>();
        Debug.Log($"{Data?.Ship_Name} ({Data?.Ship_Index}) initialized. Logs: {Logs.Count}");

        for (int i = 0; i < Logs.Count; i++)
        {
            if (Logs[i].Time <= simTime)
            {
                if (!string.IsNullOrWhiteSpace(Logs[i].Location))
                {
                    controller.MoveTo(Logs[i].Location);
                    triggeredLogIndices.Add(i);
                }
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
            if (log.Time <= currentSimTime && !triggeredLogIndices.Contains(i))
            {
                Debug.Log($"[T={log.Time}] {Data.Ship_Name}: Jump to {log.Location}");

                if (!string.IsNullOrWhiteSpace(log.Location))
                    controller.MoveTo(log.Location);

                triggeredLogIndices.Add(i);
            }
        }
    }
}