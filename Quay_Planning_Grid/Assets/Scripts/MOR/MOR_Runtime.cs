using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MOR_Runtime : MonoBehaviour
{
    public ShipData Data;
    public List<SimulationData> Logs = new List<SimulationData>();
    private MOR_Controller controller;
    private HashSet<int> triggeredLogIndices = new HashSet<int>();

    public int lossCost = 0;
    public int delayCost = 0;
    public int moveCost = 0;
    //var shipTypeScores = quayScoreDB.shipTypeScores.Find(st => st.shipType == ship.Data.Ship_Type);

    private int moveCount = 0;
    public QuayData quayScoreDB; // assign in inspector or get reference somehow
    MOR_Visualizer currentVisualizer;

    void Awake()
    {

        if (quayScoreDB == null)
        {
            quayScoreDB = Resources.Load<QuayData>("QuayData");
            if (quayScoreDB == null)
                Debug.LogWarning("QuayData DB not assigned or not found in Resources!");
        }
    }
    // Call this every time the ship moves (e.g., from AIController.MoveTo)
    //public void RegisterMove()
    //{
    //    moveCount++;
    //}

    //Call this regularly (e.g., Update or when sim time changes)
    public void UpdateCosts(int currentSimDay)
    {
        CalculateLossCost(currentSimDay);
        CalculateDelayCost(currentSimDay);
        CalculateMoveCost();
    }
    public int LaunchDay => Data?.Launching_Date ?? -1;

    public void StartFrom(int simTime)
    {
        controller = GetComponent<MOR_Controller>();
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
                //                Debug.Log($"Checking log location (len={log.Location?.Length}): '{log.Location}'");

                if (string.IsNullOrWhiteSpace(log.Location) || log.Location.Trim().Length == 0)
                {
                    //                    Debug.LogWarning($"Skipped empty or whitespace-only location at log index {i}");
                    triggeredLogIndices.Add(i);  // <--- Mark as triggered here
                    continue;
                }

                Debug.Log($"[T={log.Time}] {Data.Ship_Name}: Jump to {log.Location}");

                controller.MoveTo(log.Location);

                triggeredLogIndices.Add(i);
            }
        }
        CalculateLossCost(currentSimTime);
        CalculateDelayCost(currentSimTime);
        CalculateMoveCost();

    }

    private void CalculateLossCost(int currentSimDay)
    {
        lossCost = 0;

        if (quayScoreDB == null)
        {
            Debug.LogWarning("QuayData DB not assigned!");
            return;
        }

        var shipTypeEntry = quayScoreDB.shipTypeScores.Find(s => s.shipType == Data.Ship_Type);
        if (shipTypeEntry == null)
        {
            Debug.LogWarning($"Ship type '{Data.Ship_Type}' not found in QuayData DB.");
            return;
        }

        MOR_Controller ai = GetComponent<MOR_Controller>();
        if (ai == null)
        {
            Debug.LogWarning("Controller not found on ship.");
            return;
        }

        string currentQuay = ai.currentTarget;  // Where the ship currently is

        if (string.IsNullOrWhiteSpace(currentQuay) || currentQuay.Equals("S", StringComparison.OrdinalIgnoreCase))
        {
            // Ship not currently at any quay
            return;
        }

        // Find operation active now based on simulation time
        int simTime = SimulationClock.Instance.simulationTime;
        int currentOpIndex = -1;
        for (int i = 0; i < Data.Start_Dates.Count; i++)
        {
            if (simTime >= Data.Start_Dates[i] && simTime < Data.Finish_Dates[i])
            {
                currentOpIndex = i;
                break;
            }
        }
        if (currentOpIndex < 0)
        {
            // No current operation found at this time
            return;
        }

        string currentOperationName = Data.Operation_Type[currentOpIndex];

        var operationEntry = shipTypeEntry.operations.Find(o => o.operationName == currentOperationName);
        if (operationEntry == null)
        {
            Debug.LogWarning($"Operation '{currentOperationName}' not found for ship type '{Data.Ship_Type}'.");
            return;
        }

        // Find quay index in quayWallNames
        int quayIndex = quayScoreDB.quayWallNames.FindIndex(q => q.Trim().Equals(currentQuay.Trim(), StringComparison.OrdinalIgnoreCase));
        if (quayIndex < 0 || quayIndex >= operationEntry.quayScores.Count)
        {
            Debug.LogWarning($"Quay '{currentQuay}' not found or invalid index.");
            return;
        }

        QuayScoreGrade grade = operationEntry.quayScores[quayIndex];
        if (grade == QuayScoreGrade.C || grade == QuayScoreGrade.D || grade == QuayScoreGrade.E)
        {
            lossCost = 15000; // fixed cost per operation at low-priority quay
        }
        else
        {
            lossCost = 0;
        }
    }

    private void CalculateDelayCost(int currentSimDay)
    {
        delayCost = 0;
        if (currentSimDay > Data.Delivery_Date)
        {
            int delayedDays = currentSimDay - Data.Delivery_Date;
            delayCost = delayedDays * 30000;
        }
    }

    public void IncrementMoveCount()
    {
        moveCount++;
        CalculateMoveCost();
    }
    private void CalculateMoveCost()
    {
        moveCost = moveCount * 30000;
    }
}
