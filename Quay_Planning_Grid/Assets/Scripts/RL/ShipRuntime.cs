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
 
    public int lossCost = 0;
    public int delayCost = 0;
    public int moveCost = 0;
    //var shipTypeScores = quayScoreDB.shipTypeScores.Find(st => st.shipType == ship.Data.Ship_Type);

    private int moveCount = 0;
    public QuayData quayScoreDB; // assign in inspector or get reference somehow
    QuayVisualizer currentVisualizer;

    public bool IsDelivered { get; private set; } = false;
    public int finalLossCost = 0;
    public int finalDelayCost = 0;
    public int finalMoveCost = 0;

    public int TotalCost =>
    IsDelivered ? (finalLossCost + finalDelayCost + finalMoveCost)
                : (lossCost + delayCost + moveCost);

    public void MarkDelivered(int sinkDay)
    {
        if (IsDelivered) return; // already frozen

        IsDelivered = true;

        // Freeze current costs
        finalLossCost = lossCost;
        finalMoveCost = moveCost;

        // Delay cost rule: only if delivery is later than planned
        if (sinkDay > Data.Delivery_Date)
            finalDelayCost = (sinkDay - Data.Delivery_Date) * 15000;
        else
            finalDelayCost = 0;

        Debug.Log($"{Data.Ship_Name} delivered on {sinkDay}. Frozen costs: L={finalLossCost}, D={finalDelayCost}, M={finalMoveCost}");
    }


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
    public void RegisterMove()
    {
        moveCount++;
    }

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
        if(Logs == null || Logs.Count == 0) return;

        int currentSimTime = SimulationClock.Instance.simulationTime;

        for (int i = 0; i < Logs.Count; i++)
        {
            var log = Logs[i];
            if (log.Time <= currentSimTime && !triggeredLogIndices.Contains(i))
            {
                if (string.IsNullOrWhiteSpace(log.Location) || log.Location.Trim().Length == 0)
                {
                    triggeredLogIndices.Add(i);  // <--- Mark as triggered here
                    continue;
                }
                Debug.Log($"[T={log.Time}] {Data.Ship_Name}: Jump to {log.Location}");
                controller.MoveTo(log.Location);
                triggeredLogIndices.Add(i);
            }
        }
     }

    public void CalculateLossCost(int currentSimDay)
    {
        if (IsDelivered) return;
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

        AIController ai = GetComponent<AIController>();
        if (ai == null)
        {
            Debug.LogWarning("AIController not found on ship.");
            return;
        }

        string currentQuay = ai.currentTarget;  // Where the ship currently is

        if (string.IsNullOrWhiteSpace(currentQuay))
        {
            // Ship not currently at any quay
            return;
        }
        if (currentQuay.Equals("Sink", StringComparison.OrdinalIgnoreCase))
        {
            MarkDelivered(currentSimDay);
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
            return;
        }

        QuayScoreGrade grade = operationEntry.quayScores[quayIndex];
        if (grade == QuayScoreGrade.C || grade == QuayScoreGrade.D || grade == QuayScoreGrade.E)
        {
            if (IsDelivered) return;

            lossCost += 15000;            // fixed cost per operation at low-priority quay
        }
        
    }

    public void CalculateDelayCost(int currentSimDay)
    {
        // Delay is ONLY applied when the ship is delivered to sink (per your requirement).
        if (IsDelivered) return;

        // while not delivered we do not apply delay cost
        delayCost = 0;
    }

    public void IncrementMoveCount()
    {
        moveCount++;
        CalculateMoveCost();
    }

    public void CalculateMoveCost()
    {
        if (IsDelivered) return;  // Don't change anything after delivery

        moveCost = moveCount * 15000;
    }
}
