using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;  // or UnityEngine.UI if you use regular UI text

public class SPTInfoPanel : MonoBehaviour
{
   
    public TMP_Text shipIdText;
    public TMP_Text shipTypeText;
    //public TMP_Text prevOperationText;
    public TMP_Text currentOperationText;
    //public TMP_Text nextOperationText;
    public TMP_Text percentCompleteText;
    public TMP_Text daysToCompletionText;
    public TMP_Text lossCostText;
    public TMP_Text delayCostText;
    public TMP_Text moveCostText;
    //public TMP_Text totalCostText;
    private SPT_ShipRuntime currentShip;
    private int lastSimDay = -1;

    public void UpdateShipInfo(SPT_ShipRuntime selectedShip, int currentSimDay)
    {
        currentShip = selectedShip;
        lastSimDay = currentSimDay;

        Debug.Log($"UpdateShipInfo called for ship: {selectedShip.Data.Ship_Name} type: {selectedShip.Data.Ship_Type}");
        var data = selectedShip.Data;
        shipIdText.text = $"Ship ID: {data.Ship_Name}";
        shipTypeText.text = $"Ship Type: {data.Ship_Type}";

        // Find current, previous, and next operation indexes relative to currentSimDay
        int currentIndex = GetCurrentOperationIndex(selectedShip, currentSimDay);

        // Previous operation or "-"
        //prevOperationText.text = "Previous Operation: " + (currentIndex > 0 ? FormatOperation(selectedShip, currentIndex - 1): "-");

        // Current operation or "-"
        currentOperationText.text = "Operation: " + (FormatOperation(selectedShip, currentIndex));

        // Next operation or "-"
        //nextOperationText.text = "Next Operation" +((currentIndex >= 0 && currentIndex < selectedShip.Data.Operation_Name.Count - 1)
        //    ? FormatOperation(selectedShip, currentIndex + 1)
        //    : "-");
    }

    private int GetCurrentOperationIndex(SPT_ShipRuntime ship, int currentSimDay)
    {
        // We assume ship.Data.Start_Dates and Finish_Dates correspond to operations
        for (int i = 0; i < ship.Data.Start_Dates.Count; i++)
        {
            if (currentSimDay >= ship.Data.Start_Dates[i] && currentSimDay <= ship.Data.Finish_Dates[i])
                return i;
        }

        // If none found, try to find closest operation already finished or upcoming
        for (int i = 0; i < ship.Data.Finish_Dates.Count; i++)
        {
            if (ship.Data.Finish_Dates[i] < currentSimDay) continue;
            return i;  // first upcoming operation
        }

        return -1; // no current operation
    }

    string FormatOperation(SPT_ShipRuntime ship, int index)
    {
        if (index >= 0 && index < ship.Data.Operation_Name.Count && index < ship.Data.Operation_Type.Count)
        {
            return $"{ship.Data.Operation_Name[index]} ({ship.Data.Operation_Type[index]})";
        }
        else
        {
            return "-";
        }
    }

    public void UpdateCosts(SPT_ShipRuntime ship)
    {
        lossCostText.text = $"Loss: ${ship.lossCost:N0}";
        delayCostText.text = $"Delay: ${ship.delayCost:N0}";
        moveCostText.text = $"Move: ${ship.moveCost:N0}";
        //totalCostText.text = $"Total Cost: ${ship.totalCost:N0}";
    }
    void Update()
    {
        if (currentShip == null) return;

        int simDay = SimulationClock.Instance.simulationTime;
        if (simDay == lastSimDay) return;

        lastSimDay = simDay;

        var data = currentShip.Data;

        // Completion %
        int deliveryDate = data.Delivery_Date;
        int launchDate = data.Launching_Date;
        int totalDuration = deliveryDate - launchDate;
        int daysPassed = Mathf.Clamp(simDay - launchDate, 0, totalDuration);
        float percentComplete = totalDuration > 0 ? (daysPassed / (float)totalDuration) * 100f : 0f;
        percentCompleteText.text = $"% of work complete: {percentComplete:F1}%";

        // Days to completion
        int daysLeft = Mathf.Max(deliveryDate - simDay, 0);
        daysToCompletionText.text = $"Days to Completion: {daysLeft}";

        // Costs
        UpdateCosts(currentShip);
    }
}
