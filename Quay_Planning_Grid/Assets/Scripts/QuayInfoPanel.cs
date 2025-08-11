using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class QuayInfoPanel : MonoBehaviour
{
    public static QuayInfoPanel Instance { get; private set; }

    public TMP_Text quayIdText;
    public TMP_Text statusText;
    public TMP_Text currentShipText;
    public TMP_Text currentOperationText;
    public TMP_Text timeRemainingText;

    private ShipRuntime currentShip;
    private int lastSimDay = -1;

    private string currentQuayId = "";
    private int currentQuayIndex = -1;

    void Awake()
    {
        Instance = this;
    }

    public int CurrentQuayIndex => currentQuayIndex;

    public void ShowQuay(string quayId, int quayIndex, ShipRuntime ship, int currentSimDay)
    {
        currentQuayId = quayId;
        currentQuayIndex = quayIndex;
        UpdateQuayWallInfo(quayId, ship, currentSimDay);
    }
    public void UpdateQuayWallInfo(string quayId, ShipRuntime ship, int currentSimDay)
    {
        quayIdText.text = $"Quay: {quayId}";
        currentShip = ship;   
        lastSimDay = currentSimDay;
        if (ship != null)
        {
            statusText.text = "Engaged";
            statusText.color = Color.green;

            currentShipText.text = $"Ship: {ship.Data.Ship_Name}";

            int opIndex = GetCurrentOperationIndex(ship, currentSimDay);
            if (opIndex >= 0)
            {
                string opName = ship.Data.Operation_Name[opIndex];
                string opType = ship.Data.Operation_Type[opIndex];
                currentOperationText.text = $"Operation: {opName} ({opType})";

                int end = ship.Data.Finish_Dates[opIndex];
                int timeLeft = Mathf.Max(0, end - currentSimDay);
                timeRemainingText.text = $"Time Left: {timeLeft} day(s)";
            }
        }
        else
        {
            statusText.text = "Disengaged";
            statusText.color = Color.black;

            currentShipText.text = "Ship: -";
            currentOperationText.text = "Operation: -";
            timeRemainingText.text = "Time Left: -";
        }
    }

    int GetCurrentOperationIndex(ShipRuntime ship, int simDay)
    {
        for (int i = 0; i < ship.Data.Start_Dates.Count; i++)
        {
            int start = ship.Data.Start_Dates[i];
            int end = ship.Data.Finish_Dates[i];
            if (simDay >= start && simDay < end)
                return i;
        }
        return -1;
    }
    void Update()
    {
        int simDay = SimulationClock.Instance.simulationTime;

        if (currentShip != null && simDay != lastSimDay)
        {
            lastSimDay = simDay;

            int opIndex = GetCurrentOperationIndex(currentShip, simDay);
            if (opIndex >= 0)
            {
                int end = currentShip.Data.Finish_Dates[opIndex];
                int timeLeft = Mathf.Max(0, end - simDay);
                timeRemainingText.text = $"Time Left: {timeLeft} day(s)";
            }
        }
    }
    

}
