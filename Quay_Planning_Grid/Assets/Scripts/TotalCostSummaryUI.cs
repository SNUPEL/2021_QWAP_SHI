using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class TotalCostSummaryUI : MonoBehaviour
{
    public TMP_Text lossCostText;
    public TMP_Text delayCostText;
    public TMP_Text moveCostText;
    public TMP_Text totalCostText;
    public TextMeshProUGUI counterText;

    private List<ShipRuntime> activeShips = new List<ShipRuntime>();

    void Start()
    {
        RefreshActiveShipList(); // populate once at start (or trigger manually if needed
    }

    void Update()
    {
        float totalLoss = 0f;
        float totalDelay = 0f;
        float totalMove = 0f;

        foreach (var ship in activeShips)
        {
            if (ship == null) continue;
            totalLoss += ship.lossCost;
            totalDelay += ship.delayCost;
            totalMove += ship.moveCost;
        }

        float total = totalLoss + totalDelay + totalMove;

        // Format and display
        lossCostText.text = $"Loss: {(totalLoss > 0 ? $"${totalLoss:N0}" : "$0")}";
        delayCostText.text = $"Delay: {(totalDelay > 0 ? $"${totalDelay:N0}" : "$0")}";
        moveCostText.text = $"Move: {(totalMove > 0 ? $"${totalMove:N0}" : "$0")}";
        totalCostText.text = $"Total: ${(total > 0 ? total.ToString("N0") : "0")}";

        int count = AIController.DeliveredCount;

        counterText.text = "Delivered: " + count;

    }

    public void RefreshActiveShipList()
    {
        activeShips.Clear();
        activeShips.AddRange(FindObjectsOfType<ShipRuntime>());
    }


}
