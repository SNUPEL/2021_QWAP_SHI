using UnityEngine;

public class ShipController : MonoBehaviour
{
    public Camera mainCamera; // Assign in Inspector (usually MainCamera)
    public ShipInfoPanel infoPanel; // Assign via Inspector
    public GameObject infoPanelObject; // This is the GameObject with ShipInfoPanel attached
    public SimulationClock simulationClock; // assign in inspector or find in Start
    public QuayInfoPanel infoPanel_1;
    public QuayVisualizer currentVisualizer; // reference to highlight based on ship

    public int currentSimDay; // update this from your SimulationClock

    //public QuayInfoPanel quayInfoPanel; // assign via Inspector
    public GameObject quayInfoPanelObject; // the visible panel GO

    void Start()
    {
        if (simulationClock == null)
            simulationClock = FindObjectOfType<SimulationClock>();
    }

    void Update()
    {
        //if (simulationClock != null)
        //    currentSimDay = simulationClock.simulationTime;

        //if (Input.GetMouseButtonDown(0)) // Left-click
        //{
        //    Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
        //    if (Physics.Raycast(ray, out RaycastHit hit))
        //    {
        //        var ship = hit.collider.GetComponent<ShipRuntime>();
        //        if (ship != null)
        //        {
        //            OnShipSelected(ship);
        //        }
        //        GameObject clickedObj = hit.collider.gameObject;
        //        if (clickedObj.CompareTag("RL_Waypoint"))
        //        {
        //            OnQuayWallSelected(clickedObj);
        //            return;
        //        }
        //    }
        //}
        if (simulationClock != null)
            currentSimDay = simulationClock.simulationTime;

        if (Input.GetMouseButtonDown(0)) // Left-click
        {
            Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out RaycastHit hit))
            {
                GameObject clickedObj = hit.collider.gameObject;

                var ship = hit.collider.GetComponent<ShipRuntime>();
                if (ship != null)
                {
                    OnShipSelected(ship);
                    return;
                }

                if (clickedObj.CompareTag("RL_Waypoint"))
                {
                    OnQuayWallSelected(clickedObj);
                    return;
                }

                // Clicked something else - not a ship or quay
                ClearUIAndVisualizer();
            }
            else
            {
                // Clicked empty space
                ClearUIAndVisualizer();
            }
        }
    }


    void OnShipSelected(ShipRuntime selectedShip)
    {
        Debug.Log("Selected ship: " + selectedShip.name);

        if (currentVisualizer != null && infoPanel != null)
        {
            currentVisualizer.ResetGradesOnly();

            string shipType = selectedShip.Data.Ship_Type;
            string operation = selectedShip.Data.Operation_Name.Count > 0 ? selectedShip.Data.Operation_Type[0] : null;

            currentVisualizer.HighlightQuayGrades(shipType, operation);
            Debug.Log($"Calling HighlightQuayGrades with: {shipType}, {operation}");

            infoPanelObject.gameObject.SetActive(true);  // Make sure the panel is visible
            infoPanel.UpdateShipInfo(selectedShip, currentSimDay);
            Debug.Log($"Updating panel for ship {selectedShip.Data.Ship_Index}, current day = {currentSimDay}");
        }
        else
        {
            Debug.LogWarning("QuayVisualizer is not assigned.");
        }
        if (quayInfoPanelObject != null)
            quayInfoPanelObject.SetActive(false);
    }
    

    void OnQuayWallSelected(GameObject quayWall)
    {
        string quayName = quayWall.name;
        Debug.Log("Clicked quay wall: " + quayName);
        quayInfoPanelObject.SetActive(true);

        //int quayIndex = quayScoreDB.quayWallNames.IndexOf(quayName);
        //if (quayIndex < 0)
        //{
        //    Debug.LogWarning($"Quay name '{quayName}' not found in quayWallNames list.");
        //    return;
        //}
        ShipRuntime foundShip = currentVisualizer.FindShipAtQuay(quayName);

        if (infoPanel_1 != null)
            infoPanel_1.UpdateQuayWallInfo(quayName, foundShip, currentSimDay);
        else
            Debug.LogWarning("infoPanel_1 not assigned!");

        if (currentVisualizer != null)
            currentVisualizer.ResetGradesOnly();
        if (infoPanelObject != null)
            infoPanelObject.SetActive(false);
    }
   

    //ShipRuntime FindShipAtQuay(string quayName)
    //{
    //    ShipRuntime[] ships = FindObjectsOfType<ShipRuntime>();

    //    foreach (var ship in ships)
    //    {
    //        // Find the latest log entry up to currentSimDay
    //        SimulationData latestLog = null;
    //        foreach (var log in ship.Logs)
    //        {
    //            if (log.Time <= currentSimDay)
    //            {
    //                if (latestLog == null || log.Time > latestLog.Time)
    //                    latestLog = log;
    //            }
    //        }

    //        // If latest log exists and location matches quayName, return this ship
    //        if (latestLog != null && latestLog.Location == quayName)
    //        {
    //            return ship;
    //        }
    //    }

    //    return null;
    //}
    void ClearUIAndVisualizer()
    {
        Debug.Log("Clicked outside of ship or quay — clearing visuals and UI.");

        // Reset quay visuals
        if (currentVisualizer != null)
            currentVisualizer.ResetGradesOnly();

        // Hide both panels
        if (infoPanelObject != null)
            infoPanelObject.SetActive(false);

        if (quayInfoPanelObject != null)
            quayInfoPanelObject.SetActive(false);
    }

}