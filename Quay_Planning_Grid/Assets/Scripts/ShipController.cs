using UnityEngine;

public class ShipController : MonoBehaviour
{
    //public Camera mainCamera; // Assign in Inspector (usually MainCamera)
    public ShipInfoPanel infoPanel; // Assign via Inspector
    public GameObject infoPanelObject; // This is the GameObject with ShipInfoPanel attached
    public SPTInfoPanel sptinfoPanel; // Assign via Inspector
    public GameObject sptinfoPanelObject; // This is the GameObject with ShipInfoPanel attached
    public MORInfoPanel morinfoPanel; // Assign via Inspector
    public GameObject morinfoPanelObject; // This is the GameObject with ShipInfoPanel attached
    public MWKRInfoPanel mwkrinfoPanel; // Assign via Inspector
    public GameObject mwkrinfoPanelObject; // This is the GameObject with ShipInfoPanel attached
    public Camera cam1;
    public Camera cam2;
    public Camera cam3;
    public Camera cam4;

    private Camera activeCam; // default to cam1
    public SimulationClock simulationClock; // assign in inspector or find in Start
    public QuayInfoPanel infoPanel_1;
    //public QuayVisualizer currentVisualizer; // reference to highlight based on ship

    public int currentSimDay; // update this from your SimulationClock

    //public QuayInfoPanel quayInfoPanel; // assign via Inspector
    public GameObject quayInfoPanelObject; // the visible panel GO
    public QuayVisualizer rlVisualizer;
    public SPT_Visualizer sptVisualizer;
    public MOR_Visualizer morVisualizer;
    public MWKR_Visualizer mwkrVisualizer;
    public bool highlightActive = false;
    private Renderer lastSelectedRenderer;
    private Color originalShipColor;

    void Start()
    {
        if (simulationClock == null)
            simulationClock = FindObjectOfType<SimulationClock>();

        activeCam = cam1; // now it’s safe

    }

    void Update()
    {
        if (simulationClock != null)
            currentSimDay = simulationClock.simulationTime;

        Camera activeCam = FindObjectOfType<DashboardUI>().ActiveCamera;
        if (activeCam == null) return;

        if (Input.GetMouseButtonDown(0))
        {
            Ray ray = activeCam.ScreenPointToRay(Input.mousePosition);
            if (Physics.Raycast(ray, out RaycastHit hit))
            {
                GameObject clickedObj = hit.collider.gameObject;

                // Ship selection based on active camera + tag/component
                if (activeCam == cam1 && clickedObj.CompareTag("RL_Ship"))
                {
                    if (clickedObj.TryGetComponent(out ShipRuntime rlShip))
                    {
                        OnRShipSelected(rlShip);
                    }
                }
                else if (activeCam == cam2 && clickedObj.CompareTag("SPT_Ship"))
                {
                    if (clickedObj.TryGetComponent(out SPT_ShipRuntime sptShip))
                        OnSShipSelected(sptShip);
                }
                else if (activeCam == cam3 && clickedObj.CompareTag("MOR_Ship"))
                {
                    if (clickedObj.TryGetComponent(out MOR_Runtime morShip))
                        OnMShipSelected(morShip);
                }
                else if (activeCam == cam4 && clickedObj.CompareTag("MWKR_Ship"))
                {
                    if (clickedObj.TryGetComponent(out MWKR_Runtime mwkrShip))
                        OnWShipSelected(mwkrShip);
                }

                // Quay selection based on tag (different for each sim)
                else if (clickedObj.CompareTag("RL_Waypoint") || clickedObj.CompareTag("SPT_Waypoint") ||
                            clickedObj.CompareTag("MOR_Waypoint") || clickedObj.CompareTag("MWKR_Waypoint"))
                {
                    OnQuayWallSelected(clickedObj);
                }

                else
                {
                    ClearUIAndVisualizer(); // clicked something else
                }
            }
            else
            {
                ClearUIAndVisualizer(); // clicked empty space
            }
        }
    }

    void OnSShipSelected(SPT_ShipRuntime selectedShip)
    {
        ClearUIAndVisualizer();
        Debug.Log("Selected SPT ship: " + selectedShip.name);

        // Reset previous highlight
        if (lastSelectedRenderer != null)
            lastSelectedRenderer.material.color = originalShipColor;

        // Highlight new ship
        Renderer rend = selectedShip.GetComponent<Renderer>();
        if (rend != null)
        {
            lastSelectedRenderer = rend;
            originalShipColor = rend.material.color;
            rend.material.color = new Color32(205, 92, 92, 255); // Highlight color
        }
        // Disable other visualizers, enable this one
        if (sptVisualizer != null && sptinfoPanel != null)
        {
            sptVisualizer.ResetGradesOnly();

            string shipType = selectedShip.Data.Ship_Type;
            string operation = selectedShip.Data.Operation_Type.Count > 0 ? selectedShip.Data.Operation_Type[0] : null;

            sptVisualizer.HighlightQuayGrades(shipType, operation);
            //sptVisualizer.ApplyVisualization();
            sptinfoPanelObject.gameObject.SetActive(true);
            sptinfoPanel.UpdateShipInfo(selectedShip, currentSimDay);
        }
        // Highlight the quay in the mini visualizer
        string currentQuay = selectedShip.GetComponent<SPT_Controller>()?.currentTarget;
        if (!string.IsNullOrEmpty(currentQuay))
        {
            sptVisualizer.HighlightQuayInMiniVisualizer(currentQuay);
        }
        if (quayInfoPanelObject != null)
            quayInfoPanelObject.SetActive(false);
    }
    void OnRShipSelected(ShipRuntime selectedShip)
    {
        ClearUIAndVisualizer();
        Debug.Log("Selected RL ship: " + selectedShip.name);

        // Reset previous highlight
        if (lastSelectedRenderer != null)
            lastSelectedRenderer.material.color = originalShipColor;

        // Highlight new ship
        Renderer rend = selectedShip.GetComponent<Renderer>();
        if (rend != null)
        {
            lastSelectedRenderer = rend;
            originalShipColor = rend.material.color;
            rend.material.color = new Color32(205, 92, 92, 255); // Indian Red
        }
        // Disable other visualizers, enable this one
        if (rlVisualizer != null && infoPanel != null)
        {
            rlVisualizer.ResetGradesOnly();

            string shipType = selectedShip.Data.Ship_Type;
            string operation = selectedShip.Data.Operation_Type.Count > 0 ? selectedShip.Data.Operation_Type[0] : null;

            rlVisualizer.HighlightQuayGrades(shipType, operation);
            infoPanelObject.gameObject.SetActive(true);
            infoPanel.UpdateShipInfo(selectedShip, currentSimDay);
        }
        // Highlight the quay in the mini visualizer
        string currentQuay = selectedShip.GetComponent<AIController>()?.currentTarget;
        if (!string.IsNullOrEmpty(currentQuay))
        {
            rlVisualizer.HighlightQuayInMiniVisualizer(currentQuay);
        }
        if (quayInfoPanelObject != null)
            quayInfoPanelObject.SetActive(false);
    }
    void OnMShipSelected(MOR_Runtime selectedShip)
    {
        ClearUIAndVisualizer();
        Debug.Log("Selected MOR ship: " + selectedShip.name);
        // Reset previous highlight
        if (lastSelectedRenderer != null)
            lastSelectedRenderer.material.color = originalShipColor;

        // Highlight new ship
        Renderer rend = selectedShip.GetComponent<Renderer>();
        if (rend != null)
        {
            lastSelectedRenderer = rend;
            originalShipColor = rend.material.color;
            rend.material.color = new Color32(205, 92, 92, 255); // Highlight color
        }
        if (morVisualizer != null && morinfoPanel != null)
        {
            morVisualizer.ResetGradesOnly();

            string shipType = selectedShip.Data.Ship_Type;
            string operation = selectedShip.Data.Operation_Type.Count > 0 ? selectedShip.Data.Operation_Type[0] : null;

            morVisualizer.HighlightQuayGrades(shipType, operation);
            morinfoPanelObject.gameObject.SetActive(true);
            morinfoPanel.UpdateShipInfo(selectedShip, currentSimDay);
        }
        // Highlight the quay in the mini visualizer
        string currentQuay = selectedShip.GetComponent<MOR_Controller>()?.currentTarget;
        if (!string.IsNullOrEmpty(currentQuay))
        {
            morVisualizer.HighlightQuayInMiniVisualizer(currentQuay);
        }
        if (quayInfoPanelObject != null)
            quayInfoPanelObject.SetActive(false);
    }

    void OnWShipSelected(MWKR_Runtime selectedShip)
    {
        ClearUIAndVisualizer();
        Debug.Log("Selected MWKR ship: " + selectedShip.name);
        // Reset previous highlight
        if (lastSelectedRenderer != null)
            lastSelectedRenderer.material.color = originalShipColor;

        // Highlight new ship
        Renderer rend = selectedShip.GetComponent<Renderer>();
        if (rend != null)
        {
            lastSelectedRenderer = rend;
            originalShipColor = rend.material.color;
            rend.material.color = new Color32(205, 92, 92, 255); // Highlight color
        }
        // Disable other visualizers, enable this one
        if (mwkrVisualizer != null && mwkrinfoPanel != null)
        {
            mwkrVisualizer.ResetGradesOnly();

            string shipType = selectedShip.Data.Ship_Type;
            string operation = selectedShip.Data.Operation_Type.Count > 0 ? selectedShip.Data.Operation_Type[0] : null;

            mwkrVisualizer.HighlightQuayGrades(shipType, operation);
            mwkrinfoPanelObject.gameObject.SetActive(true);
            mwkrinfoPanel.UpdateShipInfo(selectedShip, currentSimDay);
        }
        string currentQuay = selectedShip.GetComponent<MWKR_Controller>()?.currentTarget;
        if (!string.IsNullOrEmpty(currentQuay))
        {
            mwkrVisualizer.HighlightQuayInMiniVisualizer(currentQuay);
        }
        if (quayInfoPanelObject != null)
            quayInfoPanelObject.SetActive(false);
    }


    void OnQuayWallSelected(GameObject quayWall)
    {
        string quayName = quayWall.name;
        Debug.Log("Clicked quay wall: " + quayName);

        // Show the quay info panel
        quayInfoPanelObject.SetActive(true);

        ShipRuntime rlShip = rlVisualizer?.FindShipAtQuay(quayName);
        SPT_ShipRuntime sptShip = sptVisualizer?.FindShipAtQuay(quayName);
        MOR_Runtime morShip = morVisualizer?.FindShipAtQuay(quayName);
        MWKR_Runtime mwkrShip = mwkrVisualizer?.FindShipAtQuay(quayName);

        // Update info panel with whichever ship exists
        if (rlShip != null)
            infoPanel_1.UpdateQuayWallInfo(quayName, rlShip, currentSimDay);
       
        // Reset grades on all visualizers since UI is now showing the quay
        rlVisualizer?.ResetGradesOnly();
        sptVisualizer?.ResetGradesOnly();
        morVisualizer?.ResetGradesOnly();
        mwkrVisualizer?.ResetGradesOnly();

        // Hide ship info panel
        infoPanelObject.SetActive(false);
    }
    // void OnQuayWallSelected(GameObject quayWall)
    // {
    //     string quayName = quayWall.name;
    //     Debug.Log("Clicked quay wall: " + quayName);
    //     quayInfoPanelObject.SetActive(true);

    //     ShipRuntime foundShip = currentVisualizer.FindShipAtQuay(quayName);

    //     if (infoPanel_1 != null)
    //         infoPanel_1.UpdateQuayWallInfo(quayName, foundShip, currentSimDay);
    //     else
    //         Debug.LogWarning("infoPanel_1 not assigned!");

    //     if (currentVisualizer != null)
    //         currentVisualizer.ResetGradesOnly();
    //     if (infoPanelObject != null)
    //         infoPanelObject.SetActive(false);
    // }
    void ClearUIAndVisualizer()
    {
        Debug.Log("Clicked outside of ship or quay — clearing visuals and UI.");
        rlVisualizer?.ClearQuayMiniHighlights();
        sptVisualizer?.ClearQuayMiniHighlights();
        morVisualizer?.ClearQuayMiniHighlights();
        mwkrVisualizer?.ClearQuayMiniHighlights();

        // Reset ship highlight
        if (lastSelectedRenderer != null)
        {
            lastSelectedRenderer.material.color = originalShipColor;
            lastSelectedRenderer = null; // clear reference
        }

        // Reset ship highlight
        infoPanel?.ClearInfo();
        sptinfoPanel?.ClearInfo();
        morinfoPanel?.ClearInfo();
        mwkrinfoPanel?.ClearInfo();

        // Hide both panels
        if (infoPanelObject != null)
            infoPanelObject.SetActive(false);
        if (sptinfoPanelObject != null)
            sptinfoPanelObject.SetActive(false);
        if (morinfoPanelObject != null)
            morinfoPanelObject.SetActive(false);
        if (mwkrinfoPanelObject != null)
            mwkrinfoPanelObject.SetActive(false);

        // Hide quay panel
        if (quayInfoPanelObject != null)
            quayInfoPanelObject.SetActive(false);
    }

}
