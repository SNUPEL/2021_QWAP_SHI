using System.Collections;
using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class SimulationController : MonoBehaviour
{

    [SerializeField] private SimulationCountdownUI countdownUI;
    [SerializeField] private GameObject shipPanel;
    [SerializeField] private GameObject quayPanel;
    public static SimulationController Instance;

    private void Awake()
    {
        if (Instance == null) Instance = this;
        else Destroy(gameObject);
    }
    
    public void ResetSimulation()
    {
        Debug.Log("Simulation reset.");

        // 1. Destroy all active ships
        //var ships = FindObjectsOfType<ShipRuntime>();

        //foreach (var ship in ships)
        //{
        //    Destroy(ship.gameObject);
        //}

        // 2. Reset the simulation clock
        if (SimulationClock.Instance != null)
        {
            SimulationClock.Instance?.ResetTime();
        }

        // 3. Reset the ShipBuilder (if it tracks state or queues)
        ShipBuilder.Instance?.ResetBuilder();

        // 4. Reset other managers like ScheduleManager if needed
        var scheduleData = ScheduleManager.Instance?.CurrentSimulation;

        // 5. Reset the quay visualizer
        var quayVis = FindObjectOfType<QuayVisualizer>();
        if (quayVis != null)
            quayVis.ResetVisualizer();

        // 5. Reset UI/log panels if needed
        //DebugLogUI.Instance?.Clear();
        if (shipPanel != null) shipPanel.SetActive(false);
        if (quayPanel != null) quayPanel.SetActive(false);

        // 6. Reset CostSummary
        TotalCostSummaryUI costUI = FindObjectOfType<TotalCostSummaryUI>();
        if (costUI != null)
            costUI.RefreshActiveShipList();

        // 7. Reset the delivered ships counter
        AIController.ResetDeliveredCount();

        // 8. Optionally reset the camera or other views
        //Camera.main.transform.position = new Vector3(0, 50, -50); // Or whatever your default is

        // 9. Optionally disable SimulationClock (if it uses Update or Coroutines)
        //StopAllCoroutines(); // if you started any in this controller

        Debug.Log("Simulation state reset complete.");
    }

    public void StartSimulation()
    {
        Debug.Log("Simulation started.");
        StartCoroutine(DelayedSimulationStart());
    }

    private IEnumerator DelayedSimulationStart()
    {
        SimulationClock.Instance._startTime = Time.time;
        SimulationClock.Instance.simulationStarted = false;

        countdownUI.StartCountdown();
        Debug.Log("Waiting for countdown...");

        yield return new WaitForSecondsRealtime(3F); // Delay before sim clock starts
        Debug.Log("Countdown finished. Starting simulation clock.");

        // Now start simulation clock
        SimulationClock.Instance.simulationStarted = true;

        // Manually force Day 0 ship check
        ShipBuilder.Instance?.HandleTimeChanged(0);

    }

    public void PauseSimulation()
    {
        Debug.Log("Simulation paused.");
        Time.timeScale = 0f;
        SimulationClock.Instance.simulationStarted = false; // Optional
    }

    public void StopSimulation()
    {
#if UNITY_EDITOR
        EditorApplication.isPlaying = false;
#else
    Application.Quit();
#endif
    }
    public void ResumeSimulation()
    {
        Debug.Log("Simulation resumed.");
        Time.timeScale = 5f;
        // Recalculate offset to keep clock accurate after resume
        SimulationClock.Instance._startTime = Time.time - SimulationClock.Instance.simulationTime;
        SimulationClock.Instance.simulationStarted = true;
    }
}