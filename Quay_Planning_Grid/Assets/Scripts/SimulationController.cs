using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Diagnostics;
using System.Text;



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
        UnityEngine.Debug.Log("Simulation reset.");

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
        SPT_Builder.Instance?.ResetBuilder();
        MOR_Builder.Instance?.ResetBuilder();
        MWKR_Builder.Instance?.ResetBuilder();

        // 4. Reset other managers like ScheduleManager if needed
        //var scheduleData = ScheduleManager.Instance?.CurrentSimulation;

        // 5. Reset the quay visualizer
        var quayVis = FindObjectOfType<QuayVisualizer>();
        if (quayVis != null)
            quayVis.ResetVisualizer();
            
        var sptVis = FindObjectOfType<SPT_Visualizer>();
        if (sptVis != null)
            sptVis.ResetVisualizer();

        var morVis = FindObjectOfType<MOR_Visualizer>();
        if (morVis != null)
            morVis.ResetVisualizer();

        var mwkrVis = FindObjectOfType<MWKR_Visualizer>();
        if (mwkrVis != null)
            mwkrVis.ResetVisualizer();

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

        UnityEngine.Debug.Log("Simulation state reset complete.");
    }

    public void StartSimulation(string filePath)
    {
        string _pythonScriptPath = "C:\\repos\\2021_QWAP_SHI\\communicate.py";
        string _args = "";
        ProcessStartInfo psi = new ProcessStartInfo
        {
            FileName = "C:\\Users\\User\\anaconda3\\python.exe",
            Arguments = $"\"{_pythonScriptPath}\" {_args}",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            StandardOutputEncoding = Encoding.UTF8,
            StandardErrorEncoding = Encoding.UTF8
        };

        using (Process process = new Process())
        {
            process.StartInfo = psi;
            process.Start();

            string output = process.StandardOutput.ReadToEnd();
            string error = process.StandardError.ReadToEnd();

            process.WaitForExit();

            UnityEngine.Debug.Log("Python Output: " + output);
            if (!string.IsNullOrEmpty(error))
                UnityEngine.Debug.LogError("Python Error: " + error);

        }
    }

    public void Play()
    {
        UnityEngine.Debug.Log("Simulation started.");
        StartCoroutine(DelayedSimulationStart());
    }

    private IEnumerator DelayedSimulationStart()
    {
        SimulationClock.Instance._startTime = Time.time;
        SimulationClock.Instance.simulationStarted = false;

        countdownUI.StartCountdown();
        UnityEngine.Debug.Log("Waiting for countdown...");

        yield return new WaitForSecondsRealtime(3F); // Delay before sim clock starts
        UnityEngine.Debug.Log("Countdown finished. Starting simulation clock.");

        // Now start simulation clock
        SimulationClock.Instance.simulationStarted = true;

        // Manually force Day 0 ship check
        //ShipBuilder.Instance?.HandleTimeChanged(0);
        //SPT_Builder.Instance?.HandleTimeChanged(0);
        ShipBuilder.Instance?.InitializeBuilder();
        SPT_Builder.Instance?.InitializeBuilder();
        MOR_Builder.Instance?.InitializeBuilder();
        MWKR_Builder.Instance?.InitializeBuilder();
    }

    public void PauseSimulation()
    {
        UnityEngine.Debug.Log("Simulation paused.");
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
        UnityEngine.Debug.Log("Simulation resumed.");
        Time.timeScale = 5f;
        // Recalculate offset to keep clock accurate after resume
        SimulationClock.Instance._startTime = Time.time - SimulationClock.Instance.simulationTime;
        SimulationClock.Instance.simulationStarted = true;
    }
}