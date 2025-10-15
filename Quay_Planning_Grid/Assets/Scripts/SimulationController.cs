using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Diagnostics;
using System.Text;
using System.IO;
#if UNITY_EDITOR
using UnityEditor;
#endif

public class SimulationController : MonoBehaviour
{
    [SerializeField] private SimulationCountdownUI countdownUI;
    [SerializeField] private DashboardUI dashboardUI;
    [SerializeField] private ChartController chartController;
    [SerializeField] private GameObject shipPanel;
    [SerializeField] private GameObject quayPanel;
    [SerializeField] private UserSettingPanelUI userSettingPanelUI;
    [SerializeField] private ErrorPanelUI errorPanelUI;

    public static SimulationController Instance;
    public SourceType mSelectedDataSource = SourceType.None;
    private string outputDirectory;

    public string mBaseDirectory = string.Empty;
    public string mModelPath = string.Empty;
    public string mDataPath = string.Empty;
    public string mResultPath = string.Empty;

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

    private void Start()
    {
        mBaseDirectory = PlayerPrefs.GetString(userSettingPanelUI.mBaseDirectoryKey);
        mModelPath = PlayerPrefs.GetString(userSettingPanelUI.mModelPathKey);
        mDataPath = PlayerPrefs.GetString(userSettingPanelUI.mDataDirectoryKey);
        mResultPath = PlayerPrefs.GetString(userSettingPanelUI.mResultDirectoryKey);
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

        try
        {
            RunAgent();

            ShipBuilder.Instance?.InitializeBuilder();
            SPT_Builder.Instance?.InitializeBuilder();
            MOR_Builder.Instance?.InitializeBuilder();
            MWKR_Builder.Instance?.InitializeBuilder();

            chartController.RunChart(mBaseDirectory);
        } catch (Exception e)
        {
            SendError(e.Message);
        }
        
    }

    private void RunAgent()
    {
        string baseDir = mBaseDirectory;
        string parentDir = Directory.GetParent(baseDir).FullName;
        string modelPath = mModelPath;
        string dataPath = string.Empty;
        if (mSelectedDataSource == SourceType.GenerateData)
        {
            GenerateData(baseDir, dashboardUI.textNumberOfShips.text);
            dataPath = $"{mDataPath}\\{dashboardUI.textNumberOfQuays.text}-{dashboardUI.textNumberOfShips.text}\\instance-1.xlsx";
        }
        else
            dataPath = dashboardUI.textFileFullPath;
        string resPath = mResultPath;
        string pythonScript = $"{parentDir}\\communicate.py";

        SendError("Data Path: " + dataPath);
        SendError("Result Path: " + resPath);
        SendError("Model Path: " + modelPath);
        SendError("Base Directory: " + baseDir);

        ProcessStartInfo psi = new ProcessStartInfo();
        psi.FileName = Path.Combine(baseDir, "qwap_env", "python.exe");
        psi.Arguments = $"{pythonScript} --data_path \"{dataPath}\" --res_path \"{resPath}\" --model_path \"{modelPath}\"";
        psi.UseShellExecute = false;
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;
        psi.CreateNoWindow = true;

        using (Process process = Process.Start(psi))
        {
            string _output = process.StandardOutput.ReadToEnd();
            string _error = process.StandardError.ReadToEnd();
            UnityEngine.Debug.Log(_error);
            process.WaitForExit();
        }
    }

    private void GenerateData(string baseDir, string n_ships)
    {
        string configPath = $"{baseDir}\\input\\configurations\\v1\\config (m=28).xlsx";
        string testDir = $"{baseDir}\\input\\28-{n_ships}";
        string parentDir = Directory.GetParent(baseDir).FullName;
        string pythonScript = $"{parentDir}\\data_communicate.py";

        ProcessStartInfo psi = new ProcessStartInfo();
        psi.FileName = Path.Combine(baseDir, "qwap_env", "python.exe");
        psi.Arguments = $"{pythonScript} --n_ships \"{n_ships}\" --test_dir \"{testDir}\" --config_path \"{configPath}\"";
        psi.UseShellExecute = false;
        psi.RedirectStandardOutput = true;
        psi.RedirectStandardError = true;
        psi.CreateNoWindow = true;

        using (Process process = Process.Start(psi))
        {
            string output = process.StandardOutput.ReadToEnd();
            string error = process.StandardError.ReadToEnd();
            UnityEngine.Debug.Log(error);
            process.WaitForExit();
        }
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

    public void SendError(string exceptionMessage)
    {
        errorPanelUI.gameObject.SetActive(true);
        errorPanelUI.setMessage(exceptionMessage);
    }
}