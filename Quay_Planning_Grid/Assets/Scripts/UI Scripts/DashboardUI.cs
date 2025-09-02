using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class DashboardUI : MonoBehaviour
{
    public ScheduleManager scheduleManager;
    public SPT_ScheduleManager SPTScheduleManager;
    public TMP_Text algorithmNameText;

    [Header("Cameras")]
    public Camera cam1;
    public Camera cam2;
    public Camera cam3;
    public Camera cam4;
    public Camera icam1;
    public Camera icam2;
    public Camera icam3;
    public Camera icam4;
    private Camera activeCam;

    private void Start()
    {
        SetActiveCamera(cam1); // Default camera
        UpdateAlgorithmText("RL");
    }

    private void SetActiveCamera(Camera cam)
    {
        if (activeCam != null) activeCam.gameObject.SetActive(false);
        activeCam = cam;
        if (activeCam != null) activeCam.gameObject.SetActive(true);
    }

    public void OnCam1Button() 
    {
        SetActiveCamera(cam1);
        UpdateAlgorithmText("RL");
    }
    public void OnCam2Button() 
    {
        SetActiveCamera(cam2);
        UpdateAlgorithmText("SPT-MF");
    }
    public void OnCam3Button() 
    {
        SetActiveCamera(cam3);
        UpdateAlgorithmText("MOR-MF");
    }
    public void OnCam4Button() 
    {
        SetActiveCamera(cam4);
        UpdateAlgorithmText("MWKR-MF");
    }

    // Run simulation button
    public void OnRunSimulationClicked()
    {
        SimulationController.Instance?.ResetSimulation();
        //scheduleManager.LoadAllSchedules();
        scheduleManager.LoadSchedule("log-RL.csv");
        scheduleManager.LoadSchedule("log-SPT-MF.csv");
        //scheduleManager.LoadSchedule("log-MOR-MF.csv");
        //scheduleManager.LoadSchedule("log-MWKR-MF.csv");

        ShipBuilder.Instance?.InitializeBuilder();
        SPT_Builder.Instance?.InitializeBuilder();
        SimulationController.Instance?.StartSimulation();
        
    }

    public void OnPauseButtonClicked() => SimulationController.Instance?.PauseSimulation();
    public void OnStopButtonClicked() => SimulationController.Instance?.StopSimulation();
    public void OnResumeButtonClicked() => SimulationController.Instance?.ResumeSimulation();

    private void UpdateAlgorithmText(string name)
    {
        if (algorithmNameText != null)
            algorithmNameText.text = $"{name} Algorithm";
    }
}
