using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class DashboardUI : MonoBehaviour
{
    //public ScheduleManager scheduleManager;
    //public SPT_ScheduleManager SPTScheduleManager;
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
    public void OnIsoButton()
    {
        if (activeCam == cam1) SetActiveCamera(icam1);
        else if (activeCam == cam2) SetActiveCamera(icam2);
        else if (activeCam == cam3) SetActiveCamera(icam3);
        else if (activeCam == cam4) SetActiveCamera(icam4);
        else
            Debug.LogWarning("OnIsoButton: No matching perspective cam is active!");
    }
    public void OnTopButton()
    {
        if (activeCam == icam1) SetActiveCamera(cam1);
        else if (activeCam == icam2) SetActiveCamera(cam2);
        else if (activeCam == icam3) SetActiveCamera(cam3);
        else if (activeCam == icam4) SetActiveCamera(cam4);
        else
            Debug.LogWarning("OnTopButton: No matching iso cam is active!");
    }
    // Run simulation button
    public void OnRunSimulationClicked()
    {
        SimulationController.Instance?.ResetSimulation();

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
