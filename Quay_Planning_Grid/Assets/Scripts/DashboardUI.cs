using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class DashboardUI : MonoBehaviour
{
    public ScheduleManager scheduleManager;
   // public RL_ScheduleManager rlScheduleManager; // assign via Inspector

    public TMP_Text algorithmNameText; // Assign in Inspector

    public void LoadScheduleRL()
    {
        SimulationController.Instance.ResetSimulation();
        scheduleManager.LoadSchedule("log-RL.csv");
        UpdateAlgorithmText("RL");
    }

    public void LoadScheduleSHP()
    {
        SimulationController.Instance.ResetSimulation();
        //rlScheduleManager.RLLoadSchedule("log-RL.csv");
        scheduleManager.LoadSchedule("log-SPT-MF.csv");
        UpdateAlgorithmText("SPT-HP");
    }

    public void LoadScheduleSLU()
    {
        SimulationController.Instance.ResetSimulation();
        //  rlScheduleManager.RLLoadSchedule("log-RL.csv");
        scheduleManager.LoadSchedule("log-SPT-LU.csv");
        UpdateAlgorithmText("SPT-LU");
    }
     
    public void LoadScheduleSMF()
    {
        SimulationController.Instance.ResetSimulation();
        // rlScheduleManager.RLLoadSchedule("log-RL.csv");
        scheduleManager.LoadSchedule("log-SPT-MF.csv");
        UpdateAlgorithmText("SPT-MF");
    }

    public void LoadScheduleMHP()
    {
        SimulationController.Instance.ResetSimulation();
        // rlScheduleManager.RLLoadSchedule("log-RL.csv");
        scheduleManager.LoadSchedule("log-MOR-HP.csv");
        UpdateAlgorithmText("MOR-HP");
    }

    public void LoadScheduleMLU()
    {
        SimulationController.Instance.ResetSimulation();
        //  rlScheduleManager.RLLoadSchedule("log-RL.csv");
        scheduleManager.LoadSchedule("log-MOR-LU.csv");
        UpdateAlgorithmText("MOR-LU");
    }

    public void LoadScheduleMMF()
    {
        SimulationController.Instance.ResetSimulation();
        //  rlScheduleManager.RLLoadSchedule("log-RL.csv");
        scheduleManager.LoadSchedule("log-MOR-MF.csv");
        UpdateAlgorithmText("MOR-MF");
    }

    public void LoadScheduleMWHP()
    {
        SimulationController.Instance.ResetSimulation();
        //  rlScheduleManager.RLLoadSchedule("log-RL.csv");
        scheduleManager.LoadSchedule("log-MWKR-HP.csv");
        UpdateAlgorithmText("MWKR-HP");
    }

    public void LoadScheduleMWLU()
    {
        SimulationController.Instance.ResetSimulation();
        // rlScheduleManager.RLLoadSchedule("log-RL.csv");
        scheduleManager.LoadSchedule("log-MWKR-LU.csv");
        UpdateAlgorithmText("MWKR-LU");
    }

    public void LoadScheduleMWMF()
    {
      //  rlScheduleManager.RLLoadSchedule("log-RL.csv");
        scheduleManager.LoadSchedule("log-MWKR-MF.csv");
        UpdateAlgorithmText("MWKR-MF");
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
