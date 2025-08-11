using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class RL_ScheduleManager : MonoBehaviour
{
    public static RL_ScheduleManager Instance;
    //public Simulation_Data CurrentSimulation => RL_Simulation_Data.Instance;
    public RL_Simulation_Data CurrentSimulation => RL_Simulation_Data.Instance;

    private void Awake()
    {
        if (Instance == null) Instance = this;
        else Destroy(gameObject);
    }

    public void RLLoadSchedule(string fileName)
    {

        if (RL_Simulation_Data.Instance == null)
        {
            Debug.LogError($"RL schedule or file {fileName} not found or null!");
            return;
        }

        RL_Simulation_Data.Instance.LoadSimulationLogs(fileName);

        Debug.Log($"RL_ScheduleManager loaded {fileName}");

        RL_ShipBuilder.Instance?.HandleTimeChanged(0);
    }
}
