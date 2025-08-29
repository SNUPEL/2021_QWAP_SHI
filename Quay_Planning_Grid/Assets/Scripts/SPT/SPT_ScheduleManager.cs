using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SPT_ScheduleManager : MonoBehaviour
{
    public static SPT_ScheduleManager Instance;
    //public Simulation_Data CurrentSimulation => RL_Simulation_Data.Instance;
    public Simulation_Data CurrentSimulation => Simulation_Data.Instance;

    private void Awake()
    {
        if (Instance == null) Instance = this;
        else Destroy(gameObject);
    }

    public void SPTLoadSchedule(string fileName)
    {

        if (Simulation_Data.Instance == null)
        {
            Debug.LogError($"SPT schedule or file {fileName} not found or null!");
            return;
        }

        Simulation_Data.Instance.LoadSimulationLogs(fileName);

        Debug.Log($"SPT_ScheduleManager loaded {fileName}");

        //SPT_Builder.Instance?.HandleTimeChanged(0);
    }
}
