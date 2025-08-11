using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class ScheduleManager : MonoBehaviour
{
    public static ScheduleManager Instance;
    private int currentScheduleIndex = 0;
    private List<Simulation_Data> schedules = new List<Simulation_Data>();
    public Simulation_Data CurrentSimulation => Simulation_Data.Instance;
    
        private void Awake()
    {
        if (Instance == null) Instance = this;
        else Destroy(gameObject);
    }
    public void LoadSchedule(string fileName)
    {

    SimulationController.Instance?.ResetSimulation(); 

        Simulation_Data.Instance.LoadSimulationLogs(fileName);

        if (Simulation_Data.Instance == null)
        {
            Debug.LogError($"Schedule or file {fileName} not found or null!");
            return;
        }
        SimulationController.Instance?.StartSimulation();
    }
}