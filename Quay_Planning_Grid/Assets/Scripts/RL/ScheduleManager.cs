using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class ScheduleManager : MonoBehaviour
{
    public static ScheduleManager Instance;
    //private int currentScheduleIndex = 0;
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
        //ShipBuilder.Instance?.HandleTimeChanged(0);

    }
}
    //public static ScheduleManager Instance;

//private List<string> scheduleFiles = new List<string>
//{
//    "log-RL.csv",
//    "log-SPT-MF.csv",
//    "log-MOR-HP.csv",
//    "log-MWKR-LU.csv"
//};

//public List<Simulation_Data> LoadedSchedules { get; private set; } = new List<Simulation_Data>();
//public Simulation_Data CurrentSimulation => Simulation_Data.Instance;

//private void Awake()
//{
//    if (Instance == null) Instance = this;
//    else Destroy(gameObject);
//}

//// Load all 4 scenarios
//public void LoadAllSchedules()
//{
//    SimulationController.Instance?.ResetSimulation();
//    LoadedSchedules.Clear();

//    //foreach (var file in scheduleFiles)
//    //{
//    //    var simData = ScriptableObject.CreateInstance<Simulation_Data>();
//    //    simData.LoadSimulationLogs(file);
//    //    LoadedSchedules.Add(simData);
//    //}

//    //Debug.Log($"Loaded {LoadedSchedules.Count} schedules.");
//    //SimulationController.Instance?.StartSimulation();
//}

