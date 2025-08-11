using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[CreateAssetMenu(fileName = "New Ship", menuName = "Ships/Ship")]

public class SimulationData : ScriptableObject
{

    public int Time;
    public string Location;
    public string Ship_Index;
    public string Operation;
    public string Log;
    public string Weight;

}
