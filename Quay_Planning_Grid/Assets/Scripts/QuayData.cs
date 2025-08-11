using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public enum QuayScoreGrade { A, B, C, D, E, N }

[System.Serializable]
public class OperationQuayScores
{
    public string operationName; // e.g., "화물창", "P/T"
    public List<QuayScoreGrade> quayScores = new List<QuayScoreGrade>(28); // one grade per quay wall
}

[System.Serializable]
public class ShipTypeScores
{
    public string shipType; // e.g., "LNG", "VL_CONT"
    public List<OperationQuayScores> operations = new List<OperationQuayScores>();
}

[CreateAssetMenu(fileName = "QuayData", menuName = "Custom/Quay Wall Scores")]


public class QuayData : ScriptableObject
{

[Tooltip("Ordered list of quay walls (should be 28 names like A1, A2, ..., E8)")]
    public List<string> quayWallNames = new List<string>();

    [Tooltip("All ship types and their operation scores per quay wall")]
    public List<ShipTypeScores> shipTypeScores = new List<ShipTypeScores>();

}
