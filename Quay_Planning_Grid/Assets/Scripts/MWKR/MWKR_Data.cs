using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEditor;
using System;

public class MWKR_Data : MonoBehaviour
{
    public static MWKR_Data Instance;

    public List<SimulationData> alllogs = new List<SimulationData>();

    private void Awake()
    {
        if (Instance == null) Instance = this;
        else Destroy(gameObject);
    }

    public void LoadSimulationLogs(string fileName)
    {
        string fullPath = Path.Combine(Application.dataPath, "Data", fileName);
        Debug.Log("Looking for CSV at: " + fullPath);

        if (!File.Exists(fullPath))
        {
            Debug.LogError("CSV file not found at: " + fullPath);
            return;
        }

        alllogs.Clear();
        string[] allLines = File.ReadAllLines(fullPath);

        foreach (string s in allLines)
        {
            string[] splitData = s.Split(',');
            for (int i = 0; i < splitData.Length; i++)
                splitData[i] = splitData[i].Trim();

            if (splitData.Length != 6)
            {
                Debug.LogWarning($"{s} does not have the required 6 values.");
                continue;
            }

            SimulationData log = ScriptableObject.CreateInstance<SimulationData>();
            log.Time = int.Parse(splitData[0]);
            log.Location = splitData[1];
            log.Ship_Index = splitData[2];
            log.Operation = splitData[3];
            log.Log = splitData[4];
            log.Weight = splitData[5];

            alllogs.Add(log);
        }

        Debug.Log($"Loaded {alllogs.Count} entries from {fileName}");
    }

    public List<SimulationData> GetLogsForShips(string shipIndex)
    {
        string normalizedIndex = NormalizeShipID(shipIndex);
        Debug.Log($"Looking for logs matching: {normalizedIndex}");

        List<SimulationData> matching = new List<SimulationData>();

        foreach (var log in alllogs)
        {
            string logID = NormalizeShipID(log.Ship_Index);
            //            Debug.Log($"Comparing log.Ship_Index = {log.Ship_Index} → {logID}");
            if (logID == normalizedIndex)
            {
                matching.Add(log);
            }
        }

        return matching;

    }

    private string NormalizeShipID(string id)
    {
        if (string.IsNullOrEmpty(id)) return "";
        return id.Trim().ToUpperInvariant(); // Keep underscores if needed
    }
}
