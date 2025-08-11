using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEditor;

public class CSVtoSO
{
    private static string InputCSV = "/Editor/instance-1.csv";
    [MenuItem("Utilities/ Generate Ship Data")]
    public static void GenerateShips()
    {
        string[] allLines = File.ReadAllLines(Application.dataPath + InputCSV);

        Debug.Log("First few raw lines from CSV:");
        for (int i = 0; i < Mathf.Min(3, allLines.Length); i++)
        {
            Debug.Log(allLines[i]);
        }

        HashSet<int> processedIndices = new HashSet<int>(); // Store processed Ship_Indices
        Dictionary<int, ShipData> shipDict = new Dictionary<int, ShipData>(); // Store by Ship_Index

        foreach (string s in allLines)
        {
            string[] splitData = s.Split(',');
            for (int i = 0; i < splitData.Length; i++)
                splitData[i] = splitData[i].Trim();

            if (splitData.Length != 16)
            {
                Debug.Log(s + " does not have the required values.");
                return;
            }

            int shipIndex = int.Parse(splitData[2]);

            if (processedIndices.Contains(shipIndex))
            {
                shipDict[shipIndex].Start_Dates.Add(int.Parse(splitData[11]));
                shipDict[shipIndex].Finish_Dates.Add(int.Parse(splitData[12]));
                shipDict[shipIndex].Operation_Name.Add(splitData[7]);
                shipDict[shipIndex].Operation_Type.Add(splitData[9]);
                continue;
            }
            processedIndices.Add(shipIndex);

            ShipData ship = ScriptableObject.CreateInstance<ShipData>();

            ship.Entry_Index = splitData[0];
            ship.Ship_Name = splitData[1];
            ship.Ship_Index = shipIndex;
            ship.Ship_Type = splitData[3];
            ship.Category = int.Parse(splitData[4]);
            ship.Launching_Date = int.Parse(splitData[5]);
            ship.Delivery_Date = int.Parse(splitData[6]);
            ship.Operation_Name = new List<string> { splitData[7] };
            ship.Operation_Index = int.Parse(splitData[8]);
            ship.Operation_Type = new List<string> { splitData[9] };
            ship.Order = int.Parse(splitData[10]);
            ship.Start_Dates = new List<int> { int.Parse(splitData[11]) };
            ship.Finish_Dates = new List<int> { int.Parse(splitData[12]) };
            ship.Duration = int.Parse(splitData[13]);
            ship.Interruption = splitData[14];
            ship.Fixed_Duration = int.Parse(splitData[15]);

            shipDict[shipIndex] = ship;
        }

        foreach (var kvp in shipDict)
        {
            ShipData ship = kvp.Value;
            string safeName = $"{ship.Ship_Name}";
            AssetDatabase.CreateAsset(ship, $"Assets/Resources/Ships/{safeName}.asset");
        }

        AssetDatabase.SaveAssets();
    }
}