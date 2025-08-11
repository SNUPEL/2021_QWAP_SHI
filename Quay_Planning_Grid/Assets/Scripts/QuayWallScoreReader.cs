using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.IO;

public static class QuayWallScoreReader
{
    [MenuItem("Tools/Import Quay Wall Scores from CSV")]
    public static void ImportCSV()
    {
        string path = EditorUtility.OpenFilePanel("Import Quay Wall Score CSV", "", "csv");
        if (string.IsNullOrEmpty(path)) return;

        string[] lines = File.ReadAllLines(path);
        if (lines.Length < 2)
        {
            Debug.LogError("CSV file is too short.");
            return;
        }

        // First line is header
        string[] headers = lines[0].Split(',');
        if (headers.Length < 30) // ShipType + Operation + 28 QuayWalls
        {
            Debug.LogError("CSV header format is invalid.");
            return;
        }

        List<string> quayWalls = new List<string>(headers[2..]); // skip ShipType and Operation

        Dictionary<string, ShipTypeScores> shipScoresDict = new();

        // Parse rows
        for (int i = 1; i < lines.Length; i++)
        {
            string[] tokens = lines[i].Split(',');
            if (tokens.Length != headers.Length) continue;

            string shipType = tokens[0].Trim();
            string operation = tokens[1].Trim();
            var scoresRaw = tokens[2..];

            var quayScores = new List<QuayScoreGrade>();
            foreach (var s in scoresRaw)
            {
                if (System.Enum.TryParse<QuayScoreGrade>(s.Trim(), out var grade))
                    quayScores.Add(grade);
                else
                    quayScores.Add(QuayScoreGrade.N); // fallback
            }

            var opScores = new OperationQuayScores
            {
                operationName = operation,
                quayScores = quayScores
            };

            if (!shipScoresDict.ContainsKey(shipType))
            {
                shipScoresDict[shipType] = new ShipTypeScores
                {
                    shipType = shipType,
                    operations = new List<OperationQuayScores>()
                };
            }

            shipScoresDict[shipType].operations.Add(opScores);
        }

        // Create ScriptableObject
        var asset = ScriptableObject.CreateInstance<QuayData>();
        asset.quayWallNames = quayWalls;
        asset.shipTypeScores = new List<ShipTypeScores>(shipScoresDict.Values);

        // Save asset
        string assetPath = "Assets/QuayData.asset";
        AssetDatabase.CreateAsset(asset, assetPath);
        AssetDatabase.SaveAssets();
        AssetDatabase.Refresh();

        Debug.Log($"Imported QuayData with {asset.shipTypeScores.Count} ship types.");
    }
}
