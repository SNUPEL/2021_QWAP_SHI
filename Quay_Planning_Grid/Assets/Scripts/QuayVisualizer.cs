using UnityEngine;
using System.Collections.Generic;

public class QuayVisualizer : MonoBehaviour
{

    public static QuayVisualizer Instance { get; private set; }

    public QuayData quayScoreDB;
    public GradeMaterialMap materialMap;
    public List<Renderer> quayWallRenderers; // 28 renderers for A1–E8 (in same order as quayScoreDB.quayWallNames)
    private bool[] isQuayEngaged = new bool[28];
    public Material defaultMat;
    //private void Awake()
    //{
    //    engagedFlags = new bool[quayWallRenderers.Count];
    //}

    public void SetQuayEngagement(int quayIndex, bool engaged)
    {
        if (quayIndex < 0 || quayIndex >= quayWallRenderers.Count)
        {
            Debug.LogWarning($"Invalid quayIndex {quayIndex}");
            return;
        }
        isQuayEngaged[quayIndex] = engaged;
        // Update material immediately
        UpdateQuayMaterial(quayIndex);
    }

    void UpdateQuayMaterial(int quayIndex)
    {
        var currentGrade = QuayScoreGrade.N; // default fallback

        // Optional: You could get grade info from your data model here if you want to keep showing grades
        // For now, just set material based on engagement

        bool engaged = isQuayEngaged[quayIndex];
        Material mat = materialMap.GetMaterial(currentGrade, engaged);
        quayWallRenderers[quayIndex].material = mat;
    }

    // Called when a ship is selected
    public void HighlightQuayGrades(string shipType, string operation)
    {
        var shipEntry = quayScoreDB.shipTypeScores.Find(s => s.shipType == shipType);
        if (shipEntry == null)
        {
            Debug.LogWarning($"No ship type found: {shipType}");
            return;
        }

        var opEntry = shipEntry.operations.Find(o => o.operationName == operation);
        if (opEntry == null)
        {
            Debug.LogWarning($"No operation found: {operation} for {shipType}");
            return;
        }

        for (int i = 0; i < quayWallRenderers.Count && i < opEntry.quayScores.Count; i++)
        {
            if (isQuayEngaged[i])
            {
                quayWallRenderers[i].material = materialMap.engagedMaterial; // Green material for engaged
            }
            else
            {
                var grade = opEntry.quayScores[i];
                var mat = materialMap.GetMaterial(grade);
                quayWallRenderers[i].material = mat;
            }
        }
    }
    

#if UNITY_EDITOR
    [ContextMenu("Auto-Fill Quay Renderers")]
    public void AutoFillRenderers()
    {
        quayWallRenderers.Clear();
        foreach (string quayName in quayScoreDB.quayWallNames)
        {
            GameObject quayObj = GameObject.Find(quayName);
            if (quayObj != null && quayObj.TryGetComponent(out Renderer rend))
            {
                quayWallRenderers.Add(rend);
            }
            else
            {
                Debug.LogWarning($"Quay object '{quayName}' not found or has no Renderer.");
            }
        }
    }
#endif

    public void ResetVisualizer()
    {
        if (defaultMat == null)
        {
            Debug.LogWarning("Default material not assigned in QuayVisualizer!");
            return;
        }

        for (int i = 0; i < quayWallRenderers.Count; i++)
        {
            isQuayEngaged[i] = false; // Reset engagement status

            if (quayWallRenderers[i] != null)
            {
                quayWallRenderers[i].material = defaultMat; // Reset material to default
            }
            else
            {
                Debug.LogWarning($"Renderer at index {i} is null.");
            }
        }

        Debug.Log("QuayVisualizer fully reset (no green materials).");
    }
    public void ResetGradesOnly()
    {
        for (int i = 0; i < quayWallRenderers.Count; i++)
        {
            if (!isQuayEngaged[i] && quayWallRenderers[i] != null)
            {
                quayWallRenderers[i].material = defaultMat;
            }
        }

        Debug.Log("QuayVisualizer: grades cleared, engaged ones preserved.");
    }

}
