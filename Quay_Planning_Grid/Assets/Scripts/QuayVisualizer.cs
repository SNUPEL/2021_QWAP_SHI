using UnityEngine;
using System.Collections.Generic;

public class QuayVisualizer : MonoBehaviour
{
    public static QuayVisualizer Instance { get; private set; }

    public QuayData quayScoreDB;
    public GradeMaterialMap materialMap;
    public List<Renderer> quayWallRenderers; // should match quayScoreDB.quayWallNames order
    public Material defaultMat;

    // engagement state per quay
    private bool[] isQuayEngaged;

    // highlight cache (when user selects a ship)
    private string highlightedShipType;
    private string highlightedOperation;
    private bool highlightActive = false;

    private void Awake()
    {
        if (Instance == null) Instance = this;
        else Destroy(gameObject);

        // init engagement array sized to renderer list (safe)
        int n = Mathf.Max(28, quayWallRenderers?.Count ?? 28);
        isQuayEngaged = new bool[n];
    }

    // PUBLIC API --------------------------------------------------
    public void SetQuayEngagement(int quayIndex, bool engaged)
    {
        if (quayIndex < 0 || quayIndex >= quayWallRenderers.Count)
        {
            Debug.LogWarning($"Invalid quayIndex {quayIndex}");
            return;
        }

        isQuayEngaged[quayIndex] = engaged;
        // Re-apply whole visualization so we never accidentally overwrite correct visuals
        ApplyVisualization();
        //UpdateQuayMaterial(quayIndex);

        // Refresh info panel if it is showing this quay
        if (QuayInfoPanel.Instance != null && QuayInfoPanel.Instance.CurrentQuayIndex == quayIndex)
        {
            ShipRuntime ship = engaged ? FindShipAtQuay(quayIndex) : null; // You must implement this
            QuayInfoPanel.Instance.UpdateQuayWallInfo(
                quayScoreDB.quayWallNames[quayIndex],
                ship,
                SimulationClock.Instance.simulationTime
            );
        }
    }

    // Called when user selects a ship to show grades for that ship/operation
    public void HighlightQuayGrades(string shipType, string operation)
    {
        highlightedShipType = shipType;
        highlightedOperation = operation;
        highlightActive = true;
        ApplyVisualization();
    }

    //Clear any grade highlight but preserve engaged quays
    public void ResetGradesOnly()
    {
        highlightActive = false;
        highlightedShipType = null;
        highlightedOperation = null;
        ApplyVisualization();
    }

    // Full reset (used when resetting entire simulation)
    public void ResetVisualizer()
    {
        highlightActive = false;
        highlightedShipType = null;
        highlightedOperation = null;

        if (quayWallRenderers == null)
        {
            Debug.LogWarning("QuayVisualizer: quayWallRenderers is null");
            return;
        }

        for (int i = 0; i < quayWallRenderers.Count; i++)
        {
            isQuayEngaged[i] = false;
            if (quayWallRenderers[i] != null)
                quayWallRenderers[i].material = defaultMat;
        }

        Debug.Log("QuayVisualizer fully reset (no green materials).");
    }

    // INTERNAL ----------------------------------------------------
    void ApplyVisualization()
    {
        // If highlightActive, find the ship type entry & operation entry (if possible)
        ShipTypeScores shipEntry = null;
        OperationQuayScores opEntry = null;

        if (highlightActive && quayScoreDB != null)
        {
            shipEntry = quayScoreDB.shipTypeScores.Find(s => s.shipType == highlightedShipType);
            if (shipEntry != null)
                opEntry = shipEntry.operations.Find(o => o.operationName == highlightedOperation);
        }

        for (int i = 0; i < quayWallRenderers.Count; i++)
        {
            var renderer = quayWallRenderers[i];
            if (renderer == null) continue;

            // engaged quays always show the engaged material
            if (i < isQuayEngaged.Length && isQuayEngaged[i])
            {
                renderer.material = materialMap.engagedMaterial;
                continue;
            }

            // if we have a highlight and an opEntry, show grade for that quay
            if (opEntry != null && i < opEntry.quayScores.Count)
            {
                var grade = opEntry.quayScores[i];
                renderer.material = materialMap.GetMaterial(grade);
                continue;
            }

            // default fall back material
            renderer.material = defaultMat;
        }
    }

    public ShipRuntime FindShipAtQuay(int quayIndex)
    {
        foreach (var shipGO in ShipBuilder.Instance.ActiveShips)
        {
            ShipRuntime shipRuntime = shipGO.GetComponent<ShipRuntime>();
            if (shipRuntime != null)
            {
                AIController aiController = shipGO.GetComponent<AIController>();
                if (aiController != null && aiController.currIndex == quayIndex)
                {
                    return shipRuntime;
                }
            }
        }
        return null;
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

        // ensure array length matches
        isQuayEngaged = new bool[Mathf.Max(28, quayWallRenderers.Count)];
    }
#endif
}
//using UnityEngine;
//using System.Collections.Generic;

//public class QuayVisualizer : MonoBehaviour
//{

//    public static QuayVisualizer Instance { get; private set; }

//    public QuayData quayScoreDB;
//    public GradeMaterialMap materialMap;
//    public List<Renderer> quayWallRenderers; // 28 renderers for A1–E8 (in same order as quayScoreDB.quayWallNames)
//    private bool[] isQuayEngaged = new bool[28];
//    public Material defaultMat;
//    //private void Awake()
//    //{
//    //    engagedFlags = new bool[quayWallRenderers.Count];
//    //}

//    public void SetQuayEngagement(int quayIndex, bool engaged)
//    {
//        if (quayIndex < 0 || quayIndex >= quayWallRenderers.Count)
//        {
//            Debug.LogWarning($"Invalid quayIndex {quayIndex}");
//            return;
//        }
//        isQuayEngaged[quayIndex] = engaged;
//        // Update material immediately
//        UpdateQuayMaterial(quayIndex);
//    }

//    void UpdateQuayMaterial(int quayIndex)
//    {
//        var currentGrade = QuayScoreGrade.N; // default fallback

//        // Optional: You could get grade info from your data model here if you want to keep showing grades
//        // For now, just set material based on engagement

//        bool engaged = isQuayEngaged[quayIndex];
//        Material mat = materialMap.GetMaterial(currentGrade, engaged);
//        quayWallRenderers[quayIndex].material = mat;
//    }

//    // Called when a ship is selected
//    public void HighlightQuayGrades(string shipType, string operation)
//    {
//        var shipEntry = quayScoreDB.shipTypeScores.Find(s => s.shipType == shipType);
//        if (shipEntry == null)
//        {
//            Debug.LogWarning($"No ship type found: {shipType}");
//            return;
//        }

//        var opEntry = shipEntry.operations.Find(o => o.operationName == operation);
//        if (opEntry == null)
//        {
//            Debug.LogWarning($"No operation found: {operation} for {shipType}");
//            return;
//        }

//        for (int i = 0; i < quayWallRenderers.Count && i < opEntry.quayScores.Count; i++)
//        {
//            if (isQuayEngaged[i])
//            {
//                quayWallRenderers[i].material = materialMap.engagedMaterial; // Green material for engaged
//            }
//            else
//            {
//                var grade = opEntry.quayScores[i];
//                var mat = materialMap.GetMaterial(grade);
//                quayWallRenderers[i].material = mat;
//            }
//        }
//    }


//#if UNITY_EDITOR
//    [ContextMenu("Auto-Fill Quay Renderers")]
//    public void AutoFillRenderers()
//    {
//        quayWallRenderers.Clear();
//        foreach (string quayName in quayScoreDB.quayWallNames)
//        {
//            GameObject quayObj = GameObject.Find(quayName);
//            if (quayObj != null && quayObj.TryGetComponent(out Renderer rend))
//            {
//                quayWallRenderers.Add(rend);
//            }
//            else
//            {
//                Debug.LogWarning($"Quay object '{quayName}' not found or has no Renderer.");
//            }
//        }
//    }
//#endif

//    public void ResetVisualizer()
//    {
//        if (defaultMat == null)
//        {
//            Debug.LogWarning("Default material not assigned in QuayVisualizer!");
//            return;
//        }

//        for (int i = 0; i < quayWallRenderers.Count; i++)
//        {
//            isQuayEngaged[i] = false; // Reset engagement status

//            if (quayWallRenderers[i] != null)
//            {
//                quayWallRenderers[i].material = defaultMat; // Reset material to default
//            }
//            else
//            {
//                Debug.LogWarning($"Renderer at index {i} is null.");
//            }
//        }

//        Debug.Log("QuayVisualizer fully reset (no green materials).");
//    }
//    public void ResetGradesOnly()
//    {
//        for (int i = 0; i < quayWallRenderers.Count; i++)
//        {
//            if (!isQuayEngaged[i] && quayWallRenderers[i] != null)
//            {
//                quayWallRenderers[i].material = defaultMat;
//            }
//        }

//        Debug.Log("QuayVisualizer: grades cleared, engaged ones preserved.");
//    }
//public void ResetVisualizer()
//{
//    if (defaultMat == null)
//    {
//        Debug.LogWarning("Default material not assigned in QuayVisualizer!");
//        return;
//    }

//    for (int i = 0; i < quayWallRenderers.Count; i++)
//    {
//        isQuayEngaged[i] = false; // Reset engagement status

//        if (quayWallRenderers[i] != null)
//        {
//            quayWallRenderers[i].material = defaultMat; // Reset material to default
//        }
//        else
//        {
//            Debug.LogWarning($"Renderer at index {i} is null.");
//        }
//    }

//    Debug.Log("QuayVisualizer fully reset (no green materials).");
//}
//public void ResetGradesOnly()
//{
//    for (int i = 0; i < quayWallRenderers.Count; i++)
//    {
//        if (!isQuayEngaged[i] && quayWallRenderers[i] != null)
//        {
//            quayWallRenderers[i].material = defaultMat;
//        }
//    }

//    Debug.Log("QuayVisualizer: grades cleared, engaged ones preserved.");
//}
//}
