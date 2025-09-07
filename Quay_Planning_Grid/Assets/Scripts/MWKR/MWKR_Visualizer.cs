using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System;

public class MWKR_Visualizer : MonoBehaviour
{
    public static MWKR_Visualizer Instance { get; private set; }

    public QuayData quayScoreDB;
    public GradeMaterialMap materialMap;
    public List<Renderer> quayWallRenderers; // should match quayScoreDB.quayWallNames order

    [Header("Mini Visualizers (UI Panels)")]
    public List<Image> miniVisualizerImages;  // 28 slots, assign in Inspector for RL
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
        int n = Mathf.Max(
    quayScoreDB?.quayWallNames?.Count ?? 28,
    quayWallRenderers?.Count ?? 28
);
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
        // Refresh info panel if it is showing this quay
        if (QuayInfoPanel.Instance != null && QuayInfoPanel.Instance.CurrentQuayIndex == quayIndex)
        {
            MWKR_Runtime wship = engaged ? FindShipAtQuay(quayScoreDB.quayWallNames[quayIndex]) : null;
            /* QuayInfoPanel.Instance.UpdateQuayWallInfo(
                quayScoreDB.quayWallNames[quayIndex],
                wship,
                SimulationClock.Instance.simulationTime */
            //);
        }
    }

    public void SetQuayEngagement(string quayName, bool engaged)
    {
        if (string.IsNullOrWhiteSpace(quayName))
        {
            return;
        }
        // Optionally ignore special names like "Source"
        if (string.Equals(quayName.Trim(), "Source", StringComparison.OrdinalIgnoreCase))
        {
            // Debug.Log($"Ignoring non-quay name: {quayName}");
            return;
        }
        if (string.Equals(quayName.Trim(), "S", StringComparison.OrdinalIgnoreCase))
        {
            // Debug.Log($"Ignoring non-quay name: {quayName}");
            return;
        }
        if (string.Equals(quayName.Trim(), "Sink", StringComparison.OrdinalIgnoreCase))
        {
            // Debug.Log($"Ignoring non-quay name: {quayName}");
            return;
        }
        int quayIndex = quayScoreDB.quayWallNames.FindIndex(q => string.Equals(q.Trim(), quayName.Trim(), StringComparison.OrdinalIgnoreCase));
        if (quayIndex < 0)
        {
            Debug.LogWarning($"SetQuayEngagement: quayName '{quayName}' not found or invalid index");
            return;
        }
        SetQuayEngagement(quayIndex, engaged);
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
            // Reset mini visualizers to default color
            if (i < miniVisualizerImages.Count && miniVisualizerImages[i] != null)
                miniVisualizerImages[i].color = defaultMat.color;    
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
            }
             else
            {
                renderer.material = defaultMat;
            }
            // if we have a highlight and an opEntry, show grade for that quay
            if (opEntry != null && i < opEntry.quayScores.Count)
            {
                var grade = opEntry.quayScores[i];
                //renderer.material = materialMap.GetMaterial(grade);
                Color gradeColor = materialMap.GetMaterial(grade).color;

                if (i < miniVisualizerImages.Count && miniVisualizerImages[i] != null)
                    miniVisualizerImages[i].color = gradeColor;
            }
            else
          {
                if (i < miniVisualizerImages.Count && miniVisualizerImages[i] != null)
                    miniVisualizerImages[i].color = Color.white; // fallback
            }
            // default fall back material
        }
    }
    public void HighlightQuayInMiniVisualizer(string quayName)
    {
        if (string.IsNullOrWhiteSpace(quayName)) return;

        int quayIndex = quayScoreDB.quayWallNames.FindIndex(
            q => string.Equals(q.Trim(), quayName.Trim(), System.StringComparison.OrdinalIgnoreCase)
        );
        if (quayIndex < 0 || quayIndex >= miniVisualizerImages.Count) return;

        // Reset all outlines first
        for (int i = 0; i < miniVisualizerImages.Count; i++)
        {
            Outline outline = miniVisualizerImages[i].GetComponent<Outline>();
            if (outline != null) outline.enabled = false;
        }

        // Add/enable outline for the selected quay
        Image targetImg = miniVisualizerImages[quayIndex];
        if (targetImg != null)
        {
            Outline outline = targetImg.GetComponent<Outline>();
            if (outline == null) outline = targetImg.gameObject.AddComponent<Outline>();
            outline.effectColor =new Color32(205, 92, 92, 255);
            outline.effectDistance = new Vector2(2, -2);
            outline.enabled = true;
        }
    }

    public void ClearQuayMiniHighlights()
    {
        foreach (var img in miniVisualizerImages)
        {
            if (img == null) continue;
            Outline outline = img.GetComponent<Outline>();
            if (outline != null) outline.enabled = false;
        }
    }
    public MWKR_Runtime FindShipAtQuay(string quayName)
    {
        if (string.IsNullOrWhiteSpace(quayName))
        {
            Debug.LogWarning("FindShipAtQuay called with null or empty quayName");
            return null;
        }

        // Ignore reserved names
        if (string.Equals(quayName.Trim(), "Source", StringComparison.OrdinalIgnoreCase))
        {
            Debug.Log($"Ignoring non-quay name: {quayName}");
            return null;
        }

        // Normalize and find the index
        string normalizedName = quayName.Trim();
        int quayIndex = quayScoreDB.quayWallNames.FindIndex(
            q => string.Equals(q.Trim(), normalizedName, StringComparison.OrdinalIgnoreCase)
        );

        if (quayIndex < 0)
        {
            Debug.LogWarning($"Quay name '{quayName}' not found in quayWallNames list.");
            return null;
        }

        // Look for a ship whose AI target matches (ignoring case and spaces)
        foreach (MWKR_Runtime wship in FindObjectsOfType<MWKR_Runtime>())
        { 
            MWKR_Controller ai = wship.GetComponent<MWKR_Controller>();
            if (ai != null && string.Equals(ai.currentTarget?.Trim(), normalizedName, StringComparison.OrdinalIgnoreCase))
            {
                return wship;
            }
        }

        return null;
    }

#if UNITY_EDITOR
    [ContextMenu("Auto-Fill MWKR_Waypoint Renderers")]
    private void AutoFillSPTRenderers()
    {
        AutoFillRenderersByTag("MWKR_Waypoint");
    }

    private void AutoFillRenderersByTag(string tagToUse)
    {
        if (string.IsNullOrWhiteSpace(tagToUse))
        {
            Debug.LogWarning("No tag specified for AutoFillRenderers.");
            return;
        }

        quayWallRenderers.Clear();
        GameObject[] taggedObjects = GameObject.FindGameObjectsWithTag(tagToUse);

        foreach (string quayName in quayScoreDB.quayWallNames)
        {
            GameObject quayObj = Array.Find(taggedObjects,
                go => string.Equals(go.name.Trim(), quayName.Trim(), StringComparison.OrdinalIgnoreCase));

            if (quayObj != null && quayObj.TryGetComponent(out Renderer rend))
            {
                quayWallRenderers.Add(rend);
            }
            else
            {
                Debug.LogWarning($"Quay object '{quayName}' with tag '{tagToUse}' not found or has no Renderer.");
            }
        }

        isQuayEngaged = new bool[Mathf.Max(28, quayWallRenderers.Count)];
        Debug.Log($"Auto-filled {quayWallRenderers.Count} renderers for tag '{tagToUse}'.");
    }
#endif
}
