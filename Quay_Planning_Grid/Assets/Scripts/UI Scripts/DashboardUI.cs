using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System.IO;
using UnityEngine.UI;
using UnityEngine.UIElements;
using System;
#if !UNITY_EDITOR_OSX
using System.Windows.Forms;
#endif
public enum SimulationMode { RL, SPT, MOR, MWKR }


public class DashboardUI : MonoBehaviour
{
    public static DashboardUI Instance { get; private set; }
    public SimulationMode CurrentMode { get; private set; } = SimulationMode.RL;

    //public ScheduleManager scheduleManager;
    //public SPT_ScheduleManager SPTScheduleManager;
    public TMP_Text algorithmNameText;
    public TextMeshProUGUI textFilePath;
    public string textFileFullPath;
    public string textFileFullDirectory;
    public TextMeshProUGUI textFileName;
    public TextMeshProUGUI textNumberOfQuays;
    public TextMeshProUGUI textNumberOfShips;
    public UnityEngine.UI.Button SimulateButton;
    public UnityEngine.UI.Button PlayButton;

    private string FilePath = string.Empty;

    [Header("Cameras")]
    public Camera cam1;
    public Camera cam2;
    public Camera cam3;
    public Camera cam4;
    public Camera icam1;
    public Camera icam2;
    public Camera icam3;
    public Camera icam4;
    private Camera activeCam;
    [Header("Ship Data UI")]
    public GameObject RL;
    public GameObject SPT;
    public GameObject MOR;
    public GameObject MWKR;

    private GameObject activeImage;

    public Camera ActiveCamera => activeCam; // getter property

    private void Awake()
    {
        if (Instance == null) Instance = this;
    }
   
    private void Start()
    {
        SetActiveCamera(cam1); // Default camera
        UpdateAlgorithmText("RL");
        SetImageActive(RL);
        CurrentMode = SimulationMode.RL;
        SimulateButton.interactable = false;
        PlayButton.interactable = false;
    }
    private void SetActiveCamera(Camera cam)
    {
        if (activeCam != null) activeCam.gameObject.SetActive(false);
        activeCam = cam;
        if (activeCam != null) activeCam.gameObject.SetActive(true);
    }
    private void SetImageActive(GameObject panel)
    {
        // deactivate all heuristic panels
        RL.SetActive(false);
        SPT.SetActive(false);
        MOR.SetActive(false);
        MWKR.SetActive(false);

        // activate only the selected one
        activeImage = panel;
        if (activeImage != null)
            activeImage.SetActive(true);
    }
    public void OnCam1Button()
    {
        SetActiveCamera(cam1);
        UpdateAlgorithmText("RL");
        SetImageActive(RL);
        CurrentMode = SimulationMode.RL;
    }
    public void OnCam2Button()
    {
        SetActiveCamera(cam2);
        UpdateAlgorithmText("SPT-MF");
        SetImageActive(SPT);
        CurrentMode = SimulationMode.SPT;
    }
    public void OnCam3Button()
    {
        SetActiveCamera(cam3);
        UpdateAlgorithmText("MOR-MF");
        SetImageActive(MOR);
        CurrentMode = SimulationMode.MOR;
    }
    public void OnCam4Button()
    {
        SetActiveCamera(cam4);
        UpdateAlgorithmText("MWKR-MF");
        //SetImageActive(mwkrminiVisualizer);
        CurrentMode = SimulationMode.MWKR;
    }
    public void OnIsoButton()
    {
        if (activeCam == cam1) SetActiveCamera(icam1);
        else if (activeCam == cam2) SetActiveCamera(icam2);
        else if (activeCam == cam3) SetActiveCamera(icam3);
        else if (activeCam == cam4) SetActiveCamera(icam4);
        else
            Debug.LogWarning("OnIsoButton: No matching perspective cam is active!");
    }
    public void OnTopButton()
    {
        if (activeCam == icam1) SetActiveCamera(cam1);
        else if (activeCam == icam2) SetActiveCamera(cam2);
        else if (activeCam == icam3) SetActiveCamera(cam3);
        else if (activeCam == icam4) SetActiveCamera(cam4);
        else
            Debug.LogWarning("OnTopButton: No matching iso cam is active!");
    }
    // Run simulation button
    public void OnRunSimulationClicked()
    {
        SimulationController.Instance?.ResetSimulation();
        SimulationController.Instance?.Simulate();
    }

    public void OnPauseButtonClicked() => SimulationController.Instance?.PauseSimulation();
    public void OnStopButtonClicked() => SimulationController.Instance?.StopSimulation();
    public void OnResumeButtonClicked() => SimulationController.Instance?.ResumeSimulation();

    private void UpdateAlgorithmText(string name)
    {
        if (algorithmNameText != null)
            algorithmNameText.text = $"{name} Algorithm";
    }

    public void OnTypeBClicked()
    {
        SimulationController.Instance?.ResetSimulation();
#if !UNITY_EDITOR_OSX
        try
        {
            using (OpenFileDialog ofd = new OpenFileDialog())
            {
                ofd.Filter = "All Files (*.*)|*.*";
                ofd.Title = "Select a file";

                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    string fileName = Path.GetFileName(ofd.FileName);
                    DirectoryInfo dir = new DirectoryInfo(Path.GetDirectoryName(ofd.FileName));
                    string dir1 = dir?.Name;
                    string dir2 = dir?.Parent?.Name;
                    textFileFullPath = ofd.FileName;
                    textFileFullDirectory = dir.FullName;
                    if (!string.IsNullOrEmpty(dir2))
                        textFilePath.text = $"{dir2}/{dir1}/{fileName}";
                    if (!string.IsNullOrEmpty(dir1))
                        textFilePath.text = $"{dir1}/{fileName}";
                    textFileName.text = Path.GetFileNameWithoutExtension(ofd.FileName);
                    this.textNumberOfShips.text = "-";
                    FilePath = ofd.FileName;
                    SimulateButton.interactable = true;
                    SimulationController.Instance.mSelectedDataSource = SourceType.FindData;
                }
            }
        } catch (Exception e)
        {
            SimulationController.Instance.SendError(e.Message);
        }
#endif
    }
    public void OnTypeAButtonClicked(UserInputPanelUI ui)
    {
        ui.NumberOfShips.value = ui.SelectedNumberOfShips;
    }

    public void OnUserInputPanelOkButtonClicked(UserInputPanelUI ui)
    {
        SimulationController.Instance.mSelectedDataSource = SourceType.GenerateData;
        this.textNumberOfQuays.text = ui.textNumberOfWalls.text;
        this.textNumberOfShips.text = ui.textNumberOfShips.text;
        ui.SelectedNumberOfShips = (int)ui.NumberOfShips.value;
        textFilePath.text = "> No file selected";
        textFileName.text = "-";
        textFileFullPath = string.Empty;
        textFileFullDirectory = string.Empty;
        FilePath = string.Empty;
        SimulateButton.interactable = true;
    }

}
