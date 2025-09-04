using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;
using System.IO;
using UnityEngine.UI;


#if UNITY_EDITOR_WIN
using System.Windows.Forms;
#endif

public class DashboardUI : MonoBehaviour
{
    //public ScheduleManager scheduleManager;
    //public SPT_ScheduleManager SPTScheduleManager;
    public TMP_Text algorithmNameText;
    public TextMeshProUGUI textFilePath;
    public TextMeshProUGUI textFileName;
    public TextMeshProUGUI textNumberOfQuays;
    public TextMeshProUGUI textNumberOfShips;

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

    private void Start()
    {
        SetActiveCamera(cam1); // Default camera
        UpdateAlgorithmText("RL");
    }

    private void SetActiveCamera(Camera cam)
    {
        if (activeCam != null) activeCam.gameObject.SetActive(false);
        activeCam = cam;
        if (activeCam != null) activeCam.gameObject.SetActive(true);
    }

    public void OnCam1Button() 
    {
        SetActiveCamera(cam1);
        UpdateAlgorithmText("RL");
    }
    public void OnCam2Button() 
    {
        SetActiveCamera(cam2);
        UpdateAlgorithmText("SPT-MF");
    }
    public void OnCam3Button() 
    {
        SetActiveCamera(cam3);
        UpdateAlgorithmText("MOR-MF");
    }
    public void OnCam4Button() 
    {
        SetActiveCamera(cam4);
        UpdateAlgorithmText("MWKR-MF");
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

        SimulationController.Instance?.Play();


        SimulationController.Instance.StartSimulation(FilePath);
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
#if UNITY_EDITOR_WIN
        using (OpenFileDialog ofd = new OpenFileDialog())
        {
            ofd.Filter = "All Files (*.*)|*.*";
            ofd.Title = "Select a file";

            if (ofd.ShowDialog() == DialogResult.OK)
            {
                // = ofd.FileName;  // 선택한 파일 경로 반환
                                                   // 파일명 추출
                string fileName = Path.GetFileName(ofd.FileName);

                // 상위 디렉토리 2개 추출
                DirectoryInfo dir = new DirectoryInfo(Path.GetDirectoryName(ofd.FileName));
                string dir1 = dir?.Name;               // 현재 디렉토리
                string dir2 = dir?.Parent?.Name;       // 상위 디렉토리

                if (!string.IsNullOrEmpty(dir2))
                    textFilePath.text = $"{dir2}/{dir1}/{fileName}";
                if (!string.IsNullOrEmpty(dir1))
                    textFilePath.text = $"{dir1}/{fileName}";
                textFileName.text = Path.GetFileNameWithoutExtension(ofd.FileName);
                FilePath = ofd.FileName;
            }
        }
#endif
    }
    public void OnTypeAButtonClicked(UserInputPanelUI ui)
    {
        ui.NumberOfShips.value = ui.SelectedNumberOfShips;
    }

    public void OnUserInputPanelOkButtonClicked(UserInputPanelUI ui)
    {
        this.textNumberOfQuays.text = ui.textNumberOfWalls.text;
        this.textNumberOfShips.text = ui.textNumberOfShips.text;
        ui.SelectedNumberOfShips = (int)ui.NumberOfShips.value;
        textFilePath.text = "> No file selected";
        textFileName.text = "-";
        FilePath = string.Empty;
    }

}
