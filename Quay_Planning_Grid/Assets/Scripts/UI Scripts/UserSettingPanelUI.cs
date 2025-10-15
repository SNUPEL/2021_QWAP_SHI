using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
#if !UNITY_EDITOR_OSX
using System.Windows.Forms;
#endif
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.UIElements;


public class UserSettingPanelUI : MonoBehaviour
{

    [SerializeField] private TMP_InputField mInputFieldBaseDirectory;
    [SerializeField] private TMP_InputField mInputFieldModelPath;
    [SerializeField] private TMP_InputField mInputFieldDataDirectory;
    [SerializeField] private TMP_InputField mInputFieldResultDirectory;

    [SerializeField] private UnityEngine.UI.Button mButtonSearchBaseDirectory;
    [SerializeField] private UnityEngine.UI.Button mButtonSearchModelPath;
    [SerializeField] private UnityEngine.UI.Button mButtonSearchDataDirectory;
    [SerializeField] private UnityEngine.UI.Button mButtonSearchResultDirectory;

    public string mBaseDirectoryKey = "Base Directory Key";
    public string mModelPathKey = "Model Path Key";
    public string mDataDirectoryKey = "Data Directory Key";
    public string mResultDirectoryKey = "Result Directory Key";

    

    // Start is called before the first frame update
    void Start()
    {
        SimulationController.Instance.mBaseDirectory = PlayerPrefs.GetString(mBaseDirectoryKey);
        SimulationController.Instance.mModelPath = PlayerPrefs.GetString(mModelPathKey);
        SimulationController.Instance.mDataPath = PlayerPrefs.GetString(mDataDirectoryKey);
        SimulationController.Instance.mResultPath = PlayerPrefs.GetString(mResultDirectoryKey);

        mInputFieldBaseDirectory.text = SimulationController.Instance.mBaseDirectory;
        mInputFieldModelPath.text = SimulationController.Instance.mModelPath;
        mInputFieldDataDirectory.text = SimulationController.Instance.mDataPath;
        mInputFieldResultDirectory.text = SimulationController.Instance.mResultPath;
    }

    public void OnUserSettingPanelUIShowed()
    {
        mInputFieldBaseDirectory.text = SimulationController.Instance.mBaseDirectory;
        mInputFieldModelPath.text = SimulationController.Instance.mModelPath;
        mInputFieldDataDirectory.text = SimulationController.Instance.mDataPath;
        mInputFieldResultDirectory.text = SimulationController.Instance.mResultPath;
    }

    public void OnOKButtonClicked()
    {
        SimulationController.Instance.mBaseDirectory = mInputFieldBaseDirectory.text;
        SimulationController.Instance.mModelPath = mInputFieldModelPath.text;
        SimulationController.Instance.mDataPath = mInputFieldDataDirectory.text;
        SimulationController.Instance.mResultPath= mInputFieldResultDirectory.text;

        PlayerPrefs.SetString(mBaseDirectoryKey, SimulationController.Instance.mBaseDirectory);
        PlayerPrefs.SetString(mModelPathKey, SimulationController.Instance.mModelPath);
        PlayerPrefs.SetString(mDataDirectoryKey, SimulationController.Instance.mDataPath);
        PlayerPrefs.SetString(mResultDirectoryKey, SimulationController.Instance.mResultPath);
        PlayerPrefs.Save();
    }

    public void OnSearchBaseDirectoryButtonClicked()
    {
#if !UNITY_EDITOR_OSX
        try
        {
            using (FolderBrowserDialog folderBrowserDialog = new FolderBrowserDialog())
            {
                folderBrowserDialog.Description = "Select a folder";
                folderBrowserDialog.ShowNewFolderButton = true;

                if (folderBrowserDialog.ShowDialog() == DialogResult.OK)
                {
                    string _selectedPath = folderBrowserDialog.SelectedPath;

                    mInputFieldBaseDirectory.text = _selectedPath;
                    SimulationController.Instance.mBaseDirectory = _selectedPath;
                }
            }
        }
        catch (Exception e)
        {
            SimulationController.Instance.SendError(e.Message);    
        }

#endif
    }

    public void OnSearchModelPathButtonClicked()
    {
#if !UNITY_EDITOR_OSX
        try
        {
            using (OpenFileDialog ofd = new OpenFileDialog())
            {
                ofd.Filter = "PyTorch Model Files (*.pt; *.pth)|*.pt; *.pth";
                ofd.Title = "Select a file";

                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    mInputFieldModelPath.text = ofd.FileName;
                    SimulationController.Instance.mModelPath = ofd.FileName;
                }
            }
        }
        catch (Exception e) 
        {
            SimulationController.Instance.SendError($"{e.Message}");
        }
#endif
    }

    public void OnSearchDataDirectoryButtonClicked()
    {
#if !UNITY_EDITOR_OSX
        try
        {
            using (FolderBrowserDialog folderBrowserDialog = new FolderBrowserDialog())
            {
                folderBrowserDialog.Description = "Select a folder";
                folderBrowserDialog.ShowNewFolderButton = true;

                if (folderBrowserDialog.ShowDialog() == DialogResult.OK)
                {
                    string _selectedPath = folderBrowserDialog.SelectedPath;

                    mInputFieldDataDirectory.text = _selectedPath;
                    SimulationController.Instance.mDataPath = _selectedPath;
                }
            }
        }
        catch (Exception e)
        {
            SimulationController.Instance.SendError(e.Message);
        }
#endif
    }

    public void OnSearchResultDirectoryButtonClicked()
    {
#if !UNITY_EDITOR_OSX
        using (FolderBrowserDialog folderBrowserDialog = new FolderBrowserDialog())
        {
            folderBrowserDialog.Description = "Select a folder";
            folderBrowserDialog.ShowNewFolderButton = true;

            if (folderBrowserDialog.ShowDialog() == DialogResult.OK)
            {
                string _selectedPath = folderBrowserDialog.SelectedPath;

                mInputFieldResultDirectory.text = _selectedPath;
                SimulationController.Instance.mResultPath = _selectedPath;
            }
        }
#endif
    }
}
