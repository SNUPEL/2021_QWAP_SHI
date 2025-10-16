


using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;

public class PostBuildCopy : IPostprocessBuildWithReport 
{
    public int callbackOrder => 0;

    public void OnPostprocessBuild(BuildReport report)
    {
        string buildDir = Path.GetDirectoryName(report.summary.outputPath);
        string[] _requirements = { "input", "output", "model", "configuration" };
        foreach(string _requirement in _requirements)
        {
            string _targetDir = Path.Combine(buildDir, _requirement);
            Directory.CreateDirectory(_targetDir);
        }
        string _modelSavePath = Path.Combine(buildDir, _requirements[2], Path.GetFileName(SimulationController.Instance.mModelPath));
        if (File.Exists(SimulationController.Instance.mModelPath) && !File.Exists(_modelSavePath))
            File.Copy(SimulationController.Instance.mModelPath, _modelSavePath);

        string _configSavePath = Path.Combine(buildDir, _requirements[3], Path.GetFileName(SimulationController.Instance.mConfigPath));
        if (File.Exists(SimulationController.Instance.mConfigPath) && !File.Exists(_configSavePath))
            File.Copy(SimulationController.Instance.mConfigPath, _configSavePath);
    }

}
