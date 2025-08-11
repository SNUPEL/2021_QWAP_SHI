//using UnityEditor;
//using UnityEngine;

//[CustomEditor(typeof(WPManagerScriptableObject))]
//public class WPManagerValidator : Editor
//{
//    public override void OnInspectorGUI()
//    {
//        DrawDefaultInspector();

//        if (GUILayout.Button("Trim Waypoint Names"))
//        {
//            WPManagerScriptableObject wpManager = (WPManagerScriptableObject)target;

//            for (int i = 0; i < wpManager.waypoints.Length; i++)
//            {
//                GameObject wp = wpManager.waypoints[i];
//                if (wp != null)
//                {
//                    string originalName = wp.name;
//                    wp.name = wp.name.Trim();

//                    if (originalName != wp.name)
//                    {
//                        Debug.Log($"Trimmed waypoint name: '{originalName}' ➜ '{wp.name}'");
//                        EditorUtility.SetDirty(wp);
//                    }
//                }
//            }

//            EditorUtility.SetDirty(wpManager);
//            AssetDatabase.SaveAssets();
//        }
//    }
//}