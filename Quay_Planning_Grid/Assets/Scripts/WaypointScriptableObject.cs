using UnityEngine;

public enum DockDirection { None, Horizontal, Vertical }
public enum DockSide { Negative = -1, Positive = 1 }

[CreateAssetMenu(menuName = "Waypoint")]
public class WaypointScriptableObject : ScriptableObject
{
    public string waypointName;
    //public Vector3 RLPosition;
    //public Vector3 SelectedPosition;
    public DockDirection dockDirection;
    public DockSide dockSide = DockSide.Negative;
    public float dockDistance = 0.1f; // How far to move along the wall
    public Vector3 forward;
    public float preDockDistance = 1f;
    public int steps = 4;

    [Header("Optional custom docking angle")]
    public bool useDockAngle = false;
    [Range(-180f, 180f)]
    public float dockAngle = 0f;
}
