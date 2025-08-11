using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.AI;
using System;

public class RL_Controller : MonoBehaviour
{
    GameObject currentNode;
    public NavMeshAgent agent;
    public string currentTarget = "";
    int shipsdelivered = 0;
    private bool dockingInProgress = false;
    private bool isMoving;
    private List<Vector3> recordedPath = new List<Vector3>();

    public void MoveTo(string locationName)
    {
        if (string.IsNullOrWhiteSpace(locationName)) return;

        if (agent == null) agent = GetComponent<NavMeshAgent>();
        string cleanName = locationName.Trim();
        currentTarget = cleanName;

        GameObject target = RLWP_Manager.Instance.GetWaypoint(cleanName);

        if (target == null)
        {
            Debug.LogWarning($"{gameObject.name}: No RL waypoint found named '{cleanName}'");
            return;
        }

        if (NavMesh.SamplePosition(target.transform.position, out var hit, 1f, NavMesh.AllAreas))
        {
            isMoving = true;
            agent.SetDestination(hit.position);
            StartCoroutine(CheckPathReady());
        }
        else
        {
            Debug.LogWarning($"{gameObject.name}: RL Waypoint '{cleanName}' is unreachable via NavMesh.");
        }
    }

    IEnumerator CheckPathReady()
    {
        while (agent.pathPending)
            yield return null;

        if (agent.pathStatus != NavMeshPathStatus.PathComplete)
        {
            Debug.LogWarning($"{gameObject.name}: RL Path not complete ({agent.pathStatus}).");
        }
    }

    void Awake()
    {
        if (agent == null)
            agent = GetComponent<NavMeshAgent>();
    }

    void Start()
    {
        Time.timeScale = 20f;
    }

    void Update()
    {
        if (agent == null || !agent.isOnNavMesh || string.IsNullOrEmpty(currentTarget)) return;

        if (!agent.pathPending && agent.remainingDistance <= agent.stoppingDistance)
        {
            if (!agent.hasPath || agent.velocity.sqrMagnitude == 0f)
            {
                recordedPath.Add(transform.position);
                isMoving = false;

                WaypointScriptableObject data = RLWP_Manager.Instance.GetWaypointSO(currentTarget);

                if (data != null)
                {
                    if (currentTarget.Equals("Sink", StringComparison.OrdinalIgnoreCase))
                    {
                        Debug.Log($"{gameObject.name} (RL) arrived at Sink. Destroying.");
                        Destroy(gameObject);
                        shipsdelivered++;
                        return;
                    }

                    if (!dockingInProgress)
                        StartCoroutine(PerformDockingManeuver(data));
                }
                else if (!isMoving)
                {
                    MoveTo(currentTarget);
                }
            }
        }
    }

    IEnumerator PerformDockingManeuver(WaypointScriptableObject data)
    {
        dockingInProgress = true;
        Vector3 slideDir = Vector3.zero;
        Quaternion targetRot = transform.rotation;
        int side = (int)data.dockSide;

        switch (data.dockDirection)
        {
            case DockDirection.Horizontal: slideDir = Vector3.right * side; break;
            case DockDirection.Vertical: slideDir = Vector3.forward * side; break;
            case DockDirection.None: yield break;
        }

        if (data.useDockAngle)
            targetRot = Quaternion.Euler(0f, data.dockAngle, 0f);
        else
            targetRot = Quaternion.LookRotation(slideDir == Vector3.right * side ? Vector3.forward : Vector3.right);

        Quaternion startRot = transform.rotation;
        float rotDur = 3f;
        for (float t = 0f; t < rotDur; t += Time.deltaTime)
        {
            transform.rotation = Quaternion.Slerp(startRot, targetRot, t / rotDur);
            yield return null;
        }
        transform.rotation = targetRot;

        Vector3 start = transform.position;
        Vector3 end = start + slideDir.normalized * data.dockDistance;
        float slideDur = 0.5f;
        for (float t = 0f; t < slideDur; t += Time.deltaTime)
        {
            transform.position = Vector3.Lerp(start, end, t / slideDur);
            yield return null;
        }
        transform.position = end;
        dockingInProgress = false;
    }
}