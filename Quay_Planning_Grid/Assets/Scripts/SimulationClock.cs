using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimulationClock : MonoBehaviour
{
    public float realSecondsPerSimDay = 0.5f; // 0.5 real second = 1 sim day
    public int simulationTime { get; private set; }

    public float _startTime;
    private int _lastReportedTime;

    public static SimulationClock Instance { get; private set; }

    //public System.DateTime CurrentDate { get; private set; }

    public delegate void TimeChanged(int newTime);
    public event TimeChanged OnTimeChanged;
    public bool simulationStarted = false; // Add this

    //Day 0 ??Time = 0s
    //Day 1 ??Time = 0.5s
    //Day 2 ??Time = 1.0s
    //Day 10 ??Time = 5.0s
    void Start()
    {
        _startTime = 0;
        simulationStarted = false;
    }
    void Awake()
    {
        if (Instance == null) Instance = this;
        _startTime = Time.time - 0.45f;
    }

    // (Time.time - (Time.time - 1.0f)) = 1.0f
    // 1.0 / 0.5 = 2.0 ??FloorToInt ??2

    void Update()
    {
        if (!simulationStarted) return;

        int newSimTime = Mathf.FloorToInt((Time.time - _startTime) / realSecondsPerSimDay);

        if (newSimTime != _lastReportedTime)
        {
            simulationTime = newSimTime;
            _lastReportedTime = newSimTime;
            OnTimeChanged?.Invoke(simulationTime);

        }
    }
    public void ResetTime()
    {
        simulationStarted = false;
        _startTime = 0f;
        simulationTime = 0;
        OnTimeChanged?.Invoke(0);
        Debug.Log("SimulationClock reset.");
    }

}