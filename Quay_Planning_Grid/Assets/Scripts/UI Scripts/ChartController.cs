using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using ChartAndGraph;
using System;

public class ChartController : MonoBehaviour
{
    public GraphChart[] charts;
    //public GraphChart[] chartMove;
    //public GraphChart chartDelay;
    //public GraphChart chartPreference;
    //public GraphChart chartTotalCost;

    private string rl = "RL";
    private string sptmf = "SPT MF";
    private string mormf = "MOR MF";
    private string mwkrmf = "MWKR MF";

    private float Timer = 1f;
    private float X = 4f;

    // Start is called before the first frame update
    void Start()
    {
        StartBatch();
        ClearCategory();
        EndBatch();

        if (SimulationClock.Instance != null)
        {
            SimulationClock.Instance.OnTimeChanged += ChartChanged;
        }
    }

    private void ChartChanged(int newTime)
    {
        if ( newTime % 10 != 0)
        {
            return;
        }
        if (!SimulationClock.Instance.simulationStarted)
        {
            return;
        }
        foreach (var chart in charts)
        {
            chart.DataSource.AddPointToCategoryRealtime(rl, X, UnityEngine.Random.value, 1f);
            chart.DataSource.AddPointToCategoryRealtime(sptmf, X, UnityEngine.Random.value, 1f);
            chart.DataSource.AddPointToCategoryRealtime(mormf, X, UnityEngine.Random.value, 1f);
            chart.DataSource.AddPointToCategoryRealtime(mwkrmf, X, UnityEngine.Random.value, 1f);
        }
    }

    private void StartBatch()
    {
        foreach (var chart in charts)
            chart.DataSource.StartBatch();
        
    }

    private void EndBatch()
    { 
        foreach (var chart in charts)
            chart.DataSource.EndBatch();
    }

    private void ClearCategory()
    {
        foreach(var chart in charts)
        {
            chart.DataSource.ClearCategory(rl);
            chart.DataSource.ClearCategory(sptmf);
            chart.DataSource.ClearCategory(mormf);
            chart.DataSource.ClearCategory(mwkrmf);
        }
    }



    // Update is called once per frame
    void Update()
    {
        Timer -= Time.deltaTime;
        if (Timer <= 0f)
        {
            Timer = 1f;
            
            X++;
        }
    }
}
