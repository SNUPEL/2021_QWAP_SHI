using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using ChartAndGraph;
using System;
using ExcelDataReader;
using System.IO;
using System.Data;

public class Chart
{
    public float rl;
    public float sptmf;
    public float mormf;
    public float mwkrmf;
}

public class ChartController : MonoBehaviour
{
    public GraphChart[] charts;

    private string rl = "RL";
    private string sptmf = "SPT MF";
    private string mormf = "MOR MF";
    private string mwkrmf = "MWKR MF";

    private float Timer = 1f;
    private int X = 0;

    private Dictionary<int, Chart> MoveCost = new Dictionary<int, Chart>();
    private Dictionary<int, Chart> DelayCost = new Dictionary<int, Chart>();
    private Dictionary<int, Chart> PreferenceCost = new Dictionary<int, Chart>();
    private Dictionary<int, Chart> TotalCost = new Dictionary<int, Chart>();

    private int index_RL = 3;
    private int index_SPTMF = 4;
    private int index_MORMF = 1;
    private int index_MWKRMF = 2;
    private bool mStarted = false;

    private void Start()
    {
        StartBatch();
        ClearCategory();
        EndBatch();

        if (SimulationClock.Instance != null)
        {
            SimulationClock.Instance.OnTimeChanged += ChartChanged;
        }

    }

    public void RunChart(string baseDir)
    {
        mStarted = true;
        X = 0;
        string filePath_DelayLog = Path.Combine(baseDir, "output/DelayLog.xlsx");
        string filePath_CostLog = Path.Combine(baseDir, "output/CostLog.xlsx");
        string filePath_PriorityLogReverse = Path.Combine(baseDir, "output/PriorityLogReverse.xlsx");
        string filePath_MoveLog = Path.Combine(baseDir, "output/MoveLog.xlsx");

        MoveCost = ReadExcel(filePath_DelayLog);
        DelayCost = ReadExcel(filePath_CostLog);
        PreferenceCost = ReadExcel(filePath_PriorityLogReverse);
        TotalCost = ReadExcel(filePath_MoveLog);
    }

    private Dictionary<int, Chart> ReadExcel(string filePath)
    {
        Dictionary<int, Chart> cost = new Dictionary<int, Chart>();
        using (var stream = File.Open(filePath, FileMode.Open, FileAccess.Read))
        {
            using (var reader = ExcelReaderFactory.CreateReader(stream))
            {
                var result = reader.AsDataSet();
                DataTable table = result.Tables[0];

                for (int i = 1; i < table.Rows.Count; i++)
                {
                    Chart chart = new Chart();
                    chart.rl = float.Parse(table.Rows[i][index_RL].ToString());
                    chart.sptmf = float.Parse(table.Rows[i][index_SPTMF].ToString());
                    chart.mormf = float.Parse(table.Rows[i][index_MORMF].ToString());
                    chart.mwkrmf = float.Parse(table.Rows[i][index_MWKRMF].ToString());
                    cost.Add(int.Parse(table.Rows[i][0].ToString()), chart);
                }
            }
        }
        return cost;
    }

    private void ChartChanged(int newTime)
    {
        X = newTime;
        if ( newTime % 10 != 0)
        {
            return;
        }
        if (!SimulationClock.Instance.simulationStarted)
        {
            return;
        }

        if (X >= MoveCost.Count || X >= DelayCost.Count || X >= PreferenceCost.Count || X >= TotalCost.Count)
        {
            X = 0;
            ClearCategory();
        }

        charts[0].DataSource.AddPointToCategoryRealtime(rl, X, MoveCost[X].rl, 1f);
        charts[0].DataSource.AddPointToCategoryRealtime(sptmf, X, MoveCost[X].sptmf, 1f);
        charts[0].DataSource.AddPointToCategoryRealtime(mormf, X, MoveCost[X].mormf, 1f);
        charts[0].DataSource.AddPointToCategoryRealtime(mwkrmf, X, MoveCost[X].mwkrmf, 1f);
        charts[0].HorizontalValueToStringMap[X] = X.ToString();

        charts[1].DataSource.AddPointToCategoryRealtime(rl, X, DelayCost[X].rl, 1f);
        charts[1].DataSource.AddPointToCategoryRealtime(sptmf, X, DelayCost[X].sptmf, 1f);
        charts[1].DataSource.AddPointToCategoryRealtime(mormf, X, DelayCost[X].mormf, 1f);
        charts[1].DataSource.AddPointToCategoryRealtime(mwkrmf, X, DelayCost[X].mwkrmf, 1f);
        charts[1].HorizontalValueToStringMap[X] = X.ToString();

        charts[2].DataSource.AddPointToCategoryRealtime(rl, X, PreferenceCost[X].rl, 1f);
        charts[2].DataSource.AddPointToCategoryRealtime(sptmf, X, PreferenceCost[X].sptmf, 1f);
        charts[2].DataSource.AddPointToCategoryRealtime(mormf, X, PreferenceCost[X].mormf, 1f);
        charts[2].DataSource.AddPointToCategoryRealtime(mwkrmf, X, PreferenceCost[X].mwkrmf, 1f);
        charts[2].HorizontalValueToStringMap[X] = X.ToString();

        charts[3].DataSource.AddPointToCategoryRealtime(rl, X, TotalCost[X].rl, 1f);
        charts[3].DataSource.AddPointToCategoryRealtime(sptmf, X, TotalCost[X].sptmf, 1f);
        charts[3].DataSource.AddPointToCategoryRealtime(mormf, X, TotalCost[X].mormf, 1f);
        charts[3].DataSource.AddPointToCategoryRealtime(mwkrmf, X, TotalCost[X].mwkrmf, 1f);
        charts[3].HorizontalValueToStringMap[X] = X.ToString();
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

    public void ClearCategory()
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
        if (!mStarted) return;
        Timer -= Time.deltaTime;
        if (Timer <= 0f)
        {
            Timer = 1f;
            
            //X++;
        }
    }
}
