using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;

public class ShipData : ScriptableObject
{
    public List<int> Start_Dates = new List<int>();
    public List<int> Finish_Dates = new List<int>();
    public List<string> Operation_Type = new List<string>();
    public List<string> Operation_Name = new List<string>();

    public string Entry_Index;
    public string Ship_Name;
    public int Ship_Index;
    public string Ship_Type;
    public int Category;
    public int Launching_Date;
    public int Delivery_Date;
    public int Operation_Index;
    public int Order;
    public int Duration;
    public string Interruption;
    public int Fixed_Duration;
  
}
