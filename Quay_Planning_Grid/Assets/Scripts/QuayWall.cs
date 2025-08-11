//using UnityEngine;

//public class QuayWall : MonoBehaviour
//{
//    public string quayName;
//    public bool isEngaged;
//    public string currentShipID;
//    public string currentOperation;
//    public float timeLeft;

//    public void UpdateStatus(float currentTime)
//    {
//        var log = ShipRuntime.Instance.GetQuayStatusAtTime(quayName, currentTime);
//        if (log != null)
//        {
//            isEngaged = true;
//            currentShipID = log.shipID;
//            currentOperation = log.operation;
//            timeLeft = log.endTime - currentTime;
//        }
//        else
//        {
//            isEngaged = false;
//            currentShipID = "";
//            currentOperation = "";
//            timeLeft = 0;
//        }
//    }
//}