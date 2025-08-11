//using UnityEngine;
//using System.Collections.Generic;

//public class QuayBuilder : MonoBehaviour
//{
//    public GameObject quayPrefab; // Simple cube or mesh with a collider
//    public int numberOfQuays = 28;
//    public Transform parent; // Assign in inspector to organize in hierarchy

//    public static Dictionary<string, GameObject> QuayLookup = new Dictionary<string, GameObject>();

//    void Start()
//    {
//        BuildQuays();
//    }

//    void BuildQuays()
//    {
//        QuayLookup.Clear();

//        for (int i = 0; i < numberOfQuays; i++)
//        {
//            string quayId = $"Q{i + 1:00}"; // e.g., Q01, Q02...

//            Vector3 position = new Vector3(i * 3, 0, 0); // customize layout
//            GameObject quay = Instantiate(quayPrefab, position, Quaternion.identity, parent);
//            quay.name = quayId;

//            // Add identifier
//            quay.AddComponent<QuayIdentifier>().quayId = quayId;

//            QuayLookup[quayId] = quay;
//        }
//    }
//}