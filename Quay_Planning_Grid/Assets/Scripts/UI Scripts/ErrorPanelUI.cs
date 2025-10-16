using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class ErrorPanelUI : MonoBehaviour
{
    [SerializeField] private TMP_InputField mInputField;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void clearMessage()
    {
        mInputField.text = string.Empty;
    }

    public void setMessage(string error)
    {
        mInputField.text = $"{error}";
    }
}
