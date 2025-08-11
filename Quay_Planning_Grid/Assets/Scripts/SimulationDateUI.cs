using UnityEngine;
using TMPro;
using System.Collections;

public class SimulationDateUI : MonoBehaviour
{
    [SerializeField] private TMP_Text dateText;

    private void OnEnable()
    {
      //  SimulationClock.Instance.OnTimeChanged += UpdateDateDisplay;
    }

    private void OnDisable()
    {
        if (SimulationClock.Instance != null)
            SimulationClock.Instance.OnTimeChanged -= UpdateDateDisplay;
    }

    private void UpdateDateDisplay(int simDay)
    {
        dateText.text = $"Simulation Day: {simDay}";
    }

    private IEnumerator Start()
    {
        yield return new WaitUntil(() => SimulationClock.Instance != null);
        SimulationClock.Instance.OnTimeChanged += UpdateDateDisplay;
    }
}
