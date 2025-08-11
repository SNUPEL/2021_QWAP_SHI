using System.Collections;
using UnityEngine;
using TMPro; // or use UnityEngine.UI for legacy Text

public class SimulationCountdownUI : MonoBehaviour
{
    public TextMeshProUGUI countdownText; // Assign in Inspector
    public float countdownDuration = 2f;
    public GameObject panel;

    public void StartCountdown()
    {
        panel.SetActive(true); // Enable UI
        StartCoroutine(CountdownRoutine());
    }

    public IEnumerator CountdownRoutine()
    {
        int seconds = Mathf.CeilToInt(countdownDuration);

        while (seconds > 0)
        {
            countdownText.text = $"Simulation starting in {seconds}...";
            yield return new WaitForSecondsRealtime(1f); // Use real time in case Time.timeScale is 0
            seconds--;
        }

        countdownText.text = "Starting!";
        yield return new WaitForSecondsRealtime(1f);

        countdownText.text = "";
        panel.SetActive(false);
    }
}