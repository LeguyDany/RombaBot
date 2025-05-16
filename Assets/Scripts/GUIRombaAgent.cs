using UnityEngine;

public class GUIRombaAgent : MonoBehaviour
{
    [SerializeField] private RombaAgent _rombaAgent;

    private GUIStyle _defaultStyle = new GUIStyle();
    private GUIStyle _positiveStyle = new GUIStyle();
    
    private GUIStyle _negativeStyle = new GUIStyle();

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {
        _defaultStyle.fontSize = 20;
        _defaultStyle.normal.textColor = Color.white;

        _positiveStyle.fontSize = 20;
        _positiveStyle.normal.textColor = Color.green;

        _negativeStyle.fontSize = 20;
        _negativeStyle.normal.textColor = Color.red;
    }

    private void OnGUI()
    {
        string debugEpisode = "Episode: " + _rombaAgent.CurrentEpisde + "- Step: " + _rombaAgent.StepCount;
        string debugReward = "Reward: " + _rombaAgent.CumulativeReward.ToString();

        GUIStyle rewardStyle = _rombaAgent.CumulativeReward > 0 ? _positiveStyle : _negativeStyle;

        GUI.Label(new Rect(10, 10, 300, 20), debugEpisode, _defaultStyle);
        GUI.Label(new Rect(10, 30, 300, 20), debugReward, rewardStyle);
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
