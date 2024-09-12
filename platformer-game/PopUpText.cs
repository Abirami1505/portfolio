using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class PopUpText : MonoBehaviour
{
    public GameObject popupPanel;
    public TMP_Text popupText;

    // Start is called before the first frame update
    void Start()
    {
        popupPanel.SetActive(false);
    }

    public void ShowPopup(string message)
    {
        popupText.text = message;
        popupPanel.SetActive(true);
    }
}
