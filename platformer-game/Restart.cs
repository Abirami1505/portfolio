using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Restart : MonoBehaviour
{
    public PopUpText popup;
    private Vector3 initialPos;
    private int lives;
    
    void Start()
    {
        initialPos = transform.position;
        lives = 3;
        popup=FindObjectOfType<PopUpText>();
        
    }

    // Update is called once per frame
    void Update()
    {
        if (transform.position.y < -100)
        {
            lives -= 1;
            if (lives > 0)
            {
                transform.position = initialPos;
            }
            else
            {
                popup.ShowPopup("Game Over");
            }
        }
    }

    
}
