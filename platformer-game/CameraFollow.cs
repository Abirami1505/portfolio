using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraFollow : MonoBehaviour
{
    public Transform player;
    public Vector3 offset;
    private void LateUpdate()
    {
        transform.position = player.position + offset;
        if (player.position.y < 0)
        {
            transform.position-=new Vector3(0f,player.position.y,0f);
        }
    }
}
