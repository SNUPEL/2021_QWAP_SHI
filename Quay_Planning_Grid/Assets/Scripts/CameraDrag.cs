using UnityEngine;

[RequireComponent(typeof(Camera))]
public class CameraDrag : MonoBehaviour
{
    [Header("Drag Settings")]
    public float dragSpeed = 0.5f; // Increase for faster movement
    private Vector3 dragOrigin;

    [Header("Zoom Settings")]
    public float zoomSpeed = 50f; // Increase for faster zoom
    public float minZoom = 10f;
    public float maxZoom = 80f;

    private Camera cam;

    void Awake()
    {
        cam = GetComponent<Camera>();
    }

    void Update()
    {
        HandleDrag();
        HandleZoom();
    }

    void HandleDrag()
    {
        if (Input.GetMouseButtonDown(0))
        {
            dragOrigin = Input.mousePosition;
            return;
        }

        if (!Input.GetMouseButton(0))
            return;

        Vector3 difference = Input.mousePosition - dragOrigin;
        dragOrigin = Input.mousePosition;

        // Move opposite to drag direction for natural feel
        Vector3 move = new Vector3(-difference.x, 0, -difference.y) * dragSpeed * Time.deltaTime;
        transform.Translate(move, Space.World);
    }

    void HandleZoom()
    {
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        if (Mathf.Abs(scroll) < 0.001f)
            return;

        if (!cam.orthographic)
        {
            cam.fieldOfView -= scroll * zoomSpeed * Time.deltaTime;
            cam.fieldOfView = Mathf.Clamp(cam.fieldOfView, minZoom, maxZoom);
        }
        else
        {
            cam.orthographicSize -= scroll * (zoomSpeed / 100f) * Time.deltaTime;
            cam.orthographicSize = Mathf.Clamp(cam.orthographicSize, minZoom / 10f, maxZoom / 10f);
        }
    }
}