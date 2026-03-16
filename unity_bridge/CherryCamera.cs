using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// Stationary camera controller that simulates the source image camera.
///
/// Position is fixed at initialization — the camera never moves.
/// Left-click and drag to rotate:
///   Horizontal (yaw):  clamped to ±90° from the initial forward direction (180° total sweep)
///   Vertical   (pitch): clamped to ±90° from horizontal                   (180° total sweep)
///
/// This lets the model map objects seen in real images into 3D Unity space
/// from the same vantage point as the original photo camera.
/// </summary>
public class CherryCamera : MonoBehaviour
{
    [Header("Rotation Sensitivity")]
    public float sensitivity = 1.5f;

    // Yaw/pitch relative to the initial forward direction (degrees)
    private float _yaw   = 0f;
    private float _pitch = 0f;

    // World position is locked here forever
    private Vector3 _lockedPosition;

    // Orientation the camera starts at (before any user rotation)
    private Quaternion _baseRotation;

    void Start()
    {
        _lockedPosition = transform.position;
        _baseRotation   = transform.rotation;
    }

    void Update()
    {
        // Enforce fixed position every frame
        transform.position = _lockedPosition;

        // Rotate only while the left mouse button is held
        var mouse = Mouse.current;
        if (mouse == null || !mouse.leftButton.isPressed) return;

        var delta = mouse.delta.ReadValue();
        _yaw   += delta.x * sensitivity;
        _pitch -= delta.y * sensitivity;

        // Clamp both axes to ±90° (180° total sweep each)
        _yaw   = Mathf.Clamp(_yaw,   -90f, 90f);
        _pitch = Mathf.Clamp(_pitch, -90f, 90f);

        // Apply rotation relative to the camera's initial orientation
        transform.rotation = _baseRotation * Quaternion.Euler(_pitch, _yaw, 0f);
    }

    /// <summary>
    /// Resets the camera back to its initial orientation (yaw=0, pitch=0).
    /// </summary>
    public void ResetOrientation()
    {
        _yaw   = 0f;
        _pitch = 0f;
        transform.rotation = _baseRotation;
    }
}
