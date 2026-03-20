using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// CherryUnityBridge — HTTP server that receives commands from Python to place objects in a 3D scene.
///
/// Attach this script to any GameObject in your Unity scene.
///
/// Coordinate System:
///   X axis: Left (−) / Right (+)
///   Y axis: Ground (0) / Up (+)  — Y=0 is floor level; sphere on ground at Y=0.5
///   Z axis: Near (0) / Far (+)   — Z=0 is at the camera; Z=20 is far background
///
/// Scene bounds: X ∈ [−10, 10], Y ∈ [0, 10], Z ∈ [0, 20]
///
/// HTTP endpoints (localhost:5555):
///   GET  /health         — health check
///   POST /place_object   — place a labeled sphere {"label","x","y","z","color","scale"}
///   POST /remove_object  — remove object nearest to {"x","y","z"}
///   POST /move_object    — move object nearest to {"x","y","z","new_x","new_y","new_z"}
///   POST /clear_scene    — remove all placed spheres
///   GET  /scene_state    — list all placed objects
///   POST /initialize     — clear scene
/// </summary>
public class CherryUnityBridge : MonoBehaviour
{
    [Header("Server Settings")]
    public int port = 5555;

    [Header("Scene Settings")]
    public float sceneSize  = 20f;
    public float sphereRadius = 0.5f;

    // ── HTTP server ──────────────────────────────────────────────────────────
    private HttpListener _listener;
    private Thread       _listenerThread;
    private volatile bool _running = false;

    private readonly Queue<PendingCommand> _commandQueue = new Queue<PendingCommand>();
    private readonly object                _queueLock    = new object();

    // ── Scene state ──────────────────────────────────────────────────────────
    private GameObject _objectsRoot;

    [Serializable]
    private class PlacedObjectRecord
    {
        public int id; public string label;
        public float x, y, z, scale; public string colorHex;
    }
    private readonly List<PlacedObjectRecord> _records = new List<PlacedObjectRecord>();

    private class PendingCommand
    {
        public string Type, Body, Result;
        public readonly ManualResetEventSlim Signal = new ManualResetEventSlim(false);
    }

    private static readonly Color[] Palette = {
        new Color(0.90f,0.20f,0.20f), new Color(0.20f,0.50f,0.90f),
        new Color(0.20f,0.80f,0.25f), new Color(0.95f,0.75f,0.10f),
        new Color(0.70f,0.20f,0.90f), new Color(0.95f,0.50f,0.10f),
        new Color(0.10f,0.80f,0.80f), new Color(0.90f,0.20f,0.70f),
        new Color(0.60f,0.40f,0.20f), new Color(0.50f,0.90f,0.50f),
    };

    // ════════════════════════════════════════════════════════════════════════
    // Unity lifecycle
    // ════════════════════════════════════════════════════════════════════════

    // Only one instance may run the server at a time
    private static CherryUnityBridge _activeInstance;

    void Start()
    {
        if (_activeInstance != null && _activeInstance != this)
        {
            Debug.LogWarning($"[CherryBridge] Duplicate instance on '{gameObject.name}' — disabled. Keep only one Bridge GameObject in the scene.");
            enabled = false;
            return;
        }
        _activeInstance = this;

        // Keep running even when the Unity editor window loses focus
        // (required so Update() processes HTTP commands while Python runs in another window)
        Application.runInBackground = true;

        // Create the objects container immediately so place_object works at once
        _objectsRoot = new GameObject("PlacedObjects");

        // Start the HTTP server before anything else so Update() is free to respond
        StartServer();

        // Build the grid/camera in a coroutine so it doesn't block the first frame
        StartCoroutine(BuildSceneAsync());
    }

    void Update()
    {
        PendingCommand cmd = null;
        lock (_queueLock)
        {
            if (_commandQueue.Count > 0)
                cmd = _commandQueue.Dequeue();
        }
        if (cmd == null) return;

        try   { cmd.Result = DispatchCommand(cmd.Type, cmd.Body); }
        catch (Exception ex) { cmd.Result = JsonError(ex.Message); }
        finally { cmd.Signal.Set(); }
    }

    void OnDestroy()
    {
        if (_activeInstance == this) _activeInstance = null;
        StopServer();
    }
    void OnApplicationQuit() => StopServer();

    private void StopServer()
    {
        _running = false;
        try { _listener?.Stop();  } catch { }
        try { _listener?.Close(); } catch { }
        _listenerThread?.Join(1000);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Scene setup (runs after first frame so HTTP is already responsive)
    // ════════════════════════════════════════════════════════════════════════

    private IEnumerator BuildSceneAsync()
    {
        yield return null; // let Update() run at least once first
        SetupCamera();
        BuildFloor();
        BuildCornerLight();
        yield return StartCoroutine(BuildGridAsync());
        Debug.Log("[CherryBridge] Scene ready.");
    }

    private void BuildCornerLight()
    {
        var lightGo = new GameObject("CornerLight");
        // Positioned at top-left-far corner, angled down toward the scene centre
        lightGo.transform.position = new Vector3(-10f, 10f, 20f);
        lightGo.transform.rotation = Quaternion.LookRotation(
            new Vector3(10f, -10f, -10f).normalized); // points toward scene centre
        var light = lightGo.AddComponent<Light>();
        light.type      = LightType.Directional;
        light.color     = new Color(1f, 0.95f, 0.85f); // warm white
        light.intensity = 1.2f;
    }

    private void BuildFloor()
    {
        // Solid semi-transparent plane covering the 20×20 scene at Y=0.
        // Unity's Plane primitive is 10 units wide at scale 1, so scale (2,1,2) → 20×20.
        // Centre: X=0, Z=sceneSize/2 so it spans Z=[0,20].
        var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "Floor";
        floor.transform.position   = new Vector3(0f, 0f, sceneSize / 2f);
        floor.transform.localScale = new Vector3(sceneSize / 10f, 1f, sceneSize / 10f);
        floor.GetComponent<Renderer>().material =
            MakeTransparentMaterial(new Color(0.92f, 0.92f, 0.95f, 0.75f));
        Destroy(floor.GetComponent<Collider>());
    }

    private IEnumerator BuildGridAsync()
    {
        var gridRoot = new GameObject("Grid");
        var mat      = MakeTransparentMaterial(new Color(1f, 1f, 1f, 0f));

        const float step = 2f;
        float halfXZ = sceneSize / 2f;
        float halfY  = Mathf.Ceil(sceneSize / 4f / step) * step;
        int xSteps   = Mathf.RoundToInt(halfXZ / step);
        int zCount   = Mathf.RoundToInt(sceneSize / step);
        int ySteps   = Mathf.RoundToInt(halfY / step);

        // Yield after each Y-level so Update() stays responsive during grid construction
        for (int jj = -ySteps; jj <= ySteps; jj++)
        {
            float y = jj * step;
            for (int ii = -xSteps; ii <= xSteps; ii++)
                SpawnGridLine(gridRoot, mat,
                    new Vector3(ii * step, y, halfXZ),
                    new Vector3(0.02f, 0.02f, sceneSize));
            for (int kk = 0; kk <= zCount; kk++)
                SpawnGridLine(gridRoot, mat,
                    new Vector3(0f, y, kk * step),
                    new Vector3(sceneSize, 0.02f, 0.02f));
            yield return null;
        }
        for (int ii = -xSteps; ii <= xSteps; ii++)
        {
            for (int kk = 0; kk <= zCount; kk++)
                SpawnGridLine(gridRoot, mat,
                    new Vector3(ii * step, 0f, kk * step),
                    new Vector3(0.02f, halfY * 2f, 0.02f));
            yield return null;
        }
    }

    private void SetupCamera()
    {
        foreach (var c in Camera.allCameras) Destroy(c.gameObject);

        var camGo = new GameObject("CherryCamera");
        camGo.transform.position = new Vector3(0f, 5f, 0f);
        camGo.transform.rotation = Quaternion.identity;
        var cam = camGo.AddComponent<Camera>();
        cam.backgroundColor = new Color(0.12f, 0.12f, 0.17f);
        cam.clearFlags  = CameraClearFlags.SolidColor;
        cam.farClipPlane = 200f;
        cam.fieldOfView  = 60f;
        camGo.tag = "MainCamera";
        camGo.AddComponent<CherryCamera>();
    }

    private Material MakeTransparentMaterial(Color color)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Lit")
                     ?? Shader.Find("HDRP/Lit")
                     ?? Shader.Find("Standard")
                     ?? Shader.Find("Sprites/Default");
        if (shader == null) return new Material(Shader.Find("Hidden/InternalErrorShader"));
        var mat = new Material(shader);
        mat.color = color;
        if (mat.HasProperty("_Surface"))
        {
            mat.SetFloat("_Surface", 1f); mat.SetFloat("_Blend", 0f);
            mat.SetFloat("_SrcBlend", 5f); mat.SetFloat("_DstBlend", 10f);
            mat.SetFloat("_ZWrite", 0f);
            mat.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
        }
        else if (mat.HasProperty("_Mode"))
        {
            mat.SetFloat("_Mode", 3f); mat.SetFloat("_SrcBlend", 5f);
            mat.SetFloat("_DstBlend", 10f); mat.SetFloat("_ZWrite", 0f);
            mat.DisableKeyword("_ALPHATEST_ON"); mat.DisableKeyword("_ALPHAPREMULTIPLY_ON");
            mat.EnableKeyword("_ALPHABLEND_ON");
            mat.renderQueue = (int)UnityEngine.Rendering.RenderQueue.Transparent;
        }
        return mat;
    }

    private void SpawnGridLine(GameObject parent, Material mat, Vector3 pos, Vector3 scale)
    {
        var line = GameObject.CreatePrimitive(PrimitiveType.Cube);
        line.transform.SetParent(parent.transform);
        line.transform.position   = pos;
        line.transform.localScale = scale;
        line.GetComponent<Renderer>().material = mat;
        Destroy(line.GetComponent<Collider>());
    }

    // ════════════════════════════════════════════════════════════════════════
    // HTTP server
    // ════════════════════════════════════════════════════════════════════════

    private void StartServer()
    {
        _listener = new HttpListener();
        _listener.Prefixes.Add($"http://localhost:{port}/");
        try { _listener.Start(); }
        catch (HttpListenerException ex)
        {
            Debug.LogError($"[CherryBridge] Port {port} in use: {ex.Message}. Stop Play, wait a moment, try again.");
            return;
        }
        _running = true;
        _listenerThread = new Thread(ListenLoop) { IsBackground = true };
        _listenerThread.Start();
        Debug.Log($"[CherryBridge] HTTP bridge listening on http://localhost:{port}/");
    }

    private void ListenLoop()
    {
        while (_running)
        {
            try
            {
                var ctx = _listener.GetContext();
                ThreadPool.QueueUserWorkItem(_ => HandleRequest(ctx));
            }
            catch (HttpListenerException) { break; }
            catch (Exception ex) { Debug.LogError($"[CherryBridge] {ex.Message}"); }
        }
    }

    private void HandleRequest(HttpListenerContext ctx)
    {
        string body = "";
        if (ctx.Request.HasEntityBody)
            using (var r = new System.IO.StreamReader(ctx.Request.InputStream, ctx.Request.ContentEncoding))
                body = r.ReadToEnd();

        string path = ctx.Request.Url.AbsolutePath.TrimStart('/');
        var cmd = new PendingCommand { Type = path, Body = body };
        lock (_queueLock) { _commandQueue.Enqueue(cmd); }

        if (!cmd.Signal.Wait(15000))
        {
            SendJson(ctx.Response, 504, JsonError("timeout"));
            return;
        }
        SendJson(ctx.Response, 200, cmd.Result);
    }

    private static void SendJson(HttpListenerResponse resp, int status, string body)
    {
        try
        {
            resp.StatusCode    = status;
            resp.ContentType   = "application/json; charset=utf-8";
            byte[] bytes       = Encoding.UTF8.GetBytes(body);
            resp.ContentLength64 = bytes.Length;
            resp.OutputStream.Write(bytes, 0, bytes.Length);
        }
        finally { resp.Close(); }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Command dispatch
    // ════════════════════════════════════════════════════════════════════════

    private string DispatchCommand(string type, string body)
    {
        switch (type.ToLowerInvariant())
        {
            case "health":         return Health();
            case "place_object":   return PlaceObject(body);
            case "remove_object":  return RemoveObject(body);
            case "move_object":    return MoveObject(body);
            case "clear_scene":    return ClearScene();
            case "scene_state":    return GetSceneState();
            case "initialize":     ClearScene(); return "{\"status\":\"reinitialized\"}";
            case "capture_view":   return CaptureView();
            case "rotate_camera":  return RotateCamera(body);
            case "reset_camera":   return ResetCamera();
            default:               return JsonError($"unknown endpoint: {type}");
        }
    }

    private string Health() =>
        $"{{\"status\":\"ok\",\"objects_placed\":{_records.Count}}}";

    private string PlaceObject(string body)
    {
        var req = JsonUtility.FromJson<PlaceObjectRequest>(body);
        if (req == null) return JsonError("invalid JSON");

        string label = string.IsNullOrEmpty(req.label) ? "object" : req.label;
        float scale  = req.scale <= 0f ? 1f : req.scale;
        bool isCube  = !string.IsNullOrEmpty(req.shape) && req.shape.ToLowerInvariant() == "cube";

        var sphere = GameObject.CreatePrimitive(isCube ? PrimitiveType.Cube : PrimitiveType.Sphere);
        sphere.name = $"Obj_{label}_{_records.Count}";
        sphere.transform.SetParent(_objectsRoot.transform);
        sphere.transform.position   = new Vector3(req.x, req.y, req.z);
        sphere.transform.localScale = Vector3.one * (sphereRadius * 2f * scale);

        Color color = ResolveColor(req.color, _records.Count);
        ApplyColor(sphere, color);

        var labelGo = new GameObject($"Label_{label}");
        labelGo.transform.SetParent(_objectsRoot.transform);
        labelGo.transform.position = new Vector3(req.x, req.y + sphereRadius * scale + 0.35f, req.z);
        var tm = labelGo.AddComponent<TextMesh>();
        tm.text = label; tm.fontSize = 26; tm.characterSize = 0.13f;
        tm.color = Color.white; tm.anchor = TextAnchor.MiddleCenter;

        string hex = "#" + ColorUtility.ToHtmlStringRGB(color);
        _records.Add(new PlacedObjectRecord { id = _records.Count, label = label,
            x = req.x, y = req.y, z = req.z, colorHex = hex, scale = scale });

        return $"{{\"status\":\"placed\",\"id\":{_records.Count - 1},\"label\":\"{label}\"," +
               $"\"position\":{{\"x\":{req.x},\"y\":{req.y},\"z\":{req.z}}},\"color\":\"{hex}\"}}";
    }

    [Serializable]
    private class PlaceObjectRequest
    {
        public string label = ""; public float x, y, z; public string color = ""; public float scale = 1f; public string shape = "sphere";
    }

    [Serializable]
    private class RemoveObjectRequest { public float x, y, z; }

    [Serializable]
    private class MoveObjectRequest { public float x, y, z, new_x, new_y, new_z; }

    // Returns the index in _records of the object nearest to (x,y,z), or -1 if empty.
    private int FindClosestRecord(float x, float y, float z, out float distance)
    {
        int best = -1;
        float bestDist = float.MaxValue;
        for (int i = 0; i < _records.Count; i++)
        {
            var r = _records[i];
            float d = Vector3.Distance(new Vector3(x, y, z), new Vector3(r.x, r.y, r.z));
            if (d < bestDist) { bestDist = d; best = i; }
        }
        distance = bestDist;
        return best;
    }

    private string RemoveObject(string body)
    {
        var req = JsonUtility.FromJson<RemoveObjectRequest>(body);
        if (req == null) return JsonError("invalid JSON");

        int idx = FindClosestRecord(req.x, req.y, req.z, out float _);
        if (idx < 0) return JsonError("no objects in scene");

        var rec = _records[idx];

        // Destroy the sphere and its label — both share the same X/Z as the record.
        var toDestroy = new List<Transform>();
        foreach (Transform child in _objectsRoot.transform)
        {
            Vector3 p = child.position;
            if (Mathf.Abs(p.x - rec.x) < 0.01f && Mathf.Abs(p.z - rec.z) < 0.01f)
                toDestroy.Add(child);
        }
        foreach (var t in toDestroy) Destroy(t.gameObject);

        string label = rec.label;
        int id       = rec.id;
        _records.RemoveAt(idx);

        return $"{{\"status\":\"removed\",\"id\":{id},\"label\":\"{label}\"}}";
    }

    private string MoveObject(string body)
    {
        var req = JsonUtility.FromJson<MoveObjectRequest>(body);
        if (req == null) return JsonError("invalid JSON");

        int idx = FindClosestRecord(req.x, req.y, req.z, out float _);
        if (idx < 0) return JsonError("no objects in scene");

        var rec = _records[idx];
        float oldX = rec.x, oldY = rec.y, oldZ = rec.z;

        // Move the sphere and its label, preserving each child's Y offset.
        foreach (Transform child in _objectsRoot.transform)
        {
            Vector3 p = child.position;
            if (Mathf.Abs(p.x - oldX) < 0.01f && Mathf.Abs(p.z - oldZ) < 0.01f)
            {
                float yOffset = p.y - oldY;
                child.position = new Vector3(req.new_x, req.new_y + yOffset, req.new_z);
            }
        }

        rec.x = req.new_x; rec.y = req.new_y; rec.z = req.new_z;

        return $"{{\"status\":\"moved\",\"id\":{rec.id},\"label\":\"{rec.label}\"," +
               $"\"from\":{{\"x\":{oldX},\"y\":{oldY},\"z\":{oldZ}}}," +
               $"\"to\":{{\"x\":{req.new_x},\"y\":{req.new_y},\"z\":{req.new_z}}}}}";
    }

    private string ClearScene()
    {
        int count = _records.Count;
        foreach (Transform child in _objectsRoot.transform) Destroy(child.gameObject);
        _records.Clear();
        return $"{{\"status\":\"cleared\",\"removed\":{count}}}";
    }

    private string GetSceneState()
    {
        var sb = new StringBuilder("{\"objects\":[");
        for (int i = 0; i < _records.Count; i++)
        {
            if (i > 0) sb.Append(',');
            var r = _records[i];
            sb.Append($"{{\"id\":{r.id},\"label\":\"{r.label}\"," +
                      $"\"x\":{r.x},\"y\":{r.y},\"z\":{r.z}," +
                      $"\"color\":\"{r.colorHex}\",\"scale\":{r.scale}}}");
        }
        sb.Append($"],\"count\":{_records.Count}}}");
        return sb.ToString();
    }

    // ─── Camera capture for embodied perception ─────────────────────────────

    private Camera _mainCamera;

    private Camera MainCamera
    {
        get
        {
            if (_mainCamera == null)
                _mainCamera = Camera.main;
            return _mainCamera;
        }
    }

    private string CaptureView()
    {
        if (MainCamera == null)
            return JsonError("No main camera found");

        // Render to texture
        int width = 640;
        int height = 480;
        RenderTexture rt = new RenderTexture(width, height, 24);
        MainCamera.targetTexture = rt;
        Texture2D screenshot = new Texture2D(width, height, TextureFormat.RGB24, false);

        MainCamera.Render();
        RenderTexture.active = rt;
        screenshot.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        screenshot.Apply();

        // Reset camera
        MainCamera.targetTexture = null;
        RenderTexture.active = null;
        Destroy(rt);

        // Encode to PNG and base64
        byte[] bytes = screenshot.EncodeToPNG();
        Destroy(screenshot);
        string base64 = System.Convert.ToBase64String(bytes);

        return $"{{\"status\":\"captured\",\"width\":{width},\"height\":{height},\"image\":\"{base64}\"}}";
    }

    [Serializable]
    private class RotateCameraRequest
    {
        public float yaw = 0f;
        public float pitch = 0f;
    }

    private string RotateCamera(string body)
    {
        var req = JsonUtility.FromJson<RotateCameraRequest>(body);
        if (req == null)
            return JsonError("Invalid JSON for rotate_camera");

        // Find CherryCamera component
        var cherryCamera = MainCamera?.GetComponent<CherryCamera>();
        if (cherryCamera == null)
            return JsonError("CherryCamera component not found on main camera");

        // Clamp values
        float yaw = Mathf.Clamp(req.yaw, -90f, 90f);
        float pitch = Mathf.Clamp(req.pitch, -90f, 90f);

        // Apply rotation via reflection or direct field access
        // CherryCamera uses _yaw and _pitch fields
        var yawField = typeof(CherryCamera).GetField("_yaw", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        var pitchField = typeof(CherryCamera).GetField("_pitch", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);

        if (yawField != null && pitchField != null)
        {
            yawField.SetValue(cherryCamera, yaw);
            pitchField.SetValue(cherryCamera, pitch);

            // Trigger rotation update
            var baseRotField = typeof(CherryCamera).GetField("_baseRotation", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (baseRotField != null)
            {
                Quaternion baseRot = (Quaternion)baseRotField.GetValue(cherryCamera);
                MainCamera.transform.rotation = baseRot * Quaternion.Euler(pitch, yaw, 0f);
            }
        }

        return $"{{\"status\":\"rotated\",\"yaw\":{yaw},\"pitch\":{pitch}}}";
    }

    private string ResetCamera()
    {
        var cherryCamera = MainCamera?.GetComponent<CherryCamera>();
        if (cherryCamera != null)
        {
            cherryCamera.ResetOrientation();
            return "{\"status\":\"reset\",\"yaw\":0,\"pitch\":0}";
        }
        return JsonError("CherryCamera component not found");
    }

    // ════════════════════════════════════════════════════════════════════════
    // Helpers
    // ════════════════════════════════════════════════════════════════════════

    private void ApplyColor(GameObject go, Color color)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Lit")
                     ?? Shader.Find("HDRP/Lit")
                     ?? Shader.Find("Standard");
        var rend = go.GetComponent<Renderer>();
        var mat  = shader != null ? new Material(shader) : rend.material;
        mat.color = color;
        rend.material = mat;
    }

    private Color ResolveColor(string name, int index)
    {
        if (!string.IsNullOrEmpty(name))
        {
            switch (name.ToLowerInvariant().Trim())
            {
                case "red":    return new Color(0.90f,0.20f,0.20f);
                case "blue":   return new Color(0.20f,0.50f,0.90f);
                case "green":  return new Color(0.20f,0.80f,0.25f);
                case "yellow": return new Color(0.95f,0.90f,0.10f);
                case "purple": return new Color(0.70f,0.20f,0.90f);
                case "orange": return new Color(0.95f,0.50f,0.10f);
                case "cyan":   return new Color(0.10f,0.80f,0.80f);
                case "pink":   return new Color(0.90f,0.20f,0.70f);
                case "white":  return Color.white;
                case "gray":
                case "grey":   return new Color(0.60f,0.60f,0.60f);
                case "brown":  return new Color(0.55f,0.35f,0.15f);
            }
            if (ColorUtility.TryParseHtmlString(name, out Color hex)) return hex;
        }
        return Palette[index % Palette.Length];
    }

    private static string JsonError(string msg) =>
        $"{{\"error\":\"{msg.Replace("\"", "\\'")}\",\"status\":\"error\"}}";
}
