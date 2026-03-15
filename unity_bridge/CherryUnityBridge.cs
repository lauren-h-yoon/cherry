using System;
using System.Collections.Generic;
using System.Net;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// CherryUnityBridge - HTTP server that receives commands from Python to place objects in a 3D scene.
///
/// Attach this script to any GameObject in your Unity scene (e.g. an empty "Bridge" object).
/// The scene initializes automatically on Start().
///
/// Unity Coordinate System:
///   X axis: Left (negative) / Right (positive)
///   Y axis: Down (negative) / Up (positive)  -- ground plane at Y=0
///   Z axis: Toward viewer (negative) / Away from viewer (positive)
///
/// Scene bounds: X ∈ [-10, 10], Z ∈ [-10, 10], Y ∈ [0, 10]
/// All objects are placed as spheres with radius 0.5 (diameter 1.0 unit).
/// To place a sphere sitting on the ground: set Y = 0.5.
///
/// HTTP endpoints (all on localhost:5555):
///   GET  /health        -- health check
///   POST /place_object  -- place a labeled sphere
///   POST /clear_scene   -- remove all placed spheres
///   GET  /scene_state   -- get list of all placed objects
///   POST /initialize    -- re-initialize scene (clears all objects)
/// </summary>
public class CherryUnityBridge : MonoBehaviour
{
    [Header("Server Settings")]
    public int port = 5555;

    [Header("Scene Settings")]
    public float sceneSize = 20f;
    public float sphereRadius = 0.5f;

    // ── Threading infrastructure ────────────────────────────────────────────
    private HttpListener _listener;
    private Thread _listenerThread;
    private volatile bool _running = false;

    private readonly Queue<PendingCommand> _commandQueue = new Queue<PendingCommand>();
    private readonly object _queueLock = new object();

    // ── Scene state ─────────────────────────────────────────────────────────
    private GameObject _sceneRoot;      // permanent scene elements
    private GameObject _objectsRoot;    // placed objects only — wiped on clear_scene

    [Serializable]
    private class PlacedObjectRecord
    {
        public int id;
        public string label;
        public float x, y, z;
        public string colorHex;
        public float scale;
    }

    private readonly List<PlacedObjectRecord> _records = new List<PlacedObjectRecord>();

    // ── Command queue entry ─────────────────────────────────────────────────
    private class PendingCommand
    {
        public string Type;
        public string Body;
        public string Result;
        public readonly ManualResetEventSlim Signal = new ManualResetEventSlim(false);
    }

    // ── Color palette for auto-assignment ───────────────────────────────────
    private static readonly Color[] Palette = {
        new Color(0.90f, 0.20f, 0.20f),  // red
        new Color(0.20f, 0.50f, 0.90f),  // blue
        new Color(0.20f, 0.80f, 0.25f),  // green
        new Color(0.95f, 0.75f, 0.10f),  // yellow
        new Color(0.70f, 0.20f, 0.90f),  // purple
        new Color(0.95f, 0.50f, 0.10f),  // orange
        new Color(0.10f, 0.80f, 0.80f),  // cyan
        new Color(0.90f, 0.20f, 0.70f),  // pink
        new Color(0.60f, 0.40f, 0.20f),  // brown
        new Color(0.50f, 0.90f, 0.50f),  // lime
    };

    // ════════════════════════════════════════════════════════════════════════
    // Unity lifecycle
    // ════════════════════════════════════════════════════════════════════════

    void Start()
    {
        InitializeScene();
        StartServer();
    }

    void Update()
    {
        // Execute ONE queued command per frame on the main thread
        // (Unity API calls are not thread-safe)
        PendingCommand cmd = null;
        lock (_queueLock)
        {
            if (_commandQueue.Count > 0)
                cmd = _commandQueue.Dequeue();
        }

        if (cmd == null) return;

        try
        {
            cmd.Result = DispatchCommand(cmd.Type, cmd.Body);
        }
        catch (Exception ex)
        {
            cmd.Result = JsonError(ex.Message);
        }
        finally
        {
            cmd.Signal.Set();
        }
    }

    void OnDestroy() => StopServer();

    void OnApplicationQuit() => StopServer();

    private void StopServer()
    {
        _running = false;
        try { _listener?.Stop(); } catch { }
        try { _listener?.Close(); } catch { }
        _listenerThread?.Join(1000);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Scene initialization
    // ════════════════════════════════════════════════════════════════════════

    private void InitializeScene()
    {
        // Root objects
        _sceneRoot = new GameObject("CherryScene_Static");
        _objectsRoot = new GameObject("CherryScene_Objects");

        // ── Ground plane ────────────────────────────────────────────────────
        var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.SetParent(_sceneRoot.transform);
        ground.transform.position = Vector3.zero;
        // Unity Plane is 10x10 units by default; scale to sceneSize
        ground.transform.localScale = new Vector3(sceneSize / 10f, 1f, sceneSize / 10f);
        ApplyColor(ground, new Color(0.65f, 0.65f, 0.65f));

        // ── Grid overlay ────────────────────────────────────────────────────
        BuildGrid();

        // ── Axis markers ────────────────────────────────────────────────────
        BuildAxisMarkers();

        // ── Lighting ────────────────────────────────────────────────────────
        var lightObj = new GameObject("DirectionalLight");
        lightObj.transform.SetParent(_sceneRoot.transform);
        var light = lightObj.AddComponent<Light>();
        light.type = LightType.Directional;
        light.intensity = 1.2f;
        light.color = new Color(1f, 0.97f, 0.92f);
        lightObj.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.38f, 0.38f, 0.38f);

        // ── Camera ──────────────────────────────────────────────────────────
        // Positioned for a clear overview: high and slightly behind
        var cam = Camera.main;
        if (cam != null)
        {
            cam.transform.position = new Vector3(0f, 16f, -14f);
            cam.transform.rotation = Quaternion.Euler(46f, 0f, 0f);
            cam.backgroundColor = new Color(0.12f, 0.12f, 0.17f);
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.farClipPlane = 200f;
        }

        Debug.Log($"[CherryBridge] Scene initialized — {sceneSize}x{sceneSize} ground plane.");
    }

    private void BuildGrid()
    {
        var gridRoot = new GameObject("Grid");
        gridRoot.transform.SetParent(_sceneRoot.transform);

        var mat = new Material(Shader.Find("Standard"));
        mat.color = new Color(0.40f, 0.40f, 0.40f, 0.6f);

        float half = sceneSize / 2f;
        int steps = (int)(sceneSize / 2); // line every 2 units

        for (int i = -steps; i <= steps; i++)
        {
            float t = i * 2f;
            SpawnGridLine(gridRoot, mat, new Vector3(t, 0.01f, 0f), new Vector3(0.04f, 0.01f, sceneSize));
            SpawnGridLine(gridRoot, mat, new Vector3(0f, 0.01f, t), new Vector3(sceneSize, 0.01f, 0.04f));
        }
    }

    private void SpawnGridLine(GameObject parent, Material mat, Vector3 pos, Vector3 scale)
    {
        var line = GameObject.CreatePrimitive(PrimitiveType.Cube);
        line.transform.SetParent(parent.transform);
        line.transform.position = pos;
        line.transform.localScale = scale;
        line.GetComponent<Renderer>().material = mat;
        Destroy(line.GetComponent<Collider>());
    }

    private void BuildAxisMarkers()
    {
        float edge = sceneSize / 2f - 0.8f;

        // X axis (red)
        SpawnAxisMarker("+X\n(right)",  new Vector3( edge, 0.5f, 0f), new Color(0.90f, 0.20f, 0.20f));
        SpawnAxisMarker("-X\n(left)",   new Vector3(-edge, 0.5f, 0f), new Color(1.00f, 0.60f, 0.60f));

        // Z axis (blue)
        SpawnAxisMarker("+Z\n(far)",    new Vector3(0f, 0.5f,  edge), new Color(0.20f, 0.40f, 0.90f));
        SpawnAxisMarker("-Z\n(near)",   new Vector3(0f, 0.5f, -edge), new Color(0.60f, 0.70f, 1.00f));

        // Y axis (green pillar at origin)
        var pillar = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        pillar.name = "YAxisPillar";
        pillar.transform.SetParent(_sceneRoot.transform);
        pillar.transform.position = new Vector3(0f, 2f, 0f);
        pillar.transform.localScale = new Vector3(0.12f, 2f, 0.12f);
        ApplyColor(pillar, new Color(0.20f, 0.85f, 0.20f));
        Destroy(pillar.GetComponent<Collider>());
        SpawnTextLabel("+Y (up)", new Vector3(0.3f, 4.3f, 0f), new Color(0.20f, 0.95f, 0.20f));
    }

    private void SpawnAxisMarker(string text, Vector3 pos, Color color)
    {
        var cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.name = $"AxisMarker_{text.Replace("\n", "")}";
        cube.transform.SetParent(_sceneRoot.transform);
        cube.transform.position = pos;
        cube.transform.localScale = new Vector3(0.6f, 0.6f, 0.6f);
        ApplyColor(cube, color);
        Destroy(cube.GetComponent<Collider>());
        SpawnTextLabel(text, pos + Vector3.up * 0.7f, color);
    }

    private void SpawnTextLabel(string text, Vector3 pos, Color color)
    {
        var go = new GameObject($"TextLabel");
        go.transform.SetParent(_sceneRoot.transform);
        go.transform.position = pos;
        var tm = go.AddComponent<TextMesh>();
        tm.text = text;
        tm.fontSize = 28;
        tm.characterSize = 0.12f;
        tm.color = color;
        tm.anchor = TextAnchor.MiddleCenter;
        tm.alignment = TextAlignment.Center;
    }

    private void ApplyColor(GameObject go, Color color)
    {
        var mat = new Material(Shader.Find("Standard"));
        mat.color = color;
        go.GetComponent<Renderer>().material = mat;
    }

    // ════════════════════════════════════════════════════════════════════════
    // HTTP server
    // ════════════════════════════════════════════════════════════════════════

    private void StartServer()
    {
        _listener = new HttpListener();
        _listener.Prefixes.Add($"http://localhost:{port}/");
        try
        {
            _listener.Start();
        }
        catch (HttpListenerException ex)
        {
            Debug.LogError($"[CherryBridge] Could not start on port {port}: {ex.Message}. " +
                           "Stop Play mode, wait a moment, then press Play again.");
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
        {
            using var reader = new System.IO.StreamReader(
                ctx.Request.InputStream, ctx.Request.ContentEncoding);
            body = reader.ReadToEnd();
        }

        string path = ctx.Request.Url.AbsolutePath.TrimStart('/');
        var cmd = new PendingCommand { Type = path, Body = body };

        lock (_queueLock) { _commandQueue.Enqueue(cmd); }

        // Block until main thread processes it (5 s timeout)
        if (!cmd.Signal.Wait(5000))
        {
            SendJson(ctx.Response, 504, JsonError("timeout — main thread did not respond"));
            return;
        }

        SendJson(ctx.Response, 200, cmd.Result);
    }

    private static void SendJson(HttpListenerResponse resp, int status, string body)
    {
        resp.StatusCode = status;
        resp.ContentType = "application/json; charset=utf-8";
        byte[] bytes = Encoding.UTF8.GetBytes(body);
        resp.ContentLength64 = bytes.Length;
        resp.OutputStream.Write(bytes, 0, bytes.Length);
        resp.Close();
    }

    // ════════════════════════════════════════════════════════════════════════
    // Command dispatch (runs on main thread via Update())
    // ════════════════════════════════════════════════════════════════════════

    private string DispatchCommand(string type, string body)
    {
        switch (type.ToLowerInvariant())
        {
            case "health":      return Health();
            case "place_object":return PlaceObject(body);
            case "clear_scene": return ClearScene();
            case "scene_state": return GetSceneState();
            case "initialize":
                ClearScene();
                return "{\"status\":\"reinitialized\"}";
            default:
                return JsonError($"unknown endpoint: {type}");
        }
    }

    // ── /health ─────────────────────────────────────────────────────────────
    private string Health()
    {
        return $"{{\"status\":\"ok\",\"objects_placed\":{_records.Count}}}";
    }

    // ── /place_object ────────────────────────────────────────────────────────
    //  Body: {"label":"chair","x":1.0,"y":0.5,"z":2.0,"color":"blue","scale":1.0}
    private string PlaceObject(string body)
    {
        // Parse request (JsonUtility requires [Serializable] class)
        var req = JsonUtility.FromJson<PlaceObjectRequest>(body);
        if (req == null) return JsonError("invalid JSON body");

        string label = string.IsNullOrEmpty(req.label) ? "object" : req.label;
        float x = req.x;
        float y = req.y < 0f ? sphereRadius : req.y; // auto: sit on ground
        float z = req.z;
        float scale = req.scale <= 0f ? 1f : req.scale;

        // Create sphere
        var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sphere.name = $"Obj_{label}_{_records.Count}";
        sphere.transform.SetParent(_objectsRoot.transform);
        sphere.transform.position = new Vector3(x, y, z);
        sphere.transform.localScale = Vector3.one * (sphereRadius * 2f * scale);

        // Color
        Color color = ResolveColor(req.color, _records.Count);
        ApplyColor(sphere, color);

        // Text label (floats above sphere)
        var labelGo = new GameObject($"Label_{label}");
        labelGo.transform.SetParent(_objectsRoot.transform);
        float labelY = y + (sphereRadius * scale) + 0.35f;
        labelGo.transform.position = new Vector3(x, labelY, z);
        var tm = labelGo.AddComponent<TextMesh>();
        tm.text = label;
        tm.fontSize = 26;
        tm.characterSize = 0.13f;
        tm.color = Color.white;
        tm.anchor = TextAnchor.MiddleCenter;
        tm.alignment = TextAlignment.Center;

        // Record
        string hex = "#" + ColorUtility.ToHtmlStringRGB(color);
        var rec = new PlacedObjectRecord { id = _records.Count, label = label, x = x, y = y, z = z, colorHex = hex, scale = scale };
        _records.Add(rec);

        Debug.Log($"[CherryBridge] Placed '{label}' at ({x:F2}, {y:F2}, {z:F2})");

        return $"{{\"status\":\"placed\",\"id\":{rec.id},\"label\":\"{label}\"," +
               $"\"position\":{{\"x\":{x},\"y\":{y},\"z\":{z}}},\"color\":\"{hex}\"}}";
    }

    [Serializable]
    private class PlaceObjectRequest
    {
        public string label = "";
        public float x = 0f;
        public float y = -1f;  // -1 = auto (sit on ground)
        public float z = 0f;
        public string color = "";
        public float scale = 1f;
    }

    // ── /clear_scene ─────────────────────────────────────────────────────────
    private string ClearScene()
    {
        int count = _records.Count;

        // Destroy all placed object GameObjects at once via parent
        foreach (Transform child in _objectsRoot.transform)
            Destroy(child.gameObject);

        _records.Clear();

        Debug.Log($"[CherryBridge] Cleared {count} objects.");
        return $"{{\"status\":\"cleared\",\"removed\":{count}}}";
    }

    // ── /scene_state ─────────────────────────────────────────────────────────
    private string GetSceneState()
    {
        var sb = new StringBuilder();
        sb.Append("{\"objects\":[");
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

    // ════════════════════════════════════════════════════════════════════════
    // Helpers
    // ════════════════════════════════════════════════════════════════════════

    private Color ResolveColor(string name, int index)
    {
        if (!string.IsNullOrEmpty(name))
        {
            // Named colors
            switch (name.ToLowerInvariant().Trim())
            {
                case "red":    return new Color(0.90f, 0.20f, 0.20f);
                case "blue":   return new Color(0.20f, 0.50f, 0.90f);
                case "green":  return new Color(0.20f, 0.80f, 0.25f);
                case "yellow": return new Color(0.95f, 0.90f, 0.10f);
                case "purple": return new Color(0.70f, 0.20f, 0.90f);
                case "orange": return new Color(0.95f, 0.50f, 0.10f);
                case "cyan":   return new Color(0.10f, 0.80f, 0.80f);
                case "pink":   return new Color(0.90f, 0.20f, 0.70f);
                case "white":  return Color.white;
                case "gray":
                case "grey":   return new Color(0.60f, 0.60f, 0.60f);
                case "brown":  return new Color(0.55f, 0.35f, 0.15f);
            }
            // Hex color (#RRGGBB)
            if (ColorUtility.TryParseHtmlString(name, out Color hex)) return hex;
        }
        return Palette[index % Palette.Length];
    }

    private static string JsonError(string msg) =>
        $"{{\"error\":\"{msg.Replace("\"", "\\'")}\",\"status\":\"error\"}}";
}
