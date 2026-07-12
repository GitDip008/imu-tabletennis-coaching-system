// IMUReceiver.cs
// ==============
// Receives per-bone DELTA quaternions from stream_p01.py via UDP.
// Captures the Mixamo character's own T-pose at Start() — no external
// InitialPoseExport.txt file needed.
//
// Per frame:
//   bone.localRotation = tpose[bone] * received_delta
//
// UDP packet format (one bone per line, \n separated):
//   BoneName:qx,qy,qz,qw
//   BoneName:qx,qy,qz,qw
//   ...

using UnityEngine;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Globalization;

public class IMUReceiver : MonoBehaviour
{
    [Header("=== 17 Body Bones — assign Mixamo transforms ===")]
    public Transform Hips;
    public Transform Spine;
    public Transform Head;

    public Transform LeftUpperLeg;
    public Transform LeftLowerLeg;
    public Transform LeftFoot;

    public Transform RightUpperLeg;
    public Transform RightLowerLeg;
    public Transform RightFoot;

    public Transform LeftShoulder;
    public Transform LeftUpperArm;
    public Transform LeftLowerArm;
    public Transform LeftHand;

    public Transform RightShoulder;
    public Transform RightUpperArm;
    public Transform RightLowerArm;
    public Transform RightHand;

    [Header("=== Settings ===")]
    public int UdpPort = 5005;

    // ── Internals ──────────────────────────────────────────────────────────
    private UdpClient                    udp;
    private IPEndPoint                   endpoint;
    private Dictionary<string, Transform>   boneMap;
    private Dictionary<string, Quaternion>  tpose;   // captured at Start()

    void Start()
    {
        // Build bone name → Transform map
        boneMap = new Dictionary<string, Transform>
        {
            {"Hips",         Hips},
            {"Spine",        Spine},
            {"Head",         Head},
            {"LeftUpperLeg", LeftUpperLeg},
            {"LeftLowerLeg", LeftLowerLeg},
            {"LeftFoot",     LeftFoot},
            {"RightUpperLeg",RightUpperLeg},
            {"RightLowerLeg",RightLowerLeg},
            {"RightFoot",    RightFoot},
            {"LeftShoulder", LeftShoulder},
            {"LeftUpperArm", LeftUpperArm},
            {"LeftLowerArm", LeftLowerArm},
            {"LeftHand",     LeftHand},
            {"RightShoulder",RightShoulder},
            {"RightUpperArm",RightUpperArm},
            {"RightLowerArm",RightLowerArm},
            {"RightHand",    RightHand},
        };

        // Capture each bone's T-pose localRotation RIGHT NOW
        // (character starts in T-pose so this is the reference)
        tpose = new Dictionary<string, Quaternion>();
        foreach (var kv in boneMap)
        {
            if (kv.Value != null)
                tpose[kv.Key] = kv.Value.localRotation;
            else
                Debug.LogWarning($"[IMUReceiver] Bone slot '{kv.Key}' is not assigned in Inspector!");
        }

        // Open UDP socket
        udp      = new UdpClient(UdpPort);
        endpoint = new IPEndPoint(IPAddress.Any, 0);
        Debug.Log($"[IMUReceiver] Ready. T-pose captured. Listening on UDP port {UdpPort}.");
    }

    void Update()
    {
        // Drain all pending UDP packets each frame
        while (udp != null && udp.Available > 0)
        {
            byte[] data = udp.Receive(ref endpoint);
            string msg  = Encoding.ASCII.GetString(data);
            ApplyDeltas(msg);
        }
    }

    void ApplyDeltas(string msg)
    {
        var ci    = CultureInfo.InvariantCulture;
        var lines = msg.Split(new[] {'\n', '\r'},
                              System.StringSplitOptions.RemoveEmptyEntries);

        foreach (string line in lines)
        {
            var parts = line.Split(':');
            if (parts.Length != 2) continue;

            string   boneName = parts[0].Trim();
            string[] vals     = parts[1].Split(',');
            if (vals.Length != 4) continue;

            if (!boneMap.TryGetValue(boneName, out Transform bone) || bone == null) continue;
            if (!tpose.TryGetValue(boneName,   out Quaternion tp))                  continue;

            if (float.TryParse(vals[0], NumberStyles.Float, ci, out float qx) &&
                float.TryParse(vals[1], NumberStyles.Float, ci, out float qy) &&
                float.TryParse(vals[2], NumberStyles.Float, ci, out float qz) &&
                float.TryParse(vals[3], NumberStyles.Float, ci, out float qw))
            {
                // delta quaternion received from Python (already in Unity LH space)
                Quaternion delta = new Quaternion(qx, qy, qz, qw);

                // Compose T-pose + delta → final local rotation
                bone.localRotation = tp * delta;
            }
        }
    }

    void OnDisable()
    {
        udp?.Close();
        udp = null;
    }
}