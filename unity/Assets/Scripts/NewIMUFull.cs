using UnityEngine;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Globalization;

public class IMUAvatarDriver : MonoBehaviour
{
    [Header("Body Bones (17)")]
    public Transform Head;
    public Transform RightFoot;
    public Transform RightLowerLeg;
    public Transform RightUpperLeg;
    public Transform LeftFoot;
    public Transform LeftLowerLeg;
    public Transform LeftUpperLeg;
    public Transform RightShoulder;
    public Transform RightUpperArm;
    public Transform RightLowerArm;
    public Transform RightHand;
    public Transform LeftShoulder;
    public Transform LeftUpperArm;
    public Transform LeftLowerArm;
    public Transform LeftHand;
    public Transform Hips;
    public Transform Spine;

    [Header("Left Hand Fingers")]
    public Transform LeftThumbProximal;
    public Transform LeftThumbIntermediate;
    public Transform LeftThumbDistal;

    public Transform LeftIndexProximal;
    public Transform LeftIndexIntermediate;
    public Transform LeftIndexDistal;

    public Transform LeftMiddleProximal;
    public Transform LeftMiddleIntermediate;
    public Transform LeftMiddleDistal;

    public Transform LeftRingProximal;
    public Transform LeftRingIntermediate;
    public Transform LeftRingDistal;

    public Transform LeftLittleProximal;
    public Transform LeftLittleIntermediate;
    public Transform LeftLittleDistal;

    private UdpClient udpClient;
    private IPEndPoint remoteEndPoint;
    private Dictionary<string, Transform> boneMap;

    void Start()
    {
        udpClient = new UdpClient(5005);
        remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);

        boneMap = new Dictionary<string, Transform>()
        {
            {"Head", Head},
            {"RightFoot", RightFoot},
            {"RightLowerLeg", RightLowerLeg},
            {"RightUpperLeg", RightUpperLeg},
            {"LeftFoot", LeftFoot},
            {"LeftLowerLeg", LeftLowerLeg},
            {"LeftUpperLeg", LeftUpperLeg},
            {"RightShoulder", RightShoulder},
            {"RightUpperArm", RightUpperArm},
            {"RightLowerArm", RightLowerArm},
            {"RightHand", RightHand},
            {"LeftShoulder", LeftShoulder},
            {"LeftUpperArm", LeftUpperArm},
            {"LeftLowerArm", LeftLowerArm},
            {"LeftHand", LeftHand},
            {"Hips", Hips},
            {"Spine", Spine},

            // 左手手指（与你发的 hierarchy 对应）
            {"LeftThumbProximal", LeftThumbProximal},
            {"LeftThumbIntermediate", LeftThumbIntermediate},
            {"LeftThumbDistal", LeftThumbDistal},

            {"LeftIndexProximal", LeftIndexProximal},
            {"LeftIndexIntermediate", LeftIndexIntermediate},
            {"LeftIndexDistal", LeftIndexDistal},

            {"LeftMiddleProximal", LeftMiddleProximal},
            {"LeftMiddleIntermediate", LeftMiddleIntermediate},
            {"LeftMiddleDistal", LeftMiddleDistal},

            {"LeftRingProximal", LeftRingProximal},
            {"LeftRingIntermediate", LeftRingIntermediate},
            {"LeftRingDistal", LeftRingDistal},

            {"LeftLittleProximal", LeftLittleProximal},
            {"LeftLittleIntermediate", LeftLittleIntermediate},
            {"LeftLittleDistal", LeftLittleDistal},
        };
    }

    void Update()
    {
        while (udpClient.Available > 0)
        {
            byte[] data = udpClient.Receive(ref remoteEndPoint);
            string msg = Encoding.ASCII.GetString(data);
            ApplyRotations(msg);
        }
    }

    private void ApplyRotations(string msg)
    {
        var ci = CultureInfo.InvariantCulture;
        string[] lines = msg.Split(new[] { '\n', '\r' }, System.StringSplitOptions.RemoveEmptyEntries);

        foreach (string line in lines)
        {
            string[] parts = line.Split(':');
            if (parts.Length != 2) continue;

            string boneName = parts[0].Trim();
            string[] values = parts[1].Split(',');

            // hips 位移
            if (boneName == "hipsPosition" && values.Length == 3)
            {
                if (float.TryParse(values[0], NumberStyles.Float, ci, out float x) &&
                    float.TryParse(values[1], NumberStyles.Float, ci, out float y) &&
                    float.TryParse(values[2], NumberStyles.Float, ci, out float z))
                {
                    if (Hips != null) Hips.position = new Vector3(x, y, z);
                }
                continue;
            }

            if (values.Length != 4) continue;
            if (!boneMap.TryGetValue(boneName, out Transform bone) || bone == null) continue;

            if (float.TryParse(values[0], NumberStyles.Float, ci, out float qx) &&
                float.TryParse(values[1], NumberStyles.Float, ci, out float qy) &&
                float.TryParse(values[2], NumberStyles.Float, ci, out float qz) &&
                float.TryParse(values[3], NumberStyles.Float, ci, out float qw))
            {
                    bone.rotation = new Quaternion(qx, qy, qz, qw);
            }
        }
    }

    private void OnDisable()
    {
        udpClient?.Close();
    }
}
