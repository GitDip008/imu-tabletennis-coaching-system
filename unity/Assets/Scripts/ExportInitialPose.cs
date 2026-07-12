using UnityEngine;
using System.IO;
using System.Globalization;

/// <summary>
/// Attach this to any GameObject in the scene.
/// Make sure the character is in T-pose (default import pose).
/// Press Play — it writes InitialPoseExport.txt next to your imu_to_unity.py.
/// Check the Console for confirmation.
/// </summary>
public class ExportInitialPose : MonoBehaviour
{
    [Header("Assign the same bones as in IMUController")]
    public Transform Hips;
    public Transform Spine;
    public Transform Head;

    public Transform LeftUpperLeg;
    public Transform LeftLowerLeg;
    public Transform LeftFoot;
    public Transform LeftToes;

    public Transform RightUpperLeg;
    public Transform RightLowerLeg;
    public Transform RightFoot;
    public Transform RightToes;

    public Transform LeftShoulder;
    public Transform LeftUpperArm;
    public Transform LeftLowerArm;
    public Transform LeftHand;

    public Transform RightShoulder;
    public Transform RightUpperArm;
    public Transform RightLowerArm;
    public Transform RightHand;

    // Spine chain
    public Transform Chest;
    public Transform UpperChest;
    public Transform Neck;

    // Left fingers
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

    // Right fingers
    public Transform RightThumbProximal;
    public Transform RightThumbIntermediate;
    public Transform RightThumbDistal;
    public Transform RightIndexProximal;
    public Transform RightIndexIntermediate;
    public Transform RightIndexDistal;
    public Transform RightMiddleProximal;
    public Transform RightMiddleIntermediate;
    public Transform RightMiddleDistal;
    public Transform RightRingProximal;
    public Transform RightRingIntermediate;
    public Transform RightRingDistal;
    public Transform RightLittleProximal;
    public Transform RightLittleIntermediate;
    public Transform RightLittleDistal;

    [Header("Output path — folder where imu_to_unity.py lives")]
    public string OutputPath = @"E:\thesis_work\thesis_works_new\InitialPoseExport.txt";

    void Start()
    {
        ExportPose();
    }

    void ExportPose()
    {
        var ci = CultureInfo.InvariantCulture;

        // All bones with their string names — order matches BoneHierarchy.txt
        var bones = new (string name, Transform t)[]
        {
            ("Hips",                  Hips),
            ("LeftUpperLeg",          LeftUpperLeg),
            ("RightUpperLeg",         RightUpperLeg),
            ("LeftLowerLeg",          LeftLowerLeg),
            ("RightLowerLeg",         RightLowerLeg),
            ("LeftFoot",              LeftFoot),
            ("RightFoot",             RightFoot),
            ("Spine",                 Spine),
            ("Chest",                 Chest),
            ("UpperChest",            UpperChest),
            ("Neck",                  Neck),
            ("Head",                  Head),
            ("LeftShoulder",          LeftShoulder),
            ("RightShoulder",         RightShoulder),
            ("LeftUpperArm",          LeftUpperArm),
            ("RightUpperArm",         RightUpperArm),
            ("LeftLowerArm",          LeftLowerArm),
            ("RightLowerArm",         RightLowerArm),
            ("LeftHand",              LeftHand),
            ("RightHand",             RightHand),
            ("LeftToes",              LeftToes),
            ("RightToes",             RightToes),
            ("LeftThumbProximal",     LeftThumbProximal),
            ("LeftThumbIntermediate", LeftThumbIntermediate),
            ("LeftThumbDistal",       LeftThumbDistal),
            ("LeftIndexProximal",     LeftIndexProximal),
            ("LeftIndexIntermediate", LeftIndexIntermediate),
            ("LeftIndexDistal",       LeftIndexDistal),
            ("LeftMiddleProximal",    LeftMiddleProximal),
            ("LeftMiddleIntermediate",LeftMiddleIntermediate),
            ("LeftMiddleDistal",      LeftMiddleDistal),
            ("LeftRingProximal",      LeftRingProximal),
            ("LeftRingIntermediate",  LeftRingIntermediate),
            ("LeftRingDistal",        LeftRingDistal),
            ("LeftLittleProximal",    LeftLittleProximal),
            ("LeftLittleIntermediate",LeftLittleIntermediate),
            ("LeftLittleDistal",      LeftLittleDistal),
            ("RightThumbProximal",    RightThumbProximal),
            ("RightThumbIntermediate",RightThumbIntermediate),
            ("RightThumbDistal",      RightThumbDistal),
            ("RightIndexProximal",    RightIndexProximal),
            ("RightIndexIntermediate",RightIndexIntermediate),
            ("RightIndexDistal",      RightIndexDistal),
            ("RightMiddleProximal",   RightMiddleProximal),
            ("RightMiddleIntermediate",RightMiddleIntermediate),
            ("RightMiddleDistal",     RightMiddleDistal),
            ("RightRingProximal",     RightRingProximal),
            ("RightRingIntermediate", RightRingIntermediate),
            ("RightRingDistal",       RightRingDistal),
            ("RightLittleProximal",   RightLittleProximal),
            ("RightLittleIntermediate",RightLittleIntermediate),
            ("RightLittleDistal",     RightLittleDistal),
        };

        using (var writer = new StreamWriter(OutputPath, false))
        {
            foreach (var (name, t) in bones)
            {
                if (t == null)
                {
                    Debug.LogWarning($"ExportInitialPose: bone '{name}' is not assigned — skipping.");
                    continue;
                }

                // Export world-space rotation to match Zuyan's original format
                Quaternion q = t.rotation;
                writer.WriteLine(string.Format(ci,
                    "{0}:{1:F6},{2:F6},{3:F6},{4:F6}",
                    name, q.x, q.y, q.z, q.w));
            }
        }

        Debug.Log($"✅ InitialPoseExport.txt written to: {OutputPath}  ({bones.Length} bones)");
    }
}