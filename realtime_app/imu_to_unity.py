"""
imu_to_unity.py
───────────────
Reads IMU CSV data, calibrates against T-pose, computes bone
rotations, and streams them to Unity via UDP.

Runs independently — no .pyd black box needed.

Required files (all in the same folder):
  imu_data_log_*.csv
  BoneHierarchy.txt
  BoneOffsets.json
  InitialPoseExport.txt

Unity must be running with Play pressed before you run this.
"""

import csv
import json
import socket
import time
import numpy as np
from collections import defaultdict

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION  —  edit these paths to match your setup
# ─────────────────────────────────────────────────────────────
CSV_PATH       = "imu_data_log_20250624_204958.csv"
HIERARCHY_PATH = "BoneHierarchy.txt"
OFFSETS_PATH   = "BoneOffsets.json"
TPOSE_PATH     = "InitialPoseExport.txt"

UNITY_IP       = "127.0.0.1"   # same machine
UNITY_PORT     = 5005           # must match NewIMUFull.cs

CALIB_FRAMES   = 100            # frames used to compute T-pose offset
PLAYBACK_SPEED = 1.0            # 1.0 = real-time; 0.5 = half speed

# Confirmed IMU ID → bone name (discovered by discover_imu_mapping.py)
IMU_TO_BONE = {
    0:  "Head",           1:  "RightFoot",      2:  "RightLowerLeg",
    3:  "RightUpperLeg",  4:  "LeftFoot",        5:  "LeftLowerLeg",
    6:  "LeftUpperLeg",   7:  "RightHand",       8:  "RightLowerArm",
    9:  "RightUpperArm",  10: "LeftHand",        11: "LeftLowerArm",
    12: "LeftUpperArm",   13: "Hips",            14: "Spine",
    15: "RightShoulder",  16: "LeftShoulder",
}

# ─────────────────────────────────────────────────────────────
#  QUATERNION HELPERS  (format: [x, y, z, w])
# ─────────────────────────────────────────────────────────────
def q_mult(q1, q2):
    """Hamilton product."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])

def q_inv(q):
    """Inverse of a unit quaternion."""
    return np.array([-q[0], -q[1], -q[2], q[3]])

def q_norm(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([0, 0, 0, 1], dtype=float)

def q_avg(quats):
    """Simple average — accurate for small angular spread (calibration)."""
    return q_norm(np.mean(quats, axis=0))

def q_rotate(q, v):
    """Rotate 3-vector v by quaternion q."""
    qv  = np.array([v[0], v[1], v[2], 0.0])
    return q_mult(q_mult(q, qv), q_inv(q))[:3]

# ─────────────────────────────────────────────────────────────
#  FILE LOADERS
# ─────────────────────────────────────────────────────────────
def load_tpose(path):
    """InitialPoseExport.txt  →  {bone_name: np.array([x,y,z,w])}"""
    tpose = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ":" not in line:
                continue
            name, vals = line.split(":", 1)
            x, y, z, w = map(float, vals.split(","))
            tpose[name.strip()] = np.array([x, y, z, w])
    return tpose

def load_hierarchy(path):
    """BoneHierarchy.txt  →  {bone: parent_or_None}"""
    hier = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if ":" not in line:
                continue
            bone, parent = line.split(":", 1)
            hier[bone.strip()] = None if parent.strip() == "None" else parent.strip()
    return hier

def load_offsets(path):
    """
    BoneOffsets.json  →  {unity_bone_name: np.array([x,y,z])}
    Converts mixamorig: naming convention to Unity bone names.
    """
    MIXAMO_TO_UNITY = {
        "Hips": "Hips", "Spine": "Spine", "Spine1": "Chest",
        "Spine2": "UpperChest", "Neck": "Neck", "Head": "Head",
        "LeftShoulder": "LeftShoulder",  "RightShoulder": "RightShoulder",
        "LeftArm": "LeftUpperArm",       "RightArm": "RightUpperArm",
        "LeftForeArm": "LeftLowerArm",   "RightForeArm": "RightLowerArm",
        "LeftHand": "LeftHand",          "RightHand": "RightHand",
        "LeftUpLeg": "LeftUpperLeg",     "RightUpLeg": "RightUpperLeg",
        "LeftLeg": "LeftLowerLeg",       "RightLeg": "RightLowerLeg",
        "LeftFoot": "LeftFoot",          "RightFoot": "RightFoot",
        "LeftToeBase": "LeftToes",       "RightToeBase": "RightToes",
    }
    with open(path) as f:
        data = json.load(f)
    offsets = {}
    for item in data["bones"]:
        raw = item["boneName"].replace("mixamorig:", "")
        unity_name = MIXAMO_TO_UNITY.get(raw)
        if unity_name:
            offsets[unity_name] = np.array(item["localPosition"])
    return offsets

def parse_row(row):
    """CSV row  →  {imu_id: np.array([x,y,z,w])}"""
    quats = {}
    for imu_id in range(17):
        p = f"imu_{imu_id}_"
        try:
            q = np.array([
                float(row[p + "quat_x"]), float(row[p + "quat_y"]),
                float(row[p + "quat_z"]), float(row[p + "quat_w"]),
            ])
            quats[imu_id] = q_norm(q)
        except (KeyError, ValueError):
            quats[imu_id] = np.array([0.0, 0.0, 0.0, 1.0])
    return quats

# ─────────────────────────────────────────────────────────────
#  CALIBRATION
# ─────────────────────────────────────────────────────────────
def calibrate(rows, tpose_quats, n=100):
    """
    Strategy
    ─────────
    During the first n frames the subject holds T-pose.
    For each IMU, average its quaternion → q_sensor_rest.
    The T-pose bone quaternion from Unity → q_tpose.

    Calibration offset:
        q_offset = q_tpose × inv(q_sensor_rest)

    At runtime:
        q_bone = normalise(q_offset × q_sensor_live)

    This maps the sensor's world frame onto Unity's bone frame.
    """
    accum = defaultdict(list)
    for row in rows[:n]:
        for imu_id, q in parse_row(row).items():
            accum[imu_id].append(q)

    offsets = {}
    for imu_id, bone_name in IMU_TO_BONE.items():
        q_rest  = q_avg(accum[imu_id])
        q_tpose = tpose_quats.get(bone_name, np.array([0, 0, 0, 1], dtype=float))
        offsets[imu_id] = q_norm(q_mult(q_tpose, q_inv(q_rest)))

    return offsets

# ─────────────────────────────────────────────────────────────
#  FORWARD KINEMATICS  (3-D world joint positions)
# ─────────────────────────────────────────────────────────────
def forward_kinematics(bone_quats, bone_offsets, hierarchy):
    """
    Walk the skeleton tree from root (Hips) and compute the
    world-space 3-D position of every joint.

    Returns {bone_name: np.array([x, y, z])}
    """
    world_pos  = {}
    world_quat = {}

    # Topological sort (root → leaves)
    def topo(bone):
        parent = hierarchy.get(bone)
        if parent and parent not in world_pos:
            topo(parent)

        if parent is None:
            # Root
            world_pos[bone]  = bone_offsets.get(bone, np.zeros(3))
            world_quat[bone] = bone_quats.get(bone, np.array([0,0,0,1.0]))
        else:
            offset = bone_offsets.get(bone, np.zeros(3))
            rotated_offset = q_rotate(world_quat[parent], offset)
            world_pos[bone]  = world_pos[parent] + rotated_offset
            world_quat[bone] = bone_quats.get(bone, np.array([0,0,0,1.0]))

    for bone in hierarchy:
        if bone not in world_pos:
            topo(bone)

    return world_pos

# ─────────────────────────────────────────────────────────────
#  PER-FRAME PROCESSING
# ─────────────────────────────────────────────────────────────
def compute_bone_rotations(row, cal_offsets):
    """Apply calibration and return {bone_name: quaternion}."""
    sensor_quats = parse_row(row)
    bone_quats = {}
    for imu_id, bone_name in IMU_TO_BONE.items():
        q_live = sensor_quats[imu_id]
        bone_quats[bone_name] = q_norm(q_mult(cal_offsets[imu_id], q_live))
    return bone_quats

def build_udp_message(bone_quats, hips_pos=(0.0, 1.0, 0.0)):
    """
    Format expected by NewIMUFull.cs:
        BoneName:qx,qy,qz,qw
        hipsPosition:x,y,z
    """
    lines = [
        f"{bone}:{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f}"
        for bone, q in bone_quats.items()
    ]
    lines.append(
        f"hipsPosition:{hips_pos[0]:.4f},{hips_pos[1]:.4f},{hips_pos[2]:.4f}"
    )
    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  IMU → Unity skeleton streamer")
    print("=" * 50)

    # Load files
    print("\n[1/4] Loading config files...")
    tpose     = load_tpose(TPOSE_PATH)
    hierarchy = load_hierarchy(HIERARCHY_PATH)
    offsets   = load_offsets(OFFSETS_PATH)
    print(f"      T-pose: {len(tpose)} bones  |  Hierarchy: {len(hierarchy)} bones")

    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"      CSV: {len(rows)} frames")

    # Calibrate
    print(f"\n[2/4] Calibrating with first {CALIB_FRAMES} frames (keep T-pose)...")
    cal_offsets = calibrate(rows, tpose, n=CALIB_FRAMES)
    print("      Done.")

    # Set up UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    work_rows = rows[CALIB_FRAMES:]
    print(f"\n[3/4] Streaming {len(work_rows)} frames → {UNITY_IP}:{UNITY_PORT}")
    print("      Make sure Unity is open and Play is pressed!\n")
    input("      Press Enter when Unity is ready...")

    # Stream frames
    print("\n[4/4] Streaming... (Ctrl+C to stop)\n")
    prev_ts = None
    try:
        for i, row in enumerate(work_rows):
            ts = float(row["timestamp"])

            if prev_ts is not None:
                dt = (ts - prev_ts) / PLAYBACK_SPEED
                if 0 < dt < 0.5:
                    time.sleep(dt)
            prev_ts = ts

            bone_quats = compute_bone_rotations(row, cal_offsets)
            msg = build_udp_message(bone_quats)
            sock.sendto(msg.encode("ascii"), (UNITY_IP, UNITY_PORT))

            if i % 100 == 0:
                pct = 100 * i / len(work_rows)
                print(f"  Frame {i}/{len(work_rows)}  ({pct:.0f}%)")

    except KeyboardInterrupt:
        print("\n  Stopped by user.")

    sock.close()
    print("\nDone.")


if __name__ == "__main__":
    main()






# """
# imu_to_unity.py
# ───────────────
# Reads IMU CSV data, calibrates against T-pose, computes bone
# rotations, and streams them to Unity via UDP.
#
# Runs independently — no .pyd black box needed.
#
# Required files (all in the same folder):
#   imu_data_log_*.csv
#   BoneHierarchy.txt
#   BoneOffsets.json
#   InitialPoseExport.txt
#
# Unity must be running with Play pressed before you run this.
# """
#
# import csv
# import json
# import socket
# import time
# import numpy as np
# from collections import defaultdict
#
# # ─────────────────────────────────────────────────────────────
# #  CONFIGURATION
# # ─────────────────────────────────────────────────────────────
# CSV_PATH       = "imu_data_log_20250624_204958.csv"
# HIERARCHY_PATH = "BoneHierarchy.txt"
# OFFSETS_PATH   = "BoneOffsets.json"
# TPOSE_PATH     = "InitialPoseExport.txt"
#
# UNITY_IP       = "127.0.0.1"
# UNITY_PORT     = 5005
#
# CALIB_START    = 0            # frame where subject holds stable T-pose
# CALIB_FRAMES   = 100            # number of frames used for calibration
# PLAYBACK_SPEED = 1.0
#
# # ─────────────────────────────────────────────────────────────
# #  COORDINATE CONVERSION
# #  The IMU outputs quaternions in a right-handed frame.
# #  Unity uses a left-handed Y-up frame.
# #  Try each option (1–4) until motion looks correct.
# #
# #  1 = Z-up right-hand → Y-up left-hand  (x, z, y, -w)   ← try first
# #  2 = Negate X and Z                    (-x, y, -z, w)
# #  3 = Negate Y and W                    (x, -y, z, -w)
# #  4 = No conversion — raw               (x, y, z, w)
# # ─────────────────────────────────────────────────────────────
# COORD_MODE = 4
#
# def convert_imu_quat(x, y, z, w):
#     """Convert raw IMU quaternion to Unity coordinate frame."""
#     if COORD_MODE == 1:
#         return np.array([ x,  z,  y, -w])   # Z-up RH → Y-up LH
#     elif COORD_MODE == 2:
#         return np.array([-x,  y, -z,  w])   # flip X and Z
#     elif COORD_MODE == 3:
#         return np.array([ x, -y,  z, -w])   # flip Y and W
#     else:
#         return np.array([ x,  y,  z,  w])   # raw, no conversion
#
# # ─────────────────────────────────────────────────────────────
# #  IMU → BONE MAPPING
# # ─────────────────────────────────────────────────────────────
# IMU_TO_BONE = {
#     0:  "Head",           1:  "RightFoot",      2:  "RightLowerLeg",
#     3:  "RightUpperLeg",  4:  "LeftFoot",        5:  "LeftLowerLeg",
#     6:  "LeftUpperLeg",   7:  "RightHand",       8:  "RightLowerArm",
#     9:  "RightUpperArm",  10: "LeftHand",        11: "LeftLowerArm",
#     12: "LeftUpperArm",   13: "Hips",            14: "Spine",
#     15: "RightShoulder",  16: "LeftShoulder",
# }
#
# # ─────────────────────────────────────────────────────────────
# #  QUATERNION HELPERS  (format: [x, y, z, w])
# # ─────────────────────────────────────────────────────────────
# def q_mult(q1, q2):
#     x1, y1, z1, w1 = q1
#     x2, y2, z2, w2 = q2
#     return np.array([
#         w1*x2 + x1*w2 + y1*z2 - z1*y2,
#         w1*y2 - x1*z2 + y1*w2 + z1*x2,
#         w1*z2 + x1*y2 - y1*x2 + z1*w2,
#         w1*w2 - x1*x2 - y1*y2 - z1*z2,
#     ])
#
# def q_inv(q):
#     return np.array([-q[0], -q[1], -q[2], q[3]])
#
# def q_norm(q):
#     n = np.linalg.norm(q)
#     return q / n if n > 1e-9 else np.array([0, 0, 0, 1], dtype=float)
#
# def q_avg(quats):
#     return q_norm(np.mean(quats, axis=0))
#
# # ─────────────────────────────────────────────────────────────
# #  FILE LOADERS
# # ─────────────────────────────────────────────────────────────
# def load_tpose(path):
#     tpose = {}
#     with open(path) as f:
#         for line in f:
#             line = line.strip()
#             if ":" not in line:
#                 continue
#             name, vals = line.split(":", 1)
#             x, y, z, w = map(float, vals.split(","))
#             tpose[name.strip()] = np.array([x, y, z, w])
#     return tpose
#
# def load_hierarchy(path):
#     hier = {}
#     with open(path) as f:
#         for line in f:
#             line = line.strip()
#             if ":" not in line:
#                 continue
#             bone, parent = line.split(":", 1)
#             hier[bone.strip()] = None if parent.strip() == "None" else parent.strip()
#     return hier
#
# def load_offsets(path):
#     MIXAMO_TO_UNITY = {
#         "Hips": "Hips", "Spine": "Spine", "Spine1": "Chest",
#         "Spine2": "UpperChest", "Neck": "Neck", "Head": "Head",
#         "LeftShoulder": "LeftShoulder",  "RightShoulder": "RightShoulder",
#         "LeftArm": "LeftUpperArm",       "RightArm": "RightUpperArm",
#         "LeftForeArm": "LeftLowerArm",   "RightForeArm": "RightLowerArm",
#         "LeftHand": "LeftHand",          "RightHand": "RightHand",
#         "LeftUpLeg": "LeftUpperLeg",     "RightUpLeg": "RightUpperLeg",
#         "LeftLeg": "LeftLowerLeg",       "RightLeg": "RightLowerLeg",
#         "LeftFoot": "LeftFoot",          "RightFoot": "RightFoot",
#         "LeftToeBase": "LeftToes",       "RightToeBase": "RightToes",
#     }
#     with open(path) as f:
#         data = json.load(f)
#     offsets = {}
#     for item in data["bones"]:
#         raw = item["boneName"].replace("mixamorig:", "")
#         unity_name = MIXAMO_TO_UNITY.get(raw)
#         if unity_name:
#             offsets[unity_name] = np.array(item["localPosition"])
#     return offsets
#
# def parse_row(row):
#     """CSV row → {imu_id: converted quaternion in Unity frame}"""
#     quats = {}
#     for imu_id in range(17):
#         p = f"imu_{imu_id}_"
#         try:
#             q = convert_imu_quat(
#                 float(row[p + "quat_x"]),
#                 float(row[p + "quat_y"]),
#                 float(row[p + "quat_z"]),
#                 float(row[p + "quat_w"]),
#             )
#             quats[imu_id] = q_norm(q)
#         except (KeyError, ValueError):
#             quats[imu_id] = np.array([0.0, 0.0, 0.0, 1.0])
#     return quats
#
# # ─────────────────────────────────────────────────────────────
# #  CALIBRATION
# # ─────────────────────────────────────────────────────────────
# def calibrate(rows, tpose_quats, start=0, n=100):
#     accum = defaultdict(list)
#     for row in rows[start : start + n]:
#         for imu_id, q in parse_row(row).items():
#             accum[imu_id].append(q)
#
#     offsets = {}
#     for imu_id, bone_name in IMU_TO_BONE.items():
#         q_rest  = q_avg(accum[imu_id])
#         q_tpose = tpose_quats.get(bone_name, np.array([0, 0, 0, 1], dtype=float))
#         offsets[imu_id] = q_norm(q_mult(q_tpose, q_inv(q_rest)))
#
#     return offsets
#
# # ─────────────────────────────────────────────────────────────
# #  PER-FRAME PROCESSING
# # ─────────────────────────────────────────────────────────────
# def compute_bone_rotations(row, cal_offsets):
#     sensor_quats = parse_row(row)
#     bone_quats = {}
#     for imu_id, bone_name in IMU_TO_BONE.items():
#         q_live = sensor_quats[imu_id]
#         bone_quats[bone_name] = q_norm(q_mult(cal_offsets[imu_id], q_live))
#     return bone_quats
#
# def build_udp_message(bone_quats, hips_pos=(0.0, 1.0, 0.0)):
#     lines = [
#         f"{bone}:{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f}"
#         for bone, q in bone_quats.items()
#     ]
#     lines.append(f"hipsPosition:{hips_pos[0]:.4f},{hips_pos[1]:.4f},{hips_pos[2]:.4f}")
#     return "\n".join(lines)
#
# # ─────────────────────────────────────────────────────────────
# #  MAIN
# # ─────────────────────────────────────────────────────────────
# def main():
#     print("=" * 50)
#     print("  IMU → Unity skeleton streamer")
#     print(f"  Coordinate mode: {COORD_MODE}")
#     print("=" * 50)
#
#     print("\n[1/4] Loading config files...")
#     tpose     = load_tpose(TPOSE_PATH)
#     hierarchy = load_hierarchy(HIERARCHY_PATH)
#     offsets   = load_offsets(OFFSETS_PATH)
#     print(f"      T-pose: {len(tpose)} bones  |  Hierarchy: {len(hierarchy)} bones")
#
#     with open(CSV_PATH, newline="") as f:
#         rows = list(csv.DictReader(f))
#     print(f"      CSV: {len(rows)} frames")
#
#     print(f"\n[2/4] Calibrating with frames {CALIB_START}–{CALIB_START + CALIB_FRAMES} ...")
#     cal_offsets = calibrate(rows, tpose, start=CALIB_START, n=CALIB_FRAMES)
#     print("      Done.")
#
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     work_rows = rows[CALIB_START + CALIB_FRAMES:]
#     print(f"\n[3/4] Streaming {len(work_rows)} frames → {UNITY_IP}:{UNITY_PORT}")
#     print("      Make sure Unity is open and Play is pressed!\n")
#     input("      Press Enter when Unity is ready...")
#
#     print("\n[4/4] Streaming... (Ctrl+C to stop)\n")
#     prev_ts = None
#     try:
#         for i, row in enumerate(work_rows):
#             ts = float(row["timestamp"])
#             if prev_ts is not None:
#                 dt = (ts - prev_ts) / PLAYBACK_SPEED
#                 if 0 < dt < 0.5:
#                     time.sleep(dt)
#             prev_ts = ts
#
#             bone_quats = compute_bone_rotations(row, cal_offsets)
#             msg = build_udp_message(bone_quats)
#             if 700 <= i <= 705:
#                 print(f"--- Frame {i} ---")
#                 print(msg)
#                 print()
#
#
#
#             if i < 3:
#                 print(f"--- Frame {i} ---")
#                 print(msg)
#                 print()
#
#
#             sock.sendto(msg.encode("ascii"), (UNITY_IP, UNITY_PORT))
#
#             if i % 100 == 0:
#                 pct = 100 * i / len(work_rows)
#                 print(f"  Frame {i}/{len(work_rows)}  ({pct:.0f}%)")
#
#     except KeyboardInterrupt:
#         print("\n  Stopped by user.")
#
#     sock.close()
#     print("\nDone.")
#
#
# if __name__ == "__main__":
#     main()



# """
# imu_to_unity.py
# ───────────────
# Reads IMU CSV data, calibrates against T-pose, computes bone
# rotations, and streams them to Unity via UDP.
#
# Key fix: quaternion continuity check prevents sign-flip artifacts.
# """
#
# import csv
# import json
# import socket
# import time
# import numpy as np
# from collections import defaultdict
#
# # ─────────────────────────────────────────────────────────────
# #  CONFIGURATION
# # ─────────────────────────────────────────────────────────────
# CSV_PATH       = "imu_data_log_20250624_204958.csv"
# HIERARCHY_PATH = "BoneHierarchy.txt"
# OFFSETS_PATH   = "BoneOffsets.json"
# TPOSE_PATH     = "InitialPoseExport.txt"
#
# UNITY_IP       = "127.0.0.1"
# UNITY_PORT     = 5005
#
# CALIB_START    = 0
# CALIB_FRAMES   = 100
# PLAYBACK_SPEED = 1.0
#
# IMU_TO_BONE = {
#     0:  "Head",           1:  "RightFoot",      2:  "RightLowerLeg",
#     3:  "RightUpperLeg",  4:  "LeftFoot",        5:  "LeftLowerLeg",
#     6:  "LeftUpperLeg",   7:  "RightHand",       8:  "RightLowerArm",
#     9:  "RightUpperArm",  10: "LeftHand",        11: "LeftLowerArm",
#     12: "LeftUpperArm",   13: "Hips",            14: "Spine",
#     15: "RightShoulder",  16: "LeftShoulder",
# }
#
#
# # ─────────────────────────────────────────────────────────────
# #  QUATERNION HELPERS
# # ─────────────────────────────────────────────────────────────
# def q_mult(q1, q2):
#     x1, y1, z1, w1 = q1
#     x2, y2, z2, w2 = q2
#     return np.array([
#         w1*x2 + x1*w2 + y1*z2 - z1*y2,
#         w1*y2 - x1*z2 + y1*w2 + z1*x2,
#         w1*z2 + x1*y2 - y1*x2 + z1*w2,
#         w1*w2 - x1*x2 - y1*y2 - z1*z2,
#     ])
#
# def q_inv(q):
#     return np.array([-q[0], -q[1], -q[2], q[3]])
#
# def q_norm(q):
#     n = np.linalg.norm(q)
#     return q / n if n > 1e-9 else np.array([0, 0, 0, 1], dtype=float)
#
# def q_avg(quats):
#     return q_norm(np.mean(quats, axis=0))
#
# def q_ensure_continuity(q_current, q_prev):
#     """
#     Quaternion double-cover fix.
#     q and -q represent the same rotation. If the dot product with
#     the previous frame is negative, negate to prevent 360° snap flips.
#     """
#     if q_prev is None:
#         return q_current
#     if np.dot(q_current, q_prev) < 0:
#         return -q_current
#     return q_current
#
# # ─────────────────────────────────────────────────────────────
# #  FILE LOADERS
# # ─────────────────────────────────────────────────────────────
# def load_tpose(path):
#     tpose = {}
#     with open(path) as f:
#         for line in f:
#             line = line.strip()
#             if ":" not in line:
#                 continue
#             name, vals = line.split(":", 1)
#             x, y, z, w = map(float, vals.split(","))
#             tpose[name.strip()] = np.array([x, y, z, w])
#     return tpose
#
# def load_hierarchy(path):
#     hier = {}
#     with open(path) as f:
#         for line in f:
#             line = line.strip()
#             if ":" not in line:
#                 continue
#             bone, parent = line.split(":", 1)
#             hier[bone.strip()] = None if parent.strip() == "None" else parent.strip()
#     return hier
#
# def load_offsets(path):
#     MIXAMO_TO_UNITY = {
#         "Hips": "Hips", "Spine": "Spine", "Spine1": "Chest",
#         "Spine2": "UpperChest", "Neck": "Neck", "Head": "Head",
#         "LeftShoulder": "LeftShoulder",  "RightShoulder": "RightShoulder",
#         "LeftArm": "LeftUpperArm",       "RightArm": "RightUpperArm",
#         "LeftForeArm": "LeftLowerArm",   "RightForeArm": "RightLowerArm",
#         "LeftHand": "LeftHand",          "RightHand": "RightHand",
#         "LeftUpLeg": "LeftUpperLeg",     "RightUpLeg": "RightUpperLeg",
#         "LeftLeg": "LeftLowerLeg",       "RightLeg": "RightLowerLeg",
#         "LeftFoot": "LeftFoot",          "RightFoot": "RightFoot",
#         "LeftToeBase": "LeftToes",       "RightToeBase": "RightToes",
#     }
#     with open(path) as f:
#         data = json.load(f)
#     offsets = {}
#     for item in data["bones"]:
#         raw = item["boneName"].replace("mixamorig:", "")
#         unity_name = MIXAMO_TO_UNITY.get(raw)
#         if unity_name:
#             offsets[unity_name] = np.array(item["localPosition"])
#     return offsets
#
# def parse_row(row):
#     quats = {}
#     for imu_id in range(17):
#         p = f"imu_{imu_id}_"
#         try:
#             q = np.array([
#                 float(row[p + "quat_x"]),
#                 float(row[p + "quat_y"]),
#                 float(row[p + "quat_z"]),
#                 float(row[p + "quat_w"]),
#             ])
#             quats[imu_id] = q_norm(q)
#         except (KeyError, ValueError):
#             quats[imu_id] = np.array([0.0, 0.0, 0.0, 1.0])
#     return quats
#
# # ─────────────────────────────────────────────────────────────
# #  CALIBRATION
# # ─────────────────────────────────────────────────────────────
# def calibrate(rows, tpose_quats, start=0, n=100):
#     accum = defaultdict(list)
#     for row in rows[start : start + n]:
#         for imu_id, q in parse_row(row).items():
#             accum[imu_id].append(q)
#
#     offsets = {}
#     for imu_id, bone_name in IMU_TO_BONE.items():
#         q_rest  = q_avg(accum[imu_id])
#         q_tpose = tpose_quats.get(bone_name, np.array([0, 0, 0, 1], dtype=float))
#         offsets[imu_id] = q_norm(q_mult(q_tpose, q_inv(q_rest)))
#     return offsets
#
# # ─────────────────────────────────────────────────────────────
# #  PER-FRAME PROCESSING
# # ─────────────────────────────────────────────────────────────
# def compute_bone_rotations(row, cal_offsets, prev_quats):
#     """
#     Apply calibration + quaternion continuity fix.
#     prev_quats: {bone_name: previous frame quaternion} for sign-flip detection.
#     """
#     sensor_quats = parse_row(row)
#     bone_quats = {}
#     for imu_id, bone_name in IMU_TO_BONE.items():
#         q_live = sensor_quats[imu_id]
#         q_bone = q_norm(q_mult(cal_offsets[imu_id], q_live))
#
#         # Continuity fix — prevent 360° snap from quaternion sign flip
#         q_bone = q_ensure_continuity(q_bone, prev_quats.get(bone_name))
#
#         bone_quats[bone_name] = q_bone
#     return bone_quats
#
# def build_udp_message(bone_quats, hips_pos=(0.0, 1.0, 0.0)):
#     lines = [
#         f"{bone}:{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f}"
#         for bone, q in bone_quats.items()
#     ]
#     lines.append(f"hipsPosition:{hips_pos[0]:.4f},{hips_pos[1]:.4f},{hips_pos[2]:.4f}")
#     return "\n".join(lines)
#
# # ─────────────────────────────────────────────────────────────
# #  MAIN
# # ─────────────────────────────────────────────────────────────
# def main():
#     print("=" * 50)
#     print("  IMU → Unity skeleton streamer")
#     print("  Quaternion continuity fix: ON")
#     print("=" * 50)
#
#     print("\n[1/4] Loading config files...")
#     tpose     = load_tpose(TPOSE_PATH)
#     hierarchy = load_hierarchy(HIERARCHY_PATH)
#     offsets   = load_offsets(OFFSETS_PATH)
#     print(f"      T-pose: {len(tpose)} bones  |  Hierarchy: {len(hierarchy)} bones")
#
#     with open(CSV_PATH, newline="") as f:
#         rows = list(csv.DictReader(f))
#     print(f"      CSV: {len(rows)} frames")
#
#     print(f"\n[2/4] Calibrating with frames {CALIB_START}–{CALIB_START + CALIB_FRAMES} ...")
#     cal_offsets = calibrate(rows, tpose, start=CALIB_START, n=CALIB_FRAMES)
#     print("      Done.")
#
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     work_rows = rows[CALIB_START + CALIB_FRAMES:]
#     print(f"\n[3/4] Streaming {len(work_rows)} frames → {UNITY_IP}:{UNITY_PORT}")
#     print("      Make sure Unity is open and Play is pressed!\n")
#     input("      Press Enter when Unity is ready...")
#
#     print("\n[4/4] Streaming... (Ctrl+C to stop)\n")
#
#     prev_quats = {}   # tracks previous frame quaternions for continuity fix
#     prev_ts = None
#
#     try:
#         for i, row in enumerate(work_rows):
#             ts = float(row["timestamp"])
#             if prev_ts is not None:
#                 dt = (ts - prev_ts) / PLAYBACK_SPEED
#                 if 0 < dt < 0.5:
#                     time.sleep(dt)
#             prev_ts = ts
#
#             bone_quats = compute_bone_rotations(row, cal_offsets, prev_quats)
#             prev_quats = bone_quats   # update for next frame
#
#             msg = build_udp_message(bone_quats)
#             sock.sendto(msg.encode("ascii"), (UNITY_IP, UNITY_PORT))
#
#             if i % 100 == 0:
#                 pct = 100 * i / len(work_rows)
#                 print(f"  Frame {i}/{len(work_rows)}  ({pct:.0f}%)")
#
#     except KeyboardInterrupt:
#         print("\n  Stopped by user.")
#
#     sock.close()
#     print("\nDone.")
#
#
# if __name__ == "__main__":
#     main()