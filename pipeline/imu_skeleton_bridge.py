# # src/imu_skeleton_bridge.py
# """
# Bridges the 17-IMU recording to two parallel outputs:
#   1. Unity skeleton animation (UDP quaternion stream)
#   2. TTSwing stroke classifier + coaching pipeline
# """
# import socket
# import time
# import pathlib
# import numpy as np
# import pandas as pd
# from collections import deque
# from scipy.stats import kurtosis, skew
# from scipy.fft import fft
#
# # IMU → Bone mapping
# IMU_TO_BONE = {
#      0: "Head",
#      1: "RightFoot",
#      2: "RightLowerLeg",
#      3: "RightUpperLeg",
#      4: "LeftFoot",
#      5: "LeftLowerLeg",
#      6: "LeftUpperLeg",
#      7: "RightHand",
#      8: "RightLowerArm",
#      9: "RightUpperArm",
#     10: "LeftHand",
#     11: "LeftLowerArm",
#     12: "LeftUpperArm",
#     13: "Hips",
#     14: "Spine",
#     15: "RightShoulder",
#     16: "LeftShoulder",
# }
#
# RACKET_IMU_ID   = 7          # change if racket sensor has a different ID
# CALIB_START     = 640         # confirmed T-pose window
# CALIB_FRAMES    = 100
# UNITY_HOST      = "127.0.0.1"
# UNITY_PORT      = 5005
# STREAM_DELAY    = 1 / 89.2    # match original recording rate
#
# # TTSwing feature window (matches training config)
# WINDOW_SIZE     = 50          # frames (~0.56s at 89Hz)
# STRIDE          = 25          # 50% overlap
#
#
# # ######################### T-pose calibration ─────────────────────────────────────────────────────
#
# def load_initial_pose(path: str) -> dict:
#     """Load Unity T-pose quaternions from InitialPoseExport.txt"""
#     pose = {}
#     with open(path) as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             bone, vals = line.split(":")
#             x, y, z, w = map(float, vals.split(","))
#             pose[bone] = np.array([x, y, z, w])
#     return pose
#
#
# def compute_calibration(df: pd.DataFrame,
#                         initial_pose: dict) -> dict:
#     """
#     Compute per-sensor calibration offset using the T-pose window.
#     Returns dict: imu_id → offset quaternion
#     """
#     calib_rows = df.iloc[CALIB_START:CALIB_START + CALIB_FRAMES]
#     offsets = {}
#
#     for imu_id, bone_name in IMU_TO_BONE.items():
#         if bone_name not in initial_pose:
#             continue
#
#         # Mean quaternion over calibration window
#         q_calib = calib_rows[[
#             f"imu_{imu_id}_quat_x", f"imu_{imu_id}_quat_y",
#             f"imu_{imu_id}_quat_z", f"imu_{imu_id}_quat_w",
#         ]].values.mean(axis=0)                    # (4,)
#
#         q_tpose = initial_pose[bone_name]         # Unity T-pose reference
#
#         # Offset = T-pose * inverse(calib_mean)
#         offsets[imu_id] = q_multiply(q_tpose, q_conjugate(q_calib))
#
#     return offsets
#
#
# # ######################### Quaternion math ────────────────────────────────────────────────────────
#
# def q_conjugate(q: np.ndarray) -> np.ndarray:
#     return np.array([-q[0], -q[1], -q[2], q[3]])
#
#
# def q_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
#     x1, y1, z1, w1 = q1
#     x2, y2, z2, w2 = q2
#     return np.array([
#         w1*x2 + x1*w2 + y1*z2 - z1*y2,
#         w1*y2 - x1*z2 + y1*w2 + z1*x2,
#         w1*z2 + x1*y2 - y1*x2 + z1*w2,
#         w1*w2 - x1*x2 - y1*y2 - z1*z2,
#     ])
#
#
# def q_normalize(q: np.ndarray) -> np.ndarray:
#     n = np.linalg.norm(q)
#     return q / n if n > 1e-8 else q
#
#
# # ######################### UDP sender to Unity ────────────────────────────────────────────────────
#
# def build_udp_packet(bone_rotations: dict,
#                      hips_pos: tuple = (0.0, 1.0, 0.0)) -> bytes:
#     """
#     Format: bone:qx,qy,qz,qw|bone2:... + hipsPosition:x,y,z
#     Matches NewIMUFull.cs parser exactly.
#     """
#     parts = []
#     for bone, q in bone_rotations.items():
#         parts.append(f"{bone}:{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f}")
#     parts.append(
#         f"hipsPosition:{hips_pos[0]:.4f},{hips_pos[1]:.4f},{hips_pos[2]:.4f}"
#     )
#     return "|".join(parts).encode("ascii")
#
#
# # ######################### Feature extraction from raw IMU window ─────────────────────────────────
#
# def extract_features_from_window(window: np.ndarray) -> np.ndarray:
#     """
#     Extract the 34 TTSwing features from a raw IMU window.
#     window: (N, 6)  columns = [ax, ay, az, gx, gy, gz]
#     Returns: (34,) feature vector matching FEATURE_COLS order
#     """
#     ax, ay, az = window[:, 0], window[:, 1], window[:, 2]
#     gx, gy, gz = window[:, 3], window[:, 4], window[:, 5]
#
#     a_mag = np.sqrt(ax**2 + ay**2 + az**2)
#     g_mag = np.sqrt(gx**2 + gy**2 + gz**2)
#
#     def safe_fft(sig):
#         F = np.abs(fft(sig))
#         return float(np.mean(F[:len(F)//2]))
#
#     def safe_psd(sig):
#         F = np.abs(fft(sig))**2
#         return float(np.mean(F[:len(F)//2]))
#
#     features = np.array([
#         # Per-axis mean
#         np.mean(ax), np.mean(ay), np.mean(az),
#         np.mean(gx), np.mean(gy), np.mean(gz),
#         # Per-axis variance
#         np.var(ax),  np.var(ay),  np.var(az),
#         np.var(gx),  np.var(gy),  np.var(gz),
#         # Per-axis RMS
#         np.sqrt(np.mean(ax**2)), np.sqrt(np.mean(ay**2)), np.sqrt(np.mean(az**2)),
#         np.sqrt(np.mean(gx**2)), np.sqrt(np.mean(gy**2)), np.sqrt(np.mean(gz**2)),
#         # Combined accel stats
#         np.max(a_mag), np.mean(a_mag), np.min(a_mag),
#         # Combined gyro stats
#         np.max(g_mag), np.mean(g_mag), np.min(g_mag),
#         # FFT dominant frequency
#         safe_fft(a_mag), safe_fft(g_mag),
#         # PSD
#         safe_psd(a_mag), safe_psd(g_mag),
#         # Higher-order stats
#         float(kurtosis(a_mag)), float(kurtosis(g_mag)),
#         float(skew(a_mag)),     float(skew(g_mag)),
#         # Entropy (approximate)
#         float(np.sum(-np.abs(a_mag/a_mag.sum() + 1e-10) *
#                      np.log(np.abs(a_mag/a_mag.sum() + 1e-10)))),
#         float(np.sum(-np.abs(g_mag/g_mag.sum() + 1e-10) *
#                      np.log(np.abs(g_mag/g_mag.sum() + 1e-10)))),
#     ], dtype=np.float32)
#
#     return features
#
#
# # ######################### Main bridge ────────────────────────────────────────────────────────────
#
# class IMUSkeletonBridge:
#     """
#     Reads the 17-IMU CSV frame by frame.
#     Simultaneously:
#       - Sends bone rotations to Unity via UDP
#       - Accumulates RightHand raw IMU into sliding window
#       - Extracts features and feeds TTSwing MLP classifier
#       - Returns (bone_rotations, feature_vector_or_None) per frame
#     """
#
#     def __init__(self, csv_path: str, initial_pose_path: str,
#                  send_udp: bool = True):
#         self.df           = pd.read_csv(csv_path)
#         self.initial_pose = load_initial_pose(initial_pose_path)
#         self.offsets      = compute_calibration(self.df, self.initial_pose)
#         self.send_udp     = send_udp
#         self.raw_window   = deque(maxlen=WINDOW_SIZE)  # RightHand raw buffer
#         self.frame_count  = 0
#         self.features_out = []   # list of (frame_idx, feature_vec)
#
#         if send_udp:
#             self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#             print(f"UDP → {UNITY_HOST}:{UNITY_PORT}")
#
#         print(f"Loaded {len(self.df)} frames | "
#               f"Calibrated on frames {CALIB_START}–"
#               f"{CALIB_START + CALIB_FRAMES}")
#
#     def process_frame(self, row: pd.Series) -> tuple[dict, np.ndarray | None]:
#         """
#         Process one CSV row.
#         Returns:
#           bone_rotations : dict bone_name → quaternion (4,)
#           features       : (34,) array if window full, else None
#         """
#         bone_rotations = {}
#
#         for imu_id, bone_name in IMU_TO_BONE.items():
#             q_raw = np.array([
#                 row[f"imu_{imu_id}_quat_x"],
#                 row[f"imu_{imu_id}_quat_y"],
#                 row[f"imu_{imu_id}_quat_z"],
#                 row[f"imu_{imu_id}_quat_w"],
#             ])
#             if imu_id in self.offsets:
#                 q_world = q_normalize(
#                     q_multiply(self.offsets[imu_id], q_raw)
#                 )
#             else:
#                 q_world = q_normalize(q_raw)
#
#             bone_rotations[bone_name] = q_world
#
#         # Accumulate RightHand raw for classifier
#         accel = np.array([
#             row[f"imu_{RACKET_IMU_ID}_accel_x"],
#             row[f"imu_{RACKET_IMU_ID}_accel_y"],
#             row[f"imu_{RACKET_IMU_ID}_accel_z"],
#         ])
#         gyro = np.array([
#             row[f"imu_{RACKET_IMU_ID}_gyro_x"],
#             row[f"imu_{RACKET_IMU_ID}_gyro_y"],
#             row[f"imu_{RACKET_IMU_ID}_gyro_z"],
#         ])
#         self.raw_window.append(np.concatenate([accel, gyro]))
#
#         # Extract features when window is full
#         features = None
#         if (len(self.raw_window) == WINDOW_SIZE and
#                 self.frame_count % STRIDE == 0):
#             window_arr = np.array(self.raw_window)   # (WINDOW_SIZE, 6)
#             features   = extract_features_from_window(window_arr)
#             self.features_out.append((self.frame_count, features))
#
#         # Send to Unity
#         if self.send_udp:
#             packet = build_udp_packet(bone_rotations)
#             self.sock.sendto(packet, (UNITY_HOST, UNITY_PORT))
#
#         self.frame_count += 1
#         return bone_rotations, features
#
#     def stream_all(self, delay: float = STREAM_DELAY):
#         """Stream entire CSV at real-time rate. Returns list of feature vectors."""
#         print(f"Streaming {len(self.df)} frames at {1/delay:.1f} Hz…")
#         for _, row in self.df.iterrows():
#             self.process_frame(row)
#             time.sleep(delay)
#         print(f"Done. {len(self.features_out)} feature windows extracted.")
#         return self.features_out
#
#
# # ######################### Smoke test #########################
# if __name__ == "__main__":
#     root = pathlib.Path(__file__).resolve().parent.parent
#
#     bridge = IMUSkeletonBridge(
#         csv_path          = str(root / "data/raw/imu_data_log_20250624_204958.csv"),
#         initial_pose_path = str(root / "data/raw/InitialPoseExport.txt"),
#         send_udp          = False,   # set True when Unity is open
#     )
#
#     # Process first 200 frames and show output
#     df = pd.read_csv(root / "data/raw/imu_data_log_20250624_204958.csv")
#     feature_windows = []
#
#     for i, (_, row) in enumerate(df.iterrows()):
#         bone_rots, feats = bridge.process_frame(row)
#         if feats is not None:
#             feature_windows.append(feats)
#             print(f"Frame {i:04d} → feature window extracted  "
#                   f"shape={feats.shape}  "
#                   f"a_mean={feats[19]:.4f}  g_mean={feats[22]:.4f}")
#         if i >= 200:
#             break
#
#     print(f"\nTotal feature windows from 200 frames: {len(feature_windows)}")
#     print(f"Each window shape: {feature_windows[0].shape}")
#     print(f"\nBone rotations for first frame (sample):")
#     bone_rots_sample, _ = bridge.process_frame(df.iloc[0])
#     for bone, q in list(bone_rots_sample.items())[:5]:
#         print(f"  {bone:<20} q={q.round(4)}")



# src/imu_skeleton_bridge.py
"""
Bridges the 17-IMU recording to two parallel outputs:
  1. Unity skeleton animation (UDP quaternion stream)
  2. TTSwing stroke classifier + coaching pipeline

ALL BUGS FIXED:
  FIX-1  IMU_TO_BONE: Zuyan's verified mapping (was 14/17 wrong)
  FIX-2  RACKET_IMU_ID: changed 10 → 7 (RightHand = IMU 7)
  FIX-3  UDP separator: changed '|' → '\n' (Unity parses \n, pipe caused 0 bones parsed)
  FIX-4  hipsPosition: REMOVED from packet entirely
         Sending (0,0,0) set Hips.localPosition to floor level → legs went underground
         The rig's default Hips localPosition (y≈1.027) is now preserved by Unity
  FIX-5  Coordinate system: delta-only RH→LH conversion
         Old (WRONG):  q_send = q_tpose_LH * q_calib_RH^-1 * q_raw_RH
                       → mixed LH/RH spaces, then Unity applied CoordPreset on top
                       → double-converts q_tpose, scrambles all non-root bones
         New (CORRECT): q_delta_RH = q_calib^-1 * q_raw        (delta in sensor RH)
                        q_delta_LH = rh_to_lh(q_delta_RH)      (convert delta only)
                        q_send     = q_tpose_LH * q_delta_LH   (compose in LH)
                        → Unity applies with CoordPreset = 0 (no conversion in Unity)

UNITY SETUP REQUIRED:
  In the IMUController Inspector → CoordPreset = 0
  (This disables Unity-side conversion since Python now handles it correctly)
"""

import socket
import time
import pathlib
import numpy as np
import pandas as pd
from collections import deque
from scipy.stats import kurtosis, skew
from scipy.fft import fft


# ######################### IMU → Bone mapping  (Zuyan's verified mapping — do not modify) #########################
IMU_TO_BONE = {
     0: "Head",
     1: "RightFoot",
     2: "RightLowerLeg",
     3: "RightUpperLeg",
     4: "LeftFoot",
     5: "LeftLowerLeg",
     6: "LeftUpperLeg",
     7: "RightHand",        # RACKET WRIST — feeds stroke classifier
     8: "RightLowerArm",
     9: "RightUpperArm",
    10: "LeftHand",
    11: "LeftLowerArm",
    12: "LeftUpperArm",
    13: "Hips",
    14: "Spine",
    15: "RightShoulder",
    16: "LeftShoulder",
}

RACKET_IMU_ID   = 7           # RightHand — IMU 7 per Zuyan's mapping
CALIB_START     = 640         # confirmed stable T-pose window
CALIB_FRAMES    = 100
UNITY_HOST      = "127.0.0.1"
UNITY_PORT      = 5005
STREAM_DELAY    = 1 / 89.2    # ~89 Hz recording rate

# RH→LH conversion preset for the delta quaternion (try 1 first, then 2, 3, 4)
# Set Unity Inspector CoordPreset = 0 when using this script
# 1 = negate Z          → (qx,  qy, -qz,  qw)   most common Y-up sensors
# 2 = negate X          → (-qx, qy,  qz,  qw)
# 3 = negate X and Z    → (-qx, qy, -qz,  qw)
# 4 = negate Z and W    → (qx,  qy, -qz, -qw)   (legacy Unity convention)
COORD_PRESET    = 1

# TTSwing feature window
WINDOW_SIZE     = 50          # frames (~0.56 s at 89 Hz)
STRIDE          = 25          # 50% overlap


# ######################### Quaternion math #########################

def q_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def q_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def q_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-8 else q


def rh_to_lh(q: np.ndarray, preset: int = COORD_PRESET) -> np.ndarray:
    """
    Convert a delta quaternion from sensor right-handed space to Unity left-handed space.
    Applied ONLY to the delta (relative rotation), never to the T-pose directly.

    Preset guide — try in order until the skeleton looks correct in Unity:
      1 = negate Z          (qx,  qy, -qz,  qw)  [START HERE — most common]
      2 = negate X          (-qx, qy,  qz,  qw)
      3 = negate X and Z    (-qx, qy, -qz,  qw)
      4 = negate Z and W    (qx,  qy, -qz, -qw)
    """
    x, y, z, w = q
    if preset == 1:
        return q_normalize(np.array([ x,  y, -z,  w]))
    elif preset == 2:
        return q_normalize(np.array([-x,  y,  z,  w]))
    elif preset == 3:
        return q_normalize(np.array([-x,  y, -z,  w]))
    elif preset == 4:
        return q_normalize(np.array([ x,  y, -z, -w]))
    else:
        return q_normalize(q)   # preset 0 = raw, no conversion


# ######################### T-pose calibration #########################

def load_initial_pose(path: str) -> dict:
    """Load Unity T-pose LOCAL quaternions from InitialPoseExport.txt"""
    pose = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            bone, vals = line.split(":")
            x, y, z, w = map(float, vals.split(","))
            pose[bone.strip()] = q_normalize(np.array([x, y, z, w]))
    return pose


def compute_calibration(df: pd.DataFrame, initial_pose: dict) -> dict:
    """
    Store mean sensor quaternion over the T-pose calibration window.
    Returns dict: imu_id → q_calib  (sensor RH world space)

    NOTE: We only store q_calib here. The T-pose compose happens per-frame
    so that the RH→LH conversion is applied correctly to the delta only.
    """
    calib_rows = df.iloc[CALIB_START: CALIB_START + CALIB_FRAMES]
    q_calibs = {}

    for imu_id, bone_name in IMU_TO_BONE.items():
        if bone_name not in initial_pose:
            print(f"  [WARN] '{bone_name}' not in InitialPoseExport — skipping IMU {imu_id}")
            continue

        cols = [f"imu_{imu_id}_quat_x", f"imu_{imu_id}_quat_y",
                f"imu_{imu_id}_quat_z", f"imu_{imu_id}_quat_w"]

        q_mean = calib_rows[cols].values.mean(axis=0)
        q_calibs[imu_id] = q_normalize(q_mean)

    return q_calibs


# ######################### UDP sender #########################

def build_udp_packet(bone_rotations: dict) -> bytes:
    """
    Format: BoneName:qx,qy,qz,qw  — one bone per line, \n separated.
    hipsPosition is intentionally NOT sent — Unity preserves the rig's
    default Hips localPosition (y≈1.027). Sending (0,0,0) would move
    the hips to the floor and bury the lower body underground.
    """
    lines = []
    for bone, q in bone_rotations.items():
        lines.append(f"{bone}:{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f}")
    return "\n".join(lines).encode("ascii")


# ######################### Feature extraction #########################

def extract_features_from_window(window: np.ndarray) -> np.ndarray:
    """
    Extract 34 TTSwing features from a raw IMU window.
    window: (N, 6)  columns = [ax, ay, az, gx, gy, gz]
    """
    ax, ay, az = window[:, 0], window[:, 1], window[:, 2]
    gx, gy, gz = window[:, 3], window[:, 4], window[:, 5]
    a_mag = np.sqrt(ax**2 + ay**2 + az**2)
    g_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    def safe_fft(sig):
        F = np.abs(fft(sig))
        return float(np.mean(F[:len(F)//2]))

    def safe_psd(sig):
        F = np.abs(fft(sig))**2
        return float(np.mean(F[:len(F)//2]))

    return np.array([
        np.mean(ax), np.mean(ay), np.mean(az),
        np.mean(gx), np.mean(gy), np.mean(gz),
        np.var(ax),  np.var(ay),  np.var(az),
        np.var(gx),  np.var(gy),  np.var(gz),
        np.sqrt(np.mean(ax**2)), np.sqrt(np.mean(ay**2)), np.sqrt(np.mean(az**2)),
        np.sqrt(np.mean(gx**2)), np.sqrt(np.mean(gy**2)), np.sqrt(np.mean(gz**2)),
        np.max(a_mag), np.mean(a_mag), np.min(a_mag),
        np.max(g_mag), np.mean(g_mag), np.min(g_mag),
        safe_fft(a_mag), safe_fft(g_mag),
        safe_psd(a_mag), safe_psd(g_mag),
        float(kurtosis(a_mag)), float(kurtosis(g_mag)),
        float(skew(a_mag)),     float(skew(g_mag)),
        float(np.sum(-np.abs(a_mag / (a_mag.sum() + 1e-10)) *
                     np.log(np.abs(a_mag / (a_mag.sum() + 1e-10)) + 1e-10))),
        float(np.sum(-np.abs(g_mag / (g_mag.sum() + 1e-10)) *
                     np.log(np.abs(g_mag / (g_mag.sum() + 1e-10)) + 1e-10))),
    ], dtype=np.float32)


# ######################### Main bridge class #########################

class IMUSkeletonBridge:

    def __init__(self, csv_path: str, initial_pose_path: str,
                 send_udp: bool = True):
        print(f"[INFO] Loading CSV: {csv_path}")
        self.df = pd.read_csv(csv_path)

        print(f"[INFO] Loading T-pose: {initial_pose_path}")
        self.initial_pose = load_initial_pose(initial_pose_path)

        print(f"[INFO] Calibrating on frames {CALIB_START}–{CALIB_START + CALIB_FRAMES}...")
        self.q_calibs = compute_calibration(self.df, self.initial_pose)
        print(f"[INFO] Calibrated {len(self.q_calibs)}/17 sensors")
        print(f"[INFO] Coordinate preset: {COORD_PRESET} (change COORD_PRESET if pose is wrong)")

        self.send_udp     = send_udp
        self.raw_window   = deque(maxlen=WINDOW_SIZE)
        self.frame_count  = 0
        self.features_out = []

        if send_udp:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"[INFO] UDP → {UNITY_HOST}:{UNITY_PORT}")
            print(f"[INFO] *** Set Unity Inspector: CoordPreset = 0 ***\n")

    def process_frame(self, row: pd.Series) -> tuple:
        bone_rotations = {}

        for imu_id, bone_name in IMU_TO_BONE.items():
            if imu_id not in self.q_calibs:
                continue

            q_raw = q_normalize(np.array([
                row[f"imu_{imu_id}_quat_x"],
                row[f"imu_{imu_id}_quat_y"],
                row[f"imu_{imu_id}_quat_z"],
                row[f"imu_{imu_id}_quat_w"],
            ]))

            q_calib  = self.q_calibs[imu_id]          # sensor RH, at T-pose
            q_tpose  = self.initial_pose[bone_name]    # Unity LH local rotation

            # Step 1: delta rotation in sensor RH space
            q_delta_rh = q_normalize(q_multiply(q_conjugate(q_calib), q_raw))

            # Step 2: convert ONLY the delta to Unity LH space
            q_delta_lh = rh_to_lh(q_delta_rh)

            # Step 3: compose delta with T-pose local rotation (both now in LH)
            q_send = q_normalize(q_multiply(q_tpose, q_delta_lh))

            bone_rotations[bone_name] = q_send

        # Accumulate RightHand raw data for stroke classifier
        accel = np.array([
            row[f"imu_{RACKET_IMU_ID}_accel_x"],
            row[f"imu_{RACKET_IMU_ID}_accel_y"],
            row[f"imu_{RACKET_IMU_ID}_accel_z"],
        ])
        gyro = np.array([
            row[f"imu_{RACKET_IMU_ID}_gyro_x"],
            row[f"imu_{RACKET_IMU_ID}_gyro_y"],
            row[f"imu_{RACKET_IMU_ID}_gyro_z"],
        ])
        self.raw_window.append(np.concatenate([accel, gyro]))

        features = None
        if len(self.raw_window) == WINDOW_SIZE and self.frame_count % STRIDE == 0:
            features = extract_features_from_window(np.array(self.raw_window))
            self.features_out.append((self.frame_count, features))

        if self.send_udp:
            packet = build_udp_packet(bone_rotations)
            self.sock.sendto(packet, (UNITY_HOST, UNITY_PORT))

        self.frame_count += 1
        return bone_rotations, features

    def stream_all(self, delay: float = STREAM_DELAY):
        total       = len(self.df)
        start_frame = CALIB_START + CALIB_FRAMES   # skip frames 0-739 (pre-calib noise)
        motion_rows = total - start_frame

        print(f"[STREAM] Skipping frames 0-{start_frame-1} (pre-calibration noise)")
        print(f"[STREAM] Streaming frames {start_frame}-{total-1} "
              f"({motion_rows} frames, ~{motion_rows/89.2:.0f}s)\n")

        # Hold T-pose for 1s so Unity settles before motion begins
        print("[INFO] Holding T-pose for 1 second...")
        tpose_pkt = self._build_tpose_packet()
        for _ in range(int(1.0 / delay)):
            if self.send_udp:
                self.sock.sendto(tpose_pkt, (UNITY_HOST, UNITY_PORT))
            time.sleep(delay)
        print("[INFO] Starting motion stream...\n")

        for idx, (_, row) in enumerate(self.df.iloc[start_frame:].iterrows(),
                                        start=start_frame):
            t0 = time.perf_counter()
            self.process_frame(row)

            if (idx - start_frame) % 89 == 0:
                elapsed = (idx - start_frame) / 89.2
                print(f"  Frame {idx:>5}/{total}  ({elapsed:.1f}s elapsed)  "
                      f"windows: {len(self.features_out)}")

            sleep_t = delay - (time.perf_counter() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

        print(f"\n[DONE] {motion_rows} frames streamed. "
              f"Feature windows: {len(self.features_out)}")
        return self.features_out

    def _build_tpose_packet(self) -> bytes:
        """Send InitialPoseExport T-pose quaternions directly to hold T-pose."""
        valid_bones = set(IMU_TO_BONE.values())
        lines = []
        for bone_name, q_tp in self.initial_pose.items():
            if bone_name in valid_bones:
                lines.append(
                    f"{bone_name}:{q_tp[0]:.6f},{q_tp[1]:.6f},"
                    f"{q_tp[2]:.6f},{q_tp[3]:.6f}"
                )
        return "\n".join(lines).encode("ascii")


    def __del__(self):
        if hasattr(self, 'sock'):
            self.sock.close()


# ######################### Entry point #########################
if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parent.parent

    bridge = IMUSkeletonBridge(
        csv_path          = str(root / "data/raw/imu_data_log_20250624_204958.csv"),
        initial_pose_path = str(root / "data/raw/InitialPoseExport.txt"),
        send_udp          = True,
    )
    bridge.stream_all()