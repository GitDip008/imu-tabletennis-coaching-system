"""
opensrc_imu_to_unity.py
========================
Streams the Scientific Data 17-IMU HAR dataset (P#.csv) to Unity via UDP,
using the same packet format as imu_skeleton_bridge.py / NewIMUFull.cs.

PURPOSE:
  Validate whether the Unity skeleton pipeline itself is correct by feeding
  known-good open-source data through it. If Unity renders cleanly here but
  scrambles with your own hardware CSV → the bug is in your Python calibration
  or CSV format. If it scrambles here too → the bug is in NewIMUFull.cs or
  the coordinate conversion.

DATASET:
  Scientific Data (Nature, 2025): "A comprehensive IMU dataset for evaluating
  sensor layouts in human activity and intensity recognition"
  - 17 IMUs, 60 Hz, CSV per participant, quaternion columns included
  - Download: https://figshare.com  (search paper title or DOI)
  - File naming: P1.csv, P2.csv, ... P30.csv

COLUMN FORMAT in dataset CSV:
  {BodyPart}_{Signal}_{axis}
  e.g. LowerBack_Quat_x, LowerBack_Quat_y, LowerBack_Quat_z, LowerBack_Quat_w

UNITY PACKET FORMAT (matching NewIMUFull.cs boneMap):
  BoneName:qx,qy,qz,qw\n  (per bone, all in one UDP datagram per frame)
  hipsPosition:0,0,0\n     (always zero — no FK position, localPosition mode)

USAGE:
  python opensrc_imu_to_unity.py --csv P1.csv [--host 127.0.0.1] [--port 5005]
                                 [--fps 60] [--start_frame 0] [--preset 1]

COORD PRESETS (mirror NewIMUFull.cs CoordPreset):
  1 = (qx, qy, -qz, -qw)   right-hand → left-hand basic  [start here]
  2 = (-qx, qy, qz, -qw)   flip X and W
  3 = (qx, -qy, qz, -qw)   flip Y and W
  4 = (-qx, -qy, qz, qw)   flip X and Y
  0 = raw (no conversion)
"""

import argparse
import csv
import socket
import time
import sys

# ─── Bone name mapping: Dataset column prefix → Unity boneMap key ───────────
DATASET_TO_UNITY = {
    "LowerBack": "Hips",
    "UpperBack": "Spine",
    "Head": "Head",
    "LeftShoulder": "LeftShoulder",
    "RightShoulder": "RightShoulder",
    "LeftUpperArm": "LeftUpperArm",
    "RightUpperArm": "RightUpperArm",
    "LeftForeArm": "LeftLowerArm",
    "RightForeArm": "RightLowerArm",
    "LeftWrist": "LeftHand",
    "RightWrist": "RightHand",
    "LeftThigh": "LeftUpperLeg",
    "RightThigh": "RightUpperLeg",
    "LeftShank": "LeftLowerLeg",
    "RightShank": "RightLowerLeg",
    "LeftFoot": "LeftFoot",
    "RightFoot": "RightFoot",
}

# Quaternion column format: Quat_{Axis}_{BodyPart}  e.g. Quat_X_LowerBack
QUAT_AXES = ["Quat_X", "Quat_Y", "Quat_Z", "Quat_W"]


def convert_quaternion(qx, qy, qz, qw, preset):
    """
    Mirror the ConvertQuaternion() presets in NewIMUFull.cs.
    Dataset uses right-handed world frame; Unity uses left-handed.
    Change preset via --preset flag to match whatever CoordPreset
    you have set in the Unity Inspector.
    """
    if preset == 1:
        return qx, qy, -qz, -qw  # right-hand → left-hand basic
    elif preset == 2:
        return -qx, qy, qz, -qw  # flip X and W
    elif preset == 3:
        return qx, -qy, qz, -qw  # flip Y and W
    elif preset == 4:
        return -qx, -qy, qz, qw  # flip X and Y
    else:
        return qx, qy, qz, qw  # raw, no conversion (preset 0)


def build_column_index(headers):
    """
    Build a mapping: (unity_bone_name) → (col_idx_qx, col_idx_qy, col_idx_qz, col_idx_qw)
    Raises KeyError with a clear message if any expected column is missing.
    """
    col_idx = {h: i for i, h in enumerate(headers)}
    bone_cols = {}

    for dataset_bone, unity_bone in DATASET_TO_UNITY.items():
        try:
            indices = tuple(col_idx[f"{ax}_{dataset_bone}"] for ax in QUAT_AXES)
            bone_cols[unity_bone] = indices
        except KeyError as e:
            print(f"[ERROR] Column not found in CSV: {e}")
            print(f"        Expected 'Quat_X/Y/Z/W_{dataset_bone}' for Unity bone '{unity_bone}'")
            print(f"        Available columns starting with '{dataset_bone}':")
            matched = [h for h in headers if dataset_bone in h]
            print(f"        {matched if matched else 'NONE — check dataset_bone name'}")
            sys.exit(1)

    return bone_cols


def stream_csv(csv_path, host, port, fps, start_frame, preset):
    frame_delay = 1.0 / fps

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"[INFO] Opening CSV: {csv_path}")
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader)
        print(f"[INFO] CSV has {len(headers)} columns")

        # Detect if there's a leading index/timestamp column
        # (dataset may or may not have one)
        if headers[0].lower() in ("", "index", "frame", "timestamp", "time"):
            print(f"[INFO] Skipping first column '{headers[0]}' (index/timestamp)")

        bone_cols = build_column_index(headers)
        print(f"[INFO] Mapped {len(bone_cols)}/17 bones successfully")
        print(f"[INFO] Streaming to {host}:{port} at {fps} Hz | CoordPreset={preset}")
        print(f"[INFO] Press Ctrl+C to stop\n")

        rows = list(reader)  # load all rows to allow start_frame skip
        total_frames = len(rows)
        print(f"[INFO] Total frames in CSV: {total_frames}")

        if start_frame >= total_frames:
            print(f"[ERROR] start_frame={start_frame} >= total_frames={total_frames}")
            sys.exit(1)

        for frame_idx, row in enumerate(rows[start_frame:], start=start_frame):
            t_start = time.perf_counter()

            lines = []

            # ── Build bone rotation lines ────────────────────────────────────
            for unity_bone, (cx, cy, cz, cw) in bone_cols.items():
                try:
                    qx = float(row[cx])
                    qy = float(row[cy])
                    qz = float(row[cz])
                    qw = float(row[cw])
                except (ValueError, IndexError):
                    continue  # skip malformed rows silently

                qx, qy, qz, qw = convert_quaternion(qx, qy, qz, qw, preset)
                lines.append(f"{unity_bone}:{qx:.6f},{qy:.6f},{qz:.6f},{qw:.6f}")

            # ── Hips world position — zero (localPosition mode) ──────────────
            lines.append("hipsPosition:0,0,0")

            packet = "\n".join(lines).encode("ascii")
            sock.sendto(packet, (host, port))

            # ── Progress print every 60 frames ───────────────────────────────
            if frame_idx % 60 == 0:
                elapsed = frame_idx - start_frame
                print(f"  Frame {frame_idx}/{total_frames}  ({elapsed / fps:.1f}s elapsed)")

            # ── Rate control ─────────────────────────────────────────────────
            elapsed_this_frame = time.perf_counter() - t_start
            sleep_time = frame_delay - elapsed_this_frame
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("\n[INFO] Stream complete.")
    sock.close()


def main():
    parser = argparse.ArgumentParser(
        description="Stream open-source 17-IMU CSV → Unity UDP skeleton pipeline"
    )
    parser.add_argument("--csv", required=True, help="Path to P#.csv from the dataset")
    parser.add_argument("--host", default="127.0.0.1", help="Unity UDP host (default: 127.0.0.1)")
    parser.add_argument("--port", default=5005, type=int, help="Unity UDP port (default: 5005)")
    parser.add_argument("--fps", default=60, type=int, help="Playback FPS (default: 60, matches dataset rate)")
    parser.add_argument("--start_frame", default=0, type=int, help="Start from this frame index (default: 0)")
    parser.add_argument("--preset", default=1, type=int, choices=[0, 1, 2, 3, 4],
                        help="Coord conversion preset (0=raw, 1-4 mirror Unity CoordPreset). Default: 1")
    args = parser.parse_args()

    stream_csv(
        csv_path=args.csv,
        host=args.host,
        port=args.port,
        fps=args.fps,
        start_frame=args.start_frame,
        preset=args.preset,
    )


if __name__ == "__main__":
    main()