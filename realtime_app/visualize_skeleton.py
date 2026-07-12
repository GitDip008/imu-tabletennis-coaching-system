"""
visualize_skeleton.py
─────────────────────
Shows the 3-D skeleton reconstructed from IMU data using matplotlib.
Use this to verify your skeleton looks correct BEFORE connecting Unity.
Press the arrow keys or use the slider to scrub through frames.

Run:  python visualize_skeleton.py
"""

import csv
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict

# ── point these at your files ──────────────────────
CSV_PATH       = "imu_data_log_20250624_204958.csv"
HIERARCHY_PATH = "BoneHierarchy.txt"
OFFSETS_PATH   = "BoneOffsets.json"
TPOSE_PATH     = "InitialPoseExport.txt"
CALIB_FRAMES   = 100
# ───────────────────────────────────────────────────

IMU_TO_BONE = {
    0: "Head",           1: "RightFoot",      2: "RightLowerLeg",
    3: "RightUpperLeg",  4: "LeftFoot",        5: "LeftLowerLeg",
    6: "LeftUpperLeg",   7: "RightHand",       8: "RightLowerArm",
    9: "RightUpperArm",  10: "LeftHand",       11: "LeftLowerArm",
    12: "LeftUpperArm",  13: "Hips",           14: "Spine",
    15: "RightShoulder", 16: "LeftShoulder",
}

# Skeleton connections to draw as lines
SKELETON_EDGES = [
    ("Hips", "Spine"), ("Spine", "Chest"), ("Chest", "UpperChest"),
    ("UpperChest", "Neck"), ("Neck", "Head"),
    ("UpperChest", "LeftShoulder"), ("LeftShoulder", "LeftUpperArm"),
    ("LeftUpperArm", "LeftLowerArm"), ("LeftLowerArm", "LeftHand"),
    ("UpperChest", "RightShoulder"), ("RightShoulder", "RightUpperArm"),
    ("RightUpperArm", "RightLowerArm"), ("RightLowerArm", "RightHand"),
    ("Hips", "LeftUpperLeg"), ("LeftUpperLeg", "LeftLowerLeg"),
    ("LeftLowerLeg", "LeftFoot"), ("LeftFoot", "LeftToes"),
    ("Hips", "RightUpperLeg"), ("RightUpperLeg", "RightLowerLeg"),
    ("RightLowerLeg", "RightFoot"), ("RightFoot", "RightToes"),
]

# ── copy helpers from imu_to_unity.py ──────────────
def q_mult(q1, q2):
    x1,y1,z1,w1 = q1; x2,y2,z2,w2 = q2
    return np.array([w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2, w1*w2-x1*x2-y1*y2-z1*z2])
def q_inv(q):  return np.array([-q[0],-q[1],-q[2],q[3]])
def q_norm(q):
    n = np.linalg.norm(q)
    return q/n if n > 1e-9 else np.array([0,0,0,1.])
def q_avg(qs): return q_norm(np.mean(qs, axis=0))
def q_rotate(q, v):
    qv = np.array([v[0],v[1],v[2],0.])
    return q_mult(q_mult(q, qv), q_inv(q))[:3]

def load_tpose(path):
    tpose = {}
    with open(path) as f:
        for line in f:
            line=line.strip()
            if ":" not in line: continue
            name, vals = line.split(":",1)
            x,y,z,w = map(float, vals.split(","))
            tpose[name.strip()] = np.array([x,y,z,w])
    return tpose

def load_hierarchy(path):
    hier = {}
    with open(path) as f:
        for line in f:
            line=line.strip()
            if ":" not in line: continue
            bone,parent = line.split(":",1)
            hier[bone.strip()] = None if parent.strip()=="None" else parent.strip()
    return hier

def load_offsets(path):
    MIXAMO = {
        "Hips":"Hips","Spine":"Spine","Spine1":"Chest","Spine2":"UpperChest",
        "Neck":"Neck","Head":"Head","LeftShoulder":"LeftShoulder",
        "RightShoulder":"RightShoulder","LeftArm":"LeftUpperArm",
        "RightArm":"RightUpperArm","LeftForeArm":"LeftLowerArm",
        "RightForeArm":"RightLowerArm","LeftHand":"LeftHand","RightHand":"RightHand",
        "LeftUpLeg":"LeftUpperLeg","RightUpLeg":"RightUpperLeg",
        "LeftLeg":"LeftLowerLeg","RightLeg":"RightLowerLeg",
        "LeftFoot":"LeftFoot","RightFoot":"RightFoot",
        "LeftToeBase":"LeftToes","RightToeBase":"RightToes",
    }
    with open(path) as f: data=json.load(f)
    out = {}
    for item in data["bones"]:
        raw = item["boneName"].replace("mixamorig:","")
        name = MIXAMO.get(raw)
        if name: out[name] = np.array(item["localPosition"])
    return out

def parse_row(row):
    quats = {}
    for imu_id in range(17):
        p = f"imu_{imu_id}_"
        try:
            quats[imu_id] = q_norm(np.array([float(row[p+"quat_x"]),float(row[p+"quat_y"]),
                                              float(row[p+"quat_z"]),float(row[p+"quat_w"])]))
        except: quats[imu_id] = np.array([0.,0.,0.,1.])
    return quats

def calibrate(rows, tpose, n=100):
    accum = defaultdict(list)
    for row in rows[:n]:
        for imu_id, q in parse_row(row).items():
            accum[imu_id].append(q)
    offsets = {}
    for imu_id, bone in IMU_TO_BONE.items():
        q_rest  = q_avg(accum[imu_id])
        q_tp    = tpose.get(bone, np.array([0,0,0,1.]))
        offsets[imu_id] = q_norm(q_mult(q_tp, q_inv(q_rest)))
    return offsets

def compute_bone_quats(row, cal):
    sq = parse_row(row)
    return {bone: q_norm(q_mult(cal[imu_id], sq[imu_id]))
            for imu_id, bone in IMU_TO_BONE.items()}

def forward_kinematics(bone_quats, offsets, hierarchy):
    pos, quat = {}, {}
    visited = set()
    def visit(bone):
        if bone in visited: return
        parent = hierarchy.get(bone)
        if parent and parent not in visited: visit(parent)
        if parent is None:
            pos[bone]  = offsets.get(bone, np.zeros(3)).copy()
            quat[bone] = bone_quats.get(bone, np.array([0,0,0,1.]))
        else:
            off = offsets.get(bone, np.zeros(3))
            pos[bone]  = pos[parent] + q_rotate(quat[parent], off)
            quat[bone] = bone_quats.get(bone, np.array([0,0,0,1.]))
        visited.add(bone)
    for b in hierarchy: visit(b)
    return pos

# ── MAIN ───────────────────────────────────────────
print("Loading...")
tpose     = load_tpose(TPOSE_PATH)
hierarchy = load_hierarchy(HIERARCHY_PATH)
offsets   = load_offsets(OFFSETS_PATH)
with open(CSV_PATH, newline="") as f:
    rows = list(csv.DictReader(f))

print(f"Calibrating ({CALIB_FRAMES} frames)...")
cal = calibrate(rows, tpose, CALIB_FRAMES)
work = rows[CALIB_FRAMES:]

# Pre-compute all frames
print(f"Computing {len(work)} frames...")
all_frames = []
for row in work:
    bq  = compute_bone_quats(row, cal)
    pos = forward_kinematics(bq, offsets, hierarchy)
    all_frames.append(pos)
print("Ready — close the window to exit.\n")

# ── 3-D plot ────────────────────────────────────────
fig = plt.figure(figsize=(7, 9))
ax  = fig.add_subplot(111, projection="3d")
ax.set_title("IMU 3-D Skeleton", pad=12)
ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")

frame_idx = [0]

def draw_frame(idx):
    ax.cla()
    pos = all_frames[idx]

    # Joints
    xs = [p[0] for p in pos.values()]
    ys = [p[2] for p in pos.values()]   # swap Y/Z for Unity→matplotlib
    zs = [p[1] for p in pos.values()]
    ax.scatter(xs, ys, zs, c="#378ADD", s=20, depthshade=False)

    # Bones
    for a, b in SKELETON_EDGES:
        if a in pos and b in pos:
            pa, pb = pos[a], pos[b]
            ax.plot([pa[0], pb[0]], [pa[2], pb[2]], [pa[1], pb[1]], "gray", lw=1.5)

    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 2.2)
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")
    ax.set_title(f"Frame {idx + CALIB_FRAMES} / {len(rows)}")

draw_frame(0)

def on_key(event):
    if event.key == "right":  frame_idx[0] = min(frame_idx[0]+1, len(all_frames)-1)
    elif event.key == "left": frame_idx[0] = max(frame_idx[0]-1, 0)
    elif event.key == "up":   frame_idx[0] = min(frame_idx[0]+10, len(all_frames)-1)
    elif event.key == "down": frame_idx[0] = max(frame_idx[0]-10, 0)
    draw_frame(frame_idx[0])
    fig.canvas.draw()

fig.canvas.mpl_connect("key_press_event", on_key)

ani = animation.FuncAnimation(
    fig, lambda i: draw_frame(i % len(all_frames)),
    interval=1000//89, blit=False
)

print("← → arrow keys: step  |  ↑ ↓ : jump 10 frames  |  animation plays automatically")
plt.tight_layout()
plt.show()