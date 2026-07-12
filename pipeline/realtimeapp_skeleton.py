"""
TT Coaching App — Raw IMU CSV + embedded 3D matplotlib skeleton.

Skeleton source: SiriusCeption_unity_controller (same as Unity version)
  - process_row() returns (sleep_s, target_str)
  - target_str contains calibrated bone quaternions ready for Unity
  - We parse target_str directly → FK → matplotlib (no UDP interception needed)
  - Unity LH quaternion → RH for scipy: negate qz component

Layout:
    ┌──────────┬──────────┬──────────────┬──────────────────┐
    │ Live     │ Session  │  Event Log   │  3D Skeleton     │
    │ Predict  │ Stats    │  (scrolled)  │  (matplotlib)    │
    └──────────┴──────────┴──────────────┴──────────────────┘
    [CSV…] [Start Session] [End Session] [Generate Report]  status…
"""

import sys
import csv
import time
import queue
import pathlib
import threading
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial.transform import Rotation

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).resolve().parent.parent
UNITY_DIR = pathlib.Path(r"E:\thesis_work\1_new_test")
sys.path.insert(0, str(UNITY_DIR))

from SiriusCeption_unity_controller import IMUUnityController
from inference         import StrokePredictor, CLASS_NAMES, FEATURE_COLS
from summarizer        import run_session
from coaching          import get_coaching_feedback, STROKE_LABEL_MAP
from feature_extractor import SlidingWindowExtractor, WINDOW_SIZE, STEP_SIZE

# ── Config ────────────────────────────────────────────────────────────────────
SUBJECT_ID         = 10
CALIBRATION_FRAMES = 100
SKEL_UPDATE_EVERY  = 3      # redraw skeleton every N raw IMU frames
PLAYBACK_SPEED     = 1.0    # 1.0 = real-time
DEFAULT_CSV        = str(UNITY_DIR / "imu_data_log_20250624_204958.csv")

COLORS = {
    "bg"     : "#1E1E2E",
    "panel"  : "#2A2A3E",
    "accent" : "#7C3AED",
    "green"  : "#22C55E",
    "yellow" : "#EAB308",
    "red"    : "#EF4444",
    "text"   : "#E2E8F0",
    "subtext": "#94A3B8",
    "border" : "#3F3F5F",
}
CLASS_COLORS = {
    "No Stroke"    : "#64748B",
    "Stroke Type 1": "#3B82F6",
    "Stroke Type 2": "#10B981",
    "Stroke Type 3": "#F59E0B",
}

# ── Skeleton constants ─────────────────────────────────────────────────────────
# Maps SiriusCeption bone names (from target string) → IMU IDs
BONE_TO_IMU = {
    "Head"          :  0,
    "RightFoot"     :  1,
    "RightLowerLeg" :  2,
    "RightUpperLeg" :  3,
    "LeftFoot"      :  4,
    "LeftLowerLeg"  :  5,
    "LeftUpperLeg"  :  6,
    "RightHand"     :  7,
    "RightLowerArm" :  8,
    "RightUpperArm" :  9,
    "LeftHand"      : 10,
    "LeftLowerArm"  : 11,
    "LeftUpperArm"  : 12,
    "Hips"          : 13,
    "Spine"         : 14,
    "RightShoulder" : 15,
    "LeftShoulder"  : 16,
}

# T-pose joint positions  (Unity/world space: x=right, y=up, z=forward)
TPOSE = {
    13: np.array([ 0.00, 1.00,  0.00]),  # Hips  (root)
    14: np.array([ 0.00, 1.25,  0.00]),  # Spine
     0: np.array([ 0.00, 1.70,  0.00]),  # Head
    15: np.array([ 0.18, 1.50,  0.00]),  # RightShoulder
    16: np.array([-0.18, 1.50,  0.00]),  # LeftShoulder
     9: np.array([ 0.38, 1.50,  0.00]),  # RightUpperArm
     8: np.array([ 0.60, 1.30,  0.00]),  # RightLowerArm
     7: np.array([ 0.72, 1.10,  0.00]),  # RightHand (racket)
    12: np.array([-0.38, 1.50,  0.00]),  # LeftUpperArm
    11: np.array([-0.60, 1.30,  0.00]),  # LeftLowerArm
    10: np.array([-0.72, 1.10,  0.00]),  # LeftHand
     3: np.array([ 0.12, 0.58,  0.00]),  # RightUpperLeg
     2: np.array([ 0.14, 0.28,  0.00]),  # RightLowerLeg
     1: np.array([ 0.16, 0.00,  0.08]),  # RightFoot
     6: np.array([-0.12, 0.58,  0.00]),  # LeftUpperLeg
     5: np.array([-0.14, 0.28,  0.00]),  # LeftLowerLeg
     4: np.array([-0.16, 0.00,  0.08]),  # LeftFoot
}

# FK parent map (child → parent IMU id)
PARENT = {
    14: 13,  16: 14,  15: 14,   0: 14,
     9: 15,   8:  9,   7:  8,
    12: 16,  11: 12,  10: 11,
     3: 13,   2:  3,   1:  2,
     6: 13,   5:  6,   4:  5,
}

# Bones to draw: (joint_a, joint_b, colour)
BONES = [
    (13, 14, "#E2E8F0"), (14,  0, "#E2E8F0"),           # spine/head
    (14, 15, "#3B82F6"), (15,  9, "#3B82F6"),            # right arm
    ( 9,  8, "#3B82F6"), ( 8,  7, "#3B82F6"),
    (14, 16, "#06B6D4"), (16, 12, "#06B6D4"),            # left arm
    (12, 11, "#06B6D4"), (11, 10, "#06B6D4"),
    (13,  3, "#22C55E"), ( 3,  2, "#22C55E"), ( 2,  1, "#22C55E"),  # right leg
    (13,  6, "#84CC16"), ( 6,  5, "#84CC16"), ( 5,  4, "#84CC16"),  # left leg
]

RACKET_JOINTS = {7}


# ── Skeleton FK from SiriusCeption target string ───────────────────────────────

def parse_target(target: str) -> dict[int, Rotation]:
    """
    Parse the target string returned by process_row() into
    {imu_id: Rotation} using scipy (right-hand convention).

    SiriusCeption outputs Unity LH quaternions. Converting to RH
    for scipy is done by negating the z component: (qx, qy, -qz, qw).
    """
    rotations: dict[int, Rotation] = {}
    for line in target.strip().split("\n"):
        line = line.strip()
        if ":" not in line:
            continue
        bone_name, vals = line.split(":", 1)
        bone_name = bone_name.strip()
        if bone_name not in BONE_TO_IMU:
            continue
        try:
            qx, qy, qz, qw = (float(v) for v in vals.split(","))
        except ValueError:
            continue
        imu_id = BONE_TO_IMU[bone_name]
        # Unity LH → scipy RH: negate z
        rotations[imu_id] = Rotation.from_quat([qx, qy, -qz, qw])
    return rotations


def fk_positions(rotations: dict[int, Rotation]) -> dict[int, np.ndarray]:
    """
    Forward kinematics: compute world positions for all 17 joints.
    Root (Hips=13) is fixed at TPOSE[13].
    Each child position = parent_pos + parent_rot.apply(bone_vector).
    """
    positions: dict[int, np.ndarray] = {}

    def compute(jid: int):
        if jid in positions:
            return
        if jid not in PARENT:              # root
            positions[jid] = TPOSE[jid].copy()
            return
        pid = PARENT[jid]
        compute(pid)
        bone_vec = TPOSE[jid] - TPOSE[pid]
        q_parent = rotations.get(pid, Rotation.identity())
        positions[jid] = positions[pid] + q_parent.apply(bone_vec)

    for jid in TPOSE:
        compute(jid)
    return positions


# ── Background streamer ────────────────────────────────────────────────────────

class RawIMUStreamer(threading.Thread):
    """
    Uses IMUUnityController for calibration + per-frame processing.
    Pushes ("imu_row", (raw_row_dict, target_str)) into the event queue.
    target_str is None during calibration frames.
    """

    def __init__(self, csv_path: str, event_queue: queue.Queue):
        super().__init__(daemon=True)
        self.csv_path    = csv_path
        self.q           = event_queue
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        controller = None
        try:
            controller = IMUUnityController(
                bone_hierarchy_path=str(UNITY_DIR / "BoneHierarchy.txt"),
                bone_offsets_path  =str(UNITY_DIR / "BoneOffsets.json"),
                tpose_quats_path   =str(UNITY_DIR / "InitialPoseExport.txt"),
                udp_ip             ="127.0.0.1",
                udp_port           =5005,
                position_scale     =1.0,
                hips_y_scale       =1.0,
            )

            with open(self.csv_path, newline="") as f:
                reader = csv.DictReader(f)

                self.q.put(("status", f"Calibrating ({CALIBRATION_FRAMES} frames)…"))
                controller.calibrate(reader, frames=CALIBRATION_FRAMES)
                self.q.put(("status", "Streaming…"))

                first_row = True
                for row in reader:
                    if self._stop_event.is_set():
                        break
                    sleep_s, target = controller.process_row(row)

                    # Normalise target to str regardless of what the .pyd returns
                    if isinstance(target, (bytes, bytearray)):
                        target = target.decode("utf-8", errors="replace")
                    elif target is None:
                        target = ""
                    else:
                        target = str(target)

                    # Print first target to console so we can verify format
                    if first_row:
                        print("[DEBUG] first target sample:\n", target[:300])
                        first_row = False

                    self.q.put(("imu_row", (row, target)))
                    if 0 < sleep_s < 1.0:          # cap to avoid multi-second stalls
                        time.sleep(sleep_s / max(PLAYBACK_SPEED, 0.01))

        except Exception as e:
            import traceback
            full = traceback.format_exc()
            print("[STREAMER ERROR]\n", full)
            self.q.put(("error", full))
        finally:
            if controller is not None:
                controller.close()
            self.q.put(("stream_done", None))


# ── Main application ───────────────────────────────────────────────────────────

class RealTimeApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title(f"TT Coaching — Player {SUBJECT_ID}  [Skeleton + Coaching]")
        self.geometry("1280x760")
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)

        with open(ROOT / "config.yaml") as f:
            self.cfg = yaml.safe_load(f)

        self.predictor = StrokePredictor.from_subject(SUBJECT_ID)

        self.session_rows   = []
        self.session_preds  = []
        self.stroke_counts  = {CLASS_NAMES[i]: 0 for i in range(4)}
        self.event_queue    = queue.Queue()
        self.streamer       = None
        self.session_active = False
        self.csv_path       = DEFAULT_CSV
        self._extractor     = SlidingWindowExtractor(WINDOW_SIZE, STEP_SIZE)
        self._raw_frames    = 0

        self._build_ui()
        self._poll_queue()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        hdr = tk.Frame(self, bg=COLORS["accent"], pady=8)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="TT Coaching System  (SiriusCeption Skeleton)",
                 font=("Helvetica", 15, "bold"),
                 fg="white", bg=COLORS["accent"]).pack(side=tk.LEFT, padx=20)
        tk.Label(hdr, text=f"Player {SUBJECT_ID}",
                 font=("Helvetica", 11), fg="#DDD6FE",
                 bg=COLORS["accent"]).pack(side=tk.RIGHT, padx=20)

        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        for col, w in enumerate([2, 2, 2, 3]):
            body.columnconfigure(col, weight=w)
        body.rowconfigure(0, weight=1)

        self._build_left_panel(body)
        self._build_middle_panel(body)
        self._build_log_panel(body)
        self._build_skeleton_panel(body)   # canvas created here first

        footer = tk.Frame(self, bg=COLORS["bg"], pady=8)
        footer.pack(fill=tk.X, padx=12)

        self.btn_csv = tk.Button(footer, text="CSV…",
            font=("Helvetica", 10), bg=COLORS["border"], fg=COLORS["text"],
            relief=tk.FLAT, padx=10, pady=8,
            cursor="hand2", command=self._pick_csv)
        self.btn_csv.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_start = tk.Button(footer, text="Start Session",
            font=("Helvetica", 11, "bold"),
            bg=COLORS["green"], fg="white",
            relief=tk.FLAT, padx=20, pady=8,
            cursor="hand2", command=self._start_session)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_stop = tk.Button(footer, text="End Session",
            font=("Helvetica", 11, "bold"),
            bg=COLORS["red"], fg="white",
            relief=tk.FLAT, padx=20, pady=8,
            cursor="hand2", command=self._end_session,
            state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_report = tk.Button(footer, text="Generate LLM Report",
            font=("Helvetica", 11, "bold"),
            bg=COLORS["accent"], fg="white",
            relief=tk.FLAT, padx=20, pady=8,
            cursor="hand2", command=self._generate_report,
            state=tk.DISABLED)
        self.btn_report.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Ready — select CSV and press Start")
        tk.Label(footer, textvariable=self.status_var,
                 font=("Helvetica", 10), fg=COLORS["subtext"], bg=COLORS["bg"],
                 wraplength=320, anchor="e").pack(side=tk.RIGHT, padx=10)

    def _panel(self, parent, title, col):
        f = tk.LabelFrame(parent, text=f"  {title}  ",
                          font=("Helvetica", 10, "bold"),
                          fg=COLORS["subtext"], bg=COLORS["panel"],
                          bd=1, relief=tk.FLAT,
                          highlightbackground=COLORS["border"],
                          highlightthickness=1)
        f.grid(row=0, column=col, sticky="nsew", padx=5, pady=5)
        return f

    def _build_left_panel(self, body):
        p = self._panel(body, "Live Prediction", col=0)

        tk.Label(p, text="Current Stroke", font=("Helvetica", 9),
                 fg=COLORS["subtext"], bg=COLORS["panel"]).pack(pady=(14, 2))
        self.lbl_stroke = tk.Label(p, text="—", font=("Helvetica", 20, "bold"),
                                   fg=COLORS["text"], bg=COLORS["panel"])
        self.lbl_stroke.pack()

        tk.Label(p, text="Confidence", font=("Helvetica", 9),
                 fg=COLORS["subtext"], bg=COLORS["panel"]).pack(pady=(10, 2))
        self.conf_canvas = tk.Canvas(p, height=20, bg=COLORS["border"],
                                     highlightthickness=0)
        self.conf_canvas.pack(fill=tk.X, padx=16)
        self.lbl_conf = tk.Label(p, text="0.00%", font=("Helvetica", 12, "bold"),
                                 fg=COLORS["text"], bg=COLORS["panel"])
        self.lbl_conf.pack(pady=(4, 10))

        tk.Label(p, text="Class Probabilities", font=("Helvetica", 9),
                 fg=COLORS["subtext"], bg=COLORS["panel"]).pack(pady=(4, 4))
        self.prob_bars = {}
        for cid in range(4):
            name  = CLASS_NAMES[cid]
            row_f = tk.Frame(p, bg=COLORS["panel"])
            row_f.pack(fill=tk.X, padx=14, pady=2)
            tk.Label(row_f, text=name[:12], width=13, anchor="w",
                     font=("Helvetica", 8), fg=COLORS["subtext"],
                     bg=COLORS["panel"]).pack(side=tk.LEFT)
            bar = tk.Canvas(row_f, height=12, bg=COLORS["border"],
                            highlightthickness=0)
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            lbl = tk.Label(row_f, text="0%", width=5, anchor="e",
                           font=("Helvetica", 8), fg=COLORS["text"],
                           bg=COLORS["panel"])
            lbl.pack(side=tk.LEFT)
            self.prob_bars[name] = (bar, lbl)

        tk.Label(p, text="Events", font=("Helvetica", 9),
                 fg=COLORS["subtext"], bg=COLORS["panel"]).pack(pady=(12, 2))
        self.lbl_events = tk.Label(p, text="0", font=("Helvetica", 24, "bold"),
                                   fg=COLORS["accent"], bg=COLORS["panel"])
        self.lbl_events.pack()

        self.lbl_calib = tk.Label(p, text="", font=("Helvetica", 8),
                                  fg=COLORS["yellow"], bg=COLORS["panel"])
        self.lbl_calib.pack(pady=(8, 4))

    def _build_middle_panel(self, body):
        p = self._panel(body, "Session Statistics", col=1)

        tk.Label(p, text="Stroke Distribution", font=("Helvetica", 9),
                 fg=COLORS["subtext"], bg=COLORS["panel"]).pack(pady=(14, 6))
        self.dist_bars = {}
        for cid in range(4):
            name  = CLASS_NAMES[cid]
            color = CLASS_COLORS[name]
            row_f = tk.Frame(p, bg=COLORS["panel"])
            row_f.pack(fill=tk.X, padx=14, pady=3)
            tk.Label(row_f, text=name[:12], width=13, anchor="w",
                     font=("Helvetica", 8, "bold"), fg=color,
                     bg=COLORS["panel"]).pack(side=tk.LEFT)
            bar = tk.Canvas(row_f, height=14, bg=COLORS["border"],
                            highlightthickness=0)
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            lbl = tk.Label(row_f, text="0", width=6, anchor="e",
                           font=("Helvetica", 8), fg=COLORS["text"],
                           bg=COLORS["panel"])
            lbl.pack(side=tk.LEFT)
            self.dist_bars[name] = (bar, lbl, color)

        tk.Frame(p, bg=COLORS["border"], height=1).pack(
            fill=tk.X, padx=14, pady=10)
        grid = tk.Frame(p, bg=COLORS["panel"])
        grid.pack(fill=tk.X, padx=14)
        self.stat_vars = {}
        for i, (label, key) in enumerate([
            ("Total Strokes",   "total_strokes"),
            ("Dominant Stroke", "dominant"),
            ("Avg Confidence",  "avg_conf"),
            ("Tempo Pattern",   "tempo"),
            ("Low Conf Events", "low_conf"),
            ("Weak Strokes",    "weak"),
        ]):
            tk.Label(grid, text=label, font=("Helvetica", 8),
                     fg=COLORS["subtext"], bg=COLORS["panel"],
                     anchor="w").grid(row=i, column=0, sticky="w", pady=3)
            var = tk.StringVar(value="—")
            self.stat_vars[key] = var
            tk.Label(grid, textvariable=var, font=("Helvetica", 8, "bold"),
                     fg=COLORS["text"], bg=COLORS["panel"],
                     anchor="w").grid(row=i, column=1, sticky="w",
                                      padx=(10, 0), pady=3)

    def _build_log_panel(self, body):
        p = self._panel(body, "Event Log", col=2)
        self.log_box = scrolledtext.ScrolledText(
            p, font=("Courier", 8), bg="#0F0F1A", fg=COLORS["text"],
            insertbackground=COLORS["text"],
            relief=tk.FLAT, state=tk.DISABLED, wrap=tk.WORD)
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        for name, color in CLASS_COLORS.items():
            self.log_box.tag_config(name, foreground=color)
        self.log_box.tag_config("header", foreground=COLORS["accent"],
                                font=("Courier", 8, "bold"))
        self.log_box.tag_config("report", foreground="#A78BFA",
                                font=("Courier", 8))
        self.log_box.tag_config("warn", foreground=COLORS["yellow"],
                                font=("Courier", 8))

    def _build_skeleton_panel(self, body):
        p = self._panel(body, "3D Skeleton  (SiriusCeption FK)", col=3)

        fig = Figure(figsize=(4, 5), dpi=90)
        fig.patch.set_facecolor("#1E1E2E")
        self._ax = fig.add_subplot(111, projection="3d")

        # canvas FIRST so _update_skeleton_display can use it
        self._canvas = FigureCanvasTkAgg(fig, master=p)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                          padx=4, pady=4)

        # draw T-pose as initial state
        self._update_skeleton_display(
            {jid: pos.copy() for jid, pos in TPOSE.items()}
        )

    def _style_axis(self, ax):
        ax.set_facecolor("#1E1E2E")
        ax.set_xlim(-1.0,  1.0)
        ax.set_ylim(-1.0,  1.0)
        ax.set_zlim( 0.0,  2.0)
        ax.set_xlabel("X", color=COLORS["subtext"], fontsize=7)
        ax.set_ylabel("Z", color=COLORS["subtext"], fontsize=7)
        ax.set_zlabel("Y (up)", color=COLORS["subtext"], fontsize=7)
        ax.tick_params(colors=COLORS["subtext"], labelsize=6)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#3F3F5F")
        ax.view_init(elev=10, azim=-60)

    def _update_skeleton_display(self, positions: dict):
        ax = self._ax
        ax.cla()
        self._style_axis(ax)

        for j1, j2, color in BONES:
            if j1 in positions and j2 in positions:
                p1, p2 = positions[j1], positions[j2]
                # y=up in our space → z-axis in matplotlib 3D
                ax.plot([p1[0], p2[0]], [p1[2], p2[2]], [p1[1], p2[1]],
                        color=color, linewidth=2.5, solid_capstyle="round")

        for jid, pos in positions.items():
            color = "#F59E0B" if jid in RACKET_JOINTS else "#FFFFFF"
            size  = 60        if jid in RACKET_JOINTS else 22
            ax.scatter(pos[0], pos[2], pos[1],
                       color=color, s=size, zorder=5, depthshade=False)

        self._canvas.draw_idle()

    # ── Session control ───────────────────────────────────────────────────────

    def _pick_csv(self):
        path = filedialog.askopenfilename(
            title="Select raw IMU CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(UNITY_DIR))
        if path:
            self.csv_path = path
            self.status_var.set(f"CSV: {pathlib.Path(path).name}")

    def _start_session(self):
        self.session_rows  = []
        self.session_preds = []
        self.stroke_counts = {CLASS_NAMES[i]: 0 for i in range(4)}
        self._extractor.reset()
        self._raw_frames = 0

        self._log(f"Session started — Player {SUBJECT_ID}  "
                  f"|  {pathlib.Path(self.csv_path).name}\n", tag="header")

        self.session_active = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_report.config(state=tk.DISABLED)
        self.btn_csv.config(state=tk.DISABLED)
        self.lbl_calib.config(text=f"Calibrating 0/{CALIBRATION_FRAMES}")
        self.status_var.set(f"Calibrating… (0/{CALIBRATION_FRAMES})")

        self.streamer = RawIMUStreamer(self.csv_path, self.event_queue)
        self.streamer.start()

    def _end_session(self):
        if self.streamer:
            self.streamer.stop()
        self.session_active = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_csv.config(state=tk.NORMAL)
        self.btn_report.config(
            state=tk.NORMAL if self.session_preds else tk.DISABLED)
        n = len(self.session_preds)
        self.status_var.set(f"Session ended — {n} predictions")
        self.lbl_calib.config(text="")
        self._log(f"\nSession ended — {n} predictions\n", tag="header")

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                msg_type, payload = self.event_queue.get_nowait()
                if msg_type == "imu_row":
                    self._handle_imu_row(payload)
                elif msg_type == "status":
                    self._on_status(payload)
                elif msg_type == "stream_done":
                    if self.session_active:
                        self._end_session()
                elif msg_type == "error":
                    self._log(f"[ERROR] {payload}\n", tag="warn")
                    self.status_var.set(f"Error: {payload[:80]}")
        except queue.Empty:
            pass
        self.after(30, self._poll_queue)

    def _on_status(self, msg: str):
        self.status_var.set(msg)
        if msg.startswith("Calibrating"):
            pass  # streamer sends this once before calibrate()
        elif msg.startswith("Streaming"):
            self.lbl_calib.config(text="Skeleton calibrated ✓")

    # ── Core row handling ─────────────────────────────────────────────────────

    def _handle_imu_row(self, payload):
        raw_row, target = payload
        self._raw_frames += 1

        # ── Skeleton update every N frames ────────────────────────────────────
        if target and self._raw_frames % SKEL_UPDATE_EVERY == 0:
            rotations = parse_target(target)
            if rotations:
                positions = fk_positions(rotations)
                self._update_skeleton_display(positions)

        # ── Feature extraction + MLP ──────────────────────────────────────────
        features = self._extractor.add_frame(raw_row)
        if features is None:
            return

        result    = self.predictor.predict(features)
        feat_dict = dict(zip(FEATURE_COLS, features.tolist()))

        self.session_rows.append(feat_dict)
        self.session_preds.append(result)
        self.stroke_counts[result["label_name"]] += 1

        self._update_live_panel(result)
        self._update_stats_panel()
        self._log_event(len(self.session_preds), result)

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _update_live_panel(self, result: dict):
        name  = result["label_name"]
        conf  = result["confidence"]
        color = CLASS_COLORS[name]

        self.lbl_stroke.config(text=STROKE_LABEL_MAP.get(name, name), fg=color)
        self.lbl_conf.config(text=f"{conf:.2%}", fg=color)
        self.lbl_events.config(text=str(len(self.session_preds)))

        self.conf_canvas.update_idletasks()
        w = self.conf_canvas.winfo_width()
        self.conf_canvas.delete("all")
        self.conf_canvas.create_rectangle(0, 0, int(w * conf), 20,
                                          fill=color, outline="")

        for cname, prob in result["probabilities"].items():
            bar, lbl = self.prob_bars[cname]
            bar.update_idletasks()
            bw = bar.winfo_width()
            bar.delete("all")
            bar.create_rectangle(0, 0, int(bw * prob), 12,
                                  fill=CLASS_COLORS[cname], outline="")
            lbl.config(text=f"{prob:.0%}")

    def _update_stats_panel(self):
        total = len(self.session_preds)
        if total == 0:
            return

        max_count = max(self.stroke_counts.values()) or 1
        for name, (bar, lbl, color) in self.dist_bars.items():
            cnt = self.stroke_counts[name]
            bar.update_idletasks()
            bw  = bar.winfo_width()
            bar.delete("all")
            bar.create_rectangle(0, 0, int(bw * cnt / max_count), 14,
                                  fill=color, outline="")
            lbl.config(text=f"{cnt} ({cnt/total*100:.0f}%)")

        dominant = max((k for k in self.stroke_counts if k != "No Stroke"),
                       key=lambda k: self.stroke_counts[k])
        avg_conf = np.mean([p["confidence"] for p in self.session_preds])
        low_conf = sum(1 for p in self.session_preds if p["confidence"] < 0.60)

        stroke_idx = [i for i, p in enumerate(self.session_preds)
                      if p["label_id"] != 0]
        if len(stroke_idx) > 1:
            cv    = np.std(np.diff(stroke_idx)) / (np.mean(np.diff(stroke_idx)) + 1e-9)
            tempo = "consistent" if cv < 0.2 else "moderate" if cv < 0.4 else "irregular"
        else:
            tempo = "—"

        weak = [STROKE_LABEL_MAP.get(k, k)
                for k, v in self.stroke_counts.items()
                if k != "No Stroke" and v > 0
                and np.mean([p["confidence"] for p in self.session_preds
                             if p["label_name"] == k] or [1]) < 0.65]

        total_strokes = sum(v for k, v in self.stroke_counts.items()
                            if k != "No Stroke")
        self.stat_vars["total_strokes"].set(str(total_strokes))
        self.stat_vars["dominant"].set(STROKE_LABEL_MAP.get(dominant, dominant))
        self.stat_vars["avg_conf"].set(f"{avg_conf:.2%}")
        self.stat_vars["tempo"].set(tempo)
        self.stat_vars["low_conf"].set(str(low_conf))
        self.stat_vars["weak"].set(", ".join(weak) if weak else "None")

    def _log_event(self, idx: int, result: dict):
        name = result["label_name"]
        real = STROKE_LABEL_MAP.get(name, name)
        self._log(f"[{idx:04d}]  {real:<22}  conf={result['confidence']:.2%}\n",
                  tag=name)

    def _log(self, text: str, tag: str = None):
        self.log_box.config(state=tk.NORMAL)
        if tag:
            self.log_box.insert(tk.END, text, tag)
        else:
            self.log_box.insert(tk.END, text)
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    # ── LLM Report ────────────────────────────────────────────────────────────

    def _generate_report(self):
        if not self.session_preds:
            messagebox.showwarning("No Data", "No session data to report on.")
            return
        self.btn_report.config(state=tk.DISABLED, text="Generating…")
        self.status_var.set("Calling LLM — please wait…")

        def _run():
            try:
                session_df = pd.DataFrame(self.session_rows)
                summary    = run_session(self.predictor, session_df)
                feedback   = get_coaching_feedback(summary, subject_id=SUBJECT_ID,
                                                   cfg=self.cfg)
                self.after(0, lambda: self._show_report(feedback))
            except Exception as e:
                self.after(0, lambda: self._report_error(str(e)))

        threading.Thread(target=_run, daemon=True).start()

    def _show_report(self, feedback: dict):
        self.btn_report.config(state=tk.NORMAL, text="Generate LLM Report")
        self.status_var.set("Report generated")
        self._log("\n═══ LLM COACHING REPORT ═══\n", tag="header")
        self._log(f"ASSESSMENT: {feedback['assessment']}\n\n", tag="report")
        for rec in feedback["recommendations"]:
            self._log(f"[{rec['priority']}] #{rec['rank']}  {rec['text']}\n",
                      tag="report")
        self._log(f"\nNEXT FOCUS: {feedback['next_focus']}\n", tag="report")
        self._log("═" * 45 + "\n", tag="header")

        popup = tk.Toplevel(self)
        popup.title("Coaching Report")
        popup.geometry("620x480")
        popup.configure(bg=COLORS["bg"])
        tk.Label(popup, text=f"Coaching Report — Player {SUBJECT_ID}",
                 font=("Helvetica", 13, "bold"),
                 fg="white", bg=COLORS["accent"], pady=10).pack(fill=tk.X)
        txt = scrolledtext.ScrolledText(
            popup, font=("Helvetica", 10),
            bg=COLORS["panel"], fg=COLORS["text"],
            relief=tk.FLAT, padx=14, pady=14, wrap=tk.WORD)
        txt.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        body_text = (
            f"ASSESSMENT\n{'─'*40}\n{feedback['assessment']}\n\n"
            f"RECOMMENDATIONS\n{'─'*40}\n"
        )
        for rec in feedback["recommendations"]:
            body_text += f"[{rec['priority']}] #{rec['rank']}\n{rec['text']}\n\n"
        body_text += f"NEXT SESSION FOCUS\n{'─'*40}\n{feedback['next_focus']}"
        txt.insert(tk.END, body_text)
        txt.config(state=tk.DISABLED)
        tk.Button(popup, text="Close", bg=COLORS["accent"], fg="white",
                  relief=tk.FLAT, padx=16, pady=6,
                  command=popup.destroy).pack(pady=(0, 12))

    def _report_error(self, error: str):
        self.btn_report.config(state=tk.NORMAL, text="Generate LLM Report")
        self.status_var.set("Report failed — check LM Studio is running")
        messagebox.showerror("LLM Error", f"Could not connect to LLM.\n\nError: {error}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = RealTimeApp()
    app.mainloop()
