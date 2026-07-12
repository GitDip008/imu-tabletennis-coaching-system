"""
realtimeapp_new.py
──────────────────
TT Coaching System with auto-launched Unity skeleton.
On startup: launches IMU_Avatar.exe automatically (separate window).
On "Start Session": streams IMU CSV → Unity skeleton via IMUUnityController,
and runs the coaching pipeline in parallel.
"""
import sys
import csv
import time
import queue
import pathlib
import threading
import subprocess
import tkinter as tk
from tkinter import scrolledtext, messagebox, filedialog
import numpy as np
import pandas as pd
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR  = pathlib.Path(__file__).resolve().parent
ROOT     = SRC_DIR.parent

IMU_DIR  = pathlib.Path(r"K:\IMU_AVATAR_project")
UNITY_EXE = IMU_DIR / "IMU_Avatar" / "game3" / "IMU_Avatar.exe"
DEFAULT_CSV = IMU_DIR / "imu_data_log_20250624_204958.csv"
sys.path.insert(0, str(IMU_DIR))
sys.path.insert(0, str(SRC_DIR))

from SiriusCeption_unity_controller import IMUUnityController
from inference         import StrokePredictor, CLASS_NAMES, FEATURE_COLS
from summarizer        import run_session
from coaching          import get_coaching_feedback, STROKE_LABEL_MAP
from feature_extractor import SlidingWindowExtractor, WINDOW_SIZE, STEP_SIZE

# ── Config ─────────────────────────────────────────────────────────────────────
SUBJECT_ID         = 10
CALIBRATION_FRAMES = 100
UNITY_PORT         = 5005

# ── Model selection ────────────────────────────────────────────────────────────
# Set USE_SYNTHETIC_MODEL = True to load the MuJoCo-trained model+scaler
# (correct for SI-unit MuJoCo / SiriusCeption CSVs).
# Set False to fall back to the TTSWING LOSO checkpoint for SUBJECT_ID
# (only correct for TTSWING pre-extracted features).
USE_SYNTHETIC_MODEL = True
SYNTHETIC_MODEL_PATH  = pathlib.Path(
    r"E:\thesis_work\TT_thesis\tt_coaching_pipeline\mujoco_sim\output\model_synthetic.pt"
)
SYNTHETIC_SCALER_PATH = pathlib.Path(
    r"E:\thesis_work\TT_thesis\tt_coaching_pipeline\mujoco_sim\output\scaler_synthetic.pkl"
)
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

# ── IMU streaming thread ───────────────────────────────────────────────────────
class RawIMUStreamer(threading.Thread):
    """
    Uses IMUUnityController (.pyd) — same as final_version_unity.ipynb.
    Calibrates, then streams bone rotations to Unity and pushes raw rows
    into the queue for stroke classification.
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
                bone_hierarchy_path=str(IMU_DIR / "BoneHierarchy.txt"),
                bone_offsets_path  =str(IMU_DIR / "BoneOffsets.json"),
                tpose_quats_path   =str(IMU_DIR / "InitialPoseExport.txt"),
                udp_ip             ="127.0.0.1",
                udp_port           =UNITY_PORT,
                position_scale     =1.0,
                hips_y_scale       =1.0,
            )

            self.q.put(("status", f"Calibrating ({CALIBRATION_FRAMES} frames)…"))

            with open(self.csv_path, newline="") as f:
                reader = csv.DictReader(f)
                controller.calibrate(reader, frames=CALIBRATION_FRAMES)
                self.q.put(("calibrated", None))
                self.q.put(("status", "Streaming…"))

                for row in reader:
                    if self._stop_event.is_set():
                        break
                    sleep_s, _ = controller.process_row(row)
                    self.q.put(("imu_row", row))
                    if 0 < sleep_s < 1.0:
                        time.sleep(sleep_s)

        except Exception:
            import traceback
            self.q.put(("error", traceback.format_exc()))
        finally:
            if controller is not None:
                try:
                    controller.close()
                except Exception:
                    pass
            self.q.put(("stream_done", None))


# ── Main GUI ───────────────────────────────────────────────────────────────────
class RealTimeApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title(f"TT Coaching System — Player {SUBJECT_ID}")
        self.geometry("980x720")
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)

        with open(ROOT / "config.yaml") as f:
            self.cfg = yaml.safe_load(f)

        if USE_SYNTHETIC_MODEL:
            print(f"[Model] Loading MuJoCo-trained model from {SYNTHETIC_MODEL_PATH}")
            self.predictor = StrokePredictor.from_checkpoint(
                str(SYNTHETIC_MODEL_PATH),
                str(SYNTHETIC_SCALER_PATH),
                cfg=self.cfg,
            )
        else:
            print(f"[Model] Loading TTSWING LOSO checkpoint for subject {SUBJECT_ID}")
            self.predictor = StrokePredictor.from_subject(SUBJECT_ID)
        self.session_rows   = []
        self.session_preds  = []
        self.stroke_counts  = {CLASS_NAMES[i]: 0 for i in range(4)}
        self.event_queue    = queue.Queue()
        self.streamer       = None
        self.session_active = False
        self.csv_path       = str(DEFAULT_CSV)
        self._extractor     = SlidingWindowExtractor(WINDOW_SIZE, STEP_SIZE)
        self._unity_proc    = None

        self._build_ui()
        self._launch_unity()
        self._poll_queue()

    # ── Unity launch ──────────────────────────────────────────────────────────

    def _launch_unity(self):
        if not UNITY_EXE.exists():
            self.status_var.set(f"Unity exe not found: {UNITY_EXE}")
            return
        self._unity_proc = subprocess.Popen([
            str(UNITY_EXE),
            "-screen-width",      "1280",
            "-screen-height",     "720",
            "-screen-fullscreen", "0",
        ])
        self.status_var.set("Unity launched — select CSV and press Start Session")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        hdr = tk.Frame(self, bg=COLORS["accent"], pady=8)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="🏓  Table Tennis IMU Coaching System",
                 font=("Helvetica", 16, "bold"),
                 fg="white", bg=COLORS["accent"]).pack(side=tk.LEFT, padx=20)
        tk.Label(hdr, text=f"Player {SUBJECT_ID}  |  Simulation Mode",
                 font=("Helvetica", 11), fg="#DDD6FE",
                 bg=COLORS["accent"]).pack(side=tk.RIGHT, padx=20)

        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=2)
        body.columnconfigure(2, weight=3)
        body.rowconfigure(0, weight=1)

        self._build_left_panel(body)
        self._build_middle_panel(body)
        self._build_right_panel(body)

        footer = tk.Frame(self, bg=COLORS["bg"], pady=8)
        footer.pack(fill=tk.X, padx=12)

        self.btn_csv = tk.Button(
            footer, text="CSV…",
            font=("Helvetica", 10), bg=COLORS["border"], fg=COLORS["text"],
            relief=tk.FLAT, padx=10, pady=8,
            cursor="hand2", command=self._pick_csv,
        )
        self.btn_csv.pack(side=tk.LEFT, padx=(0, 6))

        self.btn_start = tk.Button(
            footer, text="▶  Start Session",
            font=("Helvetica", 11, "bold"),
            bg=COLORS["green"], fg="white",
            relief=tk.FLAT, padx=20, pady=8,
            cursor="hand2", command=self._start_session,
        )
        self.btn_start.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_stop = tk.Button(
            footer, text="■  End Session",
            font=("Helvetica", 11, "bold"),
            bg=COLORS["red"], fg="white",
            relief=tk.FLAT, padx=20, pady=8,
            cursor="hand2", command=self._end_session,
            state=tk.DISABLED,
        )
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_report = tk.Button(
            footer, text="🤖  Generate LLM Report",
            font=("Helvetica", 11, "bold"),
            bg=COLORS["accent"], fg="white",
            relief=tk.FLAT, padx=20, pady=8,
            cursor="hand2", command=self._generate_report,
            state=tk.DISABLED,
        )
        self.btn_report.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Initialising…")
        tk.Label(
            footer, textvariable=self.status_var,
            font=("Helvetica", 10), fg=COLORS["subtext"], bg=COLORS["bg"],
        ).pack(side=tk.RIGHT, padx=10)

    def _panel(self, parent, title, col):
        f = tk.LabelFrame(
            parent, text=f"  {title}  ",
            font=("Helvetica", 10, "bold"),
            fg=COLORS["subtext"], bg=COLORS["panel"],
            bd=1, relief=tk.FLAT,
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        f.grid(row=0, column=col, sticky="nsew", padx=5, pady=5)
        return f

    def _build_left_panel(self, body):
        p = self._panel(body, "Live Prediction", col=0)

        tk.Label(p, text="Current Stroke", font=("Helvetica", 9),
                 fg=COLORS["subtext"], bg=COLORS["panel"]).pack(pady=(14, 2))
        self.lbl_stroke = tk.Label(p, text="—", font=("Helvetica", 22, "bold"),
                                   fg=COLORS["text"], bg=COLORS["panel"])
        self.lbl_stroke.pack()

        tk.Label(p, text="Confidence", font=("Helvetica", 9),
                 fg=COLORS["subtext"], bg=COLORS["panel"]).pack(pady=(12, 2))
        self.conf_canvas = tk.Canvas(p, height=22, bg=COLORS["border"],
                                     highlightthickness=0)
        self.conf_canvas.pack(fill=tk.X, padx=16)
        self.lbl_conf = tk.Label(p, text="0.00%", font=("Helvetica", 13, "bold"),
                                 fg=COLORS["text"], bg=COLORS["panel"])
        self.lbl_conf.pack(pady=(4, 12))

        tk.Label(p, text="Class Probabilities", font=("Helvetica", 9),
                 fg=COLORS["subtext"], bg=COLORS["panel"]).pack(pady=(4, 6))
        self.prob_bars = {}
        for cid in range(4):
            name  = CLASS_NAMES[cid]
            row_f = tk.Frame(p, bg=COLORS["panel"])
            row_f.pack(fill=tk.X, padx=16, pady=2)
            tk.Label(row_f, text=name[:12], width=13, anchor="w",
                     font=("Helvetica", 8), fg=COLORS["subtext"],
                     bg=COLORS["panel"]).pack(side=tk.LEFT)
            bar = tk.Canvas(row_f, height=14, bg=COLORS["border"],
                            highlightthickness=0)
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            lbl = tk.Label(row_f, text="0%", width=5, anchor="e",
                           font=("Helvetica", 8), fg=COLORS["text"],
                           bg=COLORS["panel"])
            lbl.pack(side=tk.LEFT)
            self.prob_bars[name] = (bar, lbl)

        tk.Label(p, text="Events This Session", font=("Helvetica", 9),
                 fg=COLORS["subtext"], bg=COLORS["panel"]).pack(pady=(16, 2))
        self.lbl_events = tk.Label(p, text="0", font=("Helvetica", 26, "bold"),
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
            row_f.pack(fill=tk.X, padx=16, pady=3)
            tk.Label(row_f, text=name[:12], width=13, anchor="w",
                     font=("Helvetica", 8, "bold"), fg=color,
                     bg=COLORS["panel"]).pack(side=tk.LEFT)
            bar = tk.Canvas(row_f, height=16, bg=COLORS["border"],
                            highlightthickness=0)
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            lbl = tk.Label(row_f, text="0", width=6, anchor="e",
                           font=("Helvetica", 8), fg=COLORS["text"],
                           bg=COLORS["panel"])
            lbl.pack(side=tk.LEFT)
            self.dist_bars[name] = (bar, lbl, color)

        tk.Frame(p, bg=COLORS["border"], height=1).pack(fill=tk.X, padx=16, pady=12)
        grid = tk.Frame(p, bg=COLORS["panel"])
        grid.pack(fill=tk.X, padx=16)
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

    def _build_right_panel(self, body):
        p = self._panel(body, "Event Log", col=2)
        self.log_box = scrolledtext.ScrolledText(
            p, font=("Courier", 8), bg="#0F0F1A", fg=COLORS["text"],
            insertbackground=COLORS["text"],
            relief=tk.FLAT, state=tk.DISABLED, wrap=tk.WORD,
        )
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        for name, color in CLASS_COLORS.items():
            self.log_box.tag_config(name, foreground=color)
        self.log_box.tag_config("header", foreground=COLORS["accent"],
                                font=("Courier", 8, "bold"))
        self.log_box.tag_config("report", foreground="#A78BFA",
                                font=("Courier", 8))
        self.log_box.tag_config("warn", foreground=COLORS["yellow"],
                                font=("Courier", 8, "italic"))

    # ── Session control ───────────────────────────────────────────────────────

    def _pick_csv(self):
        path = filedialog.askopenfilename(
            title="Select raw IMU CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(IMU_DIR),
        )
        if path:
            self.csv_path = path
            self.status_var.set(f"CSV: {pathlib.Path(path).name}")

    def _start_session(self):
        self.session_rows   = []
        self.session_preds  = []
        self.stroke_counts  = {CLASS_NAMES[i]: 0 for i in range(4)}
        self._extractor.reset()

        self._log(
            f"═══ Session started — Player {SUBJECT_ID}  |  "
            f"{pathlib.Path(self.csv_path).name} ═══\n",
            tag="header",
        )
        self.session_active = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_report.config(state=tk.DISABLED)
        self.btn_csv.config(state=tk.DISABLED)
        self.lbl_calib.config(text="Calibrating skeleton…")
        self.status_var.set("Calibrating…")

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
        self._log(f"\n═══ Session ended — {n} predictions ═══\n", tag="header")

    def destroy(self):
        if self.streamer:
            self.streamer.stop()
        if self._unity_proc and self._unity_proc.poll() is None:
            self._unity_proc.terminate()
        super().destroy()

    # ── Queue polling ─────────────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                msg_type, payload = self.event_queue.get_nowait()
                if msg_type == "imu_row":
                    self._handle_imu_row(payload)
                elif msg_type == "calibrated":
                    self.lbl_calib.config(text="Skeleton calibrated ✓")
                elif msg_type == "status":
                    self.status_var.set(payload)
                elif msg_type == "stream_done":
                    if self.session_active:
                        self._end_session()
                elif msg_type == "error":
                    self._log(f"[ERROR]\n{payload}\n", tag="warn")
                    self.status_var.set("Streamer error — see Event Log")
        except queue.Empty:
            pass
        self.after(30, self._poll_queue)

    # ── Per-frame inference ───────────────────────────────────────────────────

    def _handle_imu_row(self, row: dict):
        features = self._extractor.add_frame(row)
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

    # ── UI update helpers ─────────────────────────────────────────────────────

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
        self.conf_canvas.create_rectangle(0, 0, int(w * conf), 22,
                                          fill=color, outline="")

        for cname, prob in result["probabilities"].items():
            bar, lbl = self.prob_bars[cname]
            bar.update_idletasks()
            bw = bar.winfo_width()
            bar.delete("all")
            bar.create_rectangle(0, 0, int(bw * prob), 14,
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
            bw = bar.winfo_width()
            bar.delete("all")
            bar.create_rectangle(0, 0, int(bw * cnt / max_count), 16,
                                  fill=color, outline="")
            lbl.config(text=f"{cnt} ({cnt / total * 100:.0f}%)")

        dominant = max(
            (k for k in self.stroke_counts if k != "No Stroke"),
            key=lambda k: self.stroke_counts[k],
        )
        avg_conf = np.mean([p["confidence"] for p in self.session_preds])
        low_conf = sum(1 for p in self.session_preds if p["confidence"] < 0.60)

        stroke_idx = [i for i, p in enumerate(self.session_preds)
                      if p["label_id"] != 0]
        if len(stroke_idx) > 1:
            cv    = np.std(np.diff(stroke_idx)) / (np.mean(np.diff(stroke_idx)) + 1e-9)
            tempo = "consistent" if cv < 0.2 else "moderate" if cv < 0.4 else "irregular"
        else:
            tempo = "—"

        weak = [
            STROKE_LABEL_MAP.get(k, k)
            for k, v in self.stroke_counts.items()
            if k != "No Stroke" and v > 0
            and np.mean([p["confidence"] for p in self.session_preds
                         if p["label_name"] == k] or [1]) < 0.65
        ]
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
        self._log(
            f"[{idx:04d}]  {real:<22}  conf={result['confidence']:.2%}\n",
            tag=name,
        )

    def _log(self, text: str, tag: str = None):
        self.log_box.config(state=tk.NORMAL)
        if tag:
            self.log_box.insert(tk.END, text, tag)
        else:
            self.log_box.insert(tk.END, text)
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    # ── LLM report ────────────────────────────────────────────────────────────

    def _generate_report(self):
        if not self.session_preds:
            messagebox.showwarning("No Data", "No session data to report on.")
            return
        self.btn_report.config(state=tk.DISABLED, text="⏳  Generating…")
        self.status_var.set("Calling LLM — please wait…")

        def _run():
            try:
                session_df = pd.DataFrame(self.session_rows)
                summary    = run_session(self.predictor, session_df)
                feedback   = get_coaching_feedback(
                    summary, subject_id=SUBJECT_ID, cfg=self.cfg)
                self.after(0, lambda: self._show_report(feedback))
            except Exception as e:
                self.after(0, lambda: self._report_error(str(e)))

        threading.Thread(target=_run, daemon=True).start()

    def _show_report(self, feedback: dict):
        self.btn_report.config(state=tk.NORMAL, text="🤖  Generate LLM Report")
        self.status_var.set("Report generated ✓")

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
        tk.Label(popup, text=f"🏓  Coaching Report — Player {SUBJECT_ID}",
                 font=("Helvetica", 13, "bold"),
                 fg="white", bg=COLORS["accent"], pady=10).pack(fill=tk.X)
        txt = scrolledtext.ScrolledText(
            popup, font=("Helvetica", 10),
            bg=COLORS["panel"], fg=COLORS["text"],
            relief=tk.FLAT, padx=14, pady=14, wrap=tk.WORD,
        )
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
        self.btn_report.config(state=tk.NORMAL, text="🤖  Generate LLM Report")
        self.status_var.set("Report failed — check LM Studio is running")
        messagebox.showerror(
            "LLM Error",
            f"Could not connect to LLM.\n\n"
            f"Make sure LM Studio server is running on port 1234.\n\n"
            f"Error: {error}",
        )


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = RealTimeApp()
    app.mainloop()
