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
IMU_DIR  = pathlib.Path(r"E:\thesis_work\1_new_test")
UNITY_EXE = IMU_DIR / "IMU_Avatar" / "game3" / "IMU_Avatar.exe"
DEFAULT_CSV = IMU_DIR / "imu_data_log_20250624_204958.csv"

sys.path.insert(0, str(IMU_DIR))
sys.path.insert(0, str(SRC_DIR))

from SiriusCeption_unity_controller import IMUUnityController
from inference         import StrokePredictor, CLASS_NAMES, FEATURE_COLS
from summarizer        import run_session
from coaching          import get_coaching_feedback, STROKE_LABEL_MAP
from feature_extractor import (
    SlidingWindowExtractor, WINDOW_SIZE, STEP_SIZE, is_idle_window,
)

# ── Config ─────────────────────────────────────────────────────────────────────
SUBJECT_ID         = 10
CALIBRATION_FRAMES = 100
UNITY_PORT         = 5005

# ── Model selection ────────────────────────────────────────────────────────────
# True  → load the MuJoCo-trained model (correct for SI-unit MuJoCo / SiriusCeption CSVs).
# False → load the TTSWING LOSO checkpoint for SUBJECT_ID (TTSWING-scale data only).
USE_SYNTHETIC_MODEL   = True
SYNTHETIC_MODEL_PATH  = ROOT / "mujoco_sim" / "output" / "model_synthetic.pt"
SYNTHETIC_SCALER_PATH = ROOT / "mujoco_sim" / "output" / "scaler_synthetic.pkl"

from ui_theme import (
    COLORS, CLASS_COLORS, FONTS,
    make_card, section_title, make_button,
    draw_progress_bar, style_header, style_footer,
)


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
        self.title("Table Tennis Coaching Live")
        self.geometry("1180x780")
        self.configure(bg=COLORS["bg"])
        self._anim_t = 0.0
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
        self._last_label_id = 0
        self.event_queue    = queue.Queue()
        self.streamer       = None
        self.session_active = False
        self.csv_path       = str(DEFAULT_CSV)
        self._extractor     = SlidingWindowExtractor(WINDOW_SIZE, STEP_SIZE)
        self._unity_proc    = None

        self._build_ui()
        self._launch_unity()
        self._poll_queue()
        self._animate_stripe()

    def _animate_stripe(self):
        if not self.winfo_exists():
            return
        try:
            import math, colorsys
            self._anim_t += 0.03
            h = (240 + 60 * math.sin(self._anim_t)) / 360.0
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
            color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            self._stripe.config(bg=color)
        except Exception:
            pass
        self.after(60, self._animate_stripe)

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
        self.status_var.set("Unity launched, select CSV and press Start Session")

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.configure(bg=COLORS["bg"])

        # Header
        hdr = style_header(self)
        tk.Label(hdr, text="◆  TABLE TENNIS COACHING",
                 font=FONTS["h1"], fg=COLORS["text"],
                 bg=COLORS["surface"]).pack(side=tk.LEFT, padx=22, pady=14)
        tk.Label(hdr, text="SESSION", font=FONTS["h3"],
                 fg=COLORS["accent"], bg=COLORS["surface"]).pack(side=tk.LEFT, pady=18)
        tk.Label(hdr,
                 text="Real-Time UDP  ·  Unity Skeleton  ·  LLM review",
                 font=FONTS["body"], fg=COLORS["text_dim"],
                 bg=COLORS["surface"]).pack(side=tk.RIGHT, padx=22)
        # animated stripe — overrides the static accent line in style_header
        self._stripe = tk.Frame(self, bg=COLORS["accent"], height=3)
        self._stripe.pack(fill=tk.X)

        # Footer
        footer = style_footer(self)
        self.status_var = tk.StringVar(value="◉ Initialising")
        tk.Label(footer, textvariable=self.status_var, font=FONTS["body"],
                 fg=COLORS["text_dim"], bg=COLORS["surface"]
                 ).pack(side=tk.RIGHT, padx=22)
        self.btn_csv    = make_button(footer, "⊞  CSV",            self._pick_csv,       kind="ghost")
        self.btn_start  = make_button(footer, "▶  START SESSION",  self._start_session,  kind="success")
        self.btn_stop   = make_button(footer, "■  END SESSION",    self._end_session,    kind="danger")
        self.btn_report = make_button(footer, "✦  LLM REPORT",     self._generate_report, kind="violet")
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_report.config(state=tk.DISABLED)
        self.btn_csv.pack(side=tk.LEFT,    padx=(22, 6), pady=14)
        self.btn_start.pack(side=tk.LEFT,  padx=6,        pady=14)
        self.btn_stop.pack(side=tk.LEFT,   padx=6,        pady=14)
        self.btn_report.pack(side=tk.LEFT, padx=6,        pady=14)

        # Body: 3 cards
        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=14)
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=2)
        body.columnconfigure(2, weight=3)
        body.rowconfigure(0, weight=1)

        self._build_left_panel(body)
        self._build_middle_panel(body)
        self._build_right_panel(body)

    def _build_left_panel(self, body):
        card = make_card(body); card.grid(row=0, column=0, sticky="nsew", padx=(0, 9))
        p = tk.Frame(card, bg=COLORS["card"])
        p.pack(fill=tk.BOTH, expand=True, padx=18, pady=16)

        # Big counter card at the top
        events_box = tk.Frame(p, bg=COLORS["card_alt"])
        events_box.pack(fill=tk.X, ipady=14)
        tk.Label(events_box, text="EVENTS THIS SESSION", font=FONTS["caption"],
                 fg=COLORS["text_muted"], bg=COLORS["card_alt"]
                 ).pack(anchor="w", padx=14)
        self.lbl_events = tk.Label(events_box, text="0", font=FONTS["value_big"],
                                   fg=COLORS["accent"], bg=COLORS["card_alt"])
        self.lbl_events.pack(anchor="w", padx=14)

        tk.Frame(p, bg=COLORS["border"], height=1).pack(fill=tk.X, pady=14)

        section_title(p, "STATUS").pack(anchor="w")
        self.lbl_calib = tk.Label(p, text="", font=FONTS["body"],
                                  fg=COLORS["warning"], bg=COLORS["card"],
                                  wraplength=240, justify="left", anchor="w")
        self.lbl_calib.pack(anchor="w", pady=(8, 0), fill=tk.X)

    def _build_middle_panel(self, body):
        card = make_card(body); card.grid(row=0, column=1, sticky="nsew", padx=9)
        p = tk.Frame(card, bg=COLORS["card"])
        p.pack(fill=tk.BOTH, expand=True, padx=18, pady=16)

        section_title(p, "STROKE DISTRIBUTION").pack(anchor="w")
        self.dist_bars = {}
        for cid in range(4):
            name = CLASS_NAMES[cid]
            color = CLASS_COLORS[name]
            row_f = tk.Frame(p, bg=COLORS["card"])
            row_f.pack(fill=tk.X, pady=5)
            tk.Label(row_f, text=name, width=14, anchor="w",
                     font=FONTS["body_sm"], fg=color,
                     bg=COLORS["card"]).pack(side=tk.LEFT)
            bar = tk.Canvas(row_f, height=10, bg=COLORS["card_dark"],
                            highlightthickness=0)
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 6))
            lbl = tk.Label(row_f, text="0", width=5, anchor="e",
                           font=FONTS["body_sm"], fg=COLORS["text_dim"],
                           bg=COLORS["card"])
            lbl.pack(side=tk.LEFT)
            self.dist_bars[name] = (bar, lbl, color)

        tk.Frame(p, bg=COLORS["border"], height=1).pack(fill=tk.X, pady=18)

        section_title(p, "SESSION STATS").pack(anchor="w", pady=(0, 6))
        grid = tk.Frame(p, bg=COLORS["card"])
        grid.pack(fill=tk.X)
        self.stat_vars = {}
        for i, (label, key) in enumerate([
            ("Total Strokes",   "total_strokes"),
            ("Dominant Stroke", "dominant"),
            ("Avg Confidence",  "avg_conf"),
            ("Tempo Pattern",   "tempo"),
            ("Low Conf Events", "low_conf"),
            ("Weak Strokes",    "weak"),
        ]):
            tk.Label(grid, text=label, font=FONTS["body_sm"],
                     fg=COLORS["text_muted"], bg=COLORS["card"],
                     anchor="w").grid(row=i, column=0, sticky="w", pady=5)
            var = tk.StringVar(value="·")
            self.stat_vars[key] = var
            tk.Label(grid, textvariable=var, font=FONTS["body"],
                     fg=COLORS["text"], bg=COLORS["card"],
                     anchor="w").grid(row=i, column=1, sticky="w",
                                      padx=(16, 0), pady=5)
        grid.columnconfigure(1, weight=1)

    def _build_right_panel(self, body):
        card = make_card(body); card.grid(row=0, column=2, sticky="nsew", padx=(9, 0))
        p = tk.Frame(card, bg=COLORS["card"])
        p.pack(fill=tk.BOTH, expand=True, padx=18, pady=16)

        section_title(p, "SHOT PREDICTIONS").pack(anchor="w", pady=(0, 8))
        self.log_box = scrolledtext.ScrolledText(
            p, font=("Cascadia Mono", 12),
            bg=COLORS["card_dark"], fg=COLORS["text"],
            insertbackground=COLORS["accent"],
            selectbackground=COLORS["accent_dim"],
            relief="flat", bd=0, padx=10, pady=10,
            state=tk.DISABLED, wrap=tk.WORD,
        )
        self.log_box.pack(fill=tk.BOTH, expand=True)
        for name, color in CLASS_COLORS.items():
            self.log_box.tag_config(name, foreground=color)
        self.log_box.tag_config("header", foreground=COLORS["accent"],
                                font=("Cascadia Mono", 12, "bold"))
        self.log_box.tag_config("report", foreground=COLORS["violet"])
        self.log_box.tag_config("warn", foreground=COLORS["warning"])
        self.log_box.tag_config("muted", foreground=COLORS["text_muted"])
        self.log_box.tag_config("danger", foreground=COLORS["danger"])

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
        self._last_label_id = 0
        self._extractor.reset()

        self._log(
            f"··  Session started, Player {SUBJECT_ID}, "
            f"{pathlib.Path(self.csv_path).name}  ··\n",
            tag="header",
        )
        self.session_active = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_report.config(state=tk.DISABLED)
        self.btn_csv.config(state=tk.DISABLED)
        self.lbl_calib.config(text="◇ Calibrating skeleton")
        self.status_var.set("◉ Calibrating")

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
        self.status_var.set(f"◉ Session ended, {n} predictions")
        self.lbl_calib.config(text="")
        self._log(f"\n··  Session ended, {n} predictions  ··\n", tag="muted")

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
                    self.lbl_calib.config(text="◆ Skeleton calibrated",
                                          fg=COLORS["success"])
                elif msg_type == "status":
                    self.status_var.set(payload)
                elif msg_type == "stream_done":
                    if self.session_active:
                        self._end_session()
                elif msg_type == "error":
                    self._log(f"[ERROR]\n{payload}\n", tag="danger")
                    self.status_var.set("⚠ Streamer error, see Shot Predictions")
        except queue.Empty:
            pass
        self.after(30, self._poll_queue)

    # ── Per-frame inference ───────────────────────────────────────────────────

    def _handle_imu_row(self, row: dict):
        features = self._extractor.add_frame(row)
        if features is None:
            return
        result = self.predictor.predict(features)

        # ── Energy gate: force No Stroke on near-motionless windows ──────────
        if is_idle_window(features):
            result = {
                "label_id"     : 0,
                "label_name"   : CLASS_NAMES[0],
                "confidence"   : result["confidence"],
                "probabilities": result["probabilities"],
            }

        feat_dict = dict(zip(FEATURE_COLS, features.tolist()))
        self.session_rows.append(feat_dict)
        self.session_preds.append(result)

        # ── Stroke-event dedup: count one stroke per run, not per window ────
        label_id = result["label_id"]
        if label_id != 0 and label_id != self._last_label_id:
            self.stroke_counts[result["label_name"]] += 1
        self._last_label_id = label_id

        self._update_live_panel(result)
        self._update_stats_panel()
        self._log_event(len(self.session_preds), result)

    # ── UI update helpers ─────────────────────────────────────────────────────

    def _update_live_panel(self, result: dict):
        # Live prediction widgets removed; per-window predictions show in the
        # Shot Predictions list. We still maintain the running event counter.
        self.lbl_events.config(text=str(len(self.session_preds)))

    def _update_stats_panel(self):
        total = len(self.session_preds)
        if total == 0:
            return

        max_count = max(self.stroke_counts.values()) or 1
        for name, (bar, lbl, color) in self.dist_bars.items():
            cnt = self.stroke_counts[name]
            bar.update_idletasks()
            draw_progress_bar(bar, cnt / max_count, color=color)
            lbl.config(text=f"{cnt}")

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
            tempo = "·"

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
        self.status_var.set("Calling LLM, please wait")

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

        self._log("\n◆◆  LLM COACHING REPORT  ◆◆\n", tag="header")
        self._log(f"ASSESSMENT: {feedback['assessment']}\n\n", tag="report")
        for rec in feedback["recommendations"]:
            self._log(f"[{rec['priority']}] #{rec['rank']}  {rec['text']}\n",
                      tag="report")
        self._log(f"\nNEXT FOCUS: {feedback['next_focus']}\n", tag="report")
        self._log("·" * 50 + "\n", tag="muted")

        popup = tk.Toplevel(self)
        popup.title("Coaching Report")
        popup.geometry("680x520")
        popup.configure(bg=COLORS["bg"])
        head = tk.Frame(popup, bg=COLORS["surface"])
        head.pack(fill=tk.X)
        tk.Label(head, text=f"◆  Coaching Report  ·  Player {SUBJECT_ID}",
                 font=FONTS["h2"], fg=COLORS["text"],
                 bg=COLORS["surface"], pady=14).pack(side=tk.LEFT, padx=20)
        tk.Frame(popup, bg=COLORS["accent"], height=2).pack(fill=tk.X)
        txt = scrolledtext.ScrolledText(
            popup, font=FONTS["body"],
            bg=COLORS["card"], fg=COLORS["text"],
            insertbackground=COLORS["accent"],
            relief="flat", padx=16, pady=16, wrap=tk.WORD,
        )
        txt.pack(fill=tk.BOTH, expand=True, padx=14, pady=14)
        body_text = (
            f"ASSESSMENT\n{'·'*40}\n{feedback['assessment']}\n\n"
            f"RECOMMENDATIONS\n{'·'*40}\n"
        )
        for rec in feedback["recommendations"]:
            body_text += f"[{rec['priority']}] #{rec['rank']}\n{rec['text']}\n\n"
        body_text += f"NEXT SESSION FOCUS\n{'·'*40}\n{feedback['next_focus']}"
        txt.insert(tk.END, body_text)
        txt.config(state=tk.DISABLED)
        make_button(popup, "CLOSE", popup.destroy, kind="accent").pack(pady=(4, 16))

    def _report_error(self, error: str):
        self.btn_report.config(state=tk.NORMAL, text="🤖  Generate LLM Report")
        self.status_var.set("Report failed, check LM Studio is running")
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
