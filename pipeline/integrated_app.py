# src/integrated_app.py
"""
Full integrated system:
  - Streams 17-IMU CSV at real-time rate
  - Sends bone rotations to Unity via UDP (skeleton animation)
  - Extracts features from RightHand IMU sliding window
  - Runs TTSwing MLP stroke classifier on each window
  - Displays live predictions + session stats in Tkinter GUI
  - Generates Mistral-7B coaching report on demand
"""
import threading
import time
import queue
import pathlib
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pandas as pd
import numpy as np
import yaml

from inference import StrokePredictor, CLASS_NAMES, FEATURE_COLS
from summarizer import run_session, format_summary_for_prompt
from coaching import get_coaching_feedback, STROKE_LABEL_MAP
from imu_skeleton_bridge import IMUSkeletonBridge, WINDOW_SIZE, STRIDE

# ── Config ─────────────────────────────────────────────────────────────────
ROOT       = pathlib.Path(__file__).resolve().parent.parent
SUBJECT_ID = 10       # MLP checkpoint to use (nearest trained subject)
SEND_UDP   = False     # set False if Unity is not open

COLORS = {
    "bg"     : "#1E1E2E", "panel"  : "#2A2A3E",
    "accent" : "#7C3AED", "green"  : "#22C55E",
    "yellow" : "#EAB308", "red"    : "#EF4444",
    "text"   : "#E2E8F0", "subtext": "#94A3B8",
    "border" : "#3F3F5F",
}
CLASS_COLORS = {
    "No Stroke"    : "#64748B", "Stroke Type 1": "#3B82F6",
    "Stroke Type 2": "#10B981", "Stroke Type 3": "#F59E0B",
}


# ── Background streaming thread ────────────────────────────────────────────

class IMUStreamThread(threading.Thread):
    """
    Reads CSV row by row at real-time rate.
    Per frame:
      - Sends bone rotations to Unity (UDP)
      - If feature window ready → pushes prediction event to GUI queue
      - Always pushes skeleton frame count for heartbeat display
    """
    def __init__(self, bridge: IMUSkeletonBridge,
                 predictor: StrokePredictor,
                 event_queue: queue.Queue,
                 df: pd.DataFrame):
        super().__init__(daemon=True)
        self.bridge    = bridge
        self.predictor = predictor
        self.q         = event_queue
        self.df        = df
        self._stop     = threading.Event()

    def stop(self): self._stop.set()

    def run(self):
        delay = 1 / 89.2
        for _, row in self.df.iterrows():
            if self._stop.is_set():
                break

            bone_rots, features = self.bridge.process_frame(row)

            # Always send skeleton heartbeat
            self.q.put(("skeleton_frame", self.bridge.frame_count))

            # Only push prediction when feature window is ready
            if features is not None:
                result = self.predictor.predict(features)
                self.q.put(("prediction", {
                    "result"    : result,
                    "frame"     : self.bridge.frame_count,
                    "features"  : features,
                    "bone_rots" : bone_rots,
                }))

            time.sleep(delay)

        self.q.put(("stream_done", None))


# ── Main GUI ───────────────────────────────────────────────────────────────

class IntegratedApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TT Coaching System — 17-IMU + Skeleton + Classification")
        self.geometry("1100x760")
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)

        # Load config
        with open(ROOT / "config.yaml") as f:
            self.cfg = yaml.safe_load(f)

        # Load model + IMU files
        self.predictor = StrokePredictor.from_subject(SUBJECT_ID)
        self.df_imu    = pd.read_csv(
            ROOT / self.cfg["imu"]["csv_path"]
        )
        self.bridge    = IMUSkeletonBridge(
            csv_path          = str(ROOT / self.cfg["imu"]["csv_path"]),
            initial_pose_path = str(ROOT / self.cfg["imu"]["initial_pose_path"]),
            send_udp          = SEND_UDP,
        )

        # Session state
        self.session_preds    = []
        self.session_features = []
        self.stroke_counts    = {CLASS_NAMES[i]: 0 for i in range(4)}
        self.skeleton_frames  = 0
        self.event_queue      = queue.Queue()
        self.stream_thread    = None
        self.session_active   = False

        self._build_ui()
        self._poll_queue()

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=COLORS["accent"], pady=8)
        hdr.pack(fill=tk.X)
        tk.Label(hdr,
                 text="🏓  Table Tennis — 17-IMU Skeleton + Coaching System",
                 font=("Helvetica", 15, "bold"),
                 fg="white", bg=COLORS["accent"]).pack(side=tk.LEFT, padx=20)
        self.lbl_skeleton = tk.Label(
            hdr, text="Skeleton: 0 frames",
            font=("Helvetica", 10), fg="#DDD6FE", bg=COLORS["accent"])
        self.lbl_skeleton.pack(side=tk.RIGHT, padx=20)

        # Body — 3 columns
        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        for c, w in enumerate([2, 2, 3]):
            body.columnconfigure(c, weight=w)
        body.rowconfigure(0, weight=1)

        self._build_live_panel(body)
        self._build_stats_panel(body)
        self._build_log_panel(body)

        # Footer
        footer = tk.Frame(self, bg=COLORS["bg"], pady=6)
        footer.pack(fill=tk.X, padx=10)

        self.btn_start = self._btn(footer, "▶  Start Session",
                                   COLORS["green"], self._start_session)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_stop = self._btn(footer, "■  End Session",
                                  COLORS["red"], self._end_session,
                                  state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=(0, 8))

        self.btn_report = self._btn(footer, "🤖  LLM Coaching Report",
                                    COLORS["accent"], self._generate_report,
                                    state=tk.DISABLED)
        self.btn_report.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Ready — press Start Session")
        tk.Label(footer, textvariable=self.status_var,
                 font=("Helvetica", 9), fg=COLORS["subtext"],
                 bg=COLORS["bg"]).pack(side=tk.RIGHT, padx=10)

    def _btn(self, parent, text, color, cmd, state=tk.NORMAL):
        return tk.Button(parent, text=text,
                         font=("Helvetica", 10, "bold"),
                         bg=color, fg="white", relief=tk.FLAT,
                         padx=16, pady=7, cursor="hand2",
                         command=cmd, state=state)

    def _panel(self, parent, title, col):
        f = tk.LabelFrame(parent, text=f"  {title}  ",
                          font=("Helvetica", 9, "bold"),
                          fg=COLORS["subtext"], bg=COLORS["panel"],
                          bd=1, relief=tk.FLAT,
                          highlightbackground=COLORS["border"],
                          highlightthickness=1)
        f.grid(row=0, column=col, sticky="nsew", padx=4, pady=4)
        return f

    def _build_live_panel(self, body):
        p = self._panel(body, "Live Prediction", 0)

        # IMU info box
        imu_box = tk.Frame(p, bg="#14142A", bd=0)
        imu_box.pack(fill=tk.X, padx=12, pady=(12, 4))
        tk.Label(imu_box, text="📡  17 IMUs Active",
                 font=("Helvetica", 9, "bold"),
                 fg=COLORS["green"], bg="#14142A").pack(side=tk.LEFT, padx=8)
        tk.Label(imu_box, text="→ Unity Skeleton",
                 font=("Helvetica", 9),
                 fg=COLORS["subtext"], bg="#14142A").pack(side=tk.LEFT)

        tk.Label(p, text="Racket Wrist (IMU 10) → Classifier",
                 font=("Helvetica", 8), fg=COLORS["subtext"],
                 bg=COLORS["panel"]).pack(pady=(4, 2))

        # Current stroke
        tk.Label(p, text="Current Stroke",
                 font=("Helvetica", 9), fg=COLORS["subtext"],
                 bg=COLORS["panel"]).pack(pady=(8, 2))
        self.lbl_stroke = tk.Label(p, text="—",
                                   font=("Helvetica", 20, "bold"),
                                   fg=COLORS["text"], bg=COLORS["panel"])
        self.lbl_stroke.pack()

        # Window info
        self.lbl_window = tk.Label(
            p, text=f"Window: {WINDOW_SIZE} frames | Stride: {STRIDE}",
            font=("Helvetica", 8), fg=COLORS["subtext"], bg=COLORS["panel"])
        self.lbl_window.pack(pady=2)

        # Confidence bar
        tk.Label(p, text="Confidence", font=("Helvetica", 9),
                 fg=COLORS["subtext"], bg=COLORS["panel"]).pack(pady=(10, 2))
        self.conf_canvas = tk.Canvas(p, height=20, bg=COLORS["border"],
                                     highlightthickness=0)
        self.conf_canvas.pack(fill=tk.X, padx=14)
        self.lbl_conf = tk.Label(p, text="0.00%",
                                 font=("Helvetica", 12, "bold"),
                                 fg=COLORS["text"], bg=COLORS["panel"])
        self.lbl_conf.pack(pady=(2, 8))

        # Probability bars
        tk.Label(p, text="Class Probabilities",
                 font=("Helvetica", 9), fg=COLORS["subtext"],
                 bg=COLORS["panel"]).pack(pady=(4, 4))
        self.prob_bars = {}
        for cid in range(4):
            name = CLASS_NAMES[cid]
            row_f = tk.Frame(p, bg=COLORS["panel"])
            row_f.pack(fill=tk.X, padx=14, pady=2)
            tk.Label(row_f, text=name[:11], width=12, anchor="w",
                     font=("Helvetica", 8), fg=COLORS["subtext"],
                     bg=COLORS["panel"]).pack(side=tk.LEFT)
            bar = tk.Canvas(row_f, height=13, bg=COLORS["border"],
                            highlightthickness=0)
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            lbl = tk.Label(row_f, text="0%", width=5, anchor="e",
                           font=("Helvetica", 8), fg=COLORS["text"],
                           bg=COLORS["panel"])
            lbl.pack(side=tk.LEFT)
            self.prob_bars[name] = (bar, lbl)

        # Prediction counter
        tk.Label(p, text="Predictions This Session",
                 font=("Helvetica", 9), fg=COLORS["subtext"],
                 bg=COLORS["panel"]).pack(pady=(14, 2))
        self.lbl_pred_count = tk.Label(p, text="0",
                                       font=("Helvetica", 24, "bold"),
                                       fg=COLORS["accent"], bg=COLORS["panel"])
        self.lbl_pred_count.pack(pady=(0, 12))

    def _build_stats_panel(self, body):
        p = self._panel(body, "Session Statistics", 1)

        tk.Label(p, text="Stroke Distribution",
                 font=("Helvetica", 9), fg=COLORS["subtext"],
                 bg=COLORS["panel"]).pack(pady=(12, 6))

        self.dist_bars = {}
        for cid in range(4):
            name  = CLASS_NAMES[cid]
            color = CLASS_COLORS[name]
            row_f = tk.Frame(p, bg=COLORS["panel"])
            row_f.pack(fill=tk.X, padx=14, pady=3)
            tk.Label(row_f, text=name[:11], width=12, anchor="w",
                     font=("Helvetica", 8, "bold"),
                     fg=color, bg=COLORS["panel"]).pack(side=tk.LEFT)
            bar = tk.Canvas(row_f, height=15, bg=COLORS["border"],
                            highlightthickness=0)
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
            lbl = tk.Label(row_f, text="0", width=7, anchor="e",
                           font=("Helvetica", 8), fg=COLORS["text"],
                           bg=COLORS["panel"])
            lbl.pack(side=tk.LEFT)
            self.dist_bars[name] = (bar, lbl, color)

        tk.Frame(p, bg=COLORS["border"], height=1).pack(
            fill=tk.X, padx=14, pady=10)

        # Stats grid
        grid = tk.Frame(p, bg=COLORS["panel"])
        grid.pack(fill=tk.X, padx=14)
        self.stat_vars = {}
        for i, (label, key) in enumerate([
            ("Skeleton Frames", "sk_frames"),
            ("IMU Sample Rate", "imu_rate"),
            ("Total Strokes",   "total_strokes"),
            ("Dominant Stroke", "dominant"),
            ("Avg Confidence",  "avg_conf"),
            ("Tempo Pattern",   "tempo"),
            ("Low Conf Events", "low_conf"),
            ("Weak Strokes",    "weak"),
        ]):
            tk.Label(grid, text=label, font=("Helvetica", 8),
                     fg=COLORS["subtext"], bg=COLORS["panel"],
                     anchor="w").grid(row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value="—")
            self.stat_vars[key] = var
            tk.Label(grid, textvariable=var,
                     font=("Helvetica", 8, "bold"),
                     fg=COLORS["text"], bg=COLORS["panel"],
                     anchor="w").grid(row=i, column=1, sticky="w",
                                      padx=(10, 0), pady=2)

    def _build_log_panel(self, body):
        p = self._panel(body, "Event Log", 2)
        self.log_box = scrolledtext.ScrolledText(
            p, font=("Courier", 8),
            bg="#0F0F1A", fg=COLORS["text"],
            relief=tk.FLAT, state=tk.DISABLED, wrap=tk.WORD)
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        for name, color in CLASS_COLORS.items():
            self.log_box.tag_config(name, foreground=color)
        self.log_box.tag_config("header",
                                foreground=COLORS["accent"],
                                font=("Courier", 8, "bold"))
        self.log_box.tag_config("skeleton",
                                foreground="#475569",
                                font=("Courier", 7))
        self.log_box.tag_config("report",
                                foreground="#A78BFA",
                                font=("Courier", 8))

    # ── Session control ────────────────────────────────────────────────────

    def _start_session(self):
        self.session_preds    = []
        self.session_features = []
        self.stroke_counts    = {CLASS_NAMES[i]: 0 for i in range(4)}
        self.skeleton_frames  = 0

        # Reset bridge
        self.bridge = IMUSkeletonBridge(
            csv_path          = str(ROOT / self.cfg["imu"]["csv_path"]),
            initial_pose_path = str(ROOT / self.cfg["imu"]["initial_pose_path"]),
            send_udp          = SEND_UDP,
        )

        self._log("═══ Session started — 17-IMU stream ═══\n", "header")
        self._log(f"  Skeleton → Unity UDP {SEND_UDP}\n", "skeleton")
        self._log(f"  Classifier → IMU {self.cfg['imu']['racket_imu_id']} "
                  f"(RightHand)\n\n", "skeleton")

        self.session_active = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_report.config(state=tk.DISABLED)
        self.status_var.set("Streaming 17-IMU data…")

        self.stream_thread = IMUStreamThread(
            self.bridge, self.predictor,
            self.event_queue, self.df_imu
        )
        self.stream_thread.start()

    def _end_session(self):
        if self.stream_thread:
            self.stream_thread.stop()
        self.session_active = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_report.config(
            state=tk.NORMAL if len(self.session_preds) > 0 else tk.DISABLED)
        n = len(self.session_preds)
        self.status_var.set(
            f"Session ended — {self.skeleton_frames} skeleton frames, "
            f"{n} predictions")
        self._log(f"\n═══ Session ended ═══\n"
                  f"  Skeleton frames : {self.skeleton_frames}\n"
                  f"  Predictions     : {n}\n\n", "header")

    # ── Queue polling ──────────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                msg_type, payload = self.event_queue.get_nowait()
                if msg_type == "prediction":
                    self._on_prediction(payload)
                elif msg_type == "skeleton_frame":
                    self._on_skeleton_frame(payload)
                elif msg_type == "stream_done":
                    self._end_session()
        except queue.Empty:
            pass
        self.after(30, self._poll_queue)

    def _on_skeleton_frame(self, frame_count: int):
        self.skeleton_frames = frame_count
        self.lbl_skeleton.config(
            text=f"Skeleton: {frame_count} frames")
        self.stat_vars["sk_frames"].set(str(frame_count))
        self.stat_vars["imu_rate"].set("89.2 Hz")

    def _on_prediction(self, payload: dict):
        result   = payload["result"]
        frame    = payload["frame"]
        features = payload["features"]

        self.session_preds.append(result)
        self.session_features.append(features)
        self.stroke_counts[result["label_name"]] += 1

        self._update_live_panel(result)
        self._update_stats_panel()
        self._log_prediction(frame, result)

    # ── UI updates ─────────────────────────────────────────────────────────

    def _update_live_panel(self, result: dict):
        name  = result["label_name"]
        conf  = result["confidence"]
        color = CLASS_COLORS[name]
        real  = STROKE_LABEL_MAP.get(name, name)

        self.lbl_stroke.config(text=real, fg=color)
        self.lbl_conf.config(text=f"{conf:.2%}", fg=color)
        self.lbl_pred_count.config(text=str(len(self.session_preds)))

        self.conf_canvas.update_idletasks()
        w = self.conf_canvas.winfo_width()
        self.conf_canvas.delete("all")
        self.conf_canvas.create_rectangle(
            0, 0, int(w * conf), 20, fill=color, outline="")

        for cname, prob in result["probabilities"].items():
            bar, lbl = self.prob_bars[cname]
            bar.update_idletasks()
            bw = bar.winfo_width()
            bar.delete("all")
            bar.create_rectangle(0, 0, int(bw * prob), 13,
                                 fill=CLASS_COLORS[cname], outline="")
            lbl.config(text=f"{prob:.0%}")

    def _update_stats_panel(self):
        total = len(self.session_preds)
        if total == 0:
            return

        total_strokes = sum(v for k, v in self.stroke_counts.items()
                            if k != "No Stroke")
        max_cnt = max(self.stroke_counts.values()) or 1

        for name, (bar, lbl, color) in self.dist_bars.items():
            cnt = self.stroke_counts[name]
            bar.update_idletasks()
            bw = bar.winfo_width()
            bar.delete("all")
            bar.create_rectangle(0, 0, int(bw * cnt / max_cnt), 15,
                                 fill=color, outline="")
            lbl.config(text=f"{cnt} ({cnt/total*100:.0f}%)")

        dominant = max(
            (k for k in self.stroke_counts if k != "No Stroke"),
            key=lambda k: self.stroke_counts[k])

        avg_conf = np.mean([p["confidence"] for p in self.session_preds])
        low_conf = sum(1 for p in self.session_preds
                       if p["confidence"] < 0.60)

        stroke_idx = [i for i, p in enumerate(self.session_preds)
                      if p["label_id"] != 0]
        if len(stroke_idx) > 1:
            cv    = np.std(np.diff(stroke_idx)) / (np.mean(np.diff(stroke_idx)) + 1e-8)
            tempo = "consistent" if cv < 0.2 else \
                    "moderate"   if cv < 0.4 else "irregular"
        else:
            tempo = "—"

        weak = [STROKE_LABEL_MAP.get(k, k)
                for k, v in self.stroke_counts.items()
                if k != "No Stroke" and v > 0 and
                np.mean([p["confidence"] for p in self.session_preds
                         if p["label_name"] == k] or [1]) < 0.65]

        self.stat_vars["total_strokes"].set(str(total_strokes))
        self.stat_vars["dominant"].set(STROKE_LABEL_MAP.get(dominant, dominant))
        self.stat_vars["avg_conf"].set(f"{avg_conf:.2%}")
        self.stat_vars["tempo"].set(tempo)
        self.stat_vars["low_conf"].set(str(low_conf))
        self.stat_vars["weak"].set(", ".join(weak) if weak else "None")

    def _log_prediction(self, frame: int, result: dict):
        name = result["label_name"]
        real = STROKE_LABEL_MAP.get(name, name)
        conf = result["confidence"]
        self._log(
            f"[F{frame:05d}]  {real:<22}  conf={conf:.2%}\n",
            tag=name)

    def _log(self, text: str, tag: str = None):
        self.log_box.config(state=tk.NORMAL)
        if tag:
            self.log_box.insert(tk.END, text, tag)
        else:
            self.log_box.insert(tk.END, text)
        self.log_box.see(tk.END)
        self.log_box.config(state=tk.DISABLED)

    # ── LLM Report ────────────────────────────────────────────────────────

    def _generate_report(self):
        if not self.session_preds:
            messagebox.showwarning("No Data", "No predictions to report on.")
            return

        self.btn_report.config(state=tk.DISABLED, text="⏳  Generating…")
        self.status_var.set("Calling Mistral-7B — please wait…")

        def _run():
            try:
                # Build synthetic session DataFrame from feature vectors
                session_df = pd.DataFrame(
                    self.session_features, columns=FEATURE_COLS)
                # Add required id column for run_session compatibility
                session_df["id"] = SUBJECT_ID

                summary  = run_session(self.predictor, session_df)
                feedback = get_coaching_feedback(
                    summary, subject_id=SUBJECT_ID, cfg=self.cfg)
                self.after(0, lambda: self._show_report(feedback, summary))
            except Exception as e:
                self.after(0, lambda: self._report_error(str(e)))

        threading.Thread(target=_run, daemon=True).start()

    def _show_report(self, feedback: dict, summary: dict):
        self.btn_report.config(state=tk.NORMAL,
                               text="🤖  LLM Coaching Report")
        self.status_var.set("Report generated ✓")

        self._log("\n═══ COACHING REPORT ═══\n", "header")
        self._log(f"{feedback['assessment']}\n\n", "report")
        for rec in feedback["recommendations"]:
            self._log(f"[{rec['priority']}] {rec['text']}\n", "report")
        self._log(f"\nFOCUS: {feedback['next_focus']}\n", "report")
        self._log("═" * 44 + "\n", "header")

        popup = tk.Toplevel(self)
        popup.title("Coaching Report")
        popup.geometry("650x500")
        popup.configure(bg=COLORS["bg"])

        tk.Label(popup,
                 text="🏓  Coaching Report — 17-IMU Session",
                 font=("Helvetica", 13, "bold"),
                 fg="white", bg=COLORS["accent"], pady=10).pack(fill=tk.X)

        # Session stats strip
        strip = tk.Frame(popup, bg=COLORS["panel"], pady=6)
        strip.pack(fill=tk.X, padx=12, pady=(8, 0))
        stats = [
            ("Skeleton Frames", f"{self.skeleton_frames}"),
            ("Predictions",     f"{len(self.session_preds)}"),
            ("Total Strokes",   f"{summary['total_strokes']}"),
            ("Avg Confidence",  f"{summary['overall_avg_confidence']:.2%}"),
        ]
        for label, val in stats:
            col = tk.Frame(strip, bg=COLORS["panel"])
            col.pack(side=tk.LEFT, expand=True)
            tk.Label(col, text=val, font=("Helvetica", 14, "bold"),
                     fg=COLORS["accent"], bg=COLORS["panel"]).pack()
            tk.Label(col, text=label, font=("Helvetica", 8),
                     fg=COLORS["subtext"], bg=COLORS["panel"]).pack()

        txt = scrolledtext.ScrolledText(
            popup, font=("Helvetica", 10),
            bg=COLORS["panel"], fg=COLORS["text"],
            relief=tk.FLAT, padx=14, pady=12, wrap=tk.WORD)
        txt.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        report_text = (
            f"ASSESSMENT\n{'─'*42}\n{feedback['assessment']}\n\n"
            f"RECOMMENDATIONS\n{'─'*42}\n")
        for rec in feedback["recommendations"]:
            report_text += f"[{rec['priority']}] #{rec['rank']}\n{rec['text']}\n\n"
        report_text += (
            f"NEXT SESSION FOCUS\n{'─'*42}\n{feedback['next_focus']}\n\n"
            f"SESSION DATA\n{'─'*42}\n"
            f"{format_summary_for_prompt(summary)}")

        txt.insert(tk.END, report_text)
        txt.config(state=tk.DISABLED)

        tk.Button(popup, text="Close",
                  bg=COLORS["accent"], fg="white",
                  relief=tk.FLAT, padx=16, pady=6,
                  command=popup.destroy).pack(pady=(0, 10))

    def _report_error(self, error: str):
        self.btn_report.config(state=tk.NORMAL,
                               text="🤖  LLM Coaching Report")
        self.status_var.set("Report failed — check LM Studio")
        messagebox.showerror("LLM Error",
                             f"Could not reach Mistral-7B.\n\n"
                             f"Make sure LM Studio server is running "
                             f"on port 1234.\n\nError: {error}")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = IntegratedApp()
    app.mainloop()