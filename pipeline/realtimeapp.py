# src/realtime_app.py
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

# ######################### Config #########################
ROOT         = pathlib.Path(__file__).resolve().parent.parent
STREAM_DELAY = 0.15          # seconds between simulated sensor rows
SUBJECT_ID   = 10            # change to any valid subject 0–92
COLORS = {
    "bg"       : "#1E1E2E",
    "panel"    : "#2A2A3E",
    "accent"   : "#7C3AED",
    "green"    : "#22C55E",
    "yellow"   : "#EAB308",
    "red"      : "#EF4444",
    "text"     : "#E2E8F0",
    "subtext"  : "#94A3B8",
    "border"   : "#3F3F5F",
}
CLASS_COLORS = {
    "No Stroke"    : "#64748B",
    "Stroke Type 1": "#3B82F6",
    "Stroke Type 2": "#10B981",
    "Stroke Type 3": "#F59E0B",
}


# ######################### Data streamer (runs in background thread) #########################

class DataStreamer(threading.Thread):
    """
    Reads TTSWING.csv rows for a given subject and pushes them
    into a queue at STREAM_DELAY intervals — simulating live IMU data.
    """
    def __init__(self, df: pd.DataFrame, event_queue: queue.Queue):
        super().__init__(daemon=True)
        self.df          = df.reset_index(drop=True)
        self.q           = event_queue
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        for _, row in self.df.iterrows():
            if self._stop_event.is_set():
                break
            self.q.put(("row", row))
            time.sleep(STREAM_DELAY)
        self.q.put(("stream_done", None))


# ######################### Main GUI #########################

class RealTimeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"TT Coaching System — Player {SUBJECT_ID}")
        self.geometry("980x720")
        self.configure(bg=COLORS["bg"])
        self.resizable(True, True)

        # Load config + model
        with open(ROOT / "config.yaml") as f:
            self.cfg = yaml.safe_load(f)

        self.predictor   = StrokePredictor.from_subject(SUBJECT_ID)
        self.df_subject  = pd.read_csv(ROOT / "data/raw/TTSWING.csv")
        self.df_subject  = self.df_subject[
            self.df_subject["id"] == SUBJECT_ID
        ].copy()

        # Session state
        self.session_rows   = []        # list of pd.Series
        self.session_preds  = []        # list of predict() dicts
        self.stroke_counts  = {CLASS_NAMES[i]: 0 for i in range(4)}
        self.event_queue    = queue.Queue()
        self.streamer       = None
        self.session_active = False

        self._build_ui()
        self._poll_queue()

    # ######################### UI construction #########################

    def _build_ui(self):
        # ######################### Header ──
        hdr = tk.Frame(self, bg=COLORS["accent"], pady=8)
        hdr.pack(fill=tk.X)
        tk.Label(
            hdr, text="🏓  Table Tennis IMU Coaching System",
            font=("Helvetica", 16, "bold"),
            fg="white", bg=COLORS["accent"],
        ).pack(side=tk.LEFT, padx=20)
        tk.Label(
            hdr, text=f"Player {SUBJECT_ID}  |  Simulation Mode",
            font=("Helvetica", 11),
            fg="#DDD6FE", bg=COLORS["accent"],
        ).pack(side=tk.RIGHT, padx=20)

        # ######################### Main body (3 columns) ──
        body = tk.Frame(self, bg=COLORS["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)
        body.columnconfigure(0, weight=2)
        body.columnconfigure(1, weight=2)
        body.columnconfigure(2, weight=3)
        body.rowconfigure(0, weight=1)

        self._build_left_panel(body)
        self._build_middle_panel(body)
        self._build_right_panel(body)

        # ######################### Footer buttons ──
        footer = tk.Frame(self, bg=COLORS["bg"], pady=8)
        footer.pack(fill=tk.X, padx=12)

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

        self.status_var = tk.StringVar(value="Ready — press Start Session")
        tk.Label(
            footer, textvariable=self.status_var,
            font=("Helvetica", 10), fg=COLORS["subtext"], bg=COLORS["bg"],
        ).pack(side=tk.RIGHT, padx=10)

    def _panel(self, parent, title, col, row=0, rowspan=1):
        frame = tk.LabelFrame(
            parent, text=f"  {title}  ",
            font=("Helvetica", 10, "bold"),
            fg=COLORS["subtext"], bg=COLORS["panel"],
            bd=1, relief=tk.FLAT,
            highlightbackground=COLORS["border"],
            highlightthickness=1,
        )
        frame.grid(row=row, column=col, rowspan=rowspan,
                   sticky="nsew", padx=5, pady=5)
        return frame

    def _build_left_panel(self, body):
        p = self._panel(body, "Live Prediction", col=0)

        # Current stroke label
        tk.Label(p, text="Current Stroke",
                 font=("Helvetica", 9), fg=COLORS["subtext"],
                 bg=COLORS["panel"]).pack(pady=(14, 2))

        self.lbl_stroke = tk.Label(
            p, text="—",
            font=("Helvetica", 22, "bold"),
            fg=COLORS["text"], bg=COLORS["panel"],
        )
        self.lbl_stroke.pack()

        # Confidence bar
        tk.Label(p, text="Confidence",
                 font=("Helvetica", 9), fg=COLORS["subtext"],
                 bg=COLORS["panel"]).pack(pady=(12, 2))

        self.conf_canvas = tk.Canvas(
            p, height=22, bg=COLORS["border"],
            highlightthickness=0,
        )
        self.conf_canvas.pack(fill=tk.X, padx=16)

        self.lbl_conf = tk.Label(
            p, text="0.00%",
            font=("Helvetica", 13, "bold"),
            fg=COLORS["text"], bg=COLORS["panel"],
        )
        self.lbl_conf.pack(pady=(4, 12))

        # Probability breakdown
        tk.Label(p, text="Class Probabilities",
                 font=("Helvetica", 9), fg=COLORS["subtext"],
                 bg=COLORS["panel"]).pack(pady=(4, 6))

        self.prob_bars = {}
        for cid in range(4):
            name = CLASS_NAMES[cid]
            row_f = tk.Frame(p, bg=COLORS["panel"])
            row_f.pack(fill=tk.X, padx=16, pady=2)
            tk.Label(row_f, text=name[:12], width=13, anchor="w",
                     font=("Helvetica", 8), fg=COLORS["subtext"],
                     bg=COLORS["panel"]).pack(side=tk.LEFT)
            bar_bg = tk.Canvas(row_f, height=14, bg=COLORS["border"],
                               highlightthickness=0)
            bar_bg.pack(side=tk.LEFT, fill=tk.X, expand=True)
            lbl = tk.Label(row_f, text="0%", width=5, anchor="e",
                           font=("Helvetica", 8), fg=COLORS["text"],
                           bg=COLORS["panel"])
            lbl.pack(side=tk.LEFT)
            self.prob_bars[name] = (bar_bg, lbl)

        # Event counter
        tk.Label(p, text="Events This Session",
                 font=("Helvetica", 9), fg=COLORS["subtext"],
                 bg=COLORS["panel"]).pack(pady=(16, 2))
        self.lbl_events = tk.Label(
            p, text="0",
            font=("Helvetica", 26, "bold"),
            fg=COLORS["accent"], bg=COLORS["panel"],
        )
        self.lbl_events.pack(pady=(0, 14))

    def _build_middle_panel(self, body):
        p = self._panel(body, "Session Statistics", col=1)

        # Stroke distribution bars
        tk.Label(p, text="Stroke Distribution",
                 font=("Helvetica", 9), fg=COLORS["subtext"],
                 bg=COLORS["panel"]).pack(pady=(14, 6))

        self.dist_bars = {}
        for cid in range(4):
            name = CLASS_NAMES[cid]
            color = CLASS_COLORS[name]
            row_f = tk.Frame(p, bg=COLORS["panel"])
            row_f.pack(fill=tk.X, padx=16, pady=3)
            tk.Label(row_f, text=name[:12], width=13, anchor="w",
                     font=("Helvetica", 8, "bold"),
                     fg=color, bg=COLORS["panel"]).pack(side=tk.LEFT)
            bar_bg = tk.Canvas(row_f, height=16, bg=COLORS["border"],
                               highlightthickness=0)
            bar_bg.pack(side=tk.LEFT, fill=tk.X, expand=True)
            lbl = tk.Label(row_f, text="0", width=6, anchor="e",
                           font=("Helvetica", 8), fg=COLORS["text"],
                           bg=COLORS["panel"])
            lbl.pack(side=tk.LEFT)
            self.dist_bars[name] = (bar_bg, lbl, color)

        # Stats grid
        tk.Frame(p, bg=COLORS["border"], height=1).pack(
            fill=tk.X, padx=16, pady=12)

        stats_grid = tk.Frame(p, bg=COLORS["panel"])
        stats_grid.pack(fill=tk.X, padx=16)

        self.stat_vars = {}
        stat_labels = [
            ("Total Strokes",   "total_strokes"),
            ("Dominant Stroke", "dominant"),
            ("Avg Confidence",  "avg_conf"),
            ("Tempo Pattern",   "tempo"),
            ("Low Conf Events", "low_conf"),
            ("Weak Strokes",    "weak"),
        ]
        for i, (label, key) in enumerate(stat_labels):
            tk.Label(stats_grid, text=label,
                     font=("Helvetica", 8), fg=COLORS["subtext"],
                     bg=COLORS["panel"], anchor="w").grid(
                row=i, column=0, sticky="w", pady=3)
            var = tk.StringVar(value="—")
            self.stat_vars[key] = var
            tk.Label(stats_grid, textvariable=var,
                     font=("Helvetica", 8, "bold"),
                     fg=COLORS["text"], bg=COLORS["panel"],
                     anchor="w").grid(row=i, column=1, sticky="w",
                                      padx=(10, 0), pady=3)

    def _build_right_panel(self, body):
        p = self._panel(body, "Event Log", col=2)

        self.log_box = scrolledtext.ScrolledText(
            p, font=("Courier", 8),
            bg="#0F0F1A", fg=COLORS["text"],
            insertbackground=COLORS["text"],
            relief=tk.FLAT, state=tk.DISABLED,
            wrap=tk.WORD,
        )
        self.log_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Tag colours for each class
        for name, color in CLASS_COLORS.items():
            self.log_box.tag_config(name, foreground=color)
        self.log_box.tag_config("header",
                                foreground=COLORS["accent"],
                                font=("Courier", 8, "bold"))
        self.log_box.tag_config("report",
                                foreground="#A78BFA",
                                font=("Courier", 8))

    # ######################### Session control #########################

    def _start_session(self):
        # Reset state
        self.session_rows  = []
        self.session_preds = []
        self.stroke_counts = {CLASS_NAMES[i]: 0 for i in range(4)}
        self._log(f"═══ Session started — Player {SUBJECT_ID} ═══\n",
                  tag="header")

        self.session_active = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_report.config(state=tk.DISABLED)
        self.status_var.set("Streaming IMU data…")

        self.streamer = DataStreamer(self.df_subject, self.event_queue)
        self.streamer.start()

    def _end_session(self):
        if self.streamer:
            self.streamer.stop()
        self.session_active = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.btn_report.config(
            state=tk.NORMAL if len(self.session_preds) > 0 else tk.DISABLED
        )
        self.status_var.set(
            f"Session ended — {len(self.session_preds)} events recorded"
        )
        self._log(f"\n═══ Session ended — {len(self.session_preds)} events ═══\n",
                  tag="header")

    # ######################### Queue polling (runs on main thread via after()) #########################

    def _poll_queue(self):
        try:
            while True:
                msg_type, payload = self.event_queue.get_nowait()
                if msg_type == "row":
                    self._process_row(payload)
                elif msg_type == "stream_done":
                    self._end_session()
        except queue.Empty:
            pass
        self.after(30, self._poll_queue)

    # ######################### Core inference per row #########################

    def _process_row(self, row: pd.Series):
        features = row[FEATURE_COLS].values.astype(np.float32)
        result   = self.predictor.predict(features)

        self.session_rows.append(row)
        self.session_preds.append(result)
        self.stroke_counts[result["label_name"]] += 1

        self._update_live_panel(result)
        self._update_stats_panel()
        self._log_event(len(self.session_preds), result)

    # def _process_row(self, row: pd.Series):
    #     features = row[FEATURE_COLS].values.astype(np.float32)
    #     result = self.predictor.predict(features)
    #

    #     self.session_rows.append(row)
    #     self.session_preds.append(result)
    #     self.stroke_counts[result["label_name"]] += 1
    #
    #     # Only refresh live panel when label changes
    #     last_label = (self.session_preds[-2]["label_name"]
    #                   if len(self.session_preds) > 1 else None)
    #     if result["label_name"] != last_label:
    #         self._update_live_panel(result)
    #
    #     self._update_stats_panel()
    #     self._log_event(len(self.session_preds), result)

    # ######################### UI update helpers #########################

    def _update_live_panel(self, result: dict):
        name  = result["label_name"]
        conf  = result["confidence"]
        color = CLASS_COLORS[name]

        self.lbl_stroke.config(text=STROKE_LABEL_MAP.get(name, name),
                               fg=color)
        self.lbl_conf.config(text=f"{conf:.2%}", fg=color)
        self.lbl_events.config(text=str(len(self.session_preds)))

        # Confidence bar
        self.conf_canvas.update_idletasks()
        w = self.conf_canvas.winfo_width()
        self.conf_canvas.delete("all")
        self.conf_canvas.create_rectangle(0, 0, int(w * conf), 22,
                                          fill=color, outline="")

        # Probability bars
        for cname, prob in result["probabilities"].items():
            bar_bg, lbl = self.prob_bars[cname]
            bar_bg.update_idletasks()
            bw = bar_bg.winfo_width()
            bar_bg.delete("all")
            bar_bg.create_rectangle(
                0, 0, int(bw * prob), 14,
                fill=CLASS_COLORS[cname], outline="",
            )
            lbl.config(text=f"{prob:.0%}")

    def _update_stats_panel(self):
        total = len(self.session_preds)
        if total == 0:
            return

        total_strokes = sum(v for k, v in self.stroke_counts.items()
                            if k != "No Stroke")
        max_count     = max(self.stroke_counts.values()) or 1

        # Distribution bars
        for name, (bar_bg, lbl, color) in self.dist_bars.items():
            cnt = self.stroke_counts[name]
            bar_bg.update_idletasks()
            bw = bar_bg.winfo_width()
            bar_bg.delete("all")
            bar_bg.create_rectangle(
                0, 0, int(bw * cnt / max_count), 16,
                fill=color, outline="",
            )
            pct = cnt / total * 100
            lbl.config(text=f"{cnt} ({pct:.0f}%)")

        # Stats
        dominant = max(
            (k for k in self.stroke_counts if k != "No Stroke"),
            key=lambda k: self.stroke_counts[k],
        )
        avg_conf  = np.mean([p["confidence"] for p in self.session_preds])
        low_conf  = sum(1 for p in self.session_preds
                        if p["confidence"] < 0.60)

        # Tempo CV
        stroke_idx = [i for i, p in enumerate(self.session_preds)
                      if p["label_id"] != 0]
        if len(stroke_idx) > 1:
            intervals = np.diff(stroke_idx)
            cv  = np.std(intervals) / np.mean(intervals)
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

    def _log_event(self, idx: int, result: dict):
        name = result["label_name"]
        conf = result["confidence"]
        real = STROKE_LABEL_MAP.get(name, name)
        self._log(
            f"[{idx:04d}]  {real:<22}  conf={conf:.2%}\n",
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

    # ######################### LLM Report #########################

    def _generate_report(self):
        if len(self.session_preds) == 0:
            messagebox.showwarning("No Data", "No session data to report on.")
            return

        self.btn_report.config(state=tk.DISABLED,
                               text="⏳  Generating…")
        self.status_var.set("Calling LLM — please wait…")

        def _run():
            try:
                session_df = pd.DataFrame(self.session_rows)
                summary    = run_session(self.predictor, session_df)
                feedback   = get_coaching_feedback(
                    summary, subject_id=SUBJECT_ID, cfg=self.cfg
                )
                self.after(0, lambda: self._show_report(feedback))
            except Exception as e:
                self.after(0, lambda: self._report_error(str(e)))

        threading.Thread(target=_run, daemon=True).start()

    def _show_report(self, feedback: dict):
        self.btn_report.config(state=tk.NORMAL,
                               text="🤖  Generate LLM Report")
        self.status_var.set("Report generated ✓")

        # Log to event log
        self._log("\n═══ LLM COACHING REPORT ═══\n", tag="header")
        self._log(f"ASSESSMENT: {feedback['assessment']}\n\n", tag="report")
        for rec in feedback["recommendations"]:
            self._log(
                f"[{rec['priority']}] #{rec['rank']}  {rec['text']}\n",
                tag="report",
            )
        self._log(
            f"\nNEXT FOCUS: {feedback['next_focus']}\n", tag="report"
        )
        self._log("═" * 45 + "\n", tag="header")

        # Popup window
        popup = tk.Toplevel(self)
        popup.title("Coaching Report")
        popup.geometry("620x480")
        popup.configure(bg=COLORS["bg"])

        tk.Label(
            popup,
            text=f"🏓  Coaching Report — Player {SUBJECT_ID}",
            font=("Helvetica", 13, "bold"),
            fg="white", bg=COLORS["accent"], pady=10,
        ).pack(fill=tk.X)

        txt = scrolledtext.ScrolledText(
            popup, font=("Helvetica", 10),
            bg=COLORS["panel"], fg=COLORS["text"],
            relief=tk.FLAT, padx=14, pady=14, wrap=tk.WORD,
        )
        txt.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        report_text = (
            f"ASSESSMENT\n{'─'*40}\n{feedback['assessment']}\n\n"
            f"RECOMMENDATIONS\n{'─'*40}\n"
        )
        for rec in feedback["recommendations"]:
            report_text += f"[{rec['priority']}] #{rec['rank']}\n{rec['text']}\n\n"
        report_text += f"NEXT SESSION FOCUS\n{'─'*40}\n{feedback['next_focus']}"

        txt.insert(tk.END, report_text)
        txt.config(state=tk.DISABLED)

        tk.Button(
            popup, text="Close",
            bg=COLORS["accent"], fg="white",
            relief=tk.FLAT, padx=16, pady=6,
            command=popup.destroy,
        ).pack(pady=(0, 12))

    def _report_error(self, error: str):
        self.btn_report.config(state=tk.NORMAL,
                               text="🤖  Generate LLM Report")
        self.status_var.set("Report failed — check LM Studio is running")
        messagebox.showerror(
            "LLM Error",
            f"Could not connect to LLM.\n\n"
            f"Make sure LM Studio server is running on port 1234.\n\n"
            f"Error: {error}",
        )


# ######################### Entry point #########################

if __name__ == "__main__":
    app = RealTimeApp()
    app.mainloop()