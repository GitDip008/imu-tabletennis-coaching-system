"""
TT Coaching App — Unity .exe embedded inside the Tkinter GUI via Win32.

Flow:
  1. Python launches IMU_Avatar.exe (windowed, no title bar)
  2. Finds Unity's HWND by process ID (reliable, no title guessing)
  3. Reparents it into a Tkinter frame with SetParent
  4. Strips window chrome → fits flush inside the panel
  5. SiriusCeption streams IMU CSV → UDP → Unity renders skeleton
  6. Same coaching panels as before (prediction / stats / log)

Requirements:
  pip install pywin32

Layout:
  ┌──────────┬──────────┬──────────────┬──────────────────────┐
  │ Live     │ Session  │  Event Log   │  Unity Skeleton      │
  │ Predict  │ Stats    │              │  (embedded .exe)     │
  └──────────┴──────────┴──────────────┴──────────────────────┘
  [CSV…] [Start Session] [End Session] [Generate Report]  status…
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

import ctypes
import ctypes.wintypes
import win32gui
import win32con
import win32process

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT      = pathlib.Path(__file__).resolve().parent.parent
UNITY_DIR = pathlib.Path(r"E:\thesis_work\1_new_test")
UNITY_EXE = pathlib.Path(r"E:\thesis_work\1_new_test\IMU_Avatar\game\IMU_Avatar.exe")

sys.path.insert(0, str(UNITY_DIR))
from SiriusCeption_unity_controller import IMUUnityController
from inference         import StrokePredictor, CLASS_NAMES, FEATURE_COLS
from summarizer        import run_session
from coaching          import get_coaching_feedback, STROKE_LABEL_MAP
from feature_extractor import SlidingWindowExtractor, WINDOW_SIZE, STEP_SIZE

# ── Config ────────────────────────────────────────────────────────────────────
SUBJECT_ID         = 10
CALIBRATION_FRAMES = 100
PLAYBACK_SPEED     = 1.0
DEFAULT_CSV        = str(UNITY_DIR / "imu_data_log_20250624_204958.csv")
UNITY_LOAD_TIMEOUT = 20      # seconds to wait for Unity window to appear

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


# ── Win32 Unity embedder ──────────────────────────────────────────────────────

class UnityEmbedder:
    """
    Launches IMU_Avatar.exe and embeds its window into a Tkinter frame.
    All Win32 calls happen on the main thread via callbacks.
    """

    def __init__(self):
        self._process:    subprocess.Popen | None = None
        self._unity_hwnd: int | None = None

    # ── launch ────────────────────────────────────────────────────────────────

    def launch(self, width: int, height: int):
        """Start Unity exe in windowed mode sized to fit our panel."""
        self._process = subprocess.Popen([
            str(UNITY_EXE),
            "-screen-fullscreen", "0",
            "-screen-width",  str(max(width,  100)),
            "-screen-height", str(max(height, 100)),
            "-popupwindow",          # no title bar
        ])

    # ── find window by PID ────────────────────────────────────────────────────

    def find_window(self, timeout: float = UNITY_LOAD_TIMEOUT) -> bool:
        """
        Poll until a visible top-level window belonging to Unity's PID appears.
        Returns True if found within timeout.
        """
        if self._process is None:
            return False

        pid   = self._process.pid
        start = time.time()

        while time.time() - start < timeout:
            found = []

            def _cb(hwnd, _):
                if not win32gui.IsWindowVisible(hwnd):
                    return
                try:
                    _, wpid = win32process.GetWindowThreadProcessId(hwnd)
                    if wpid == pid and win32gui.GetWindowText(hwnd):
                        found.append(hwnd)
                except Exception:
                    pass

            win32gui.EnumWindows(_cb, None)

            if found:
                self._unity_hwnd = found[0]
                return True

            time.sleep(0.3)

        return False

    # ── embed into the main Tk window ─────────────────────────────────────────

    def embed(self, parent_hwnd: int,
              rel_x: int, rel_y: int, w: int, h: int):
        """
        Reparent Unity under *parent_hwnd* (the main Tk window HWND) and
        position it at (rel_x, rel_y) in that window's client coordinates.

        We intentionally do NOT use the sub-frame HWND as parent because
        SetParent sends WM_SIZE messages that bypass Tkinter's geometry engine
        and collapse the frame to 1×1.  Parenting to the root Tk window is
        stable; we just place Unity exactly over the target frame area.
        """
        if not self._unity_hwnd:
            raise RuntimeError("No Unity HWND — call find_window() first.")

        # Reparent to the main Tk window
        win32gui.SetParent(self._unity_hwnd, parent_hwnd)

        # Strip outer window chrome; keep WS_CHILD | WS_CLIPSIBLINGS
        style = win32gui.GetWindowLong(self._unity_hwnd, win32con.GWL_STYLE)
        style = (
            (style & ~win32con.WS_OVERLAPPEDWINDOW)
            | win32con.WS_CHILD
            | win32con.WS_VISIBLE
            | win32con.WS_CLIPSIBLINGS   # don't bleed into adjacent panels
        )
        win32gui.SetWindowLong(self._unity_hwnd, win32con.GWL_STYLE, style)

        # Remove app-window / dropped-shadow extended style
        ex_style = win32gui.GetWindowLong(self._unity_hwnd, win32con.GWL_EXSTYLE)
        ex_style &= ~win32con.WS_EX_APPWINDOW
        win32gui.SetWindowLong(self._unity_hwnd, win32con.GWL_EXSTYLE, ex_style)

        # Position and size, on top of all sibling Tkinter widgets
        win32gui.SetWindowPos(
            self._unity_hwnd, win32con.HWND_TOP,
            rel_x, rel_y, w, h,
            win32con.SWP_SHOWWINDOW | win32con.SWP_NOACTIVATE,
        )

    # ── reposition / resize ───────────────────────────────────────────────────

    def reposition(self, rel_x: int, rel_y: int, w: int, h: int):
        """Move + resize Unity to track the target frame after a window resize."""
        if self._unity_hwnd and w > 0 and h > 0:
            try:
                win32gui.SetWindowPos(
                    self._unity_hwnd, win32con.HWND_TOP,
                    rel_x, rel_y, w, h,
                    win32con.SWP_SHOWWINDOW | win32con.SWP_NOACTIVATE,
                )
            except Exception:
                pass

    # ── teardown ──────────────────────────────────────────────────────────────

    def close(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self._unity_hwnd = None
        self._process    = None


# ── Background IMU streamer ───────────────────────────────────────────────────

class RawIMUStreamer(threading.Thread):
    """
    Calibrates via SiriusCeption, then streams rows → UDP → Unity.
    Also pushes ("imu_row", raw_row) for feature extraction.
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
                self.q.put(("calibrated", None))
                self.q.put(("status", "Streaming…"))

                for row in reader:
                    if self._stop_event.is_set():
                        break
                    sleep_s, _ = controller.process_row(row)   # sends UDP to Unity
                    self.q.put(("imu_row", row))
                    if 0 < sleep_s < 1.0:
                        time.sleep(sleep_s / max(PLAYBACK_SPEED, 0.01))

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print("[STREAMER ERROR]\n", tb)
            self.q.put(("error", tb))
        finally:
            if controller is not None:
                try:
                    controller.close()
                except Exception:
                    pass
            self.q.put(("stream_done", None))


# ── Main application ───────────────────────────────────────────────────────────

class RealTimeApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title(f"TT Coaching — Player {SUBJECT_ID}  [Unity Embedded]")
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

        self._embedder      = UnityEmbedder()
        self._unity_frame   = None   # set in _build_skeleton_panel
        self._unity_ready   = False

        self._build_ui()
        self._launch_unity()      # start Unity immediately on app open
        self._poll_queue()

    # ── Unity launch ──────────────────────────────────────────────────────────

    def _launch_unity(self):
        """Launch Unity exe and embed its window once it appears."""
        self.status_var.set("Launching Unity skeleton window…")

        # Force the UI to fully render so the frame has real pixel dimensions
        self.update()
        self.update_idletasks()
        w = max(self._unity_frame.winfo_width(),  400)
        h = max(self._unity_frame.winfo_height(), 300)
        print(f"[DEBUG] Unity frame size at launch: {w}×{h}")

        self._embedder.launch(w, h)

        # Find + embed in a background thread so the UI doesn't freeze
        def _find_and_embed():
            found = self._embedder.find_window(timeout=UNITY_LOAD_TIMEOUT)
            if found:
                self.after(0, self._do_embed)
            else:
                self.after(0, lambda: self._on_unity_not_found())

        threading.Thread(target=_find_and_embed, daemon=True).start()

    def _on_unity_not_found(self):
        self._unity_placeholder.config(
            text="❌  Unity window not found.\n\nCheck that IMU_Avatar.exe exists\n"
                 "and is not blocked by antivirus.")
        self.status_var.set("Unity not found — see skeleton panel")

    # ── coordinate helpers ────────────────────────────────────────────────────

    def _widget_client_rect(self, widget: tk.Widget):
        """
        Return (rel_x, rel_y, w, h) of *widget* in the main window's
        client-area coordinates.  We subtract winfo_rootx/y of the root
        window to convert from screen coords to client coords.
        """
        rel_x = widget.winfo_rootx() - self.winfo_rootx()
        rel_y = widget.winfo_rooty() - self.winfo_rooty()
        return rel_x, rel_y, widget.winfo_width(), widget.winfo_height()

    def _best_unity_rect(self):
        """
        Return (rel_x, rel_y, w, h) for where Unity should sit.
        Prefers _unity_frame; falls back to _unity_panel (LabelFrame) with
        a small inset so we don't overlap the panel's title label.
        Prints diagnostics each call during embed.
        """
        fx, fy, fw, fh = self._widget_client_rect(self._unity_frame)
        px, py, pw, ph = self._widget_client_rect(self._unity_panel)

        if fw > 10 and fh > 10:
            return fx, fy, fw, fh

        # Frame not usable — fall back to LabelFrame minus border+label inset
        INSET_X   = 4
        INSET_TOP = 22   # title label height
        INSET_BOT = 4
        rx = px + INSET_X
        ry = py + INSET_TOP
        rw = max(pw - 2 * INSET_X,        100)
        rh = max(ph - INSET_TOP - INSET_BOT, 100)
        return rx, ry, rw, rh

    # ── embed callback ────────────────────────────────────────────────────────

    def _do_embed(self):
        """Called on main thread once Unity window is found."""
        try:
            unity_title = win32gui.GetWindowText(self._embedder._unity_hwnd)
            self.update_idletasks()

            # ── full layout diagnostics ────────────────────────────────────
            print(f"[DEBUG] Main window : "
                  f"{self.winfo_width()}×{self.winfo_height()} "
                  f"@ screen ({self.winfo_rootx()},{self.winfo_rooty()})")
            print(f"[DEBUG] _unity_panel: "
                  f"{self._unity_panel.winfo_width()}×{self._unity_panel.winfo_height()} "
                  f"@ screen ({self._unity_panel.winfo_rootx()},"
                  f"{self._unity_panel.winfo_rooty()})")
            print(f"[DEBUG] _unity_frame: "
                  f"{self._unity_frame.winfo_width()}×{self._unity_frame.winfo_height()} "
                  f"@ screen ({self._unity_frame.winfo_rootx()},"
                  f"{self._unity_frame.winfo_rooty()})")
            print(f"[DEBUG] placeholder : "
                  f"{self._unity_placeholder.winfo_width()}×"
                  f"{self._unity_placeholder.winfo_height()}")

            rel_x, rel_y, w, h = self._best_unity_rect()
            main_hwnd = self.winfo_id()
            print(f"[DEBUG] Target rect  : ({rel_x},{rel_y})  {w}×{h}")

            # Parent to MAIN Tk window — avoids Win32 WM_SIZE collapsing frame
            self._embedder.embed(main_hwnd, rel_x, rel_y, w, h)

            # Hide placeholder text; frame stays pack'd to preserve layout ref
            self._unity_placeholder.config(text="", bg="#0F0F1A")

            self._unity_ready = True
            self.status_var.set("Unity ready — select CSV and press Start")

            self._unity_panel.bind("<Configure>", self._on_panel_resize)
            self.bind("<Configure>",              self._on_window_move)

            print(f"[DEBUG] Embed complete ✅")

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print("[EMBED ERROR]\n", tb)
            self._unity_placeholder.config(text=f"❌  Embed failed:\n{e}")
            self.status_var.set(f"Embed failed: {e}")

    def _on_panel_resize(self, event):
        """Fired when _unity_panel resizes — reposition Unity to match."""
        if not self._unity_ready:
            return
        rel_x, rel_y, w, h = self._best_unity_rect()
        if w > 10 and h > 10:
            self._embedder.reposition(rel_x, rel_y, w, h)

    # keep old name as alias so nothing breaks
    _on_frame_resize = _on_panel_resize

    def _on_window_move(self, event):
        """Fired when the main window moves — keep Unity aligned."""
        if not self._unity_ready or event.widget is not self:
            return
        rel_x, rel_y, w, h = self._best_unity_rect()
        if w > 10 and h > 10:
            self._embedder.reposition(rel_x, rel_y, w, h)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        hdr = tk.Frame(self, bg=COLORS["accent"], pady=8)
        hdr.pack(fill=tk.X)
        tk.Label(hdr, text="TT Coaching System  (Unity Skeleton Embedded)",
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
        self._build_unity_panel(body)   # must be last — sets self._unity_frame

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

        self.status_var = tk.StringVar(value="Initialising…")
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
                                font=("Courier", 8, "italic"))

    def _build_unity_panel(self, body):
        p = self._panel(body, "Unity Skeleton", col=3)
        self._unity_panel = p

        # _unity_frame is a layout anchor only — Unity is NOT reparented into
        # it (that would collapse it via Win32 WM_SIZE messages).  It stays a
        # normal Tkinter frame; Unity floats on top as a sibling child of the
        # main Tk window, positioned over this frame's screen rect.
        self._unity_frame = tk.Frame(p, bg="#0F0F1A")
        self._unity_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Placeholder is pack'd (not place'd) so the frame always has a
        # managed child and winfo_width/height() always return correct values.
        # After embed we clear the text — the frame keeps its size.
        self._unity_placeholder = tk.Label(
            self._unity_frame,
            text="⏳  Launching Unity…\n\nPlease wait, this takes ~5 seconds.",
            font=("Helvetica", 11), fg=COLORS["subtext"], bg="#0F0F1A",
            justify=tk.CENTER)
        self._unity_placeholder.pack(fill=tk.BOTH, expand=True)

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
        if not self._unity_ready:
            messagebox.showwarning("Unity Not Ready",
                                   "Unity window is still loading. Please wait.")
            return

        self.session_rows   = []
        self.session_preds  = []
        self.stroke_counts  = {CLASS_NAMES[i]: 0 for i in range(4)}
        self._extractor.reset()

        self._log(f"Session started — Player {SUBJECT_ID}  "
                  f"|  {pathlib.Path(self.csv_path).name}\n", tag="header")

        self.session_active = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.btn_report.config(state=tk.DISABLED)
        self.btn_csv.config(state=tk.DISABLED)
        self.lbl_calib.config(text=f"Calibrating 0/{CALIBRATION_FRAMES}")
        self.status_var.set("Calibrating skeleton…")

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

    # ── Graceful shutdown ─────────────────────────────────────────────────────

    def destroy(self):
        if self.streamer:
            self.streamer.stop()
        self._embedder.close()
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
                    self.status_var.set("Streamer error — see event log")
        except queue.Empty:
            pass
        self.after(30, self._poll_queue)

    # ── Core row handling ─────────────────────────────────────────────────────

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
