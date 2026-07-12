"""
live_app.py
───────────
REAL-TIME TT coaching app — reads the SiriusCeption sensors directly over
UDP (no CSV in between) and classifies strokes live.

Pipeline:
    SiriusCeption sensor (id=7, racket hand) → WiFi UDP :9999
        → Zuyan's UDPIMUServer (async, background thread)
        → poll latest packet at SAMPLE_RATE Hz → build a feature row
        → SlidingWindowExtractor (50-frame window, step 10)
        → energy gate (idle → No Stroke)
        → StrokePredictor (fine-tuned MuJoCo model)
        → OnlineStrokeCounter (one count per motion-energy peak)
        → live Tkinter dashboard + on-demand LLM report

This file is fully self-contained: it only IMPORTS from the existing modules
(inference, feature_extractor, summarizer, coaching, udp_imu_server) and does
not modify any of them. realtimeapp_new.py is untouched.

Run:
    python live_app.py
Prerequisites:
    - Sensor id=7 powered, on the same WiFi, host = this PC's IP, udp_hz=100
    - LM Studio running for the LLM report (optional)
"""
import sys
import time
import queue
import asyncio
import pathlib
import threading
import subprocess
import tkinter as tk
from tkinter import scrolledtext, messagebox

# Modern UI library — rounded widgets + dark theme
import customtkinter as ctk
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

import numpy as np
import pandas as pd
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
SRC_DIR  = pathlib.Path(__file__).resolve().parent
ROOT     = SRC_DIR.parent
IMU_DATA_DIR = pathlib.Path(r"E:\thesis_work\TT_thesis\imu_data")   # has udp_imu_server.py

UNITY_DIR = pathlib.Path(r"E:\thesis_work\1_new_test")          # Unity assets + .pyd
UNITY_EXE = UNITY_DIR / "IMU_Avatar" / "game3" / "IMU_Avatar.exe"

sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(IMU_DATA_DIR))
sys.path.insert(0, str(UNITY_DIR))

from udp_imu_server   import UDPIMUServer
from inference        import StrokePredictor, CLASS_NAMES, FEATURE_COLS
from feature_extractor import (
    SlidingWindowExtractor, WINDOW_SIZE, STEP_SIZE,
    is_idle_window, window_energy,
)
from summarizer       import run_session, PEAK_MIN_DISTANCE, PEAK_MIN_HEIGHT
from coaching         import get_coaching_feedback

# Unity skeleton controller (.pyd) — optional, skeleton disabled if unavailable
try:
    from SiriusCeption_unity_controller import IMUUnityController
    UNITY_CONTROLLER_OK = True
except Exception as _e:
    UNITY_CONTROLLER_OK = False
    print(f"[WARN] Unity controller not importable, skeleton disabled ({_e})")

# ── Config ───────────────────────────────────────────────────────────────────
SUBJECT_ID    = 10
UDP_PORT      = 9999
RACKET_ID     = 7            # RightHand sensor
SAMPLE_RATE   = 100          # Hz — must match the sensor's udp_hz
MODEL_PATH    = ROOT / "mujoco_sim" / "output" / "model_synthetic.pt"
SCALER_PATH   = ROOT / "mujoco_sim" / "output" / "scaler_synthetic.pkl"

SHOW_UNITY        = True     # embed + stream to the Unity skeleton window
UNITY_BONE_PORT   = 5005     # IMUUnityController → Unity UDP bone stream
CALIBRATION_FRAMES = 100     # frames used for the controller's T-pose calibration
TPOSE_SECONDS     = 10       # hold-T-pose countdown before classification starts

from ui_theme import (
    COLORS, CLASS_COLORS, SHORT_NAMES, FONTS,
    make_card, section_title, make_button,
    draw_progress_bar, style_header, style_footer,
)

# Optional win32 for embedding the Unity window cleanly
try:
    import win32gui
    WIN32_OK = True
except ImportError:
    WIN32_OK = False

# Return-swing rejection.
# The "bringing the arm back to ready" motion produces a LOW-energy peak
# (~20-30) compared to a real swing (140+). A modest absolute energy floor
# cleanly separates the two without rejecting genuinely soft real swings
# (which an over-aggressive relative-to-median gate would wrongly drop).
LIVE_PEAK_MIN_HEIGHT = 40.0     # raised from the offline default of 20
PEAK_REL_FRAC        = 0.0      # 0 = relative gate OFF (absolute floor handles it)


# ── Live UDP streamer (background thread + asyncio loop) ───────────────────────

class LiveIMUStreamer(threading.Thread):
    """
    Runs Zuyan's UDPIMUServer inside its own asyncio loop and polls the
    latest packet for RACKET_ID at SAMPLE_RATE Hz, pushing feature-ready
    rows into the event queue. Only slot-7 accel/gyro keys are needed by
    the feature extractor, but we include the quaternion too for safety.
    """

    def __init__(self, event_queue: queue.Queue, controller=None,
                 racket_id: int = RACKET_ID, rate_hz: int = SAMPLE_RATE):
        super().__init__(daemon=True)
        self.q          = event_queue
        self.controller = controller          # optional IMUUnityController
        self.racket_id  = racket_id
        self.rate_hz    = rate_hz
        self._stop      = threading.Event()
        self._loop      = None
        self._server    = None

    def stop(self):
        self._stop.set()

    @staticmethod
    def _build_full_row(d: dict) -> dict:
        """
        Build a full 17-slot CSV-style row. The racket slot gets the real
        packet; the other 16 slots get identity quaternion + zero accel/gyro
        (the fixed-matrix convention). Feature extraction reads slot 7 only;
        the Unity controller reads all 17.
        """
        row = {}
        for i in range(17):
            p = f"imu_{i}_"
            row[p + "quat_x"] = 0.0; row[p + "quat_y"] = 0.0
            row[p + "quat_z"] = 0.0; row[p + "quat_w"] = 1.0
            row[p + "accel_x"] = 0.0; row[p + "accel_y"] = 0.0; row[p + "accel_z"] = 0.0
            row[p + "gyro_x"]  = 0.0; row[p + "gyro_y"]  = 0.0; row[p + "gyro_z"]  = 0.0
        q = d["quat"]; a = d["accel"]; g = d["gyro"]
        p = f"imu_{RACKET_ID}_"
        row[p + "quat_w"] = q[0]; row[p + "quat_x"] = q[1]
        row[p + "quat_y"] = q[2]; row[p + "quat_z"] = q[3]
        row[p + "accel_x"] = a[0]; row[p + "accel_y"] = a[1]; row[p + "accel_z"] = a[2]
        row[p + "gyro_x"]  = g[0]; row[p + "gyro_y"]  = g[1]; row[p + "gyro_z"]  = g[2]
        return row

    async def _main(self):
        self._server = UDPIMUServer(port=UDP_PORT)
        self._server.run()                       # creates the listener task
        self.q.put(("status", f"Listening on UDP :{UDP_PORT}, waiting for sensor {self.racket_id}"))
        dt = 1.0 / self.rate_hz

        # ── wait for first packet ───────────────────────────────────────────
        while not self._stop.is_set():
            if self.racket_id in self._server.get_latest_data():
                break
            await asyncio.sleep(dt)

        # ── optional Unity calibration: collect N frames, then calibrate ─────
        if self.controller is not None:
            self.q.put(("status", f"Calibrating skeleton ({CALIBRATION_FRAMES} frames), hold T-pose"))
            calib_rows = []
            while not self._stop.is_set() and len(calib_rows) < CALIBRATION_FRAMES:
                latest = self._server.get_latest_data()
                if self.racket_id in latest:
                    calib_rows.append(self._build_full_row(latest[self.racket_id]))
                await asyncio.sleep(dt)
            try:
                self.controller.calibrate(iter(calib_rows), frames=len(calib_rows))
                self.q.put(("calibrated", None))
            except Exception as e:
                self.q.put(("status", f"Skeleton calibration failed: {e}"))
                self.controller = None

        self.q.put(("status", "Streaming, swing away"))
        # Faster poll + dedup so each pushed row is a NEW sensor packet, not a
        # duplicate hold. Without this, the model sees step-jumps as smashes.
        dt = 1.0 / (self.rate_hz * 3)
        last_signature = None
        while not self._stop.is_set():
            latest = self._server.get_latest_data()
            if self.racket_id in latest:
                d = latest[self.racket_id]
                sig = tuple(d["accel"]) + tuple(d["gyro"])
                if sig != last_signature:
                    last_signature = sig
                    row = self._build_full_row(d)
                    if self.controller is not None:
                        try:
                            self.controller.process_row(row)     # streams bones to Unity
                        except Exception:
                            pass
                    self.q.put(("imu_row", row))
            await asyncio.sleep(dt)

        await self._server.stop()
        if self.controller is not None:
            try:
                self.controller.close()
            except Exception:
                pass

    def run(self):
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._main())
        except Exception:
            import traceback
            self.q.put(("error", traceback.format_exc()))
        finally:
            self.q.put(("stream_done", None))


# ── Online stroke counter (one count per motion-energy peak) ───────────────────

class OnlineStrokeCounter:
    """
    Streaming equivalent of summarizer.detect_stroke_events(): registers one
    stroke at each local energy maximum that exceeds min_height, subject to a
    refractory `min_distance` (windows). One-window detection latency.
    """

    def __init__(self, min_distance=PEAK_MIN_DISTANCE, min_height=LIVE_PEAK_MIN_HEIGHT,
                 rel_frac=PEAK_REL_FRAC):
        from collections import deque
        self.min_distance = min_distance
        self.min_height   = min_height
        self.rel_frac     = rel_frac
        self.e_prev2 = None
        self.e_prev1 = None
        self.lab_prev1  = 0
        self.conf_prev1 = 0.0
        self.idx = 0
        self.last_stroke_idx = -10 ** 9
        self.counts = {1: 0, 2: 0, 3: 0}
        self._recent_energies = deque(maxlen=8)   # for adaptive relative gate

    def update(self, energy: float, label_id: int, confidence: float):
        """Feed one window. Returns (label_id, confidence) if a stroke fired, else None."""
        fired = None
        is_local_max = (self.e_prev2 is not None and self.e_prev1 is not None
                        and self.e_prev2 < self.e_prev1 >= energy)

        if (is_local_max
                and self.e_prev1 >= self.min_height
                and (self.idx - 1) - self.last_stroke_idx >= self.min_distance
                and self.lab_prev1 != 0):

            # Adaptive return-swing rejection: once we've seen a few real
            # strokes, require this peak to be a meaningful fraction of the
            # recent typical stroke energy. Filters low-energy return motions.
            accept = True
            if len(self._recent_energies) >= 3:
                import numpy as _np
                rel_thr = self.rel_frac * float(_np.median(self._recent_energies))
                if self.e_prev1 < rel_thr:
                    accept = False

            if accept:
                self.last_stroke_idx = self.idx - 1
                self.counts[self.lab_prev1] = self.counts.get(self.lab_prev1, 0) + 1
                self._recent_energies.append(self.e_prev1)
                fired = (self.lab_prev1, self.conf_prev1)

        self.e_prev2, self.e_prev1 = self.e_prev1, energy
        self.lab_prev1, self.conf_prev1 = label_id, confidence
        self.idx += 1
        return fired

    def total(self):
        return sum(self.counts.values())


# ── Main GUI ───────────────────────────────────────────────────────────────────

class LiveApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Table Tennis Coaching Live")
        self.geometry("1560x880")
        self.minsize(1200, 720)
        self._unity_proc = None
        self._anim_t = 0.0       # animation clock for the colour-cycling stripe

        with open(ROOT / "config.yaml") as f:
            self.cfg = yaml.safe_load(f)

        print(f"[Model] Loading {MODEL_PATH}")
        self.predictor = StrokePredictor.from_checkpoint(
            str(MODEL_PATH), str(SCALER_PATH), cfg=self.cfg)

        self.event_queue   = queue.Queue()
        self.streamer      = None
        self.session_active = False
        self._extractor    = SlidingWindowExtractor(WINDOW_SIZE, STEP_SIZE)
        self._counter      = OnlineStrokeCounter()
        self.session_rows  = []     # FEATURE_COLS dicts for LLM report
        self.session_preds = []
        self.n_windows     = 0
        self._warmup_until = 0.0    # wall-clock end of the T-pose countdown
        self._unity_hwnd   = None   # cached HWND of the embedded Unity window

        self._build_ui()
        self.update_idletasks()
        self._launch_unity()
        self._poll_queue()
        self._animate_stripe()       # start the colour-cycling header stripe
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Background animation: colour-cycling accent stripe ────────────────
    def _animate_stripe(self):
        if not self.winfo_exists():
            return
        try:
            import math, colorsys
            # Oscillate hue between cyan (180) and magenta (300) slowly
            self._anim_t += 0.03
            h = (240 + 60 * math.sin(self._anim_t)) / 360.0   # 180..300
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
            color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
            self._stripe_canvas.configure(bg=color)
        except Exception:
            pass
        self.after(60, self._animate_stripe)

    def _on_close(self):
        try:
            if self.streamer:
                self.streamer.stop()
        except Exception:
            pass
        try:
            if self._unity_proc and self._unity_proc.poll() is None:
                self._unity_proc.terminate()
        except Exception:
            pass
        self.destroy()

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.configure(fg_color=COLORS["bg"])

        # ─── HEADER ──────────────────────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color=COLORS["surface"],
                           corner_radius=0, height=66)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        ctk.CTkLabel(hdr, text="◆  TABLE TENNIS COACHING",
                     font=ctk.CTkFont("Segoe UI Semibold", 20),
                     text_color=COLORS["text"]
                     ).pack(side=tk.LEFT, padx=24)
        ctk.CTkLabel(hdr, text="LIVE",
                     font=ctk.CTkFont("Segoe UI Semibold", 12),
                     text_color=COLORS["accent"]
                     ).pack(side=tk.LEFT, padx=(2, 0))
        ctk.CTkLabel(hdr,
                     text="Real-Time UDP  ·  Unity Skeleton  ·  LLM review",
                     font=ctk.CTkFont("Segoe UI", 11),
                     text_color=COLORS["text_dim"]
                     ).pack(side=tk.RIGHT, padx=24)
        # Animated accent stripe (colour-cycles slowly)
        self._stripe_canvas = tk.Canvas(self, height=3,
                                        bg=COLORS["accent"],
                                        highlightthickness=0, bd=0)
        self._stripe_canvas.pack(fill=tk.X)

        # ─── FOOTER ──────────────────────────────────────────────────────
        footer = ctk.CTkFrame(self, fg_color=COLORS["surface"],
                              corner_radius=0, height=72)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)
        self.status_var = tk.StringVar(value="◉ Ready, press Start Live")
        ctk.CTkLabel(footer, textvariable=self.status_var,
                     font=ctk.CTkFont("Segoe UI", 11),
                     text_color=COLORS["text_dim"]
                     ).pack(side=tk.RIGHT, padx=24)

        def _ctk_btn(text, cmd, color, hover):
            return ctk.CTkButton(
                footer, text=text, command=cmd,
                font=ctk.CTkFont("Segoe UI Semibold", 12),
                fg_color=color, hover_color=hover,
                text_color=COLORS["text_inv"],
                corner_radius=10, height=40, width=160)

        self.btn_start  = _ctk_btn("▶  START LIVE",  self._start,
                                   COLORS["success"], "#1ACA8C")
        self.btn_stop   = _ctk_btn("■  STOP",        self._stop,
                                   COLORS["danger"], "#E14A6A")
        self.btn_report = ctk.CTkButton(
            footer, text="✦  LLM REPORT", command=self._generate_report,
            font=ctk.CTkFont("Segoe UI Semibold", 12),
            fg_color=COLORS["violet"], hover_color="#7C4DEC",
            text_color=COLORS["text"], corner_radius=10, height=40, width=170)
        self.btn_stop.configure(state="disabled")
        self.btn_start.pack(side=tk.LEFT, padx=(24, 8), pady=14)
        self.btn_stop.pack(side=tk.LEFT, padx=8, pady=14)
        self.btn_report.pack(side=tk.LEFT, padx=8, pady=14)

        # ─── BODY ────────────────────────────────────────────────────────
        body = ctk.CTkFrame(self, fg_color=COLORS["bg"], corner_radius=0)
        body.pack(fill=tk.BOTH, expand=True, padx=14, pady=12)

        # LEFT column: stroke counts only
        left = ctk.CTkFrame(body, fg_color=COLORS["card"],
                            corner_radius=14, border_width=1,
                            border_color=COLORS["border"],
                            width=270)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        left.pack_propagate(False)
        lin = ctk.CTkFrame(left, fg_color="transparent")
        lin.pack(fill=tk.BOTH, expand=True, padx=16, pady=14)

        ctk.CTkLabel(lin, text="STROKE COUNTS",
                     font=ctk.CTkFont("Segoe UI Semibold", 12),
                     text_color=COLORS["accent"]
                     ).pack(anchor="w")

        self.count_labels = {}
        for cid in (1, 2, 3):
            row = ctk.CTkFrame(lin, fg_color="transparent")
            row.pack(fill=tk.X, pady=8)
            ctk.CTkLabel(row, text="●",
                         font=ctk.CTkFont("Segoe UI", 18),
                         text_color=CLASS_COLORS[CLASS_NAMES[cid]]
                         ).pack(side=tk.LEFT)
            ctk.CTkLabel(row, text=SHORT_NAMES[cid],
                         font=ctk.CTkFont("Segoe UI", 13),
                         text_color=COLORS["text"]
                         ).pack(side=tk.LEFT, padx=(10, 0))
            val = ctk.CTkLabel(row, text="0",
                               font=ctk.CTkFont("Segoe UI Semibold", 22),
                               text_color=CLASS_COLORS[CLASS_NAMES[cid]])
            val.pack(side=tk.RIGHT)
            self.count_labels[cid] = val

        # Totals
        totals = ctk.CTkFrame(lin, fg_color=COLORS["card_alt"], corner_radius=10)
        totals.pack(fill=tk.X, pady=(14, 0))
        c1 = ctk.CTkFrame(totals, fg_color="transparent"); c1.pack(side=tk.LEFT, padx=14, pady=12)
        ctk.CTkLabel(c1, text="TOTAL",
                     font=ctk.CTkFont("Segoe UI", 10),
                     text_color=COLORS["text_muted"]).pack(anchor="w")
        self.lbl_total = ctk.CTkLabel(c1, text="0",
                                      font=ctk.CTkFont("Segoe UI Semibold", 24),
                                      text_color=COLORS["accent"])
        self.lbl_total.pack(anchor="w")
        c2 = ctk.CTkFrame(totals, fg_color="transparent"); c2.pack(side=tk.RIGHT, padx=14, pady=12)
        ctk.CTkLabel(c2, text="WINDOWS",
                     font=ctk.CTkFont("Segoe UI", 10),
                     text_color=COLORS["text_muted"]).pack(anchor="e")
        self.lbl_windows = ctk.CTkLabel(c2, text="0",
                                        font=ctk.CTkFont("Segoe UI Semibold", 24),
                                        text_color=COLORS["text"])
        self.lbl_windows.pack(anchor="e")

        # T-pose countdown banner (replaces the old prediction box)
        self.lbl_countdown = ctk.CTkLabel(
            lin, text="",
            font=ctk.CTkFont("Segoe UI Semibold", 22),
            text_color=COLORS["warning"])
        self.lbl_countdown.pack(anchor="w", pady=(18, 0))

        # ─── RESIZABLE SPLITTER: Unity ↔ Event Log ───────────────────────
        # tk.PanedWindow gives a draggable sash (the bar between panes)
        paned = tk.PanedWindow(
            body, orient=tk.HORIZONTAL,
            bg=COLORS["bg"], bd=0,
            sashwidth=8, sashrelief=tk.FLAT,
            sashpad=0,
            handlepad=8, handlesize=20, showhandle=False,
        )
        paned.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Style the sash so it's visible as an accent line
        # (sash is drawn with bg colour; we use a slightly lighter shade)
        paned.config(bg=COLORS["border_glow"])

        # CENTRE PANE: Unity skeleton
        mid_card = ctk.CTkFrame(
            paned, fg_color=COLORS["card"], corner_radius=14,
            border_width=1, border_color=COLORS["border"])
        ctk.CTkLabel(mid_card, text="SKELETON  •  UNITY",
                     font=ctk.CTkFont("Segoe UI Semibold", 12),
                     text_color=COLORS["accent"]
                     ).pack(anchor="w", padx=18, pady=(14, 8))
        wrap = ctk.CTkFrame(mid_card, fg_color=COLORS["card_dark"],
                            corner_radius=10)
        wrap.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))
        # The actual embed target must remain a tk.Frame so we have an HWND.
        self.unity_panel = tk.Frame(wrap, bg="#000000",
                                    width=720, height=560,
                                    highlightthickness=0, bd=0)
        self.unity_panel.place(relx=0.5, rely=0.5, anchor="center",
                               relwidth=1.0, relheight=1.0)
        self.unity_panel.bind("<Configure>", self._on_unity_resize)
        paned.add(mid_card, minsize=320, stretch="always")

        # RIGHT PANE: Event log (bigger fonts, narrower default width)
        right_card = ctk.CTkFrame(
            paned, fg_color=COLORS["card"], corner_radius=14,
            border_width=1, border_color=COLORS["border"])
        ctk.CTkLabel(right_card, text="SHOT PREDICTIONS",
                     font=ctk.CTkFont("Segoe UI Semibold", 12),
                     text_color=COLORS["accent"]
                     ).pack(anchor="w", padx=18, pady=(14, 8))
        self.log = scrolledtext.ScrolledText(
            right_card,
            font=("Cascadia Mono", 12),     # ⬆ noticeably larger
            bg=COLORS["card_dark"], fg=COLORS["text"],
            insertbackground=COLORS["accent"],
            selectbackground=COLORS["accent_dim"],
            relief="flat", bd=0, padx=12, pady=12)
        self.log.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))
        self.log.tag_config("stroke", foreground=COLORS["accent"])
        self.log.tag_config("muted",  foreground=COLORS["text_muted"])
        self.log.tag_config("warn",   foreground=COLORS["warning"])
        self.log.tag_config("danger", foreground=COLORS["danger"])
        paned.add(right_card, minsize=260, stretch="never")

        # Set initial sash position (Unity ≈ 2× event log)
        self.after(80, lambda: paned.sash_place(0,
                                                int(self.winfo_width() * 0.62),
                                                0))
        self._paned = paned

    # ── Unity embedding helpers ─────────────────────────────────────────────
    def _find_unity_child_hwnd(self):
        """Enumerate child windows of the embed panel; Unity is the first one."""
        if not WIN32_OK:
            return None
        try:
            parent = int(self.unity_panel.winfo_id())
            found = []
            def cb(hwnd, _):
                found.append(hwnd); return True
            win32gui.EnumChildWindows(parent, cb, None)
            return found[0] if found else None
        except Exception:
            return None

    def _fit_unity(self):
        """Resize the embedded Unity window to fill the panel, centred."""
        if not WIN32_OK or self._unity_proc is None:
            return
        if self._unity_hwnd is None:
            self._unity_hwnd = self._find_unity_child_hwnd()
        if self._unity_hwnd:
            w = max(self.unity_panel.winfo_width(), 1)
            h = max(self.unity_panel.winfo_height(), 1)
            try:
                win32gui.MoveWindow(self._unity_hwnd, 0, 0, w, h, True)
            except Exception:
                pass
        else:
            # Unity not visible yet — retry shortly
            self.after(400, self._fit_unity)

    def _on_unity_resize(self, _evt):
        # Debounce: schedule a fit pass after the resize settles
        self.after(60, self._fit_unity)

    # ── Unity embedding ───────────────────────────────────────────────────────
    def _launch_unity(self):
        """Launch the Unity exe embedded into the unity_panel via -parentHWND."""
        if not SHOW_UNITY or not UNITY_EXE.exists():
            self.status_var.set("◉ Unity exe not found, skeleton panel disabled")
            return
        try:
            hwnd = int(self.unity_panel.winfo_id())
            w = max(self.unity_panel.winfo_width(), 320)
            h = max(self.unity_panel.winfo_height(), 320)
            self._unity_proc = subprocess.Popen([
                str(UNITY_EXE),
                "-parentHWND", str(hwnd), "delayed",
                "-screen-width",  str(w),
                "-screen-height", str(h),
                "-screen-fullscreen", "0",
            ])
            self.status_var.set("◉ Unity launched, press Start Live")
            # Try to find + fit the Unity window once it appears
            self.after(800, self._fit_unity)
        except Exception as e:
            self.status_var.set(f"⚠ Unity launch failed: {e}")

    # ── Session control ───────────────────────────────────────────────────────
    def _start(self):
        if self.session_active:
            return
        self._extractor.reset()
        self._counter = OnlineStrokeCounter()
        self.session_rows.clear(); self.session_preds.clear(); self.n_windows = 0
        for cid in (1, 2, 3):
            self.count_labels[cid].configure(text="0")
        self.lbl_total.configure(text="0")
        self.lbl_windows.configure(text="0")
        self.log.delete("1.0", tk.END)
        self._log("··  Live session started  ··\n", tag="muted")
        self.session_active = True
        self.btn_start.configure(state="disabled"); self.btn_stop.configure(state="normal")

        # Build the Unity bone-stream controller (optional, exception-safe)
        controller = None
        if SHOW_UNITY and UNITY_CONTROLLER_OK:
            try:
                controller = IMUUnityController(
                    bone_hierarchy_path=str(UNITY_DIR / "BoneHierarchy.txt"),
                    bone_offsets_path  =str(UNITY_DIR / "BoneOffsets.json"),
                    tpose_quats_path   =str(UNITY_DIR / "InitialPoseExport.txt"),
                    udp_ip             ="127.0.0.1",
                    udp_port           =UNITY_BONE_PORT,
                    position_scale     =1.0,
                    hips_y_scale       =1.0,
                )
            except Exception as e:
                self._log(f"[WARN] Unity controller init failed, skeleton off ({e})\n",
                          tag="warn")
                controller = None

        self.streamer = LiveIMUStreamer(self.event_queue, controller=controller)
        self.streamer.start()

        # Start the T-pose countdown — classification is gated until it ends.
        self._warmup_until = time.time() + TPOSE_SECONDS
        self._update_countdown()

    def _update_countdown(self):
        if not self.session_active:
            return
        remaining = self._warmup_until - time.time()
        if remaining > 0:
            self.lbl_countdown.configure(
                text=f"T-POSE  {int(remaining)+1}s",
                text_color=COLORS["warning"])
            self.status_var.set("◉ Hold T-pose, arms straight out to the sides")
            self.after(200, self._update_countdown)
        else:
            self.lbl_countdown.configure(text="GO  ·  swing away",
                                         text_color=COLORS["success"])
            self.status_var.set("◉ Streaming")
            # Clear the banner after a few seconds
            self.after(2500, lambda: self.lbl_countdown.configure(text=""))

    def _stop(self):
        if self.streamer:
            self.streamer.stop()
        self.session_active = False
        self.btn_start.configure(state="normal"); self.btn_stop.configure(state="disabled")
        self._log(f"··  Session ended, {self._counter.total()} strokes  ··\n",
                  tag="muted")
        self.status_var.set("◉ Session ended")

    # ── Queue polling ───────────────────────────────────────────────────────
    def _poll_queue(self):
        try:
            for _ in range(500):                  # drain available rows
                msg_type, payload = self.event_queue.get_nowait()
                if msg_type == "imu_row":
                    self._handle_row(payload)
                elif msg_type == "status":
                    self.status_var.set(payload)
                elif msg_type == "calibrated":
                    self._log("◇  Skeleton calibrated\n", tag="stroke")
                elif msg_type == "error":
                    self._log(f"[ERROR]\n{payload}\n", tag="danger")
                    self.status_var.set("⚠ Streamer error, see log")
                elif msg_type == "stream_done":
                    if self.session_active:
                        self._stop()
        except queue.Empty:
            pass
        self.after(30, self._poll_queue)

    # ── Per-window inference ────────────────────────────────────────────────
    def _handle_row(self, row: dict):
        features = self._extractor.add_frame(row)
        if features is None:
            return
        # T-pose warm-up: keep filling the window buffer but don't classify/count
        if time.time() < self._warmup_until:
            return
        self.n_windows += 1
        result = self.predictor.predict(features)

        energy = window_energy(features)
        if is_idle_window(features):
            result = {"label_id": 0, "label_name": CLASS_NAMES[0],
                      "confidence": result["confidence"],
                      "probabilities": result["probabilities"]}

        # store for LLM report
        self.session_rows.append(dict(zip(FEATURE_COLS, features.tolist())))
        self.session_preds.append(result)

        # online stroke counting (peak based)
        fired = self._counter.update(energy, result["label_id"], result["confidence"])
        if fired is not None:
            lab_id, conf = fired
            self.count_labels[lab_id].configure(text=str(self._counter.counts[lab_id]))
            self.lbl_total.configure(text=str(self._counter.total()))
            self._log(f"  ✦  #{self._counter.total():<3} "
                      f"{SHORT_NAMES[lab_id]:<11}  conf {conf:.0%}\n",
                      tag="stroke")

        self._update_live(result)
        self.lbl_windows.configure(text=str(self.n_windows))

    def _update_live(self, result):
        # Live prediction panel removed; per-window predictions are logged in
        # the Shot Predictions list. This method is kept as a no-op so the
        # _handle_row call site does not need to change.
        return

    def _log(self, text, tag=None):
        if tag:
            self.log.insert(tk.END, text, tag)
        else:
            self.log.insert(tk.END, text)
        self.log.see(tk.END)

    # ── LLM report (reuses summarizer + coaching, unchanged) ──────────────────
    def _generate_report(self):
        if not self.session_rows:
            messagebox.showwarning("No Data", "No session data to report on.")
            return
        self.btn_report.configure(state="disabled", text="⌛  GENERATING")
        self.status_var.set("◉ Calling LLM, please wait")

        def _run():
            try:
                session_df = pd.DataFrame(self.session_rows)
                summary    = run_session(self.predictor, session_df)
                feedback   = get_coaching_feedback(summary, subject_id=SUBJECT_ID, cfg=self.cfg)
                self.after(0, lambda: self._show_report(feedback))
            except Exception as e:
                self.after(0, lambda: self._report_error(str(e)))

        threading.Thread(target=_run, daemon=True).start()

    def _show_report(self, feedback):
        self.btn_report.configure(state="normal", text="✦  LLM REPORT")
        self.status_var.set("◉ Report generated")
        self._log("\n◆◆  LLM COACHING REPORT  ◆◆\n", tag="stroke")
        self._log(f"ASSESSMENT  →  {feedback['assessment']}\n\n")
        for rec in feedback["recommendations"]:
            self._log(f"  [{rec['priority']}]  #{rec['rank']}  {rec['text']}\n")
        self._log(f"\nNEXT FOCUS  →  {feedback['next_focus']}\n")
        self._log("─" * 50 + "\n", tag="muted")

    def _report_error(self, msg):
        self.btn_report.configure(state="normal", text="✦  LLM REPORT")
        self.status_var.set("⚠ LLM error, is LM Studio running?")
        self._log(f"[LLM ERROR] {msg}\n", tag="danger")


if __name__ == "__main__":
    app = LiveApp()
    app.mainloop()
