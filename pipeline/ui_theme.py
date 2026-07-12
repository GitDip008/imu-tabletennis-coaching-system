"""
Shared futuristic UI theme for the TT-Coaching GUIs.

Used by:
    realtimeapp_new.py
    live_app.py
    live_app_nounity.py

The theme provides:
    - A cohesive dark / cyber palette
    - A common font stack
    - Class-colour mapping
    - Small helpers for cards, section titles, buttons, and gradient bars
"""
import tkinter as tk

# ── Palette ──────────────────────────────────────────────────────────────────
COLORS = {
    "bg":          "#080B17",   # deep space background
    "surface":     "#0F1424",   # slight lift over bg
    "card":        "#141A2E",   # main card
    "card_alt":    "#1A2138",   # alternate card / sub-section
    "card_dark":   "#0C1020",   # darkest container (event log)
    "border":      "#2A3358",   # subtle card outline
    "border_glow": "#3D4D8A",   # subtle highlight outline

    "accent":      "#00E5FF",   # cyan accent (primary)
    "accent_dim":  "#0096B3",   # darker cyan (for pressed)
    "violet":      "#8B5CF6",   # secondary violet
    "magenta":     "#FF4DD2",   # pop magenta
    "success":     "#22F0A6",   # neon green
    "warning":     "#FBBF24",   # amber
    "danger":      "#FF5C7A",   # alert pink-red

    "text":        "#E8ECF7",   # primary text
    "text_dim":    "#9CA4BD",   # secondary text
    "text_muted":  "#5E6786",   # muted hints
    "text_inv":    "#080B17",   # text on bright buttons
}

# Class → colour mapping used everywhere
CLASS_COLORS = {
    "No Stroke":     "#5E6786",
    "Stroke Type 1": "#00E5FF",   # FH topspin → cyan
    "Stroke Type 2": "#22F0A6",   # BH drive   → neon green
    "Stroke Type 3": "#FF4DD2",   # FH smash   → magenta
}

# Short names (used by live_app variants)
SHORT_NAMES = {0: "no_stroke", 1: "fh_topspin", 2: "bh_drive", 3: "fh_smash"}

# Font stack — uses Segoe UI on Windows, gracefully falls back elsewhere
FONTS = {
    "h1":        ("Segoe UI Semibold", 18),
    "h2":        ("Segoe UI Semibold", 13),
    "h3":        ("Segoe UI Semibold", 11),
    "body":      ("Segoe UI", 10),
    "body_sm":   ("Segoe UI", 9),
    "value":     ("Segoe UI Semibold", 13),
    "value_big": ("Segoe UI Semibold", 30),
    "value_md":  ("Segoe UI Semibold", 18),
    "caption":   ("Segoe UI", 9),
    "mono":      ("Cascadia Mono", 9),
    "button":    ("Segoe UI Semibold", 10),
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_card(parent, **kwargs):
    """A card frame with a subtle border and dark surface."""
    return tk.Frame(
        parent,
        bg=COLORS["card"],
        highlightbackground=COLORS["border"],
        highlightthickness=1,
        bd=0,
        **kwargs,
    )


def section_title(parent, text, color=None, anchor="w"):
    """Small uppercase-ish accent label used as a card section header."""
    return tk.Label(
        parent,
        text=text,
        font=FONTS["h3"],
        fg=color or COLORS["accent"],
        bg=COLORS["card"],
        anchor=anchor,
    )


def make_button(parent, text, command, kind="accent", width=None):
    """Flat futuristic button. kind ∈ {accent, success, danger, violet, ghost}."""
    palette = {
        "accent":  (COLORS["accent"],  COLORS["text_inv"]),
        "success": (COLORS["success"], COLORS["text_inv"]),
        "danger":  (COLORS["danger"],  COLORS["text_inv"]),
        "violet":  (COLORS["violet"],  COLORS["text"]),
        "ghost":   (COLORS["card_alt"], COLORS["text"]),
    }[kind]
    bg, fg = palette
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        font=FONTS["button"],
        bg=bg,
        fg=fg,
        activebackground=COLORS["accent_dim"] if kind == "accent" else bg,
        activeforeground=fg,
        relief="flat",
        bd=0,
        padx=20,
        pady=9,
        cursor="hand2",
    )
    if width is not None:
        btn.config(width=width)
    return btn


def draw_progress_bar(canvas, fraction, color=None, height=None, glow=True):
    """Render a thin futuristic progress bar onto a Canvas.

    Call with canvas already created. Re-call to redraw after value change.
    """
    canvas.delete("all")
    w = max(int(canvas.winfo_width()), 1)
    h = max(int(canvas.winfo_height()), 1)
    if height is None:
        height = h
    fraction = max(0.0, min(1.0, float(fraction)))
    fg = color or COLORS["accent"]
    # track
    canvas.create_rectangle(0, 0, w, height, fill=COLORS["card_dark"],
                            outline=COLORS["border"], width=0)
    fill_w = int(w * fraction)
    if fill_w > 0:
        canvas.create_rectangle(0, 0, fill_w, height, fill=fg, outline="")
        # subtle highlight at top
        canvas.create_line(0, 0, fill_w, 0, fill=COLORS["text"], width=1)


def style_header(parent):
    """Top banner — gradient-feel via solid accent + subtle inner border."""
    hdr = tk.Frame(parent, bg=COLORS["surface"], height=64)
    hdr.pack(fill=tk.X)
    hdr.pack_propagate(False)
    # accent stripe at the bottom edge
    stripe = tk.Frame(parent, bg=COLORS["accent"], height=2)
    stripe.pack(fill=tk.X)
    return hdr


def style_footer(parent):
    """Bottom action strip."""
    f = tk.Frame(parent, bg=COLORS["surface"], height=64)
    f.pack(fill=tk.X, side=tk.BOTTOM)
    f.pack_propagate(False)
    return f
