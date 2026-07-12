"""
Run this FIRST to check if IMU_Avatar.exe launches and can be found.
No Tkinter, no embedding — just raw diagnostics.

python unity_diagnostics.py
"""

import subprocess
import time
import win32gui
import win32process
import win32con
import pathlib

UNITY_EXE = pathlib.Path(r"E:\thesis_work\1_new_test\IMU_Avatar\game\IMU_Avatar.exe")

def all_visible_windows():
    """Return list of (hwnd, title, pid) for every visible window."""
    results = []
    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
            except Exception:
                pid = -1
            if title:
                results.append((hwnd, title, pid))
    win32gui.EnumWindows(cb, None)
    return results

# ── 1. Check exe exists ───────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Check exe path")
print(f"  Path : {UNITY_EXE}")
print(f"  Exists: {UNITY_EXE.exists()}")
if not UNITY_EXE.exists():
    print("  ❌ File not found. Fix the path and re-run.")
    exit(1)
print("  ✅ File exists.")

# ── 2. Snapshot windows BEFORE launch ─────────────────────────────────────────
print("\nSTEP 2: Windows open BEFORE launching Unity:")
before = {hwnd for hwnd, _, _ in all_visible_windows()}
for hwnd, title, pid in all_visible_windows():
    print(f"  [{hwnd:>8}]  {title}")

# ── 3. Launch WITHOUT -popupwindow first ─────────────────────────────────────
print("\nSTEP 3: Launching IMU_Avatar.exe (windowed, no -popupwindow)…")
proc = subprocess.Popen([
    str(UNITY_EXE),
    "-screen-fullscreen", "0",
    "-screen-width",  "800",
    "-screen-height", "600",
])
print(f"  PID: {proc.pid}")

# ── 4. Poll for new window ────────────────────────────────────────────────────
print("\nSTEP 4: Polling for Unity window (up to 20 seconds)…")
found_hwnd  = None
found_title = None
timeout = 20
start   = time.time()

while time.time() - start < timeout:
    elapsed = time.time() - start
    current = all_visible_windows()
    new_wins = [(h, t, p) for h, t, p in current if h not in before]

    # Also look by PID regardless of 'before' snapshot
    pid_wins = []
    def cb2(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            try:
                _, wpid = win32process.GetWindowThreadProcessId(hwnd)
                if wpid == proc.pid:
                    pid_wins.append((hwnd, win32gui.GetWindowText(hwnd)))
            except Exception:
                pass
    win32gui.EnumWindows(cb2, None)

    if pid_wins:
        for h, t in pid_wins:
            if t:  # has a title
                found_hwnd  = h
                found_title = t
                break
        if found_hwnd:
            print(f"  ✅ Found by PID after {elapsed:.1f}s")
            print(f"     HWND : {found_hwnd}")
            print(f"     Title: '{found_title}'")
            break

    if new_wins:
        print(f"  [{elapsed:.1f}s] New windows appeared:")
        for h, t, p in new_wins:
            print(f"    [{h:>8}] '{t}'  pid={p}")

    print(f"  [{elapsed:.1f}s] Waiting… (process alive={proc.poll() is None})")
    time.sleep(1)

if not found_hwnd:
    print("\n  ❌ No Unity window found after 20 seconds.")
    print("  All current visible windows:")
    for hwnd, title, pid in all_visible_windows():
        print(f"    [{hwnd:>8}]  pid={pid}  '{title}'")
    print("\n  Possible causes:")
    print("  1. Unity crashed on launch — check for error dialogs")
    print("  2. Build is fullscreen — check Player Settings")
    print("  3. Unity window has no title (rare) — check build name")
    proc.terminate()
    exit(1)

# ── 5. Check window rect ───────────────────────────────────────────────────────
print("\nSTEP 5: Window rect (position and size):")
try:
    rect = win32gui.GetWindowRect(found_hwnd)
    x, y, x2, y2 = rect
    w, h = x2 - x, y2 - y
    print(f"  Position : ({x}, {y})")
    print(f"  Size     : {w} × {h}")
    if w < 10 or h < 10:
        print("  ⚠️  Window is tiny — may be minimised or headless")
    else:
        print("  ✅ Window has valid size")
except Exception as e:
    print(f"  Error reading rect: {e}")

# ── 6. Done ────────────────────────────────────────────────────────────────────
print("\nSTEP 6: Result summary")
print(f"  EXE        : {UNITY_EXE.name}")
print(f"  PID        : {proc.pid}")
print(f"  HWND       : {found_hwnd}")
print(f"  Title      : '{found_title}'")
print("\nUnity should be visible on screen now.")
print("Press Enter to close it…")
input()
proc.terminate()
print("Done.")
