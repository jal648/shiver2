"""
Debug overlay drawing and info panel for the Shiver 2 vision debug window.
"""

import cv2
import numpy as np

from detector import BlockDetector


DBG_WIN = "Shiver2 — vision debug"

COLOR_NAMES = ["red", "green", "blue"]

BLOCK_COLOR_BGR = {
    "red":     (60,  60,  255),
    "green":   (50,  210, 80),
    "blue":    (255, 100, 40),
    "unknown": (160, 160, 160),
}


def draw_debug_overlay(display: np.ndarray, corners: list, blocks) -> np.ndarray:
    """Draw zone polygon and block contour outlines onto a copy of the frame."""
    out = display.copy()
    fh, fw = out.shape[:2]

    pts = np.int32([[int(c[0] * fw), int(c[1] * fh)] for c in corners])
    cv2.polylines(out, [pts], isClosed=True, color=(200, 220, 0), thickness=2)
    for pt in pts:
        cv2.circle(out, tuple(pt), 5, (200, 220, 0), -1)

    for b in blocks:
        x1 = int((b.x - b.w / 2) * fw)
        y1 = int((b.y - b.h / 2) * fh)
        x2 = int((b.x + b.w / 2) * fw)
        y2 = int((b.y + b.h / 2) * fh)
        bgr = BLOCK_COLOR_BGR.get(b.color, (160, 160, 160))
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(out, f"#{b.id} {b.color}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)
    return out


def switch_color(idx: int, detector: BlockDetector, sel_color_idx: list) -> None:
    """Switch selected color and snap trackbars to that color's current values."""
    sel_color_idx[0] = idx
    name = COLOR_NAMES[idx]
    for entry in detector.color_ranges:
        if entry[0] == name:
            cv2.setTrackbarPos("brightness", DBG_WIN, entry[1])
            cv2.setTrackbarPos("ratio",      DBG_WIN, entry[2])
            break


def build_info_panel(
    height: int,
    detector: BlockDetector,
    blocks,
    active_count: int,
    sel_color: str,
    view_mode: str,
) -> np.ndarray:
    W = 250
    panel = np.zeros((height, W, 3), dtype=np.uint8)

    C_TITLE = (200, 220, 0)
    C_WHITE = (220, 220, 220)
    C_DIM   = (100, 100, 100)
    C_GOOD  = (80,  200, 80)
    C_WARN  = (0,   180, 255)
    C_SEP   = (55,  55,  55)

    FONT  = cv2.FONT_HERSHEY_SIMPLEX
    SCALE = 0.80
    BOLD  = cv2.LINE_AA
    PAD_X = 8
    LINE  = 16

    y = 0

    def sep():
        nonlocal y
        y += 4
        cv2.line(panel, (0, y), (W, y), C_SEP, 1)
        y += 6

    def title(text):
        nonlocal y
        y += 12
        cv2.putText(panel, text, (PAD_X, y), FONT, 0.42, C_TITLE, 1, BOLD)
        y += 3
        cv2.line(panel, (PAD_X, y), (W - PAD_X, y), C_TITLE, 1)
        y += LINE - 2

    def line(text, color=C_WHITE, indent=0):
        nonlocal y
        if y + LINE > height:
            return
        cv2.putText(panel, text, (PAD_X + indent, y), FONT, SCALE, color, 1, BOLD)
        y += LINE

    def gap(px=6):
        nonlocal y
        y += px

    # ── KEYBOARD ──────────────────────────────────────────────────────
    title("KEYBOARD")
    line("1 / 2 / 3   red / green / blue", C_DIM)
    line("Tab         cycle color ->", C_DIM)
    line("v           cycle view", C_DIM)
    line("b / e       binary / edge view", C_DIM)
    line("s  save    r  reset zone", C_DIM)
    line("q / w       perspective taper", C_DIM)
    line("ESC  quit", C_DIM)
    gap()

    # ── VIEW ──────────────────────────────────────────────────────────
    sep()
    title("VIEW  [v to cycle]")
    views = [("camera", "normal"), ("binary mask", "binary"), ("edges", "edges")]
    for label, key in views:
        active = (view_mode == key)
        c = C_GOOD if active else C_DIM
        marker = ">" if active else " "
        line(f"  {marker} {label}", c)
    gap()

    # ── COLOR CHANNELS ────────────────────────────────────────────────
    sep()
    title("COLOR CHANNELS  [1/2/3 Tab]")

    key_num = {"red": "1", "green": "2", "blue": "3"}
    for name in COLOR_NAMES:
        is_sel = (name == sel_color)
        brightness_min, ratio = 0, 0
        for entry in detector.color_ranges:
            if entry[0] == name:
                brightness_min, ratio = entry[1], entry[2]
                break
        bgr    = BLOCK_COLOR_BGR.get(name, C_WHITE)
        c      = bgr if is_sel else C_DIM
        num    = key_num.get(name, "?")
        marker = ">" if is_sel else " "
        line(f"  {marker} [{num}] {name.upper()}", c)
        line(f"       bright>={brightness_min}  ratio>={ratio}%", c)
        if is_sel:
            gap(2)
    gap(3)

    # ── TUNING SLIDERS ────────────────────────────────────────────────
    sep()
    title(f"TUNING: {sel_color.upper()}")
    line("Sliders above adjust selected color.", C_DIM)
    line("brightness  dominant channel floor (0-255)", C_DIM)
    line("ratio       % of total R+G+B light (0-100)", C_DIM)
    gap(3)

    # ── BLOCKS ────────────────────────────────────────────────────────
    sep()
    title("BLOCKS")
    line(f"contours : {detector.last_contour_count}", C_DIM)
    line(f"detected : {detector.last_detection_count}",
         C_GOOD if detector.last_detection_count > 0 else C_WARN)
    line(f"in zone  : {active_count}",
         C_GOOD if active_count > 0 else C_WARN)
    gap(3)
    for b in blocks[:6]:
        bgr = BLOCK_COLOR_BGR.get(b.color, (160, 160, 160))
        line(f"  #{b.id:<2} {b.color:<6} ({b.x:.2f}, {b.y:.2f})", bgr)

    return panel
