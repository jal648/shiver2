"""
calibrate.py — Interactive zone calibration with perspective correction.

Click and drag to define the base rectangle, then use Q/W to taper the top
edge inward/outward to match the camera's angled view of the table.
The zone is saved to zone.json as 4 corner points (trapezoid).

Controls:
  Click + drag  — draw a new base rectangle
  Q             — taper top edge inward  (more perspective / camera tilted down)
  W             — taper top edge outward (less perspective / flatten back)
  S             — save zone to zone.json and quit
  R             — reset to full frame, perspective = 0
  ESC           — quit without saving

Usage:
    python calibrate.py [--camera 0]
"""

import argparse
import json
import os

import cv2
import numpy as np

from main import TABLE_ROI

ZONE_FILE = os.path.join(os.path.dirname(__file__), "zone.json")
PERSP_STEP  = 0.02   # how much each Q/W press shifts the perspective factor
PERSP_CLAMP = 0.50   # max |perspective factor|

# ── State ─────────────────────────────────────────────────────────────────────

state = {
    "drawing":    False,
    "drag_start": None,
    "drag_end":   None,
    "base":  {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},  # normalised rectangle
    "persp": 0.0,   # perspective factor: 0 = rectangle, +0.3 = top 40% narrower
}


def compute_corners(base, persp):
    """Return [[TL, TR, BR, BL]] in normalised image coords.

    persp > 0 → top edge is narrower (top of table appears smaller in frame).
    persp < 0 → top edge is wider.
    """
    x, y, w, h = base["x"], base["y"], base["w"], base["h"]
    p = persp * w / 2          # inset amount on each side of the top edge
    return [
        [round(x + p,     4), round(y,     4)],  # TL
        [round(x + w - p, 4), round(y,     4)],  # TR
        [round(x + w,     4), round(y + h, 4)],  # BR
        [round(x,         4), round(y + h, 4)],  # BL
    ]


def load_zone_state():
    """Load zone.json and return (base_rect, persp_factor), or None if missing."""
    try:
        with open(ZONE_FILE) as f:
            d = json.load(f)
    except FileNotFoundError:
        return None

    if "corners" in d:
        tl, tr, br, bl = d["corners"]
        x_left  = bl[0]
        x_right = br[0]
        y_top   = tl[1]
        y_bot   = bl[1]
        total_w = x_right - x_left
        total_h = y_bot - y_top
        top_w   = tr[0] - tl[0]
        base  = {"x": x_left, "y": y_top, "w": max(total_w, 0.01), "h": max(total_h, 0.01)}
        persp = round(1.0 - top_w / base["w"], 4) if base["w"] > 0 else 0.0
        return base, persp

    # Old {x,y,w,h} rectangle format — no perspective
    base = {k: d.get(k, v) for k, v in [("x",0.0),("y",0.0),("w",1.0),("h",1.0)]}
    return base, 0.0


def save_zone(corners):
    data = {"corners": corners}
    with open(ZONE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[calibrate] Saved zone: {data}")


def norm_rect_from_drag(p1, p2, fw, fh):
    x0 = max(0, min(p1[0], p2[0]))
    y0 = max(0, min(p1[1], p2[1]))
    x1 = min(fw, max(p1[0], p2[0]))
    y1 = min(fh, max(p1[1], p2[1]))
    return {
        "x": round(x0 / fw, 4),
        "y": round(y0 / fh, 4),
        "w": round((x1 - x0) / fw, 4),
        "h": round((y1 - y0) / fh, 4),
    }


def draw_overlay(frame, persp, corners, drag_start=None, drag_end=None):
    out = frame.copy()
    fh, fw = out.shape[:2]

    # Trapezoid fill (subtle tint) + solid outline
    if corners:
        pts = np.array(
            [[int(c[0] * fw), int(c[1] * fh)] for c in corners],
            dtype=np.int32,
        )
        tinted = out.copy()
        cv2.fillPoly(tinted, [pts], (200, 220, 0))
        out = cv2.addWeighted(tinted, 0.08, out, 0.92, 0)
        cv2.polylines(out, [pts], isClosed=True, color=(200, 220, 0), thickness=2)
        for pt in pts:
            cv2.circle(out, tuple(pt), 5, (200, 220, 0), -1)

    # In-progress drag (thin cyan rectangle)
    if drag_start and drag_end:
        cv2.rectangle(out, drag_start, drag_end, (0, 255, 255), 1)

    # HUD
    lines = [
        "Drag: set base rectangle",
        f"Q / W: perspective  (P = {persp:+.2f})",
        "S: save & quit  |  R: reset  |  ESC: quit",
    ]
    for i, line in enumerate(lines):
        cv2.putText(out, line, (8, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)
    return out


# ── Mouse callback ─────────────────────────────────────────────────────────────

_frame_size = [1, 1]


def on_mouse(event, x, y, flags, _param):
    fw, fh = _frame_size
    if event == cv2.EVENT_LBUTTONDOWN:
        state["drawing"] = True
        state["drag_start"] = (x, y)
        state["drag_end"]   = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and state["drawing"]:
        state["drag_end"] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and state["drawing"]:
        state["drawing"] = False
        state["drag_end"] = (x, y)
        rect = norm_rect_from_drag(state["drag_start"], state["drag_end"], fw, fh)
        if rect["w"] > 0.01 and rect["h"] > 0.01:
            state["base"] = rect
            print(f"[calibrate] Base rect updated: {rect}  (P={state['persp']:+.2f})")


# ── Helpers ───────────────────────────────────────────────────────────────────

def crop_roi(frame, roi):
    if roi is None:
        return frame
    x, y, w, h = roi
    return frame[y:y + h, x:x + w]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(camera_index: int):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    loaded = load_zone_state()
    if loaded:
        state["base"], state["persp"] = loaded
        print(f"[calibrate] Loaded zone — base={state['base']}  P={state['persp']:+.2f}")
    else:
        print("[calibrate] No zone.json found — starting with full frame")

    print("[calibrate] Drag=zone  Q/W=perspective  S=save  R=reset  ESC=quit")

    cv2.namedWindow("Shiver2 — calibrate")
    cv2.setMouseCallback("Shiver2 — calibrate", on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        roi_frame = crop_roi(frame, TABLE_ROI)
        fh, fw = roi_frame.shape[:2]
        _frame_size[0] = fw
        _frame_size[1] = fh

        corners = compute_corners(state["base"], state["persp"])
        ds = state["drag_start"] if state["drawing"] else None
        de = state["drag_end"]   if state["drawing"] else None
        display = draw_overlay(roi_frame, state["persp"], corners, ds, de)
        cv2.imshow("Shiver2 — calibrate", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("[calibrate] Quit without saving.")
            break
        elif key in (ord("s"), ord("S")):
            save_zone(compute_corners(state["base"], state["persp"]))
            break
        elif key in (ord("r"), ord("R")):
            state["base"]  = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
            state["persp"] = 0.0
            print("[calibrate] Reset — full frame, no perspective.")
        elif key in (ord("q"), ord("Q")):
            state["persp"] = min(PERSP_CLAMP, round(state["persp"] + PERSP_STEP, 4))
            print(f"[calibrate] P = {state['persp']:+.2f}")
        elif key in (ord("w"), ord("W")):
            state["persp"] = max(-PERSP_CLAMP, round(state["persp"] - PERSP_STEP, 4))
            print(f"[calibrate] P = {state['persp']:+.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shiver 2 — zone calibration")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    main(args.camera)
