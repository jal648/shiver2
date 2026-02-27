"""
Shiver 2 — Vision component entry point.

Captures frames from the webcam, detects Duplo blocks, and broadcasts
their positions over WebSocket to the browser audio/visual client.

Usage:
    python main.py [--camera 0] [--port 8765] [--debug]

Environment:
    - Matte black table under directional light (nighttime)
    - Camera angled downward at the table
    - Configure TABLE_ROI to crop to just the table surface
"""

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict

import cv2
import numpy as np

from detector import BlockDetector, DEFAULT_COLOR_RANGES
from server import WebSocketServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Crop rectangle for the table ROI in pixel coords: (x, y, width, height)
# Set to None to use the full frame.
TABLE_ROI: tuple[int, int, int, int] | None = None

TARGET_FPS = 30

ZONE_FILE   = os.path.join(os.path.dirname(__file__), "zone.json")
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

_DEFAULT_CONFIG = {
    "min_area_frac":       0.002,
    "max_area_frac":       0.08,
    "max_aspect":          4.0,
    "expected_block_frac": 0.0,
    "color_ranges":        DEFAULT_COLOR_RANGES,
}

# ── Config helpers ────────────────────────────────────────────────────────────

def load_config() -> dict:
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        return {**_DEFAULT_CONFIG, **cfg}
    except FileNotFoundError:
        return dict(_DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Saved config: %s", cfg)

# ── Zone helpers ──────────────────────────────────────────────────────────────

def crop_roi(frame, roi):
    if roi is None:
        return frame
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]


def load_zone() -> dict:
    try:
        with open(ZONE_FILE) as f:
            d = json.load(f)
    except FileNotFoundError:
        return {"corners": [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]}
    if "corners" in d:
        return d
    x, y, w, h = d.get("x",0.0), d.get("y",0.0), d.get("w",1.0), d.get("h",1.0)
    return {"corners": [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]}


def save_zone(corners: list) -> None:
    data = {"corners": corners}
    with open(ZONE_FILE, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved zone: %s", data)


def build_homography(zone: dict) -> np.ndarray:
    src = np.float32(zone["corners"])
    dst = np.float32([[0,0],[1,0],[1,1],[0,1]])
    return cv2.getPerspectiveTransform(src, dst)


def apply_homography(block, H: np.ndarray) -> tuple[float, float]:
    pt  = np.array([[[block.x, block.y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0][0][0]), float(out[0][0][1])


def build_state(blocks, zone: dict, timestamp: float) -> dict:
    return {"blocks": [asdict(b) for b in blocks], "zone": zone, "timestamp": timestamp}

# ── Debug constants ───────────────────────────────────────────────────────────

_DBG_WIN    = "Shiver2 — vision debug"
_VIEW_MODES = [
    "normal", "binary", "cleaned",
    "hsv_h", "hsv_s", "hsv_v",
    "mask_red", "mask_yellow", "mask_green", "mask_blue",
]

_COLOR_NAMES = ["red", "yellow", "green", "blue"]   # trackbar order

_BLOCK_COLOR_BGR = {
    "red":     (60,  60,  255),
    "blue":    (255, 100, 40),
    "green":   (50,  210, 80),
    "yellow":  (0,   230, 255),
    "unknown": (160, 160, 160),
}

_PERSP_STEP  = 0.02
_PERSP_CLAMP = 0.50

# ── Zone drawing helpers ──────────────────────────────────────────────────────

def _compute_corners(base: dict, persp: float) -> list:
    x, y, w, h = base["x"], base["y"], base["w"], base["h"]
    p = persp * w / 2
    return [
        [round(x + p,     4), round(y,     4)],
        [round(x + w - p, 4), round(y,     4)],
        [round(x + w,     4), round(y + h, 4)],
        [round(x,         4), round(y + h, 4)],
    ]


def _norm_rect(p1, p2, fw: int, fh: int) -> dict:
    x0 = max(0, min(p1[0], p2[0]))
    y0 = max(0, min(p1[1], p2[1]))
    x1 = min(fw, max(p1[0], p2[0]))
    y1 = min(fh, max(p1[1], p2[1]))
    return {
        "x": round(x0 / fw, 4), "y": round(y0 / fh, 4),
        "w": round((x1 - x0) / fw, 4), "h": round((y1 - y0) / fh, 4),
    }

# ── Camera overlay ────────────────────────────────────────────────────────────

def _draw_debug_overlay(display: np.ndarray, corners: list, blocks) -> np.ndarray:
    """Draw zone polygon + block boxes onto a copy of the display frame."""
    out = display.copy()
    fh, fw = out.shape[:2]

    pts = np.int32([[int(c[0]*fw), int(c[1]*fh)] for c in corners])
    cv2.polylines(out, [pts], isClosed=True, color=(200, 220, 0), thickness=2)
    for pt in pts:
        cv2.circle(out, tuple(pt), 5, (200, 220, 0), -1)

    for b in blocks:
        x1 = int((b.x - b.w / 2) * fw)
        y1 = int((b.y - b.h / 2) * fh)
        x2 = int((b.x + b.w / 2) * fw)
        y2 = int((b.y + b.h / 2) * fh)
        bgr = _BLOCK_COLOR_BGR.get(b.color, (160, 160, 160))
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(out, f"#{b.id} {b.color}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)
    return out

# ── Info panel ────────────────────────────────────────────────────────────────

def _build_info_panel(
    height: int,
    detector: BlockDetector,
    blocks,
    active_count: int,
    persp: float,
    view_name: str,
    selected_color_idx: int,
) -> np.ndarray:
    W = 300
    panel = np.zeros((height, W, 3), dtype=np.uint8)

    C_TITLE = (200, 220, 0)
    C_WHITE = (220, 220, 220)
    C_DIM   = (100, 100, 100)
    C_GOOD  = (80,  200, 80)
    C_WARN  = (0,   180, 255)
    C_SEP   = (50,  50,  50)

    FONT  = cv2.FONT_HERSHEY_SIMPLEX
    SCALE = 0.42
    BOLD  = cv2.LINE_AA
    PAD_X = 8
    LINE  = 17

    y = 0

    def sep():
        nonlocal y
        y += 2
        cv2.line(panel, (0, y), (W, y), C_SEP, 1)
        y += 5

    def title(text):
        nonlocal y
        y += 12
        cv2.putText(panel, text, (PAD_X, y), FONT, 0.44, C_TITLE, 1, BOLD)
        y += 4
        cv2.line(panel, (PAD_X, y), (W - PAD_X, y), C_TITLE, 1)
        y += LINE - 2

    def line(text, color=C_WHITE, scale=SCALE, indent=0):
        nonlocal y
        if y + LINE > height:
            return
        cv2.putText(panel, text, (PAD_X + indent, y), FONT, scale, color, 1, BOLD)
        y += LINE

    def hint(text):
        line(text, C_DIM, SCALE - 0.03, indent=10)

    def gap(px=6):
        nonlocal y
        y += px

    # ── VIEW ─────────────────────────────────────────────────────────────
    title("VIEW  [v = cycle]")
    for i, m in enumerate(_VIEW_MODES):
        active = (m == view_name)
        c      = C_GOOD if active else C_DIM
        line(f"{'>' if active else ' '} {i:2}: {m}", c)
    gap()

    # ── COLOR RANGES ─────────────────────────────────────────────────────
    sep()
    title("COLOR RANGES  [Color trackbar]")

    # Build a summary dict: name → (h_ranges, s_min, v_min)
    color_summary: dict[str, dict] = {}
    for name in _COLOR_NAMES:
        h_ranges = []
        s_min = v_min = 255
        for n, h_lo, h_hi, sm, vm in detector.color_ranges:
            if n == name:
                h_ranges.append((h_lo, h_hi))
                s_min = min(s_min, sm)
                v_min = min(v_min, vm)
        color_summary[name] = {"h_ranges": h_ranges, "s_min": s_min, "v_min": v_min}

    sel_name = _COLOR_NAMES[selected_color_idx]
    for i, name in enumerate(_COLOR_NAMES):
        is_sel = (name == sel_name)
        c      = C_GOOD if is_sel else C_WHITE
        info   = color_summary.get(name, {})
        h_str  = "/".join(f"{lo}-{hi}" for lo, hi in info.get("h_ranges", []))
        sm     = info.get("s_min", 0)
        vm     = info.get("v_min", 0)
        prefix = ">" if is_sel else " "
        line(f"{prefix} {name.upper():<7} H:{h_str}", c)
        line(f"        S>={sm}  V>={vm}", c, indent=0)
    gap(4)

    # Live values for selected color
    sel_info = color_summary.get(sel_name, {})
    line(f"Editing: {sel_name.upper()}", C_GOOD)
    h_ranges = sel_info.get("h_ranges", [])
    if h_ranges:
        lo, hi = h_ranges[0]
        line(f"  H: {lo} - {hi}", C_WHITE)
        if len(h_ranges) > 1:
            lo2, hi2 = h_ranges[1]
            hint(f"  + wrap: {lo2} - {hi2}")
    line(f"  S min: {sel_info.get('s_min', 0)}", C_WHITE)
    line(f"  V min: {sel_info.get('v_min', 0)}", C_WHITE)
    gap(4)
    hint("Raise S min to reject grey/table.")
    hint("Raise V min to reject dark areas.")
    hint("View mask_<color> to see result.")
    gap()

    # ── AREA FILTER + BLOCK SIZE ─────────────────────────────────────────
    sep()
    title("AREA FILTER  [trackbars]")
    line(f"Min:{detector.min_area_frac*1000:.1f}‰  "
         f"Max:{detector.max_area_frac*1000:.1f}‰  of frame")
    gap(3)
    raw  = detector.last_contour_count
    kept = detector.last_detection_count
    line(f"Raw contours:  {raw}")
    line(f"Passed filter: {kept}", C_GOOD if kept > 0 else C_WARN)
    if raw - kept > 0:
        line(f"Rejected:      {raw - kept}", C_DIM)
    gap(3)
    hint("Raise Min if noise is detected.")
    hint("Lower Min if blocks are missed.")
    hint("Raise Max if large blobs merge.")
    gap(4)
    bf = detector.expected_block_frac
    if bf > 0:
        line(f"Block size: {bf*100:.1f}% ({bf*1000:.0f}‰)", C_GOOD)
        hint("Watershed split ACTIVE.")
        hint("Overlapping blocks will be split.")
    else:
        line("Block size: OFF (0‰)", C_DIM)
        hint("Set Block size > 0 to split")
        hint("overlapping blocks.")
        hint("Duplo 1x1 = ~31.8mm.")
        hint("e.g. if table = 600mm wide,")
        hint("set to ~53 (5.3% of frame).")
    gap()

    # ── ZONE ─────────────────────────────────────────────────────────────
    sep()
    title("ZONE  [drag to draw]")
    line(f"Perspective: P = {persp:+.2f}")
    hint("drag = draw new zone rect")
    hint("q / w = taper top edge")
    hint("r = reset to full frame")
    gap()

    # ── BLOCKS ───────────────────────────────────────────────────────────
    sep()
    title("BLOCKS")
    line(f"Detected:{kept}  In zone:{active_count}",
         C_GOOD if active_count > 0 else C_WARN)
    gap(3)
    for b in blocks[:10]:
        bgr = _BLOCK_COLOR_BGR.get(b.color, (160, 160, 160))
        line(f"#{b.id:<3} {b.color:<8} {b.x:.2f},{b.y:.2f}", bgr)
    gap()

    # ── KEYS ─────────────────────────────────────────────────────────────
    sep()
    title("KEYS")
    line("s  = save zone + config")
    line("v  = cycle view mode")
    line("q / w  = persp taper")
    line("r  = reset zone")
    line("ESC = quit", C_WARN)

    return panel

# ── Main loop ─────────────────────────────────────────────────────────────────

async def camera_loop(camera_index: int, ws_server: WebSocketServer, debug: bool) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    logger.info("Camera opened (index %d)", camera_index)

    cfg = load_config()
    detector = BlockDetector(
        min_area_frac=cfg["min_area_frac"],
        max_area_frac=cfg["max_area_frac"],
        max_aspect=cfg["max_aspect"],
        color_ranges=cfg["color_ranges"],
        expected_block_frac=cfg.get("expected_block_frac", 0.0),
    )
    zone = load_zone()
    H    = build_homography(zone)
    logger.info("Active zone corners: %s", zone["corners"])

    # ── Debug-mode setup ──────────────────────────────────────────────────────
    if debug:
        tl, tr, br, bl = zone["corners"]
        x_left  = bl[0];  x_right = br[0]
        y_top   = tl[1];  y_bot   = bl[1]
        total_w = max(x_right - x_left, 0.01)
        total_h = max(y_bot   - y_top,  0.01)
        top_w   = tr[0] - tl[0]
        dbg_base  = {"x": x_left, "y": y_top, "w": total_w, "h": total_h}
        dbg_persp = round(1.0 - top_w / total_w, 4) if total_w > 0 else 0.0

        dbg_drag: dict = {"drawing": False, "start": None, "end": None}
        dbg_fs:   list = [1, 1]
        view_idx: list = [0]
        prev_color_sel: list = [-1]  # track last Color trackbar value

        def on_mouse(event, mx, my, _flags, _param):
            fw, fh = dbg_fs
            if event == cv2.EVENT_LBUTTONDOWN:
                dbg_drag["drawing"] = True
                dbg_drag["start"]   = (mx, my)
                dbg_drag["end"]     = (mx, my)
            elif event == cv2.EVENT_MOUSEMOVE and dbg_drag["drawing"]:
                dbg_drag["end"] = (mx, my)
            elif event == cv2.EVENT_LBUTTONUP and dbg_drag["drawing"]:
                dbg_drag["drawing"] = False
                dbg_drag["end"] = (mx, my)
                rect = _norm_rect(dbg_drag["start"], dbg_drag["end"], fw, fh)
                if rect["w"] > 0.01 and rect["h"] > 0.01:
                    nonlocal dbg_base, dbg_persp, H, zone
                    dbg_base  = rect
                    dbg_persp = 0.0
                    zone      = {"corners": _compute_corners(dbg_base, dbg_persp)}
                    H         = build_homography(zone)
                    logger.info("Zone updated: %s", zone["corners"])

        cv2.namedWindow(_DBG_WIN)
        cv2.setMouseCallback(_DBG_WIN, on_mouse)

        def _noop(_): pass

        # Color selector + per-color sliders
        cv2.createTrackbar("Color (0-3)", _DBG_WIN, 0, 3, _noop)
        cv2.createTrackbar("H lo",        _DBG_WIN, detector.color_ranges[0][1], 180, _noop)
        cv2.createTrackbar("H hi",        _DBG_WIN, detector.color_ranges[0][2], 180, _noop)
        cv2.createTrackbar("S min",       _DBG_WIN, detector.color_ranges[0][3], 255, _noop)
        cv2.createTrackbar("V min",       _DBG_WIN, detector.color_ranges[0][4], 255, _noop)
        # Area filter
        cv2.createTrackbar("Min area %",  _DBG_WIN, int(detector.min_area_frac * 1000), 50,  _noop)
        cv2.createTrackbar("Max area %",  _DBG_WIN, int(detector.max_area_frac * 1000), 200, _noop)
        # Watershed block splitting (0 = disabled; value is expected_block_frac * 1000)
        cv2.createTrackbar("Block size %", _DBG_WIN, int(detector.expected_block_frac * 1000), 200, _noop)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame — skipping")
                await asyncio.sleep(1 / TARGET_FPS)
                continue

            roi_frame = crop_roi(frame, TABLE_ROI)

            # ── Debug: read trackbars → push to detector ──────────────────────
            if debug:
                fh_px, fw_px = roi_frame.shape[:2]
                dbg_fs[0] = fw_px
                dbg_fs[1] = fh_px

                color_sel  = cv2.getTrackbarPos("Color (0-3)", _DBG_WIN)
                color_name = _COLOR_NAMES[color_sel]

                # When selection changes, snap the per-color sliders to current values
                if color_sel != prev_color_sel[0]:
                    # Find first entry for this color
                    for entry in detector.color_ranges:
                        if entry[0] == color_name:
                            cv2.setTrackbarPos("H lo",  _DBG_WIN, entry[1])
                            cv2.setTrackbarPos("H hi",  _DBG_WIN, entry[2])
                            cv2.setTrackbarPos("S min", _DBG_WIN, entry[3])
                            cv2.setTrackbarPos("V min", _DBG_WIN, entry[4])
                            break
                    prev_color_sel[0] = color_sel

                # Write trackbar values back into detector.color_ranges
                h_lo = cv2.getTrackbarPos("H lo",  _DBG_WIN)
                h_hi = cv2.getTrackbarPos("H hi",  _DBG_WIN)
                s_min = cv2.getTrackbarPos("S min", _DBG_WIN)
                v_min = cv2.getTrackbarPos("V min", _DBG_WIN)
                for entry in detector.color_ranges:
                    if entry[0] == color_name:
                        entry[3] = s_min
                        entry[4] = v_min
                # H range only applies to the first matching entry
                first_seen = False
                for entry in detector.color_ranges:
                    if entry[0] == color_name and not first_seen:
                        entry[1] = h_lo
                        entry[2] = h_hi
                        first_seen = True

                detector.min_area_frac       = cv2.getTrackbarPos("Min area %",   _DBG_WIN) / 1000
                detector.max_area_frac       = cv2.getTrackbarPos("Max area %",   _DBG_WIN) / 1000
                detector.expected_block_frac = cv2.getTrackbarPos("Block size %", _DBG_WIN) / 1000

            blocks = detector.detect(roi_frame)

            # Apply perspective correction; keep only blocks inside [0,1]²
            active = []
            for b in blocks:
                b.x, b.y = apply_homography(b, H)
                if 0.0 <= b.x <= 1.0 and 0.0 <= b.y <= 1.0:
                    active.append(b)

            broadcast_zone = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
            state = build_state(active, broadcast_zone, time.time())
            await ws_server.broadcast(state)

            if debug:
                view_name = _VIEW_MODES[view_idx[0]]
                if view_name == "normal":
                    display = roi_frame
                else:
                    stages  = detector.get_stages(roi_frame)
                    display = stages[view_name]

                corners   = _compute_corners(dbg_base, dbg_persp)
                annotated = _draw_debug_overlay(display, corners, blocks)

                if dbg_drag["drawing"] and dbg_drag["start"] and dbg_drag["end"]:
                    cv2.rectangle(annotated, dbg_drag["start"], dbg_drag["end"],
                                  (0, 255, 255), 1)

                color_sel = cv2.getTrackbarPos("Color (0-3)", _DBG_WIN)
                panel = _build_info_panel(
                    annotated.shape[0], detector, blocks,
                    len(active), dbg_persp, view_name, color_sel,
                )
                cv2.imshow(_DBG_WIN, np.hstack([annotated, panel]))
                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    break
                elif key in (ord("v"), ord("V")):
                    view_idx[0] = (view_idx[0] + 1) % len(_VIEW_MODES)
                elif key in (ord("q"), ord("Q")):
                    dbg_persp = min(_PERSP_CLAMP, round(dbg_persp + _PERSP_STEP, 4))
                    zone = {"corners": _compute_corners(dbg_base, dbg_persp)}
                    H    = build_homography(zone)
                elif key in (ord("w"), ord("W")):
                    dbg_persp = max(-_PERSP_CLAMP, round(dbg_persp - _PERSP_STEP, 4))
                    zone = {"corners": _compute_corners(dbg_base, dbg_persp)}
                    H    = build_homography(zone)
                elif key in (ord("r"), ord("R")):
                    dbg_base  = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
                    dbg_persp = 0.0
                    zone      = {"corners": _compute_corners(dbg_base, dbg_persp)}
                    H         = build_homography(zone)
                    logger.info("Zone reset to full frame")
                elif key in (ord("s"), ord("S")):
                    save_zone(_compute_corners(dbg_base, dbg_persp))
                    save_config({
                        "min_area_frac":       detector.min_area_frac,
                        "max_area_frac":       detector.max_area_frac,
                        "max_aspect":          detector.max_aspect,
                        "expected_block_frac": detector.expected_block_frac,
                        "color_ranges":        detector.color_ranges,
                    })

            await asyncio.sleep(0)

    finally:
        cap.release()
        if debug:
            cv2.destroyAllWindows()


async def main(camera_index: int, port: int, debug: bool) -> None:
    server = WebSocketServer(port=port)
    await server.start()
    try:
        await camera_loop(camera_index, server, debug)
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        await server.stop()


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shiver 2 — vision server")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--port",   type=int, default=8765, help="WebSocket port")
    parser.add_argument("--debug",  action="store_true", help="Show OpenCV debug window")
    args = parser.parse_args()
    asyncio.run(main(args.camera, args.port, args.debug))
