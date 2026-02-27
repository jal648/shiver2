"""
Shiver 2 — Vision component entry point.

Captures frames from the webcam, detects Duplo blocks, and broadcasts
their positions over WebSocket to the browser audio/visual client.

Usage:
    python main.py [--camera 0] [--port 8765] [--debug]

Debug mode keys:
    1/2/3/4  select color (red/yellow/green/blue)
    b        toggle binary mask view
    e        toggle edge overlay view
    s        save config + zone
    r        reset zone to full frame
    q / w    adjust perspective taper
    ESC      quit
"""

import argparse
import asyncio
import json
import logging
import os
import queue
import threading
import time
from dataclasses import asdict, replace as dc_replace

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
    "min_area_frac": 0.002,
    "max_area_frac": 0.08,
    "max_aspect":    4.0,
    "color_ranges":  DEFAULT_COLOR_RANGES,
}

# ── Config helpers ─────────────────────────────────────────────────────────────

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

# ── Zone helpers ───────────────────────────────────────────────────────────────

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
        return {"corners": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]}
    if "corners" in d:
        return d
    x, y, w, h = d.get("x", 0.0), d.get("y", 0.0), d.get("w", 1.0), d.get("h", 1.0)
    return {"corners": [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]}


def save_zone(corners: list) -> None:
    with open(ZONE_FILE, "w") as f:
        json.dump({"corners": corners}, f, indent=2)
    logger.info("Saved zone corners: %s", corners)


def build_homography(zone: dict) -> np.ndarray:
    src = np.float32(zone["corners"])
    dst = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    return cv2.getPerspectiveTransform(src, dst)


def apply_homography(block, H: np.ndarray) -> tuple[float, float]:
    pt  = np.array([[[block.x, block.y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0][0][0]), float(out[0][0][1])


def build_state(blocks, zone: dict, timestamp: float) -> dict:
    return {"blocks": [asdict(b) for b in blocks], "zone": zone, "timestamp": timestamp}

# ── WebSocket background thread ────────────────────────────────────────────────

def _run_ws_thread(server: WebSocketServer, bcast_queue: queue.SimpleQueue) -> None:
    """
    Runs the WebSocket server on a private asyncio event loop in a daemon thread.
    Pulls state dicts from bcast_queue and broadcasts them to connected clients.
    """
    async def _ws_main():
        await server.start()
        loop = asyncio.get_running_loop()
        while True:
            state = await loop.run_in_executor(None, bcast_queue.get)
            if state is None:
                break
            await server.broadcast(state)

    asyncio.run(_ws_main())

# ── Debug constants ────────────────────────────────────────────────────────────

_DBG_WIN = "Shiver2 — vision debug"

_COLOR_NAMES = ["red", "yellow", "green", "blue"]

_BLOCK_COLOR_BGR = {
    "red":     (60,  60,  255),
    "blue":    (255, 100, 40),
    "green":   (50,  210, 80),
    "yellow":  (0,   230, 255),
    "unknown": (160, 160, 160),
}

_PERSP_STEP  = 0.02
_PERSP_CLAMP = 0.50

# ── Zone helpers ───────────────────────────────────────────────────────────────

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

# ── Debug overlay ──────────────────────────────────────────────────────────────

def _draw_debug_overlay(display: np.ndarray, corners: list, blocks) -> np.ndarray:
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
        bgr = _BLOCK_COLOR_BGR.get(b.color, (160, 160, 160))
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 2)
        cv2.putText(out, f"#{b.id} {b.color}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)
    return out

# ── Info panel ─────────────────────────────────────────────────────────────────

def _build_info_panel(
    height: int,
    detector: BlockDetector,
    blocks,
    active_count: int,
    sel_color: str,
    view_mode: str,
) -> np.ndarray:
    W = 220
    panel = np.zeros((height, W, 3), dtype=np.uint8)

    C_TITLE = (200, 220, 0)
    C_WHITE = (220, 220, 220)
    C_DIM   = (90,  90,  90)
    C_GOOD  = (80,  200, 80)
    C_WARN  = (0,   180, 255)
    C_SEP   = (50,  50,  50)

    FONT  = cv2.FONT_HERSHEY_SIMPLEX
    SCALE = 0.40
    BOLD  = cv2.LINE_AA
    PAD_X = 8
    LINE  = 16

    y = 0

    def sep():
        nonlocal y
        y += 3
        cv2.line(panel, (0, y), (W, y), C_SEP, 1)
        y += 5

    def title(text):
        nonlocal y
        y += 11
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

    def gap(px=5):
        nonlocal y
        y += px

    # ── VIEW ──────────────────────────────────────────────────────────
    title("VIEW")
    for label, key in [("normal", ""), ("binary", "b"), ("edges", "e")]:
        active = (view_mode == label)
        c = C_GOOD if active else C_DIM
        marker = ">" if active else " "
        key_str = f"[{key}]" if key else "   "
        line(f"{marker} {key_str} {label}", c)
    gap()

    # ── COLOR  ────────────────────────────────────────────────────────
    sep()
    title("COLOR  [1/2/3/4]")

    color_info: dict[str, dict] = {}
    for name in _COLOR_NAMES:
        ranges, s_min, v_min = [], 255, 255
        for n, h_lo, h_hi, sm, vm in detector.color_ranges:
            if n == name:
                ranges.append((h_lo, h_hi))
                s_min = min(s_min, sm)
                v_min = min(v_min, vm)
        color_info[name] = {"ranges": ranges, "s_min": s_min, "v_min": v_min}

    for name in _COLOR_NAMES:
        is_sel = (name == sel_color)
        c = C_GOOD if is_sel else C_WHITE
        info = color_info[name]
        h_str = "/".join(f"{lo}-{hi}" for lo, hi in info["ranges"])
        prefix = ">" if is_sel else " "
        line(f"{prefix} {name.upper():<7} H:{h_str}", c)
        if is_sel:
            line(f"    S>={info['s_min']}  V>={info['v_min']}", C_GOOD, indent=0)
    gap(3)

    # ── SLIDERS  ──────────────────────────────────────────────────────
    sep()
    title("SLIDERS  (editing above)")
    line("H lo / H hi  0-180")
    line("S min / V min  0-255")
    gap(3)

    # ── BLOCKS ────────────────────────────────────────────────────────
    sep()
    title("BLOCKS")
    line(f"contours: {detector.last_contour_count}", C_DIM)
    line(f"detected: {detector.last_detection_count}",
         C_GOOD if detector.last_detection_count > 0 else C_WARN)
    line(f"in zone:  {active_count}",
         C_GOOD if active_count > 0 else C_WARN)
    gap(3)
    for b in blocks[:8]:
        bgr = _BLOCK_COLOR_BGR.get(b.color, (160, 160, 160))
        line(f"#{b.id:<2} {b.color:<8} {b.x:.2f},{b.y:.2f}", bgr)
    gap()

    # ── KEYS ──────────────────────────────────────────────────────────
    sep()
    title("KEYS")
    for text in [
        "1-4  select color",
        "b    binary view",
        "e    edge view",
        "s    save",
        "r    reset zone",
        "q/w  perspective",
        "ESC  quit",
    ]:
        line(text, C_DIM)

    return panel

# ── Camera loop (synchronous) ──────────────────────────────────────────────────

def camera_loop(
    camera_index: int,
    bcast_queue: queue.SimpleQueue,
    debug: bool,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    logger.info("Camera opened (index %d)", camera_index)

    cfg      = load_config()
    detector = BlockDetector(
        min_area_frac=cfg["min_area_frac"],
        max_area_frac=cfg["max_area_frac"],
        max_aspect=cfg["max_aspect"],
        color_ranges=cfg["color_ranges"],
    )
    zone = load_zone()
    H    = build_homography(zone)
    logger.info("Active zone corners: %s", zone["corners"])

    # ── Debug setup ───────────────────────────────────────────────────────────
    if debug:
        # Reconstruct dbg_base from zone corners
        tl, tr, br, bl = zone["corners"]
        x_left  = bl[0];  x_right = br[0]
        y_top   = tl[1];  y_bot   = bl[1]
        total_w = max(x_right - x_left, 0.01)
        total_h = max(y_bot   - y_top,  0.01)
        top_w   = tr[0] - tl[0]
        dbg_base  = {"x": x_left, "y": y_top, "w": total_w, "h": total_h}
        dbg_persp = round(1.0 - top_w / total_w, 4) if total_w > 0 else 0.0

        # State: mutable via closures
        sel_color_idx = [0]     # index into _COLOR_NAMES
        view_mode     = ["normal"]  # "normal" | "binary" | "edges"
        dbg_drag: dict = {"drawing": False, "start": None, "end": None}
        dbg_fs:   list = [1, 1]

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

        cv2.namedWindow(_DBG_WIN)
        cv2.setMouseCallback(_DBG_WIN, on_mouse)

        def _noop(_): pass

        # Only 4 trackbars — no color selector trackbar
        def _first_entry(name):
            for e in detector.color_ranges:
                if e[0] == name:
                    return e
            return detector.color_ranges[0]

        e0 = _first_entry(_COLOR_NAMES[sel_color_idx[0]])
        cv2.createTrackbar("H lo",  _DBG_WIN, e0[1], 180, _noop)
        cv2.createTrackbar("H hi",  _DBG_WIN, e0[2], 180, _noop)
        cv2.createTrackbar("S min", _DBG_WIN, e0[3], 255, _noop)
        cv2.createTrackbar("V min", _DBG_WIN, e0[4], 255, _noop)

    # ── Frame loop ────────────────────────────────────────────────────────────
    frame_interval = 1.0 / TARGET_FPS
    try:
        while True:
            t_start = time.monotonic()

            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame — skipping")
                time.sleep(frame_interval)
                continue

            roi_frame = crop_roi(frame, TABLE_ROI)

            # ── Read trackbars → push into detector ───────────────────────────
            if debug:
                fh_px, fw_px = roi_frame.shape[:2]
                dbg_fs[0] = fw_px
                dbg_fs[1] = fh_px

                sel_name = _COLOR_NAMES[sel_color_idx[0]]
                h_lo  = cv2.getTrackbarPos("H lo",  _DBG_WIN)
                h_hi  = cv2.getTrackbarPos("H hi",  _DBG_WIN)
                s_min = cv2.getTrackbarPos("S min", _DBG_WIN)
                v_min = cv2.getTrackbarPos("V min", _DBG_WIN)

                # Write values into the selected color's first entry
                first_seen = False
                for entry in detector.color_ranges:
                    if entry[0] == sel_name and not first_seen:
                        entry[1] = h_lo
                        entry[2] = h_hi
                        entry[3] = s_min
                        entry[4] = v_min
                        first_seen = True
                    elif entry[0] == sel_name:
                        # Keep S/V in sync for second entry (red's wrap range)
                        entry[3] = s_min
                        entry[4] = v_min

            blocks = detector.detect(roi_frame)

            # Apply perspective correction; keep only blocks inside [0,1]²
            # Use dc_replace so original block coords (ROI-space) are preserved
            # for the debug overlay drawn against the raw camera frame.
            active = []
            for b in blocks:
                zx, zy = apply_homography(b, H)
                if 0.0 <= zx <= 1.0 and 0.0 <= zy <= 1.0:
                    active.append(dc_replace(b, x=zx, y=zy))

            broadcast_zone = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
            bcast_queue.put(build_state(active, broadcast_zone, time.time()))

            # ── Debug display ─────────────────────────────────────────────────
            if debug:
                sel_name = _COLOR_NAMES[sel_color_idx[0]]
                mode     = view_mode[0]
                corners  = _compute_corners(dbg_base, dbg_persp)

                if mode == "binary":
                    binary, _ = detector.get_color_edges(sel_name, roi_frame)
                    display   = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                elif mode == "edges":
                    _, edges = detector.get_color_edges(sel_name, roi_frame)
                    display  = roi_frame.copy()
                    edge_bgr = _BLOCK_COLOR_BGR.get(sel_name, (255, 255, 255))
                    display[edges > 0] = edge_bgr
                else:
                    display = roi_frame

                annotated = _draw_debug_overlay(display, corners, blocks)

                if dbg_drag["drawing"] and dbg_drag["start"] and dbg_drag["end"]:
                    cv2.rectangle(annotated, dbg_drag["start"], dbg_drag["end"],
                                  (0, 255, 255), 1)

                panel = _build_info_panel(
                    annotated.shape[0], detector, blocks,
                    len(active), sel_name, mode,
                )
                cv2.imshow(_DBG_WIN, np.hstack([annotated, panel]))

                key = cv2.waitKey(1) & 0xFF

                if key == 27:  # ESC
                    break
                elif key == ord("1"):
                    _switch_color(0, detector, sel_color_idx)
                elif key == ord("2"):
                    _switch_color(1, detector, sel_color_idx)
                elif key == ord("3"):
                    _switch_color(2, detector, sel_color_idx)
                elif key == ord("4"):
                    _switch_color(3, detector, sel_color_idx)
                elif key in (ord("b"), ord("B")):
                    view_mode[0] = "normal" if mode == "binary" else "binary"
                elif key in (ord("e"), ord("E")):
                    view_mode[0] = "normal" if mode == "edges" else "edges"
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
                elif key in (ord("s"), ord("S")):
                    save_zone(_compute_corners(dbg_base, dbg_persp))
                    save_config({
                        "min_area_frac": detector.min_area_frac,
                        "max_area_frac": detector.max_area_frac,
                        "max_aspect":    detector.max_aspect,
                        "color_ranges":  detector.color_ranges,
                    })
            else:
                # Non-debug: pace the loop
                elapsed = time.monotonic() - t_start
                sleep_t = frame_interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

    finally:
        cap.release()
        if debug:
            cv2.destroyAllWindows()


def _switch_color(idx: int, detector: BlockDetector, sel_color_idx: list) -> None:
    """Switch selected color and snap trackbars to that color's current values."""
    sel_color_idx[0] = idx
    name = _COLOR_NAMES[idx]
    for entry in detector.color_ranges:
        if entry[0] == name:
            cv2.setTrackbarPos("H lo",  _DBG_WIN, entry[1])
            cv2.setTrackbarPos("H hi",  _DBG_WIN, entry[2])
            cv2.setTrackbarPos("S min", _DBG_WIN, entry[3])
            cv2.setTrackbarPos("V min", _DBG_WIN, entry[4])
            break

# ── Entry point ────────────────────────────────────────────────────────────────

def main(camera_index: int, port: int, debug: bool) -> None:
    server      = WebSocketServer(port=port)
    bcast_queue: queue.SimpleQueue = queue.SimpleQueue()

    ws_thread = threading.Thread(
        target=_run_ws_thread,
        args=(server, bcast_queue),
        daemon=True,
        name="ws-server",
    )
    ws_thread.start()
    logger.info("WebSocket server thread started on port %d", port)

    try:
        camera_loop(camera_index, bcast_queue, debug)
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        # Poison pill to unblock the WS thread's queue.get()
        bcast_queue.put(None)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shiver 2 — vision server")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--port",   type=int, default=8765, help="WebSocket port")
    parser.add_argument("--debug",  action="store_true", help="Show OpenCV debug window")
    args = parser.parse_args()
    main(args.camera, args.port, args.debug)
