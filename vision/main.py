"""
Shiver 2 — Vision component entry point.

Captures frames from the webcam, detects Duplo blocks, and broadcasts
their positions over WebSocket to the browser audio/visual client.

Usage:
    python main.py [--camera 0] [--port 8765] [--debug]

Debug mode keys:
    1/2/3    select color (red/green/blue)
    Tab      cycle color forward
    v        cycle view (camera → binary mask → edges)
    b / e    jump to binary / edge view
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
from debug_panel import (
    DBG_WIN, COLOR_NAMES, BLOCK_COLOR_BGR,
    draw_debug_overlay, build_info_panel, switch_color,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Crop rectangle for the table ROI in pixel coords: (x, y, width, height)
# Set to None to use the full frame.
TABLE_ROI: tuple[int, int, int, int] | None = None

TARGET_FPS = 12

ZONE_FILE   = os.path.join(os.path.dirname(__file__), "zone.json")
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.json")

DEFAULT_CONFIG = {
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
        return {**DEFAULT_CONFIG, **cfg}
    except FileNotFoundError:
        return dict(DEFAULT_CONFIG)


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

def run_ws_thread(server: WebSocketServer, bcast_queue: queue.SimpleQueue) -> None:
    """
    Runs the WebSocket server on a private asyncio event loop in a daemon thread.
    Pulls state dicts from bcast_queue and broadcasts them to connected clients.
    """
    async def ws_main():
        await server.start()
        loop = asyncio.get_running_loop()
        while True:
            state = await loop.run_in_executor(None, bcast_queue.get)
            if state is None:
                break
            await server.broadcast(state)

    asyncio.run(ws_main())

# ── Debug constants ────────────────────────────────────────────────────────────

PERSP_STEP  = 0.02
PERSP_CLAMP = 0.50

# ── Zone helpers ───────────────────────────────────────────────────────────────

def compute_corners(base: dict, persp: float) -> list:
    x, y, w, h = base["x"], base["y"], base["w"], base["h"]
    p = persp * w / 2
    return [
        [round(x + p,     4), round(y,     4)],
        [round(x + w - p, 4), round(y,     4)],
        [round(x + w,     4), round(y + h, 4)],
        [round(x,         4), round(y + h, 4)],
    ]


def norm_rect(p1, p2, fw: int, fh: int) -> dict:
    x0 = max(0, min(p1[0], p2[0]))
    y0 = max(0, min(p1[1], p2[1]))
    x1 = min(fw, max(p1[0], p2[0]))
    y1 = min(fh, max(p1[1], p2[1]))
    return {
        "x": round(x0 / fw, 4), "y": round(y0 / fh, 4),
        "w": round((x1 - x0) / fw, 4), "h": round((y1 - y0) / fh, 4),
    }

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
        sel_color_idx = [0]         # index into COLOR_NAMES
        view_mode     = ["normal"]  # "normal" | "binary" | "edges"
        dbg_drag: dict = {"drawing": False, "start": None, "end": None}
        dbg_fs:   list = [1, 1]

        def on_mouse(event, mx, my, *args):
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
                rect = norm_rect(dbg_drag["start"], dbg_drag["end"], fw, fh)
                if rect["w"] > 0.01 and rect["h"] > 0.01:
                    nonlocal dbg_base, dbg_persp, H, zone
                    dbg_base  = rect
                    dbg_persp = 0.0
                    zone      = {"corners": compute_corners(dbg_base, dbg_persp)}
                    H         = build_homography(zone)

        cv2.namedWindow(DBG_WIN)
        cv2.setMouseCallback(DBG_WIN, on_mouse)

        def first_entry(name):
            for e in detector.color_ranges:
                if e[0] == name:
                    return e
            return detector.color_ranges[0]

        e0 = first_entry(COLOR_NAMES[sel_color_idx[0]])
        cv2.createTrackbar("brightness", DBG_WIN, e0[1], 255, lambda x: None)
        cv2.createTrackbar("ratio",      DBG_WIN, e0[2], 100, lambda x: None)

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

                sel_name   = COLOR_NAMES[sel_color_idx[0]]
                brightness = cv2.getTrackbarPos("brightness", DBG_WIN)
                ratio      = cv2.getTrackbarPos("ratio",      DBG_WIN)

                for entry in detector.color_ranges:
                    if entry[0] == sel_name:
                        entry[1] = brightness
                        entry[2] = ratio
                        break

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
                sel_name = COLOR_NAMES[sel_color_idx[0]]
                mode     = view_mode[0]
                corners  = compute_corners(dbg_base, dbg_persp)

                if mode == "binary":
                    binary, _ = detector.get_color_edges(sel_name, roi_frame)
                    display   = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                elif mode == "edges":
                    _, edges = detector.get_color_edges(sel_name, roi_frame)
                    display  = roi_frame.copy()
                    edge_bgr = BLOCK_COLOR_BGR.get(sel_name, (255, 255, 255))
                    display[edges > 0] = edge_bgr
                else:
                    display = roi_frame

                annotated = draw_debug_overlay(display, corners, blocks)

                if dbg_drag["drawing"] and dbg_drag["start"] and dbg_drag["end"]:
                    cv2.rectangle(annotated, dbg_drag["start"], dbg_drag["end"],
                                  (0, 255, 255), 1)

                panel = build_info_panel(
                    annotated.shape[0], detector, blocks,
                    len(active), sel_name, mode,
                )
                cv2.imshow(DBG_WIN, np.hstack([annotated, panel]))

                elapsed  = time.monotonic() - t_start
                wait_ms  = max(1, int((frame_interval - elapsed) * 1000))
                key = cv2.waitKey(wait_ms) & 0xFF

                if key == 27:  # ESC
                    break
                elif key == ord("1"):
                    switch_color(0, detector, sel_color_idx)
                elif key == ord("2"):
                    switch_color(1, detector, sel_color_idx)
                elif key == ord("3"):
                    switch_color(2, detector, sel_color_idx)
                elif key == 9:  # Tab — cycle color forward
                    switch_color((sel_color_idx[0] + 1) % len(COLOR_NAMES), detector, sel_color_idx)
                elif key in (ord("v"), ord("V")):
                    views = ["normal", "binary", "edges"]
                    view_mode[0] = views[(views.index(mode) + 1) % len(views)]
                elif key in (ord("b"), ord("B")):
                    view_mode[0] = "normal" if mode == "binary" else "binary"
                elif key in (ord("e"), ord("E")):
                    view_mode[0] = "normal" if mode == "edges" else "edges"
                elif key in (ord("q"), ord("Q")):
                    dbg_persp = min(PERSP_CLAMP, round(dbg_persp + PERSP_STEP, 4))
                    zone = {"corners": compute_corners(dbg_base, dbg_persp)}
                    H    = build_homography(zone)
                elif key in (ord("w"), ord("W")):
                    dbg_persp = max(-PERSP_CLAMP, round(dbg_persp - PERSP_STEP, 4))
                    zone = {"corners": compute_corners(dbg_base, dbg_persp)}
                    H    = build_homography(zone)
                elif key in (ord("r"), ord("R")):
                    dbg_base  = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
                    dbg_persp = 0.0
                    zone      = {"corners": compute_corners(dbg_base, dbg_persp)}
                    H         = build_homography(zone)
                elif key in (ord("s"), ord("S")):
                    save_zone(compute_corners(dbg_base, dbg_persp))
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

# ── Entry point ────────────────────────────────────────────────────────────────

def main(camera_index: int, port: int, debug: bool) -> None:
    server      = WebSocketServer(port=port)
    bcast_queue: queue.SimpleQueue = queue.SimpleQueue()

    ws_thread = threading.Thread(
        target=run_ws_thread,
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
