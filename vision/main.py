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

from detector import BlockDetector
from server import WebSocketServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Crop rectangle for the table ROI in pixel coords: (x, y, width, height)
# Set to None to use the full frame.
TABLE_ROI: tuple[int, int, int, int] | None = None

# Detection tuning — adjust after seeing the actual lighting/camera
THRESHOLD = 60          # brightness cutoff for block detection
MIN_AREA_FRAC = 0.002   # smallest valid block (fraction of ROI area)
MAX_AREA_FRAC = 0.08    # largest valid block

TARGET_FPS = 30         # camera capture target

ZONE_FILE = os.path.join(os.path.dirname(__file__), "zone.json")

# ── Helpers ───────────────────────────────────────────────────────────────────

def crop_roi(frame, roi):
    if roi is None:
        return frame
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]


def load_zone() -> dict:
    """Load zone.json and always return {"corners": [[TL,TR,BR,BL]...]}.

    Accepts both the new corners format and the old {x,y,w,h} rectangle format.
    Defaults to the full frame (identity transform) if the file is missing.
    """
    try:
        with open(ZONE_FILE) as f:
            d = json.load(f)
    except FileNotFoundError:
        return {"corners": [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]}

    if "corners" in d:
        return d

    # Convert old rectangle format to corners
    x, y, w, h = d.get("x",0.0), d.get("y",0.0), d.get("w",1.0), d.get("h",1.0)
    return {"corners": [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]}


def build_homography(zone: dict) -> np.ndarray:
    """Compute perspective transform matrix that maps the zone trapezoid → [0,1]²."""
    src = np.float32(zone["corners"])                          # TL TR BR BL in image space
    dst = np.float32([[0,0],[1,0],[1,1],[0,1]])                # unit square
    return cv2.getPerspectiveTransform(src, dst)


def apply_homography(block, H: np.ndarray) -> tuple[float, float]:
    """Return the perspective-corrected (x, y) for a block."""
    pt  = np.array([[[block.x, block.y]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    return float(out[0][0][0]), float(out[0][0][1])


def build_state(blocks, zone: dict, timestamp: float) -> dict:
    return {
        "blocks": [asdict(b) for b in blocks],
        "zone": zone,
        "timestamp": timestamp,
    }

# ── Main loop ─────────────────────────────────────────────────────────────────

async def camera_loop(camera_index: int, ws_server: WebSocketServer, debug: bool) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    logger.info("Camera opened (index %d)", camera_index)

    detector = BlockDetector(
        threshold=THRESHOLD,
        min_area_frac=MIN_AREA_FRAC,
        max_area_frac=MAX_AREA_FRAC,
    )
    zone = load_zone()
    H    = build_homography(zone)
    logger.info("Active zone corners: %s", zone["corners"])

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame — skipping")
                await asyncio.sleep(1 / TARGET_FPS)
                continue

            roi_frame = crop_roi(frame, TABLE_ROI)
            blocks = detector.detect(roi_frame)

            # Apply perspective correction and keep only blocks inside the unit square
            active = []
            for b in blocks:
                b.x, b.y = apply_homography(b, H)
                if 0.0 <= b.x <= 1.0 and 0.0 <= b.y <= 1.0:
                    active.append(b)

            # Corrected coordinates span the full [0,1]² area
            broadcast_zone = {"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
            state = build_state(active, broadcast_zone, time.time())
            await ws_server.broadcast(state)

            if debug:
                annotated = detector.debug_frame(roi_frame, active)
                # Draw trapezoid zone on debug frame
                fh, fw = annotated.shape[:2]
                pts = np.int32([[int(c[0]*fw), int(c[1]*fh)] for c in zone["corners"]])
                cv2.polylines(annotated, [pts], isClosed=True, color=(200, 220, 0), thickness=2)
                cv2.imshow("Shiver2 — vision debug", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                block_count = len(blocks)
                if block_count:
                    logger.debug(
                        "Blocks: %d — %s",
                        block_count,
                        [(f"id={b.id} x={b.x:.2f} y={b.y:.2f}") for b in blocks],
                    )

            # Yield control so the WebSocket server can handle connections
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
    parser.add_argument("--port", type=int, default=8765, help="WebSocket port")
    parser.add_argument("--debug", action="store_true", help="Show OpenCV debug window")
    args = parser.parse_args()

    asyncio.run(main(args.camera, args.port, args.debug))
