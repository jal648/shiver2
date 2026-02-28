"""
Block detector — finds bright Duplo blocks on a matte black table.

Pipeline (per color):
  BGR frame → channel-ratio mask → morphological cleanup
  → contour detection → area/aspect filter → NMS dedup
  → stable ID assignment

Detection uses BGR channel ratio: for a red block, the R channel must
exceed a brightness floor AND account for more than `ratio`% of the total
R+G+B light. This handles off-primary colors (lime green, cyan blue) that
the simple margin approach misses.

Color is implicit: each block inherits the name of the mask it came from.
"""

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class Block:
    id: int
    x: float   # normalized 0–1 (center, relative to ROI)
    y: float
    w: float   # normalized width
    h: float   # normalized height
    vx: float = 0.0   # velocity (change per second)
    vy: float = 0.0
    color: str = "unknown"  # "red" | "green" | "blue" | "unknown"


# Default per-color BGR ranges — each entry: [name, brightness_min, ratio]
#   brightness_min : dominant channel must exceed this floor (0–255)
#   ratio          : dominant channel as % of R+G+B must exceed this (0–100)
#                    33 = any channel slightly dominant; 50 = fairly saturated
DEFAULT_COLOR_RANGES: list[list] = [
    ["red",   80, 38],   # 38% → tolerates pinkish-red
    ["green", 60, 33],   # 33% → tolerates lime green
    ["blue",  80, 38],   # 38% → tolerates cyan-blue
]

# BGR channel index for each color (cv2.split returns b, g, r)
_DOMINANT_IDX = {"red": 2, "green": 1, "blue": 0}


def _nms(
    detections: list[tuple],
    threshold: float = 0.05,
) -> list[tuple]:
    """
    Remove duplicate detections of the same color that are too close together.

    Greedy: keep largest-area detection first, suppress anything within
    `threshold` normalized units of the same color.

    detections: list of (cx, cy, bw, bh, color)
    """
    sorted_dets = sorted(detections, key=lambda d: d[2] * d[3], reverse=True)
    kept: list[tuple] = []
    for det in sorted_dets:
        cx, cy, _, _, color = det
        duplicate = False
        for k in kept:
            if k[4] == color:
                dist = ((cx - k[0]) ** 2 + (cy - k[1]) ** 2) ** 0.5
                if dist < threshold:
                    duplicate = True
                    break
        if not duplicate:
            kept.append(det)
    return kept


class BlockDetector:
    """
    Detects Duplo blocks in a camera frame and tracks them across frames
    with stable IDs via nearest-neighbour matching.

    Detection uses BGR channel dominance — each color's dominant channel
    (R for red, G for green, B for blue) must clear a brightness floor
    and exceed the other two channels by a configurable margin.
    """

    def __init__(
        self,
        min_area_frac: float = 0.001,
        max_area_frac: float = 0.10,
        max_aspect: float = 4.0,
        color_ranges: Optional[list] = None,
        smoothing: float = 0.4,   # EMA alpha: 0=frozen, 1=no smoothing
    ):
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.max_aspect    = max_aspect
        self.smoothing     = smoothing
        self.color_ranges: list[list] = (
            [list(r) for r in color_ranges]
            if color_ranges is not None
            else [list(r) for r in DEFAULT_COLOR_RANGES]
        )

        self._next_id = 1
        self._prev_blocks: list[Block] = []
        self._prev_time: Optional[float] = None

        # Stats updated each detect() call — readable by debug panel
        self.last_contour_count   = 0
        self.last_detection_count = 0

    # ------------------------------------------------------------------
    def _color_mask(self, name: str, frame: np.ndarray) -> np.ndarray:
        """Return binary mask using BGR channel ratio for one named color.

        A pixel passes if its dominant channel exceeds `brightness_min` AND
        accounts for more than `ratio`% of the total R+G+B light.  The ratio
        criterion handles off-primary colours (lime green, cyan blue) that a
        simple per-channel margin would miss.
        """
        dom_idx = _DOMINANT_IDX.get(name)
        if dom_idx is None:
            return np.zeros(frame.shape[:2], dtype=np.uint8)

        channels = cv2.split(frame)   # (b, g, r)
        for n, brightness_min, ratio in self.color_ranges:
            if n != name:
                continue
            dominant  = channels[dom_idx].astype(np.int32)
            total     = (channels[0].astype(np.int32)
                       + channels[1].astype(np.int32)
                       + channels[2].astype(np.int32) + 1)
            ratio_pct = (dominant * 100) // total   # 0–100
            return (
                (dominant > brightness_min) & (ratio_pct > ratio)
            ).astype(np.uint8) * 255

        return np.zeros(frame.shape[:2], dtype=np.uint8)

    # ------------------------------------------------------------------
    def _make_binary(self, frame: np.ndarray) -> np.ndarray:
        """Return combined binary mask (OR of all colors) — used by get_stages()."""
        combined = np.zeros(frame.shape[:2], dtype=np.uint8)
        seen: set[str] = set()
        for name, *_ in self.color_ranges:
            if name not in seen:
                seen.add(name)
                combined = cv2.bitwise_or(combined, self._color_mask(name, frame))
        return combined

    # ------------------------------------------------------------------
    def get_color_edges(
        self, name: str, frame: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (binary_mask, edge_mask) for one named color.

        binary_mask — morphologically cleaned color binary (255 = color present)
        edge_mask   — Canny edges of the binary (255 = edge pixel)

        Used by the debug edge-view overlay.
        """
        raw     = self._color_mask(name, frame)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(raw,     cv2.MORPH_OPEN,  kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        edges   = cv2.Canny(cleaned, 30, 100)
        return cleaned, edges

    # ------------------------------------------------------------------
    def get_stages(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """
        Return intermediate pipeline images as BGR frames for the debug viewer.

        Keys: binary, cleaned, ch_r, ch_g, ch_b,
              mask_red, mask_green, mask_blue
        """
        binary  = self._make_binary(frame)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(binary,  cv2.MORPH_OPEN,  kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

        def gray_bgr(ch: np.ndarray) -> np.ndarray:
            return cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR)

        b_ch, g_ch, r_ch = cv2.split(frame)

        stages: dict[str, np.ndarray] = {
            "binary":  gray_bgr(binary),
            "cleaned": gray_bgr(cleaned),
            "ch_r":    gray_bgr(r_ch),
            "ch_g":    gray_bgr(g_ch),
            "ch_b":    gray_bgr(b_ch),
        }
        for color in ("red", "green", "blue"):
            cm = self._color_mask(color, frame)
            stages[f"mask_{color}"] = cv2.bitwise_and(frame, frame, mask=cm)

        return stages

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[Block]:
        """
        Process one BGR frame and return tracked blocks.

        Uses BGR channel dominance per color — no HSV conversion needed.

        Args:
            frame: BGR image (already cropped to table ROI by caller)

        Returns:
            List of Block objects with normalized coordinates, velocity, and color.
        """
        h, w     = frame.shape[:2]
        roi_area = w * h
        kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        raw: list[tuple[float, float, float, float, str]] = []
        total_contours = 0

        for entry in self.color_ranges:
            name = entry[0]
            mask = self._color_mask(name, frame)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_contours += len(cnts)

            for cnt in cnts:
                area = cv2.contourArea(cnt)
                if not (self.min_area_frac <= area / roi_area <= self.max_area_frac):
                    continue
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect = max(bw, bh) / max(min(bw, bh), 1)
                if aspect > self.max_aspect:
                    continue
                cx = (bx + bw / 2) / w
                cy = (by + bh / 2) / h
                raw.append((cx, cy, bw / w, bh / h, name))

        self.last_contour_count = total_contours

        deduped = _nms(raw, threshold=0.05)
        self.last_detection_count = len(deduped)

        now = time.monotonic()
        dt  = (now - self._prev_time) if self._prev_time is not None else 0.0
        self._prev_time = now

        blocks = self._assign_ids(deduped, dt)
        self._prev_blocks = blocks
        return blocks

    # ------------------------------------------------------------------
    def _assign_ids(
        self,
        detections: list[tuple[float, float, float, float, str]],
        dt: float,
    ) -> list[Block]:
        """
        Greedy nearest-neighbour matching between previous and current detections.
        Unmatched previous blocks are dropped; new detections get fresh IDs.
        """
        if not self._prev_blocks:
            blocks = []
            for d in detections:
                blocks.append(Block(id=self._next_id, x=d[0], y=d[1], w=d[2], h=d[3], color=d[4]))
                self._next_id += 1
            return blocks

        matched: list[Block] = []
        used_prev = set()

        for cx, cy, bw, bh, color in detections:
            best_idx, best_dist = None, float("inf")
            for i, prev in enumerate(self._prev_blocks):
                if i in used_prev:
                    continue
                dist = ((cx - prev.x) ** 2 + (cy - prev.y) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = i

            if best_idx is not None and best_dist < 0.15:
                prev = self._prev_blocks[best_idx]
                a  = self.smoothing
                sx = a * cx + (1 - a) * prev.x
                sy = a * cy + (1 - a) * prev.y
                sw = a * bw + (1 - a) * prev.w
                sh = a * bh + (1 - a) * prev.h
                vx = (sx - prev.x) / dt if dt > 0 else 0.0
                vy = (sy - prev.y) / dt if dt > 0 else 0.0
                matched.append(Block(id=prev.id, x=sx, y=sy, w=sw, h=sh, vx=vx, vy=vy, color=color))
                used_prev.add(best_idx)
            else:
                matched.append(Block(id=self._next_id, x=cx, y=cy, w=bw, h=bh, color=color))
                self._next_id += 1

        return matched

    # ------------------------------------------------------------------
    def debug_frame(self, frame: np.ndarray, blocks: list[Block]) -> np.ndarray:
        """Draw detected block bounding boxes on a copy of the frame."""
        COLOR_BGR = {
            "red":     (0,   0,   255),
            "green":   (0,   200, 50),
            "blue":    (255, 80,  20),
            "unknown": (0,   255, 0),
        }
        out = frame.copy()
        fh, fw = out.shape[:2]
        for b in blocks:
            x1 = int((b.x - b.w / 2) * fw)
            y1 = int((b.y - b.h / 2) * fh)
            x2 = int((b.x + b.w / 2) * fw)
            y2 = int((b.y + b.h / 2) * fh)
            bgr = COLOR_BGR.get(b.color, (0, 255, 0))
            cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(out, f"{b.id} {b.color}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, bgr, 2)
        return out
