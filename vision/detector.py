"""
Block detector — finds bright Duplo blocks on a matte black table.

Pipeline:
  BGR frame → grayscale → threshold → morphological cleanup
  → contour detection → area/aspect filter → stable ID assignment
"""

import time
from dataclasses import dataclass, field
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
    vx: float = 0.0  # velocity (change per second)
    vy: float = 0.0


class BlockDetector:
    """
    Detects Duplo blocks in a camera frame and tracks them across frames
    with stable IDs via nearest-neighbour matching.
    """

    def __init__(
        self,
        threshold: int = 60,          # brightness threshold (0–255)
        min_area_frac: float = 0.001,  # min block area as fraction of ROI
        max_area_frac: float = 0.10,   # max block area as fraction of ROI
        max_aspect: float = 4.0,       # max width/height ratio
    ):
        self.threshold = threshold
        self.min_area_frac = min_area_frac
        self.max_area_frac = max_area_frac
        self.max_aspect = max_aspect

        self._next_id = 1
        self._prev_blocks: list[Block] = []
        self._prev_time: Optional[float] = None

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[Block]:
        """
        Process one BGR frame and return tracked blocks.

        Args:
            frame: BGR image (already cropped to table ROI by caller)

        Returns:
            List of Block objects with normalized coordinates and velocity.
        """
        h, w = frame.shape[:2]
        roi_area = w * h

        # --- 1. Threshold bright regions on the dark table
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, self.threshold, 255, cv2.THRESH_BINARY)

        # --- 2. Morphological cleanup (remove noise, fill gaps)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # --- 3. Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- 4. Filter contours by area and aspect ratio
        raw_detections: list[tuple[float, float, float, float]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            area_frac = area / roi_area
            if not (self.min_area_frac <= area_frac <= self.max_area_frac):
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > self.max_aspect:
                continue

            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            raw_detections.append((cx, cy, bw / w, bh / h))

        # --- 5. Match to previous blocks and assign stable IDs
        now = time.monotonic()
        dt = (now - self._prev_time) if self._prev_time is not None else 0.0
        self._prev_time = now

        blocks = self._assign_ids(raw_detections, dt)
        self._prev_blocks = blocks
        return blocks

    # ------------------------------------------------------------------
    def _assign_ids(
        self,
        detections: list[tuple[float, float, float, float]],
        dt: float,
    ) -> list[Block]:
        """
        Greedy nearest-neighbour matching between previous and current detections.
        Unmatched previous blocks are dropped; new detections get fresh IDs.
        """
        if not self._prev_blocks:
            return [Block(id=self._next_id + i, x=d[0], y=d[1], w=d[2], h=d[3])
                    for i, d in enumerate(detections)]

        matched: list[Block] = []
        used_prev = set()

        for cx, cy, bw, bh in detections:
            best_idx, best_dist = None, float("inf")
            for i, prev in enumerate(self._prev_blocks):
                if i in used_prev:
                    continue
                dist = ((cx - prev.x) ** 2 + (cy - prev.y) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx is not None and best_dist < 0.15:  # max match distance
                prev = self._prev_blocks[best_idx]
                vx = (cx - prev.x) / dt if dt > 0 else 0.0
                vy = (cy - prev.y) / dt if dt > 0 else 0.0
                matched.append(Block(id=prev.id, x=cx, y=cy, w=bw, h=bh, vx=vx, vy=vy))
                used_prev.add(best_idx)
            else:
                matched.append(Block(id=self._next_id, x=cx, y=cy, w=bw, h=bh))
                self._next_id += 1

        return matched

    # ------------------------------------------------------------------
    def debug_frame(self, frame: np.ndarray, blocks: list[Block]) -> np.ndarray:
        """Draw detected block bounding boxes on a copy of the frame."""
        out = frame.copy()
        h, w = out.shape[:2]
        for b in blocks:
            x1 = int((b.x - b.w / 2) * w)
            y1 = int((b.y - b.h / 2) * h)
            x2 = int((b.x + b.w / 2) * w)
            y2 = int((b.y + b.h / 2) * h)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, str(b.id), (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return out
