"""
Block detector — finds bright Duplo blocks on a matte black table.

Pipeline:
  BGR frame → HSV → per-color masks → OR → morphological cleanup
  → contour detection → optional watershed split → area/aspect filter
  → pixel-vote color classification → stable ID assignment
"""

import math
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
    color: str = "unknown"  # "red" | "blue" | "green" | "yellow" | "unknown"


# Default per-color HSV ranges — each entry: [name, h_lo, h_hi, s_min, v_min]
# OpenCV HSV: H=0–180, S/V=0–255.
# Red wraps around the hue circle so it has two entries.
DEFAULT_COLOR_RANGES: list[list] = [
    ["red",    0,   10,  80,  80],
    ["red",    170, 180, 80,  80],
    ["yellow", 20,  38,  80,  100],
    ["green",  40,  80,  80,  50],
    ["blue",   100, 130, 80,  50],
]


class BlockDetector:
    """
    Detects Duplo blocks in a camera frame and tracks them across frames
    with stable IDs via nearest-neighbour matching.

    Detection uses per-color HSV masks (OR-ed together), so only saturated,
    on-hue pixels are included — black table and specular highlights are ignored.

    Classification uses pixel voting (count matching pixels per color) which
    correctly handles red's hue wraparound and is robust to partial shadows.

    When expected_block_frac > 0, merged blobs are split via watershed using
    the known physical block size as the minimum seed distance.
    """

    def __init__(
        self,
        min_area_frac: float = 0.001,   # min block area as fraction of ROI
        max_area_frac: float = 0.10,    # max block area as fraction of ROI
        max_aspect: float = 4.0,        # max width/height ratio
        color_ranges: Optional[list] = None,   # per-color HSV ranges
        expected_block_frac: float = 0.0,      # expected block width as fraction of ROI width
    ):
        self.min_area_frac      = min_area_frac
        self.max_area_frac      = max_area_frac
        self.max_aspect         = max_aspect
        self.expected_block_frac = expected_block_frac
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
    def _make_binary(self, hsv: np.ndarray) -> np.ndarray:
        """
        Return raw binary mask before morphological cleanup.

        Pixel is 255 if it matches ANY per-color H/S/V range, 0 otherwise.
        The black table (low S+V) and specular highlights (high V, low S) are
        both excluded since they don't match any color's S_min/V_min.
        """
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for _, h_lo, h_hi, s_min, v_min in self.color_ranges:
            m = cv2.inRange(
                hsv,
                np.array([h_lo, s_min, v_min], dtype=np.uint8),
                np.array([h_hi, 255,   255  ], dtype=np.uint8),
            )
            mask = cv2.bitwise_or(mask, m)
        return mask

    # ------------------------------------------------------------------
    def _color_mask(self, name: str, hsv: np.ndarray) -> np.ndarray:
        """Return binary mask for one named color (OR of its H-range entries)."""
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for n, h_lo, h_hi, s_min, v_min in self.color_ranges:
            if n == name:
                mask = cv2.bitwise_or(
                    mask,
                    cv2.inRange(
                        hsv,
                        np.array([h_lo, s_min, v_min], dtype=np.uint8),
                        np.array([h_hi, 255,   255  ], dtype=np.uint8),
                    ),
                )
        return mask

    # ------------------------------------------------------------------
    def _classify_color(self, hsv_frame: np.ndarray, contour: np.ndarray) -> str:
        """
        Return the dominant Duplo color for the region covered by `contour`.

        Uses pixel voting: count how many pixels inside the contour match each
        color's H/S/V range, then return the winner.  This avoids the hue-mean
        wraparound bug (a red blob spanning H=2 and H=178 would average to
        H≈90 = green using cv2.mean, but pixel counting handles it correctly).
        Both red range entries naturally accumulate into the same "red" bucket.
        """
        contour_mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        total = cv2.countNonZero(contour_mask)
        if total == 0:
            return "unknown"

        votes: dict[str, int] = {}
        for name, h_lo, h_hi, s_min, v_min in self.color_ranges:
            color_px = cv2.inRange(
                hsv_frame,
                np.array([h_lo, s_min, v_min], dtype=np.uint8),
                np.array([h_hi, 255,   255  ], dtype=np.uint8),
            )
            count = cv2.countNonZero(cv2.bitwise_and(contour_mask, color_px))
            votes[name] = votes.get(name, 0) + count   # two red entries accumulate

        if not votes:
            return "unknown"
        best_name  = max(votes, key=votes.get)
        best_count = votes[best_name]
        # Require at least 15% of contour pixels to match (avoids noise wins)
        if best_count / total < 0.15:
            return "unknown"
        return best_name

    # ------------------------------------------------------------------
    def _split_blob(
        self,
        frame: np.ndarray,
        blob_mask: np.ndarray,
        block_radius_px: float,
    ) -> Optional[list]:
        """
        Use watershed to split a merged blob into individual block contours.

        Finds local maxima in the distance transform separated by at least
        block_radius_px pixels, then segments the blob accordingly.

        Returns a list of contours (one per block), or None if no split found.
        """
        dist = cv2.distanceTransform(blob_mask, cv2.DIST_L2, 5)

        # Dilate-and-compare local maxima with kernel sized to block radius
        kern_sz = max(3, int(block_radius_px * 2) | 1)   # must be odd
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kern_sz, kern_sz))
        dilated = cv2.dilate(dist, kernel)

        # Peak pixels: must be a local maximum AND deep enough in the blob
        min_depth = max(1.0, block_radius_px * 0.3)
        peaks = np.uint8((dist >= dilated - 0.001) & (dist > min_depth)) * 255

        n_labels, markers = cv2.connectedComponents(peaks)
        if n_labels <= 2:   # 0=background + 1 blob = no split
            return None

        # Watershed needs a 3-channel image; unknown region = 0 in markers
        markers = markers.astype(np.int32)
        markers[blob_mask == 0] = -1   # mark background as -1
        # Reset unknown interior to 0 for watershed to fill
        interior = cv2.erode(blob_mask, kernel, iterations=1)
        markers[(blob_mask > 0) & (interior == 0) & (peaks == 0)] = 0

        cv2.watershed(frame, markers)

        result = []
        for label in range(1, n_labels):
            seg = np.uint8(markers == label) * 255
            cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cnts:
                result.append(max(cnts, key=cv2.contourArea))

        return result if len(result) > 1 else None

    # ------------------------------------------------------------------
    def get_stages(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """
        Return intermediate pipeline images as BGR frames for the debug viewer.

        Keys:
          "binary"      — raw combined color mask
          "cleaned"     — after morphological open+close
          "hsv_h"       — hue channel as false-colour
          "hsv_s"       — saturation channel (greyscale)
          "hsv_v"       — value channel (greyscale)
          "mask_red"    — original pixels where red mask fires
          "mask_yellow" — original pixels where yellow mask fires
          "mask_green"  — original pixels where green mask fires
          "mask_blue"   — original pixels where blue mask fires
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        binary  = self._make_binary(hsv)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        cleaned = cv2.morphologyEx(binary,  cv2.MORPH_OPEN,  kernel, iterations=2)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

        def gray_bgr(ch: np.ndarray) -> np.ndarray:
            return cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR)

        h_norm = (hsv[:, :, 0].astype(np.float32) * (255 / 180)).clip(0, 255).astype(np.uint8)
        hsv_h_vis = cv2.cvtColor(
            cv2.merge([h_norm,
                       np.full_like(h_norm, 200),
                       np.full_like(h_norm, 200)]),
            cv2.COLOR_HSV2BGR,
        )

        stages: dict[str, np.ndarray] = {
            "binary":  gray_bgr(binary),
            "cleaned": gray_bgr(cleaned),
            "hsv_h":   hsv_h_vis,
            "hsv_s":   gray_bgr(hsv[:, :, 1]),
            "hsv_v":   gray_bgr(hsv[:, :, 2]),
        }

        for color in ("red", "yellow", "green", "blue"):
            cm = self._color_mask(color, hsv)
            stages[f"mask_{color}"] = cv2.bitwise_and(frame, frame, mask=cm)

        return stages

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[Block]:
        """
        Process one BGR frame and return tracked blocks.

        Args:
            frame: BGR image (already cropped to table ROI by caller)

        Returns:
            List of Block objects with normalized coordinates, velocity, and color.
        """
        h, w = frame.shape[:2]
        roi_area = w * h

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- 1. Build binary mask from per-color HSV ranges
        binary = self._make_binary(hsv_frame)

        # --- 2. Morphological cleanup (remove noise, fill gaps)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # --- 3. Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.last_contour_count = len(contours)

        # --- 3b. Optional watershed split for overlapping blocks
        block_radius_px = self.expected_block_frac * min(w, h) / 2
        if block_radius_px > 0:
            expected_area = math.pi * block_radius_px ** 2
            split_contours: list = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > expected_area * 1.6:
                    blob_mask = np.zeros((h, w), dtype=np.uint8)
                    cv2.drawContours(blob_mask, [cnt], -1, 255, cv2.FILLED)
                    parts = self._split_blob(frame, blob_mask, block_radius_px)
                    if parts:
                        split_contours.extend(parts)
                        continue
                split_contours.append(cnt)
            contours = split_contours

        # --- 4. Filter by area and aspect ratio; classify color via pixel vote
        raw_detections: list[tuple[float, float, float, float, str]] = []
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
            color = self._classify_color(hsv_frame, cnt)
            raw_detections.append((cx, cy, bw / w, bh / h, color))

        self.last_detection_count = len(raw_detections)

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
                    best_idx = i

            if best_idx is not None and best_dist < 0.15:
                prev = self._prev_blocks[best_idx]
                vx = (cx - prev.x) / dt if dt > 0 else 0.0
                vy = (cy - prev.y) / dt if dt > 0 else 0.0
                matched.append(Block(id=prev.id, x=cx, y=cy, w=bw, h=bh, vx=vx, vy=vy, color=color))
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
            "blue":    (255, 80,  20),
            "green":   (0,   200, 50),
            "yellow":  (0,   220, 255),
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
