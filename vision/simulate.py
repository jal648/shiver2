"""
simulate.py — Mouse-driven WebSocket simulator for testing the web client.

Starts the same WebSocket server as main.py (port 8765) and lets you drive
block positions with the mouse — no camera or OpenCV required.

Controls:
  Mouse hover        — moves Block 1 (always present)
  Left-click         — add a new draggable block (up to 5 total)
  Left-click+drag    — move an existing block
  Double-click block — cycle height (z=0 → 1 → 2 → 0)
  Right-click        — remove a block

Usage:
    python simulate.py [--port 8765]
"""

import argparse
import asyncio
import json
import os
import time
import threading
import tkinter as tk
from dataclasses import dataclass, field

from server import WebSocketServer

ZONE_FILE = os.path.join(os.path.dirname(__file__), "zone.json")


def load_zone() -> dict:
    """Load zone.json; always return {"corners": [...]}."""
    try:
        with open(ZONE_FILE) as f:
            d = json.load(f)
    except FileNotFoundError:
        return {"corners": [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]]}

    if "corners" in d:
        return d

    # Convert old {x,y,w,h} format
    x, y, w, h = d.get("x",0.0), d.get("y",0.0), d.get("w",1.0), d.get("h",1.0)
    return {"corners": [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]}


def _in_zone(block, zone: dict) -> bool:
    corners = zone.get("corners", [])
    if corners:
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return min(xs) <= block.x <= max(xs) and min(ys) <= block.y <= max(ys)
    return True

# ── Constants ─────────────────────────────────────────────────────────────────

CANVAS_SIZE = 600
TICK_MS = 33           # ~30 fps
BLOCK_W = 0.05         # normalized block size
BLOCK_H = 0.05
MAX_BLOCKS = 15
VELOCITY_ALPHA = 0.3   # exponential smoothing for velocity (0=frozen, 1=raw)
DRAG_HIT_RADIUS = 0.06 # normalized distance to grab a block

HEIGHTS = 3                # cycle 0 → 1 → 2 → 0
DOUBLE_CLICK_S = 0.30      # seconds between presses to count as double-click
HEIGHT_SCALE = [1.0, 1.25, 1.5]   # visual block size multiplier per height
HEIGHT_LABELS = ["", "↑", "↑↑"]  # shown above block for z > 0

DUPLO_COLORS = ["red", "blue", "green", "yellow"]

# Map Duplo color names → tkinter hex colors for the canvas
TK_COLORS = {
    "red":    "#ff4444",
    "blue":   "#4488ff",
    "green":  "#44cc66",
    "yellow": "#ffdd44",
}


# ── Block dataclass ───────────────────────────────────────────────────────────

@dataclass
class SimBlock:
    id: int
    x: float
    y: float
    color: str = "red"
    w: float = BLOCK_W
    h: float = BLOCK_H
    vx: float = 0.0
    vy: float = 0.0
    z: int = 0                                       # height level: 0, 1, 2
    _px: float = field(default=0.0, repr=False)      # prev position for velocity
    _py: float = field(default=0.0, repr=False)
    _last_click_t: float = field(default=0.0, repr=False)  # for double-click detection

    def __post_init__(self):
        self._px = self.x
        self._py = self.y

    def to_dict(self) -> dict:
        return {"id": self.id, "x": self.x, "y": self.y,
                "w": self.w, "h": self.h, "vx": self.vx, "vy": self.vy,
                "z": self.z, "color": self.color}


# ── Simulator app ─────────────────────────────────────────────────────────────

class SimulatorApp:
    def __init__(self, root: tk.Tk, ws_server: WebSocketServer, ws_loop: asyncio.AbstractEventLoop):
        self.root = root
        self.server = ws_server
        self.ws_loop = ws_loop

        self._next_id = 2          # Block 1 is the mouse-follower
        self._blocks: dict[int, SimBlock] = {
            1: SimBlock(id=1, x=0.5, y=0.5, color="red")
        }
        self._dragging: int | None = None  # id of block being dragged
        self._last_tick = time.monotonic()
        self._zone = load_zone()

        self._build_ui()
        self._tick()

    # ------------------------------------------------------------------
    def _build_ui(self):
        self.root.title("Shiver 2 — Simulator")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            self.root,
            width=CANVAS_SIZE, height=CANVAS_SIZE,
            bg="#000000", cursor="crosshair",
            highlightthickness=0,
        )
        self.canvas.pack()

        self.status_var = tk.StringVar(value="0 clients connected | 1 block")
        status_bar = tk.Label(
            self.root, textvariable=self.status_var,
            bg="#111", fg="#888", font=("monospace", 10),
            anchor="w", padx=8, pady=4,
        )
        status_bar.pack(fill=tk.X)

        # Mouse bindings
        # self.canvas.bind("<Motion>",        self._on_motion)
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>",     self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)

        hint = tk.Label(
            self.root,
            text="hover=move block 1 · click=add · drag=move · dbl-click=height · right-click=remove",
            bg="#111", fg="#444", font=("monospace", 9),
        )
        hint.pack(pady=(0, 4))

    # ------------------------------------------------------------------
    # Event handlers

    def _canvas_to_norm(self, cx: float, cy: float) -> tuple[float, float]:
        return cx / CANVAS_SIZE, cy / CANVAS_SIZE

    def _norm_to_canvas(self, x: float, y: float) -> tuple[int, int]:
        return int(x * CANVAS_SIZE), int(y * CANVAS_SIZE)

    def _block_at(self, nx: float, ny: float) -> int | None:
        """Return the id of the block closest to (nx, ny), if within hit radius."""
        best_id, best_dist = None, float("inf")
        for b in self._blocks.values():
            d = ((nx - b.x) ** 2 + (ny - b.y) ** 2) ** 0.5
            if d < best_dist:
                best_dist = d
                best_id = b.id
        if best_dist < DRAG_HIT_RADIUS:
            return best_id
        return None

    def _on_motion(self, event):
        nx, ny = self._canvas_to_norm(event.x, event.y)
        self._blocks[1].x = max(0.0, min(1.0, nx))
        self._blocks[1].y = max(0.0, min(1.0, ny))

    def _on_left_press(self, event):
        nx, ny = self._canvas_to_norm(event.x, event.y)
        hit = self._block_at(nx, ny)
        now = time.monotonic()
        if hit is not None:
            b = self._blocks[hit]
            if now - b._last_click_t < DOUBLE_CLICK_S:
                # Double-click: cycle height, don't drag
                b.z = (b.z + 1) % HEIGHTS
                b._last_click_t = 0.0
                return
            b._last_click_t = now
            self._dragging = hit
        elif len(self._blocks) < MAX_BLOCKS:
            bid = self._next_id
            self._next_id += 1
            color = DUPLO_COLORS[len(self._blocks) % len(DUPLO_COLORS)]
            self._blocks[bid] = SimBlock(id=bid, x=nx, y=ny, color=color)
            self._dragging = bid

    def _on_drag(self, event):
        if self._dragging is not None and self._dragging in self._blocks:
            nx, ny = self._canvas_to_norm(event.x, event.y)
            b = self._blocks[self._dragging]
            b.x = max(0.0, min(1.0, nx))
            b.y = max(0.0, min(1.0, ny))

    def _on_left_release(self, _event):
        self._dragging = None

    def _on_right_click(self, event):
        nx, ny = self._canvas_to_norm(event.x, event.y)
        hit = self._block_at(nx, ny)
        if hit is not None and hit != 1:  # can't remove the mouse block
            del self._blocks[hit]

    # ------------------------------------------------------------------
    # Tick: update velocities, broadcast, redraw

    def _tick(self):
        now = time.monotonic()
        dt = now - self._last_tick
        self._last_tick = now

        for b in self._blocks.values():
            raw_vx = (b.x - b._px) / dt if dt > 0 else 0.0
            raw_vy = (b.y - b._py) / dt if dt > 0 else 0.0
            b.vx = b.vx * (1 - VELOCITY_ALPHA) + raw_vx * VELOCITY_ALPHA
            b.vy = b.vy * (1 - VELOCITY_ALPHA) + raw_vy * VELOCITY_ALPHA
            b._px, b._py = b.x, b.y

        active = [b for b in self._blocks.values() if _in_zone(b, self._zone)]
        state = {
            "blocks": [b.to_dict() for b in active],
            "zone": self._zone,
            "timestamp": time.time(),
        }
        asyncio.run_coroutine_threadsafe(self.server.broadcast(state), self.ws_loop)

        n_clients = len(self.server._clients)
        n_blocks = len(active)
        self.status_var.set(f"{n_clients} client{'s' if n_clients != 1 else ''} connected | {n_blocks} block{'s' if n_blocks != 1 else ''}")

        self._draw()
        self.root.after(TICK_MS, self._tick)

    # ------------------------------------------------------------------
    # Drawing

    def _draw(self):
        self.canvas.delete("all")

        # Grid lines (subtle table reference)
        for i in range(1, 4):
            v = int(i * CANVAS_SIZE / 4)
            self.canvas.create_line(v, 0, v, CANVAS_SIZE, fill="#111", width=1)
            self.canvas.create_line(0, v, CANVAS_SIZE, v, fill="#111", width=1)

        # Active zone boundary — drawn as a (possibly trapezoidal) polygon
        corners = self._zone.get("corners", [])
        if corners:
            pts = [(int(c[0] * CANVAS_SIZE), int(c[1] * CANVAS_SIZE)) for c in corners]
            for i in range(len(pts)):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % len(pts)]
                self.canvas.create_line(x1, y1, x2, y2,
                                        fill="#00ffcc", width=2, dash=(6, 4))

        for b in self._blocks.values():
            color = TK_COLORS.get(b.color, "#e8f4ff")
            cx, cy = self._norm_to_canvas(b.x, b.y)
            scale = HEIGHT_SCALE[b.z]
            hw = int(b.w * CANVAS_SIZE / 2 * scale)
            hh = int(b.h * CANVAS_SIZE / 2 * scale)

            # Glow
            speed = (b.vx ** 2 + b.vy ** 2) ** 0.5
            glow_r = int(hw * 2 + speed * 200)
            self.canvas.create_oval(
                cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r,
                fill="", outline=color, width=1,
            )

            # Block
            self.canvas.create_rectangle(
                cx - hw, cy - hh, cx + hw, cy + hh,
                fill=color, outline="#fff", width=max(1, b.z + 1),
            )

            # Height studs: small circles on top face for z > 0
            stud_r = max(3, hw // 4)
            if b.z > 0:
                offsets = [0] if b.z == 1 else [-stud_r, stud_r]
                for ox in offsets:
                    self.canvas.create_oval(
                        cx + ox - stud_r, cy - stud_r,
                        cx + ox + stud_r, cy + stud_r,
                        fill="#fff", outline="#000", width=1,
                    )

            # ID label (shift down slightly when studs are present)
            self.canvas.create_text(
                cx, cy + (stud_r if b.z > 0 else 0),
                text=str(b.id),
                fill="#000", font=("monospace", max(8, hw)),
            )

            # Height label above block
            if b.z > 0:
                self.canvas.create_text(
                    cx, cy - hh - 8,
                    text=HEIGHT_LABELS[b.z],
                    fill=color, font=("monospace", 11),
                )

            # Velocity arrow
            if speed > 0.01:
                arrow_scale = 30
                ex = int(cx + b.vx * arrow_scale)
                ey = int(cy + b.vy * arrow_scale)
                self.canvas.create_line(cx, cy, ex, ey, fill=color, width=2, arrow=tk.LAST)


# ── Entry point ───────────────────────────────────────────────────────────────

def main(port: int):
    # Start asyncio event loop on a daemon thread
    ws_loop = asyncio.new_event_loop()

    def run_loop():
        asyncio.set_event_loop(ws_loop)
        ws_loop.run_forever()

    t = threading.Thread(target=run_loop, daemon=True)
    t.start()

    # Start WebSocket server inside that loop
    server = WebSocketServer(port=port)
    future = asyncio.run_coroutine_threadsafe(server.start(), ws_loop)
    future.result(timeout=5)

    # Build and run tkinter UI on the main thread
    root = tk.Tk()
    SimulatorApp(root, server, ws_loop)

    try:
        root.mainloop()
    finally:
        ws_loop.call_soon_threadsafe(ws_loop.stop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shiver 2 — mouse simulator")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    main(args.port)
