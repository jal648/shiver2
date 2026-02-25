"""
simulate.py — Mouse-driven WebSocket simulator for testing the web client.

Starts the same WebSocket server as main.py (port 8765) and lets you drive
block positions with the mouse — no camera or OpenCV required.

Controls:
  Mouse hover      — moves Block 1 (always present)
  Left-click       — add a new draggable block (up to 5 total)
  Left-click+drag  — move an existing block
  Right-click      — remove a block

Usage:
    python simulate.py [--port 8765]
"""

import argparse
import asyncio
import time
import threading
import tkinter as tk
from dataclasses import dataclass, field

from server import WebSocketServer

# ── Constants ─────────────────────────────────────────────────────────────────

CANVAS_SIZE = 600
TICK_MS = 33           # ~30 fps
BLOCK_W = 0.05         # normalized block size
BLOCK_H = 0.05
MAX_BLOCKS = 5
VELOCITY_ALPHA = 0.3   # exponential smoothing for velocity (0=frozen, 1=raw)
DRAG_HIT_RADIUS = 0.06 # normalized distance to grab a block

COLORS = ["#4fc3f7", "#81c784", "#ffb74d", "#f06292", "#ce93d8"]


# ── Block dataclass ───────────────────────────────────────────────────────────

@dataclass
class SimBlock:
    id: int
    x: float
    y: float
    w: float = BLOCK_W
    h: float = BLOCK_H
    vx: float = 0.0
    vy: float = 0.0
    _px: float = field(default=0.0, repr=False)  # prev position for velocity
    _py: float = field(default=0.0, repr=False)

    def __post_init__(self):
        self._px = self.x
        self._py = self.y

    def to_dict(self) -> dict:
        return {"id": self.id, "x": self.x, "y": self.y,
                "w": self.w, "h": self.h, "vx": self.vx, "vy": self.vy}


# ── Simulator app ─────────────────────────────────────────────────────────────

class SimulatorApp:
    def __init__(self, root: tk.Tk, ws_server: WebSocketServer, ws_loop: asyncio.AbstractEventLoop):
        self.root = root
        self.server = ws_server
        self.ws_loop = ws_loop

        self._next_id = 2          # Block 1 is the mouse-follower
        self._blocks: dict[int, SimBlock] = {
            1: SimBlock(id=1, x=0.5, y=0.5)
        }
        self._dragging: int | None = None  # id of block being dragged
        self._last_tick = time.monotonic()

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
        self.canvas.bind("<Motion>",        self._on_motion)
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>",     self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)

        hint = tk.Label(
            self.root,
            text="hover=move block 1 · left-click=add · drag=move · right-click=remove",
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
        if hit is not None:
            self._dragging = hit
        elif len(self._blocks) < MAX_BLOCKS:
            bid = self._next_id
            self._next_id += 1
            self._blocks[bid] = SimBlock(id=bid, x=nx, y=ny)
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

        state = {
            "blocks": [b.to_dict() for b in self._blocks.values()],
            "timestamp": time.time(),
        }
        asyncio.run_coroutine_threadsafe(self.server.broadcast(state), self.ws_loop)

        n_clients = len(self.server._clients)
        n_blocks = len(self._blocks)
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

        for b in self._blocks.values():
            color = COLORS[(b.id - 1) % len(COLORS)]
            cx, cy = self._norm_to_canvas(b.x, b.y)
            hw = int(b.w * CANVAS_SIZE / 2)
            hh = int(b.h * CANVAS_SIZE / 2)

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
                fill=color, outline="#fff", width=1,
            )

            # ID label
            self.canvas.create_text(
                cx, cy, text=str(b.id),
                fill="#000", font=("monospace", max(8, hw)),
            )

            # Velocity arrow
            if speed > 0.01:
                scale = 30
                ex = int(cx + b.vx * scale)
                ey = int(cy + b.vy * scale)
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
