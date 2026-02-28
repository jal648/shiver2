# Shiver 2

An interactive music sculpture that uses a webcam to detect movement and position of duplo blocks on a table, and converts that into cool ambient audio and visualization.

## Architecture

Two components communicate over WebSocket:

```
Camera
  │
  ▼
[vision/] Python process
  ├─ OpenCV capture & block detection
  ├─ Active zone filtering
  └─ WebSocket server  ──────► ws://localhost:8765
                                      │
                                      ▼
                              [web/] Browser app
                                ├─ WebSocket client
                                ├─ Tone.js synthesis (ambient pads)
                                └─ Canvas 2D (visualization)
```

### WebSocket Message Format
```json
{
  "blocks": [
    { "id": 1, "x": 0.35, "y": 0.62, "w": 0.04, "h": 0.04,
      "vx": 0.0, "vy": 0.0, "color": "red" }
  ],
  "zone": { "x": 0.15, "y": 0.15, "w": 0.70, "h": 0.70 },
  "timestamp": 1234567890.123
}
```
Coordinates are normalized 0–1 relative to the table ROI. Only blocks inside the active zone are included.

## Project Structure

```
shiver2/
├── vision/
│   ├── requirements.txt         opencv-python, numpy, websockets
│   ├── requirements-simulate.txt  websockets only (no OpenCV)
│   ├── zone.json                Active zone config (normalized coords)
│   ├── detector.py              BlockDetector class
│   ├── server.py                WebSocketServer class
│   ├── main.py                  Entry point (asyncio camera loop)
│   ├── simulate.py              Mouse-driven simulator (no camera needed)
│   └── calibrate.py             Interactive zone calibration tool
└── web/
    ├── index.html               Fullscreen canvas shell (loads Tone.js CDN)
    ├── socket.js                SocketClient (auto-reconnects)
    ├── audio.js                 AudioEngine (Tone.js DuoSynth, ambient pads)
    ├── visual.js                Visualizer (Canvas 2D + particles + zone boundary)
    └── main.js                  App wiring
```

## Component 1: Vision (`vision/`)

- **detector.py** — `BlockDetector`: grayscale → brightness threshold → morphological cleanup → contour filtering → nearest-neighbour ID tracking → HSV color classification. Outputs normalized `Block(id, x, y, w, h, vx, vy, color)`.
- **server.py** — `WebSocketServer`: async, broadcasts JSON state to all connected clients.
- **main.py** — Asyncio camera loop. Loads `zone.json`, filters blocks to active zone before broadcast. Tunable constants: `TABLE_ROI`, `THRESHOLD`, `MIN_AREA_FRAC`, `MAX_AREA_FRAC`. `--debug` flag opens an OpenCV window with bounding boxes and zone overlay.
- **simulate.py** — tkinter mouse-driven simulator. No camera or OpenCV needed. Hover=block 1, left-click=add block (cycles R/B/G/Y), drag=move, right-click=remove. Respects and displays the active zone.
- **calibrate.py** — OpenCV tool to define the active zone on the live camera feed. Click-drag to draw zone, `S`=save, `R`=reset, `ESC`=quit.

### Block Colors
Four Duplo colors detected via HSV classification: `red`, `blue`, `green`, `yellow` (fallback: `unknown`).

HSV ranges (OpenCV H=0–180):

| Color | H range | Notes |
|-------|---------|-------|
| red | [0–10] ∪ [170–180] | wraps in HSV |
| yellow | [20–38] | |
| green | [40–80] | |
| blue | [100–130] | |

## Component 2: Web (`web/`)

- **socket.js** — Connects to `ws://localhost:8765`, fires `blocksUpdate` CustomEvents on `window` with `{ blocks, zone, timestamp }`.
- **audio.js** — `AudioEngine` using Tone.js v14. One `DuoSynth` voice per block. Y → pitch (A1–A4), X → stereo pan, velocity → vibrato + AutoFilter rate. Shared `FeedbackDelay` → `Reverb` → `Limiter`. Unlocks on first click via `Tone.start()`.
- **visual.js** — `Visualizer`: glowing colored block overlay + velocity-driven particle system (particles match block color) + dashed active zone boundary. Motion trails via partial canvas clear.
- **main.js** — Wires socket → audio + visual; handles AudioContext unlock on first user gesture.

## Running

```bash
# Test without camera (simulator)
cd vision
pip install websockets
python simulate.py

# Real camera (needs Python ≤ 3.12 for opencv-python wheels)
cd vision
pip install -r requirements.txt
python main.py --debug

# Zone calibration (run before main.py at a new venue)
python calibrate.py

# Web client (ES modules require a server, not file://)
cd web && python -m http.server 8080
# open http://localhost:8080 — click canvas to enable audio
```

## Active Zone

The active zone is a rectangular region of the table. Blocks outside it are silently ignored; only blocks inside it trigger audio and visualization.

- Config: `vision/zone.json` — normalized `{ x, y, w, h }` within the TABLE_ROI
- Calibrate: `python vision/calibrate.py` — click-drag on live feed, `S` to save
- Default: `{ x: 0.15, y: 0.15, w: 0.70, h: 0.70 }` (central 70% of table)
- The zone boundary is drawn as a dashed cyan rectangle in both the simulator and the browser

## Environment Notes

- Nighttime event with directional light on a matte black table — high contrast aids detection.
- Camera angled downward at the table.
- Set `TABLE_ROI = (x, y, w, h)` in `vision/main.py` once camera framing is known, then run `calibrate.py` to define the active zone within that frame.
- OpenCV wheels require Python ≤ 3.12. The simulator (`simulate.py`) only needs `websockets` and works on any Python version.
