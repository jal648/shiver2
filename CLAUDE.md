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
  └─ WebSocket server  ──────► ws://localhost:8765
                                      │
                                      ▼
                              [web/] Browser app
                                ├─ WebSocket client
                                ├─ Web Audio API (synthesis)
                                └─ Canvas 2D (visualization)
```

### WebSocket Message Format
```json
{
  "blocks": [
    { "id": 1, "x": 0.35, "y": 0.62, "w": 0.04, "h": 0.04, "vx": 0.0, "vy": 0.0 }
  ],
  "timestamp": 1234567890.123
}
```
Coordinates are normalized 0–1 relative to the table ROI.

## Project Structure

```
shiver2/
├── vision/
│   ├── requirements.txt    opencv-python, numpy, websockets
│   ├── detector.py         BlockDetector class
│   ├── server.py           WebSocketServer class
│   └── main.py             Entry point (asyncio)
└── web/
    ├── index.html          Fullscreen canvas shell
    ├── socket.js           SocketClient (auto-reconnects)
    ├── audio.js            AudioEngine (Web Audio API)
    ├── visual.js           Visualizer (Canvas 2D + particles)
    └── main.js             App wiring
```

## Component 1: Vision (`vision/`)

- **detector.py** — `BlockDetector`: grayscale → brightness threshold → morphological cleanup → contour filtering → nearest-neighbour ID tracking. Outputs normalized `Block(id, x, y, w, h, vx, vy)`.
- **server.py** — `WebSocketServer`: async, broadcasts JSON state to all connected clients.
- **main.py** — Asyncio camera loop. Tunable constants: `TABLE_ROI`, `THRESHOLD`, `MIN_AREA_FRAC`, `MAX_AREA_FRAC`. `--debug` flag opens an OpenCV window with bounding boxes.

## Component 2: Web (`web/`)

- **socket.js** — Connects to `ws://localhost:8765`, fires `blocksUpdate` CustomEvents on `window`.
- **audio.js** — One synth voice per block. Y → pitch (A2–A5), X → stereo pan, velocity → filter brightness. Global reverb + limiter. Unlocks on first click.
- **visual.js** — Glowing block overlay + velocity-driven particle system, motion trails via partial canvas clear.
- **main.js** — Wires socket → audio + visual; handles AudioContext unlock on first user gesture.

## Running

```bash
# Vision server
cd vision && pip install -r requirements.txt
python main.py --debug

# Web client (ES modules require a server, not file://)
cd web && python -m http.server 8080
# open http://localhost:8080 — click canvas to enable audio
```

## Environment Notes

- Nighttime event with directional light on a matte black table — high contrast aids detection.
- Camera angled downward at the table.
- Set `TABLE_ROI = (x, y, w, h)` in `vision/main.py` once camera framing is known.
