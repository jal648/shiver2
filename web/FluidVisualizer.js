/**
 * FluidVisualizer — drop-in replacement for Visualizer using the Jos Stam
 * fluid solver.  Same interface: constructor(canvas), start(), stop(),
 * update(blocks, zone).
 *
 * Enable via URL param: http://localhost:8080?fluid
 */

import { FluidSolver } from "./fluidsim/fluidsolver.js";

const NUM_CELLS    = 128;   // simulation grid resolution (NxN)
const VEL_SCALE    = 60;    // multiplier: block velocity → fluid velocity
const LIFT         = 0.15;  // constant upward force injected per block per frame
const DENSITY_AMT  = 200;   // density injected at block position per frame
const COLOR_DECAY  = 0.98;  // colorGrid fade rate per frame (< 1)
const PULSE_POWER  = 25;    // radial velocity strength for beat pulses
const PULSE_DENSITY = 350;  // density injected in a beat pulse

const BLOCK_COLORS = {
  red:     [255, 60,  60],
  blue:    [60,  130, 255],
  green:   [60,  255, 130],
  yellow:  [255, 220, 60],
  unknown: [200, 200, 200],
};

export class FluidVisualizer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.ctx.imageSmoothingEnabled = true;

    // Fluid solver
    this.fs = new FluidSolver(NUM_CELLS);
    this.fs.resetVelocity();
    // Buoyancy makes fluid rise naturally; vorticity adds swirls
    this.fs.doBuoyancy = true;
    this.fs.doVorticityConfinement = true;

    // Per-cell color overlay (3 floats per cell: R, G, B in 0–255)
    const nc = (NUM_CELLS + 2) * (NUM_CELLS + 2);
    this.colorGrid = new Float32Array(nc * 3);

    // Offscreen ImageData at grid resolution for efficient rendering
    this._offscreen = document.createElement("canvas");
    this._offscreen.width  = NUM_CELLS;
    this._offscreen.height = NUM_CELLS;
    this._offCtx = this._offscreen.getContext("2d");
    this._imgData = this._offCtx.createImageData(NUM_CELLS, NUM_CELLS);

    this._blocks  = [];
    this._running = false;
    this._rafId   = null;

    // Resize handler
    this._onResize = () => {};
    window.addEventListener("resize", this._onResize);
  }

  start() {
    if (this._running) return;
    this._running = true;
    this._loop();
  }

  stop() {
    this._running = false;
    if (this._rafId !== null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
  }

  /** Called by main.js each time a WebSocket frame arrives. */
  update(blocks /*, zone */) {
    this._blocks = blocks;
  }

  /**
   * Low-level injection API — exposed on window.setFluidSource by main.js.
   * @param {number} x       normalized 0–1
   * @param {number} y       normalized 0–1
   * @param {number} vx      horizontal velocity
   * @param {number} vy      vertical velocity
   * @param {string} color   block color name (key of BLOCK_COLORS)
   * @param {number} density density amount to inject
   */
  _inject(x, y, vx, vy, color = "unknown", density = DENSITY_AMT) {
    const fs = this.fs;
    const i = Math.round(x * NUM_CELLS) + 1;
    const j = Math.round(y * NUM_CELLS) + 1;

    if (i < 1 || i > NUM_CELLS || j < 1 || j > NUM_CELLS) return;

    const du = vx * VEL_SCALE;
    const dv = vy * VEL_SCALE;

    // Inject velocity + density into a 3×3 neighbourhood
    for (let di = -1; di <= 1; di++) {
      for (let dj = -1; dj <= 1; dj++) {
        const ni = i + di, nj = j + dj;
        if (ni < 1 || ni > NUM_CELLS || nj < 1 || nj > NUM_CELLS) continue;
        const idx = fs.I(ni, nj);
        fs.uOld[idx] = du;
        fs.vOld[idx] = dv - LIFT; // negative = up in canvas coords
        fs.dOld[idx] = density;
      }
    }

    // Inject color at center cell
    const rgb = BLOCK_COLORS[color] ?? BLOCK_COLORS.unknown;
    const ci = fs.I(i, j) * 3;
    this.colorGrid[ci]     = rgb[0];
    this.colorGrid[ci + 1] = rgb[1];
    this.colorGrid[ci + 2] = rgb[2];
  }

  /**
   * Audio-synced one-shot radial burst.  Called from beatPulse event handler.
   * @param {number} x      normalized 0–1
   * @param {number} y      normalized 0–1
   * @param {string} color  block color name
   * @param {number} freq   note frequency in Hz — lower = bigger splash
   */
  _pulse(x, y, color, freq = 220) {
    // Map freq → radius on a log scale: bass (~27 Hz) → 16 cells, treble (~2000 Hz) → 3 cells
    const normPitch = 1 - Math.min(1, Math.max(0,
      (Math.log(freq) - Math.log(27)) / (Math.log(2000) - Math.log(27))
    ));
    const radius = Math.round(3 + normPitch * 13); // 3–16 cells

    const fs  = this.fs;
    const i0  = Math.round(x * NUM_CELLS) + 1;
    const j0  = Math.round(y * NUM_CELLS) + 1;
    const rgb = BLOCK_COLORS[color] ?? BLOCK_COLORS.unknown;

    for (let di = -radius; di <= radius; di++) {
      for (let dj = -radius; dj <= radius; dj++) {
        const dist = Math.sqrt(di * di + dj * dj);
        if (dist > radius) continue;
        const ni = i0 + di, nj = j0 + dj;
        if (ni < 1 || ni > NUM_CELLS || nj < 1 || nj > NUM_CELLS) continue;
        const falloff = 1 - dist / radius;
        const idx = fs.I(ni, nj);
        // Radially outward velocity
        if (dist > 0) {
          fs.uOld[idx] += (di / dist) * PULSE_POWER * falloff;
          fs.vOld[idx] += (dj / dist) * PULSE_POWER * falloff;
        }
        fs.dOld[idx] += PULSE_DENSITY * falloff * falloff;
        // Stamp color
        const ci = idx * 3;
        this.colorGrid[ci]     = rgb[0];
        this.colorGrid[ci + 1] = rgb[1];
        this.colorGrid[ci + 2] = rgb[2];
      }
    }
  }

  // ── Private ─────────────────────────────────────────────────────────────────

  _injectBlocks() {
    for (const block of this._blocks) {
      this._inject(block.x, block.y, block.vx ?? 0, block.vy ?? 0, block.color);
    }
  }

  _render() {
    const fs      = this.fs;
    const data    = this._imgData.data;
    const cg      = this.colorGrid;

    for (let i = 1; i <= NUM_CELLS; i++) {
      for (let j = 1; j <= NUM_CELLS; j++) {
        const cellIdx = fs.I(i, j);
        const density = fs.d[cellIdx];

        // Map grid cell (i,j) to offscreen pixel (i-1, j-1)
        const px = (i - 1) + (j - 1) * NUM_CELLS;
        const pxIdx = px * 4;

        if (density > 0.001) {
          const ci = cellIdx * 3;
          // Blend: where no color was injected the grid defaults to zero → black
          // We clamp so Uint8ClampedArray handles overflow naturally
          data[pxIdx]     = density * cg[ci];
          data[pxIdx + 1] = density * cg[ci + 1];
          data[pxIdx + 2] = density * cg[ci + 2];
          data[pxIdx + 3] = 255;
        } else {
          data[pxIdx]     = 0;
          data[pxIdx + 1] = 0;
          data[pxIdx + 2] = 0;
          data[pxIdx + 3] = 255;
        }
      }
    }

    this._offCtx.putImageData(this._imgData, 0, 0);

    // Scale up to full canvas
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.ctx.drawImage(this._offscreen, 0, 0, this.canvas.width, this.canvas.height);
  }

  _loop() {
    if (!this._running) return;

    this._injectBlocks();

    this.fs.velocityStep();
    this.fs.densityStep();

    // Decay color grid
    for (let k = 0; k < this.colorGrid.length; k++) {
      this.colorGrid[k] *= COLOR_DECAY;
    }

    this._render();

    this._rafId = requestAnimationFrame(() => this._loop());
  }
}
