/**
 * visual.js — Canvas 2D visualizer
 *
 * Renders two layers:
 *   1. Table overlay  — top-down block positions as glowing squares
 *   2. Particle field — ripples/particles spawned by block movement
 *
 * `update(blocks)` is called each animation frame with the latest block data.
 */

const BLOCK_COLORS = {
  red:     { fill: "#ff4444", glow: "rgba(255, 60,  60,  0.4)", particle: "#ff8888" },
  blue:    { fill: "#4488ff", glow: "rgba(60,  100, 255, 0.4)", particle: "#88bbff" },
  green:   { fill: "#44cc66", glow: "rgba(60,  200, 100, 0.4)", particle: "#88ffaa" },
  yellow:  { fill: "#ffdd44", glow: "rgba(255, 220, 50,  0.4)", particle: "#ffee88" },
  unknown: { fill: "#e8f4ff", glow: "rgba(100, 180, 255, 0.4)", particle: "#a0d8ff" },
};

const PARTICLE_LIFETIME = 1500;  // ms
const PARTICLE_SPAWN_SPEED = 0.01;  // min velocity to spawn particles

// ── Particle ──────────────────────────────────────────────────────────────────
class Particle {
  constructor(x, y, vx, vy, color = "#a0d8ff") {
    this.x = x;
    this.y = y;
    this.vx = (vx * 0.3 + (Math.random() - 0.5) * 0.005);
    this.vy = (vy * 0.3 + (Math.random() - 0.5) * 0.005);
    this.alpha = 1.0;
    this.radius = 2 + Math.random() * 4;
    this.born = performance.now();
    this.color = color;
  }

  /** Returns false when the particle should be removed. */
  tick(now, w, h) {
    const age = now - this.born;
    if (age > PARTICLE_LIFETIME) return false;
    this.alpha = 1 - age / PARTICLE_LIFETIME;
    this.x += this.vx;
    this.y += this.vy;
    this.vy += 0.00005;  // subtle gravity
    return true;
  }

  draw(ctx, w, h) {
    ctx.save();
    ctx.globalAlpha = this.alpha * 0.8;
    ctx.fillStyle = this.color;
    ctx.beginPath();
    ctx.arc(this.x * w, this.y * h, this.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

// ── Visualizer ────────────────────────────────────────────────────────────────
export class Visualizer {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this._blocks = [];
    this._particles = [];
    this._prevBlocks = new Map();  // id → prev block, for velocity reference
    this._zone = null;
    this._running = false;
  }

  start() {
    this._running = true;
    this._resize();
    window.addEventListener("resize", () => this._resize());
    requestAnimationFrame(() => this._frame());
  }

  stop() {
    this._running = false;
  }

  /**
   * Called by main.js when new block data arrives from the WebSocket.
   * @param {Array} blocks
   */
  update(blocks, zone) {
    if (zone) this._zone = zone;
    this._spawnParticles(blocks);
    this._blocks = blocks;

    // Keep prev block map updated
    this._prevBlocks = new Map(blocks.map((b) => [b.id, b]));
  }

  // ------------------------------------------------------------------
  _resize() {
    this.canvas.width = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  _frame() {
    if (!this._running) return;
    const now = performance.now();
    const { ctx, canvas } = this;
    const w = canvas.width;
    const h = canvas.height;

    // Background fade (trails)
    ctx.fillStyle = "rgba(0, 0, 0, 0.15)";
    ctx.fillRect(0, 0, w, h);

    // Tick and draw particles
    this._particles = this._particles.filter((p) => p.tick(now, w, h));
    for (const p of this._particles) p.draw(ctx, w, h);

    // Draw active zone boundary
    if (this._zone) {
      const { x, y, w: zw, h: zh } = this._zone;
      ctx.save();
      ctx.strokeStyle = "rgba(0, 255, 200, 0.5)";
      ctx.lineWidth = 2;
      ctx.setLineDash([8, 5]);
      ctx.strokeRect(x * w, y * h, zw * w, zh * h);
      ctx.restore();
    }

    // Draw blocks
    for (const block of this._blocks) {
      this._drawBlock(block, w, h);
    }

    requestAnimationFrame(() => this._frame());
  }

  _drawBlock(block, w, h) {
    const { ctx } = this;
    const cx = block.x * w;
    const cy = block.y * h;
    const bw = block.w * w;
    const bh = block.h * h;
    const palette = BLOCK_COLORS[block.color] ?? BLOCK_COLORS.unknown;

    // Glow
    const glowSize = Math.hypot(block.vx, block.vy) * 500 + 20;
    const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, glowSize);
    gradient.addColorStop(0, palette.glow);
    gradient.addColorStop(1, "transparent");
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(cx, cy, glowSize, 0, Math.PI * 2);
    ctx.fill();

    // Block rectangle
    ctx.save();
    ctx.fillStyle = palette.fill;
    ctx.shadowColor = palette.fill;
    ctx.shadowBlur = 12;
    ctx.fillRect(cx - bw / 2, cy - bh / 2, bw, bh);

    // ID label
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#000";
    ctx.font = `${Math.max(10, bh * 0.5)}px monospace`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(block.id, cx, cy);
    ctx.restore();
  }

  _spawnParticles(blocks) {
    for (const block of blocks) {
      const speed = Math.hypot(block.vx, block.vy);
      if (speed > PARTICLE_SPAWN_SPEED) {
        const count = Math.min(Math.floor(speed * 20), 8);
        const particleColor = (BLOCK_COLORS[block.color] ?? BLOCK_COLORS.unknown).particle;
        for (let i = 0; i < count; i++) {
          this._particles.push(new Particle(block.x, block.y, block.vx, block.vy, particleColor));
        }
      }
    }
    // Cap total particle count for performance
    if (this._particles.length > 600) {
      this._particles.splice(0, this._particles.length - 600);
    }
  }
}
