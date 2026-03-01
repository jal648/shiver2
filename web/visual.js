/**
 * visual.js — Trippy ambient canvas visualizer
 *
 * Rendering layers (back to front):
 *   1. Slow dark fade       — long luminous motion trails
 *   2. Beat ripples         — expanding rings spawned by audio beatPulse events
 *   3. Particles            — glowing sparks shed by moving blocks
 *   4. Block discs          — glowing orbs with pulsing rings, no text
 *
 * All drawing uses globalCompositeOperation = "screen" for additive light blending.
 *
 * Interface:
 *   visualizer.start()            — begin RAF loop
 *   visualizer.update(blocks, zone) — called each WebSocket frame
 */

// ── Colour palette ─────────────────────────────────────────────────────────────

const COLOR = {
  red:     "#ff4040",
  blue:    "#4488ff",
  green:   "#44ffaa",
  unknown: "#e8f4ff",
};

// ── Constants ──────────────────────────────────────────────────────────────────

const FADE_ALPHA       = 0.04;   // lower = longer trails
const MAX_RIPPLES      = 40;
const MAX_PARTICLES    = 500;
const PARTICLE_LIFETIME = 2500;  // ms

// ── Ripple ─────────────────────────────────────────────────────────────────────

class Ripple {
  /**
   * @param {number} x  normalised 0–1
   * @param {number} y  normalised 0–1
   * @param {string} color  block colour key
   */
  constructor(x, y, color) {
    this.x         = x;
    this.y         = y;
    this.color     = COLOR[color] ?? COLOR.unknown;
    this.radius    = 0;                              // normalised to shorter axis
    this.maxRadius = 0.25 + Math.random() * 0.2;
    this.speed     = 0.0008 + Math.random() * 0.0008; // normalised per ms
    this.born      = performance.now();
    this.alive     = true;
  }

  tick(now) {
    const dt = now - this.born;
    this.radius = this.speed * dt;
    this.alive  = this.radius < this.maxRadius;
    return this.alive;
  }

  draw(ctx, w, h) {
    const t          = this.radius / this.maxRadius;   // 0 → 1 as ripple expands
    const alpha      = (1 - t) * 0.85;
    const lineWidth  = 2.5 * (1 - t * 0.7);
    const px         = this.x * w;
    const py         = this.y * h;
    const r          = this.radius * Math.min(w, h);

    ctx.save();
    ctx.globalAlpha  = alpha;
    ctx.strokeStyle  = this.color;
    ctx.lineWidth    = lineWidth;
    ctx.shadowColor  = this.color;
    ctx.shadowBlur   = 20;
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }
}

// ── Particle ───────────────────────────────────────────────────────────────────

class Particle {
  constructor(x, y, vx, vy, color) {
    this.x      = x;
    this.y      = y;
    this.vx     = vx * 0.25 + (Math.random() - 0.5) * 0.004;
    this.vy     = vy * 0.25 + (Math.random() - 0.5) * 0.004;
    this.radius = 1.5 + Math.random() * 3.5;
    this.born   = performance.now();
    this.color  = COLOR[color] ?? COLOR.unknown;
  }

  tick(now) {
    const age = now - this.born;
    if (age > PARTICLE_LIFETIME) return false;
    this.alpha = Math.pow(1 - age / PARTICLE_LIFETIME, 1.5);
    this.x += this.vx;
    this.y += this.vy;
    return true;
  }

  draw(ctx, w, h) {
    ctx.save();
    ctx.globalAlpha = this.alpha * 0.9;
    ctx.fillStyle   = this.color;
    ctx.shadowColor = this.color;
    ctx.shadowBlur  = 8;
    ctx.beginPath();
    ctx.arc(this.x * w, this.y * h, this.radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
}

// ── Visualizer ─────────────────────────────────────────────────────────────────

export class Visualizer {
  /** @param {HTMLCanvasElement} canvas */
  constructor(canvas) {
    this.canvas    = canvas;
    this.ctx       = canvas.getContext("2d");
    this._blocks   = [];
    this._particles = [];
    this._ripples  = [];
    this._running  = false;
    this._lastTs   = 0;

    // Listen for beat-sync pulse events from audio.js
    window.addEventListener("beatPulse", (e) => {
      if (this._ripples.length < MAX_RIPPLES) {
        this._ripples.push(new Ripple(e.detail.x, e.detail.y, e.detail.color));
      }
    });
  }

  start() {
    this._running = true;
    this._resize();
    window.addEventListener("resize", () => this._resize());
    requestAnimationFrame((ts) => this._frame(ts));
  }

  stop() { this._running = false; }

  /**
   * Called by main.js when new block data arrives from the WebSocket.
   * @param {Array}  blocks
   * @param {object} zone
   */
  update(blocks, zone) {
    this._spawnParticles(blocks);
    this._blocks = blocks;
  }

  // ── Private ──────────────────────────────────────────────────────────────────

  _resize() {
    this.canvas.width  = window.innerWidth;
    this.canvas.height = window.innerHeight;
  }

  _frame(ts) {
    if (!this._running) return;
    const now = performance.now();
    const { ctx, canvas } = this;
    const w = canvas.width;
    const h = canvas.height;

    // 1. Slow fade — long luminous trails
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = `rgba(0, 0, 0, ${FADE_ALPHA})`;
    ctx.fillRect(0, 0, w, h);

    // 2. Draw all glowing elements with additive "screen" blending
    ctx.globalCompositeOperation = "screen";

    // Ripples
    this._ripples = this._ripples.filter(r => r.tick(now));
    for (const r of this._ripples) r.draw(ctx, w, h);

    // Particles
    this._particles = this._particles.filter(p => p.tick(now));
    for (const p of this._particles) p.draw(ctx, w, h);

    // Block discs
    for (const block of this._blocks) this._drawBlock(block, w, h, now);

    // 3. Restore normal compositing
    ctx.globalCompositeOperation = "source-over";

    requestAnimationFrame((ts) => this._frame(ts));
  }

  _drawBlock(block, w, h, now) {
    const { ctx } = this;
    const cx    = block.x * w;
    const cy    = block.y * h;
    const s     = Math.min(w, h);
    const r     = 0.025 * s;                  // ring radius
    const vel   = Math.hypot(block.vx, block.vy);
    const glowR = r + vel * 300 + 0.04 * s;   // ambient glow radius
    const hex   = COLOR[block.color] ?? COLOR.unknown;

    ctx.save();

    // Wide ambient glow
    const grd = ctx.createRadialGradient(cx, cy, 0, cx, cy, glowR);
    grd.addColorStop(0, hex + "55");   // ~33% opacity core
    grd.addColorStop(1, hex + "00");   // transparent edge
    ctx.fillStyle = grd;
    ctx.beginPath();
    ctx.arc(cx, cy, glowR, 0, Math.PI * 2);
    ctx.fill();

    // Outer ring
    ctx.strokeStyle = hex;
    ctx.lineWidth   = 2.5;
    ctx.shadowColor = hex;
    ctx.shadowBlur  = 22;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    // Time-pulsing secondary ring (slower, fainter)
    const pulseR = r * (1.0 + 0.18 * Math.sin(now * 0.0025 + block.id));
    ctx.globalAlpha = 0.25;
    ctx.shadowBlur  = 0;
    ctx.beginPath();
    ctx.arc(cx, cy, pulseR, 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = 1.0;

    // Inner bright dot
    ctx.fillStyle  = hex;
    ctx.shadowBlur = 10;
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  _spawnParticles(blocks) {
    for (const block of blocks) {
      const speed = Math.hypot(block.vx, block.vy);
      if (speed > 0.005) {
        const count = Math.min(Math.floor(speed * 20), 6);
        for (let i = 0; i < count; i++) {
          this._particles.push(new Particle(block.x, block.y, block.vx, block.vy, block.color));
        }
      }
    }
    if (this._particles.length > MAX_PARTICLES) {
      this._particles.splice(0, this._particles.length - MAX_PARTICLES);
    }
  }
}
