/**
 * audio.js — Web Audio synthesis engine
 *
 * Maps Duplo block positions to ambient synthesizer voices:
 *   - Y position  → pitch (lower on table = lower frequency)
 *   - X position  → stereo pan (left/right)
 *   - velocity    → filter brightness (fast movement = brighter tone)
 *
 * One synth voice per tracked block ID. Voices fade in/out smoothly
 * when blocks appear or disappear.
 */

const RAMP_TIME = 0.12;          // seconds for parameter transitions
const MIN_FREQ = 110;            // Hz — bottom of the table (A2)
const MAX_FREQ = 880;            // Hz — top of the table (A5)
const VOICE_GAIN = 0.15;         // per-voice output level (headroom for polyphony)
const REVERB_MIX = 0.6;          // 0 = dry, 1 = wet

// ── Reverb impulse (simple algorithmic approximation) ─────────────────────────
function buildImpulseResponse(ctx, duration = 3.0, decay = 2.0) {
  const sampleRate = ctx.sampleRate;
  const length = sampleRate * duration;
  const impulse = ctx.createBuffer(2, length, sampleRate);
  for (let c = 0; c < 2; c++) {
    const data = impulse.getChannelData(c);
    for (let i = 0; i < length; i++) {
      data[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / length, decay);
    }
  }
  return impulse;
}

// ── Voice ─────────────────────────────────────────────────────────────────────
class Voice {
  constructor(ctx, destination) {
    this.ctx = ctx;

    // Oscillator (sine wave for a clean, ambient tone)
    this.osc = ctx.createOscillator();
    this.osc.type = "sine";
    this.osc.frequency.value = 440;

    // Low-pass filter (cutoff tracks velocity)
    this.filter = ctx.createBiquadFilter();
    this.filter.type = "lowpass";
    this.filter.frequency.value = 2000;
    this.filter.Q.value = 0.7;

    // Gain (for fade in/out and overall level)
    this.gain = ctx.createGain();
    this.gain.gain.setValueAtTime(0, ctx.currentTime);

    // Stereo panner
    this.panner = ctx.createStereoPanner();
    this.panner.pan.value = 0;

    // Chain: osc → filter → gain → panner → destination
    this.osc.connect(this.filter);
    this.filter.connect(this.gain);
    this.gain.connect(this.panner);
    this.panner.connect(destination);

    this.osc.start();
  }

  /** Smoothly update voice parameters from a Block object. */
  update(block) {
    const t = this.ctx.currentTime;

    const freq = MIN_FREQ + (1 - block.y) * (MAX_FREQ - MIN_FREQ);
    this.osc.frequency.linearRampToValueAtTime(freq, t + RAMP_TIME);

    const pan = block.x * 2 - 1;  // normalize 0–1 → -1 to +1
    this.panner.pan.linearRampToValueAtTime(pan, t + RAMP_TIME);

    const speed = Math.hypot(block.vx, block.vy);
    const cutoff = 500 + Math.min(speed * 3000, 6000);
    this.filter.frequency.linearRampToValueAtTime(cutoff, t + RAMP_TIME);

    this.gain.gain.linearRampToValueAtTime(VOICE_GAIN, t + RAMP_TIME);
  }

  /** Fade out and then stop. */
  release() {
    const t = this.ctx.currentTime;
    this.gain.gain.linearRampToValueAtTime(0, t + RAMP_TIME * 3);
    setTimeout(() => this.osc.stop(), RAMP_TIME * 3 * 1000 + 50);
  }
}

// ── AudioEngine ───────────────────────────────────────────────────────────────
export class AudioEngine {
  constructor() {
    this._ctx = null;
    this._voices = new Map();  // block id → Voice
    this._masterGain = null;
    this._reverb = null;
    this._reverbGain = null;
    this._dryGain = null;
    this._ready = false;
  }

  /**
   * Must be called from a user gesture (click/tap) to unlock the AudioContext.
   */
  unlock() {
    if (this._ready) return;

    this._ctx = new AudioContext();
    const ctx = this._ctx;

    // Reverb
    this._reverb = ctx.createConvolver();
    this._reverb.buffer = buildImpulseResponse(ctx);

    // Dry/wet mix
    this._dryGain = ctx.createGain();
    this._dryGain.gain.value = 1 - REVERB_MIX;
    this._reverbGain = ctx.createGain();
    this._reverbGain.gain.value = REVERB_MIX;

    // Master limiter (DynamicsCompressor at extreme settings ≈ limiter)
    this._masterGain = ctx.createGain();
    this._masterGain.gain.value = 1.0;
    const limiter = ctx.createDynamicsCompressor();
    limiter.threshold.value = -3;
    limiter.knee.value = 0;
    limiter.ratio.value = 20;
    limiter.attack.value = 0.001;
    limiter.release.value = 0.1;

    // Routing: voices → dryGain → masterGain
    //          voices → reverb → reverbGain → masterGain
    //          masterGain → limiter → destination
    this._dryGain.connect(this._masterGain);
    this._reverbGain.connect(this._masterGain);
    this._reverb.connect(this._reverbGain);
    this._masterGain.connect(limiter);
    limiter.connect(ctx.destination);

    // Voices connect to both dry and reverb sends
    this._voiceDest = this._dryGain;  // voices connect here; also tap into reverb

    this._ready = true;
    console.log("[audio] AudioContext unlocked");
  }

  /**
   * Called each frame with the current list of blocks.
   * @param {Array} blocks
   */
  update(blocks) {
    if (!this._ready) return;

    const activeIds = new Set(blocks.map((b) => b.id));

    // Remove voices for blocks that disappeared
    for (const [id, voice] of this._voices) {
      if (!activeIds.has(id)) {
        voice.release();
        this._voices.delete(id);
      }
    }

    // Add or update voices for current blocks
    for (const block of blocks) {
      if (!this._voices.has(block.id)) {
        const voice = new Voice(this._ctx, this._voiceDest);
        // Also send to reverb
        voice.panner.connect(this._reverb);
        this._voices.set(block.id, voice);
      }
      this._voices.get(block.id).update(block);
    }
  }
}
