/**
 * audio.js — BPM-synced drum sequencer + pentatonic synth engine
 *
 * Blue blocks  → DrumVoice  (Tone.Transport @ 120 BPM)
 *   y → drum type  (kick / snare / clap / open-hat / closed-hat)
 *   x → filter cutoff
 *   z → beat pattern  (0 = beat 1, 1 = beats 1+3, 2 = all four beats)
 *
 * Other blocks → SynthVoice  (continuous pentatonic pads)
 *   x → pentatonic pitch  (A minor, A2–A4)
 *   y → filter cutoff  (bright at top, dark at bottom)
 *   color → waveform  (red=sawtooth, green=square, yellow=triangle)
 *   √(vx²+vy²) → volume
 *   z → reverb wet  (0=dry, 1=medium, 2=heavy)
 *
 * Interface (same as before):
 *   audio.unlock()        — call from first user gesture
 *   audio.update(blocks)  — call each frame with current block array
 */

/* global Tone */

// ── Constants ─────────────────────────────────────────────────────────────────

const PENTATONIC = ["A2", "C3", "D3", "E3", "G3", "A3", "C4", "D4", "E4", "G4", "A4"];

const WAVEFORMS = { red: "sawtooth", green: "square", yellow: "triangle" };

// Quarter-note beat patterns — null = skip, truthy = trigger
const BEAT_PATTERNS = [
  ["X", null, null, null],  // z=0: beat 1 only
  ["X", null, "X",  null],  // z=1: beats 1 + 3
  ["X", "X",  "X",  "X" ], // z=2: all four beats
];

const REVERB_WET = [0.0, 0.4, 0.85];

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Logarithmic interpolation: t=0 → minHz, t=1 → maxHz */
function logCutoff(t, minHz, maxHz) {
  return Math.exp(Math.log(minHz) + t * (Math.log(maxHz) - Math.log(minHz)));
}

/** Return drum type index 0–4 from y position */
function drumTypeIdx(y) {
  if (y < 0.2) return 0; // kick
  if (y < 0.4) return 1; // snare
  if (y < 0.6) return 2; // clap
  if (y < 0.8) return 3; // open hi-hat
  return 4;              // closed hi-hat
}

/** Return pentatonic note name for x in [0, 1] */
function noteForX(x) {
  const i = Math.round(x * (PENTATONIC.length - 1));
  return PENTATONIC[Math.max(0, Math.min(i, PENTATONIC.length - 1))];
}

// ── DrumVoice ─────────────────────────────────────────────────────────────────

class DrumVoice {
  constructor(block, bus) {
    this._typeIdx = -1; // force initial build
    this._z = -1;       // force initial build
    this._synth = null;
    this._seq = null;
    this._trigger = () => {};

    this._filter = new Tone.Filter({ type: "lowpass", frequency: 2000, rolloff: -24 });
    this._filter.connect(bus);

    this._buildSynth(drumTypeIdx(block.y));
    this._buildSeq(block.z);
    this.update(block);
  }

  _buildSynth(typeIdx) {
    if (typeIdx === this._typeIdx) return;
    if (this._synth) { this._synth.disconnect(); this._synth.dispose(); }
    this._typeIdx = typeIdx;

    switch (typeIdx) {
      case 0: { // kick — low thud, pitch drop
        const s = new Tone.MembraneSynth({
          pitchDecay: 0.08, octaves: 6,
          envelope: { attack: 0.001, decay: 0.35, sustain: 0, release: 0.1 },
          volume: -2,
        });
        this._trigger = (t) => s.triggerAttackRelease("C1", "16n", t);
        this._synth = s;
        break;
      }
      case 1: { // snare — white noise burst
        const s = new Tone.NoiseSynth({
          noise: { type: "white" },
          envelope: { attack: 0.001, decay: 0.18, sustain: 0, release: 0.05 },
          volume: -6,
        });
        this._trigger = (t) => s.triggerAttackRelease("16n", t);
        this._synth = s;
        break;
      }
      case 2: { // clap — pink noise, slightly softer
        const s = new Tone.NoiseSynth({
          noise: { type: "pink" },
          envelope: { attack: 0.005, decay: 0.1, sustain: 0, release: 0.04 },
          volume: -8,
        });
        this._trigger = (t) => s.triggerAttackRelease("16n", t);
        this._synth = s;
        break;
      }
      case 3: { // open hi-hat — longer metallic decay
        const s = new Tone.MetalSynth({
          frequency: 400, harmonicity: 5.1, modulationIndex: 32,
          resonance: 4000, octaves: 1.5,
          envelope: { attack: 0.001, decay: 0.5, release: 0.3 },
          volume: -14,
        });
        this._trigger = (t) => s.triggerAttackRelease("8n", t);
        this._synth = s;
        break;
      }
      default: { // closed hi-hat — very short tick
        const s = new Tone.MetalSynth({
          frequency: 400, harmonicity: 5.1, modulationIndex: 32,
          resonance: 4000, octaves: 1.5,
          envelope: { attack: 0.001, decay: 0.07, release: 0.01 },
          volume: -16,
        });
        this._trigger = (t) => s.triggerAttackRelease("32n", t);
        this._synth = s;
        break;
      }
    }
    this._synth.connect(this._filter);
  }

  _buildSeq(z) {
    if (this._seq) { this._seq.stop(); this._seq.dispose(); }
    const pattern = BEAT_PATTERNS[z] ?? BEAT_PATTERNS[0];
    this._seq = new Tone.Sequence(
      (time, val) => { if (val) this._trigger(time); },
      pattern,
      "4n",
    );
    this._seq.start(0);
    this._z = z;
  }

  update(block) {
    const t = drumTypeIdx(block.y);
    if (t !== this._typeIdx) this._buildSynth(t);
    if (block.z !== this._z) this._buildSeq(block.z);
    this._filter.frequency.rampTo(logCutoff(block.x, 400, 8000), 0.1);
  }

  dispose() {
    if (this._seq)   { this._seq.stop(); this._seq.dispose(); }
    if (this._synth) { this._synth.disconnect(); this._synth.dispose(); }
    this._filter.dispose();
  }
}

// ── SynthVoice ────────────────────────────────────────────────────────────────

class SynthVoice {
  constructor(block, bus) {
    this._z = -1;
    const type = WAVEFORMS[block.color] ?? "sine";

    this._synth  = new Tone.Synth({
      oscillator: { type },
      envelope: { attack: 0.5, decay: 0.2, sustain: 0.85, release: 3.0 },
      portamento: 0.08,
    });
    this._filter = new Tone.Filter({ type: "lowpass", frequency: 3000, rolloff: -12 });
    this._reverb = new Tone.Reverb({ decay: 3.5, wet: 0.0 });
    this._panner = new Tone.Panner(0);
    this._gain   = new Tone.Gain(Tone.dbToGain(-18));

    this._synth.chain(this._filter, this._reverb, this._panner, this._gain, bus);
    this._synth.triggerAttack(noteForX(block.x));
    this.update(block);
  }

  update(block) {
    // Pentatonic pitch from x
    this._synth.frequency.rampTo(
      Tone.Frequency(noteForX(block.x)).toFrequency(), 0.1,
    );

    // Filter cutoff from y — bright at top (y≈0), dark at bottom (y≈1)
    this._filter.frequency.rampTo(logCutoff(1 - block.y, 200, 6000), 0.2);

    // Volume from velocity magnitude
    const mag = Math.hypot(block.vx, block.vy);
    const db  = -20 + Math.min(mag / 0.4, 1) * 14; // -20 dB still → -6 dB fast
    this._gain.gain.rampTo(Tone.dbToGain(db), 0.15);

    // Stereo pan from x
    this._panner.pan.rampTo(block.x * 2 - 1, 0.2);

    // Reverb wet from z (only updated on change)
    if (block.z !== this._z) {
      this._z = block.z;
      this._reverb.wet.rampTo(REVERB_WET[block.z] ?? 0, 0.3);
    }
  }

  release() {
    this._synth.triggerRelease();
    setTimeout(() => {
      try {
        this._synth.dispose();
        this._filter.dispose();
        this._reverb.dispose();
        this._panner.dispose();
        this._gain.dispose();
      } catch (_) { /* already disposed */ }
    }, 5000);
  }
}

// ── AudioEngine ───────────────────────────────────────────────────────────────

export class AudioEngine {
  constructor() {
    this._drumVoices  = new Map(); // id → DrumVoice
    this._synthVoices = new Map(); // id → SynthVoice
    this._drumBus  = null;
    this._synthBus = null;
    this._ready = false;
  }

  async unlock() {
    if (this._ready) return;
    await Tone.start();

    const limiter = new Tone.Limiter(-2).toDestination();

    // Drum bus: drums stay punchy and dry
    const drumGain = new Tone.Gain(Tone.dbToGain(-2));
    drumGain.connect(limiter);
    this._drumBus = drumGain;

    // Synth bus: subtle delay for all pads; per-voice reverb handles z→reverb
    const synthDelay = new Tone.FeedbackDelay({ delayTime: 0.375, feedback: 0.35, wet: 0.2 });
    synthDelay.connect(limiter);
    const synthGain = new Tone.Gain(Tone.dbToGain(-4));
    synthGain.connect(synthDelay);
    this._synthBus = synthGain;

    // Start transport
    const transport = Tone.getTransport();
    transport.bpm.value = 120;
    transport.start();

    this._ready = true;
    console.log("[audio] ready — transport @ 120 BPM");
  }

  update(blocks) {
    if (!this._ready) return;

    const blue  = blocks.filter(b => b.color === "blue");
    const other = blocks.filter(b => b.color !== "blue");

    this._syncVoices(
      blue, this._drumVoices,
      (b) => new DrumVoice(b, this._drumBus),
      (v) => v.dispose(),
    );
    this._syncVoices(
      other, this._synthVoices,
      (b) => new SynthVoice(b, this._synthBus),
      (v) => v.release(),
    );
  }

  /** Create new voices, update existing, remove gone — shared by both voice types. */
  _syncVoices(blocks, voices, create, remove) {
    const ids = new Set(blocks.map(b => b.id));
    for (const [id, voice] of voices) {
      if (!ids.has(id)) { remove(voice); voices.delete(id); }
    }
    for (const block of blocks) {
      if (!voices.has(block.id)) {
        voices.set(block.id, create(block));
      } else {
        voices.get(block.id).update(block);
      }
    }
  }
}
