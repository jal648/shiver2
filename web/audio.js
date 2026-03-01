/**
 * audio.js — BPM-synced voice engine
 *
 * Blue  → BlueVoice  — 808-style bass drum, throbs every quarter note
 *   y → pitch (low y = deep sub-bass, high y = tighter/higher)
 *
 * Red   → RedVoice   — Fat Van Halen "Jump" synth stab, half-note triggers
 *   x → pentatonic pitch
 *   y → filter brightness (low y = bright)
 *
 * Green → GreenVoice — Fast arpeggiator, 16th-note steps
 *   y → note count in arpeggio (higher y = more notes)
 *
 * Each voice instance is randomly varied so no two blocks sound identical.
 * Beat events are dispatched via window CustomEvent "beatPulse" for visuals.
 *
 * Interface:
 *   audio.unlock()        — call from first user gesture
 *   audio.update(blocks)  — call each frame with current block array
 */

/* global Tone */

// ── Tuning constants (edit here to tweak timing & arp behaviour) ──────────────

const BPM           = 130;   // transport tempo
const DRUM_SUBDIV   = "4n";  // blue: kick fires on every quarter note
const RED_SUBDIV    = "2n";  // red:  synth stab fires on every half note
const ARP_SUBDIV    = "16n"; // green: arpeggiator step interval
const ARP_NOTE_LEN  = "4n";  // green: sustain length of each arp note
const ARP_MIN_NOTES = 2;     // green: notes in arp when y ≈ 0
const ARP_MAX_NOTES = 8;     // green: notes in arp when y = 1
const ARP_START_IDX = 12;    // green: base note index in NOTE_SCALE (12 = C4)

// Chromatic-ish scale: CEGB across octaves 1–6 (24 notes)
const NOTE_SCALE = "123456"
  .split("")
  .reduce((acc, i) => acc.concat("CEGB".split("").map((n) => n + i)), []);

// ── Randomisation helpers ─────────────────────────────────────────────────────

/** Uniform random float in [lo, hi] */
function rnd(lo, hi) { return lo + Math.random() * (hi - lo); }

/** Pick one element at random from the arguments */
function pick(...arr) { return arr[Math.floor(Math.random() * arr.length)]; }

// ── Other helpers ─────────────────────────────────────────────────────────────

/** Logarithmic interpolation: t=0 → minHz, t=1 → maxHz */
function logCutoff(t, minHz, maxHz) {
  return Math.exp(Math.log(minHz) + t * (Math.log(maxHz) - Math.log(minHz)));
}

/** Return scale note name for x in [0, 1] */
function noteForX(x) {
  const i = Math.round(x * (NOTE_SCALE.length - 1));
  return NOTE_SCALE[Math.max(0, Math.min(i, NOTE_SCALE.length - 1))];
}

/** 808 kick pitch from y: low y → deep sub, high y → tighter */
function kickNote(y) {
  // A0 (MIDI 21) at y=0  →  A2 (MIDI 45) at y=1
  const midi = 21 + Math.round(y * 24);
  return Tone.Frequency(midi, "midi").toNote();
}

/** Number of arp notes for a given y */
function arpNoteCount(y) {
  return Math.max(ARP_MIN_NOTES, Math.round(y * ARP_MAX_NOTES));
}

/** Dispatch a beat-sync visual pulse using Tone.getDraw() for tight A/V sync */
function schedulePulse(audioTime, x, y, color, freq = 220) {
  Tone.getDraw().schedule(() => {
    window.dispatchEvent(new CustomEvent("beatPulse", { detail: { x, y, color, freq } }));
  }, audioTime);
}

// ── BlueVoice — 808 bass drum ─────────────────────────────────────────────────

class BlueVoice {
  constructor(block, bus) {
    this._note = kickNote(block.y);
    this._bx = block.x;
    this._by = block.y;

    // Each instance gets its own flavour
    const pitchDecay = rnd(0.3, 0.7);
    const octaves    = rnd(4, 6);
    const decay      = rnd(0.6, 1.1);
    const release    = rnd(0.08, 0.2);
    const volume     = rnd(-4, 2);
    const subdiv     = pick(DRUM_SUBDIV, DRUM_SUBDIV, DRUM_SUBDIV, "2n"); // mostly 4n

    this._synth = new Tone.MembraneSynth({
      pitchDecay,
      octaves,
      envelope: { attack: 0.001, decay, sustain: 0.0, release },
      volume,
    });
    this._dist = new Tone.Distortion(rnd(0.05, 0.25));
    this._synth.chain(this._dist, bus);

    this._seq = new Tone.Sequence(
      (time) => {
        this._synth.triggerAttackRelease(this._note, "8n", time);
        schedulePulse(time, this._bx, this._by, "blue",
          Tone.Frequency(this._note).toFrequency());
      },
      ["X"],
      subdiv,
    );
    this._seq.start(0);

    this.update(block);
  }

  update(block) {
    this._note = kickNote(block.y);
    this._bx = block.x;
    this._by = block.y;
  }

  dispose() {
    this._seq.stop();
    this._seq.dispose();
    this._synth.disconnect();
    this._synth.dispose();
    this._dist.dispose();
  }
}

// ── RedVoice — Jump / fatsawtooth synth stab ──────────────────────────────────

class RedVoice {
  constructor(block, bus) {
    this._note    = noteForX(block.x);
    this._bx      = block.x;
    this._by      = block.y;
    const holdLen = pick("8n", "4n");

    this._synth = new Tone.PolySynth(Tone.Synth, {
      oscillator: {
        type:   "fatsawtooth",
        count:  pick(2, 3, 3, 4),
        spread: rnd(15, 50),
      },
      envelope: {
        attack:      0.01,
        decay:       rnd(0.05, 0.2),
        sustain:     rnd(0.3, 0.7),
        release:     rnd(0.3, 0.6),
        attackCurve: "exponential",
      },
    });
    this._filter = new Tone.Filter({
      type:      "lowpass",
      frequency: rnd(2000, 6000),
      rolloff:   -12,
    });
    this._synth.chain(this._filter, bus);

    this._seq = new Tone.Sequence(
      (time) => {
        this._synth.triggerAttackRelease(this._note, holdLen, time);
        schedulePulse(time, this._bx, this._by, "red",
          Tone.Frequency(this._note).toFrequency());
      },
      ["X"],
      RED_SUBDIV,
    );
    this._seq.start(0);

    this.update(block);
  }

  update(block) {
    this._note = noteForX(block.x);
    this._bx   = block.x;
    this._by   = block.y;
    this._filter.frequency.rampTo(logCutoff(1 - block.y, 300, 8000), 0.1);
  }

  dispose() {
    this._seq.stop();
    this._seq.dispose();
    this._synth.disconnect();
    this._synth.dispose();
    this._filter.dispose();
  }
}

// ── GreenVoice — fast arpeggiator ─────────────────────────────────────────────

class GreenVoice {
  constructor(block, bus) {
    this._lastCount = 0;
    this._bx = block.x;
    this._by = block.y;

    // Randomise envelope and arp character
    const envDecay   = rnd(0.08, 0.3);
    const envRelease = rnd(0.08, 0.25);
    const direction  = pick("up", "upDown", "random");
    // Vary starting octave: offset in steps of ±4 notes (one octave in CEGB scale)
    const startOffset = pick(-4, -4, 0, 0, 0, 4, 4);
    this._arpStart = Math.max(0, Math.min(NOTE_SCALE.length - ARP_MAX_NOTES,
                               ARP_START_IDX + startOffset));

    this._synth = new Tone.Synth({
      oscillator: { type: "triangle" },
      envelope: { attack: 0.005, decay: envDecay, sustain: 0.0, release: envRelease },
    });
    this._reverb = new Tone.Reverb({ decay: rnd(1.5, 3.5), wet: 0.4 });
    this._synth.chain(this._reverb, bus);

    const initNotes = NOTE_SCALE.slice(this._arpStart, this._arpStart + ARP_MIN_NOTES);
    this._pattern = new Tone.Pattern(
      (time, note) => {
        this._synth.triggerAttackRelease(note, ARP_NOTE_LEN, time);
        schedulePulse(time, this._bx, this._by, "green",
          Tone.Frequency(note).toFrequency());
      },
      initNotes,
      direction,
    );
    this._pattern.interval = ARP_SUBDIV;
    this._pattern.start(0);

    this.update(block);
  }

  update(block) {
    this._bx = block.x;
    this._by = block.y;
    const count = arpNoteCount(block.y);
    if (count !== this._lastCount) {
      this._lastCount = count;
      this._pattern.values = NOTE_SCALE.slice(this._arpStart, this._arpStart + count);
    }
  }

  dispose() {
    this._pattern.stop();
    this._pattern.dispose();
    this._synth.disconnect();
    this._synth.dispose();
    this._reverb.dispose();
  }
}

// ── AudioEngine ───────────────────────────────────────────────────────────────

export class AudioEngine {
  constructor() {
    this._drumVoices  = new Map(); // id → BlueVoice
    this._redVoices   = new Map(); // id → RedVoice
    this._greenVoices = new Map(); // id → GreenVoice
    this._drumBus  = null;
    this._synthBus = null;
    this._ready  = false;
    this._paused = false;
  }

  pause() {
    if (!this._ready || this._paused) return;
    Tone.getTransport().stop();
    this._paused = true;
    console.log("[audio] paused (idle)");
  }

  resume() {
    if (!this._ready || !this._paused) return;
    Tone.getTransport().start();
    this._paused = false;
    console.log("[audio] resumed (motion detected)");
  }

  get paused() { return this._paused; }

  async unlock() {
    if (this._ready) return;
    await Tone.start();

    const limiter = new Tone.Limiter(-2).toDestination();

    // Drum bus — punchy and dry; slightly quieter to let synths breathe
    const drumGain = new Tone.Gain(Tone.dbToGain(-6));
    drumGain.connect(limiter);
    this._drumBus = drumGain;

    // Synth bus — more ambient: deeper delay feedback and wetter mix
    const synthDelay = new Tone.FeedbackDelay({ delayTime: "8n", feedback: 0.4, wet: 0.25 });
    synthDelay.connect(limiter);
    const synthGain = new Tone.Gain(Tone.dbToGain(-4));
    synthGain.connect(synthDelay);
    this._synthBus = synthGain;

    const transport = Tone.getTransport();
    transport.bpm.value = BPM;
    transport.start();

    this._ready = true;
    console.log(`[audio] ready — transport @ ${BPM} BPM`);
  }

  update(blocks) {
    if (!this._ready) return;

    const blue  = blocks.filter(b => b.color === "blue");
    const red   = blocks.filter(b => b.color === "red");
    const green = blocks.filter(b => b.color === "green");

    this._syncVoices(blue,  this._drumVoices,  b => new BlueVoice(b, this._drumBus),  v => v.dispose());
    this._syncVoices(red,   this._redVoices,   b => new RedVoice(b, this._synthBus),  v => v.dispose());
    this._syncVoices(green, this._greenVoices, b => new GreenVoice(b, this._synthBus), v => v.dispose());
  }

  /** Create new voices, update existing, remove gone — shared by all voice types. */
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
