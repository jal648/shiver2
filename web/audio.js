/**
 * audio.js — BPM-synced voice engine
 *
 * Blue  → BlueVoice  — 808-style bass drum, throbs on every quarter note
 *   y → pitch (low y = deep sub-bass, high y = tighter/higher)
 *
 * Red   → RedVoice   — Fat Van Halen "Jump" synth stab, quarter-note triggers
 *   x → pentatonic pitch
 *   y → filter brightness (low y = bright)
 *
 * Green → GreenVoice — Fast arpeggiator, 16th-note steps
 *   y → note count in arpeggio (higher y = more notes)
 *
 * Interface:
 *   audio.unlock()        — call from first user gesture
 *   audio.update(blocks)  — call each frame with current block array
 */

/* global Tone */

// ── Tuning constants (edit here to tweak timing & arp behaviour) ──────────────

const BPM = 130; // transport tempo
const DRUM_SUBDIV = "4n"; // blue: kick fires on every quarter note
const RED_SUBDIV = "2n"; // red:  synth stab fires on every quarter note
const ARP_SUBDIV = "16n"; // green: arpeggiator step interval
const ARP_NOTE_LEN = "4n"; // green: sustain length of each arp note
const ARP_MIN_NOTES = 2; // green: notes in arp when y ≈ 0
const ARP_MAX_NOTES = 8; // green: notes in arp when y = 1
const ARP_START_IDX = 12; // green: start note index in NOTE_SCALE (12 = C4)

// Pentatonic scale shared by red (stabs) and green (arp)
// const PENTATONIC = ["C3", "E3", "G3", "A3", "C4", "E4", "G4", "A4", "C5"];
const NOTE_SCALE = "123456"
  .split("")
  .reduce((acc, i) => acc.concat("CEGB".split("").map((n) => n + i)), []);

console.log("Using note scale:" , NOTE_SCALE)
// ── Helpers ───────────────────────────────────────────────────────────────────

/** Logarithmic interpolation: t=0 → minHz, t=1 → maxHz */
function logCutoff(t, minHz, maxHz) {
  return Math.exp(Math.log(minHz) + t * (Math.log(maxHz) - Math.log(minHz)));
}

/** Return pentatonic note name for x in [0, 1] */
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

// ── BlueVoice — 808 bass drum ─────────────────────────────────────────────────

class BlueVoice {
  constructor(block, bus) {
    this._note = kickNote(block.y);

    this._synth = new Tone.MembraneSynth({
      pitchDecay: 0.5,
      octaves: 5,
      envelope: { attack: 0.001, decay: 0.9, sustain: 0.0, release: 0.1 },
      volume: 0,
    });
    this._dist = new Tone.Distortion(0.15);
    this._synth.chain(this._dist, bus);

    this._seq = new Tone.Sequence(
      (time) => this._synth.triggerAttackRelease(this._note, "8n", time),
      ["X"],
      DRUM_SUBDIV,
    );
    this._seq.start(0);

    this.update(block);
  }

  update(block) {
    this._note = kickNote(block.y);
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
    this._note = noteForX(block.x);

    this._synth = new Tone.PolySynth(Tone.Synth, {
      oscillator: { type: "fatsawtooth", count: 3, spread: 30 },
      envelope: {
        attack: 0.01,
        decay: 0.1,
        sustain: 0.5,
        release: 0.4,
        attackCurve: "exponential",
      },
    });
    this._filter = new Tone.Filter({
      type: "lowpass",
      frequency: 4000,
      rolloff: -12,
    });
    this._synth.chain(this._filter, bus);

    this._seq = new Tone.Sequence(
      (time) => this._synth.triggerAttackRelease(this._note, "8n", time),
      ["X"],
      RED_SUBDIV,
    );
    this._seq.start(0);

    this.update(block);
  }

  update(block) {
    this._note = noteForX(block.x);
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
    this._lastCount = 0; // force initial pattern set

    this._synth = new Tone.Synth({
      oscillator: { type: "triangle" },
      envelope: { attack: 0.005, decay: 0.12, sustain: 0.0, release: 0.1 },
    });
    this._reverb = new Tone.Reverb({ decay: 2.0, wet: 0.4 });
    this._synth.chain(this._reverb, bus);
    const arpStartIx = Math.floor(Math.random()*10)+4;
    const initNotes = NOTE_SCALE.slice(arpStartIx, arpStartIx + ARP_MIN_NOTES);
    this._pattern = new Tone.Pattern(
      (time, note) =>
        this._synth.triggerAttackRelease(note, ARP_NOTE_LEN, time),
      initNotes,
      "up",
    );
    this._pattern.interval = ARP_SUBDIV;
    this._pattern.start(0);

    this.update(block);
  }

  update(block) {
    const count = arpNoteCount(block.y);
    if (count !== this._lastCount) {
      this._lastCount = count;
      this._pattern.values = NOTE_SCALE.slice(ARP_START_IDX, ARP_START_IDX + count);
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
    this._drumVoices = new Map(); // id → BlueVoice
    this._redVoices = new Map(); // id → RedVoice
    this._greenVoices = new Map(); // id → GreenVoice
    this._drumBus = null;
    this._synthBus = null;
    this._ready = false;
  }

  async unlock() {
    if (this._ready) return;
    await Tone.start();

    const limiter = new Tone.Limiter(-2).toDestination();

    // Drum bus — punchy and dry
    const drumGain = new Tone.Gain(Tone.dbToGain(-4));
    drumGain.connect(limiter);
    this._drumBus = drumGain;

    // Synth bus — shared feedback delay adds space
    const synthDelay = new Tone.FeedbackDelay({
      delayTime: "8n",
      feedback: 0.3,
      wet: 0.15,
    });
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

    const blue = blocks.filter((b) => b.color === "blue");
    const red = blocks.filter((b) => b.color === "red");
    const green = blocks.filter((b) => b.color === "green");

    this._syncVoices(
      blue,
      this._drumVoices,
      (b) => new BlueVoice(b, this._drumBus),
      (v) => v.dispose(),
    );
    this._syncVoices(
      red,
      this._redVoices,
      (b) => new RedVoice(b, this._synthBus),
      (v) => v.dispose(),
    );
    this._syncVoices(
      green,
      this._greenVoices,
      (b) => new GreenVoice(b, this._synthBus),
      (v) => v.dispose(),
    );
  }

  /** Create new voices, update existing, remove gone — shared by all voice types. */
  _syncVoices(blocks, voices, create, remove) {
    const ids = new Set(blocks.map((b) => b.id));
    for (const [id, voice] of voices) {
      if (!ids.has(id)) {
        remove(voice);
        voices.delete(id);
      }
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
