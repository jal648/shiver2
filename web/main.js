/**
 * main.js — Application entry point
 *
 * Wires together:
 *   SocketClient → AudioEngine + Visualizer
 *
 * Web Audio must be unlocked by a user gesture (browser policy),
 * so we listen for the first click/tap on the canvas.
 */

import { SocketClient } from "./socket.js";
import { AudioEngine } from "./audio.js";
import { Visualizer } from "./visual.js";
import { FluidVisualizer } from "./FluidVisualizer.js";

const canvas = document.getElementById("canvas");
const statusEl = document.getElementById("status");
const debugPanel = document.getElementById("debug-panel");

// ── Instantiate components ────────────────────────────────────────────────────
const socket_url = `ws://${window.location.hostname}:8765`;
const audio = new AudioEngine();

// Append ?fluid to URL to use the fluid simulation visualizer instead
const searchParams = new URLSearchParams(location.search);
const useFluid = searchParams.has("fluid");
const visual = useFluid ? new FluidVisualizer(canvas) : new Visualizer(canvas);

// Low-level fluid injection API (only active in fluid mode)
if (useFluid) {
  window.setFluidSource = (x, y, vx = 0, vy = 0, color = "unknown", density = 120) =>
    visual._inject(x, y, vx, vy, color, density);
}
const socket = new SocketClient(socket_url, (status) => {
  statusEl.textContent = status;
  statusEl.className = `status-${status}`;
});

// ── Wire events ───────────────────────────────────────────────────────────────

// ── Debug panel ───────────────────────────────────────────────────────────────

function fmtJson(block) {
  const COLOR_MAP = { red: "#ff6666", blue: "#66aaff", green: "#66dd88", yellow: "#ffdd55" };
  const blockColor = COLOR_MAP[block.color] ?? "#ccc";
  const lines = Object.entries(block).map(([k, v]) => {
    const keyHtml = `<span class="key">${k}:</span> `;
    const valHtml = typeof v === "string"
      ? `<span class="str">"${v}"</span>`
      : `<span class="num">${typeof v === "number" ? v.toFixed(3).replace(/\.?0+$/, "") : v}</span>`;
    return `  ${keyHtml}${valHtml}`;
  });
  return `<div class="label" style="color:${blockColor}">block ${block.id}</div>` +
         `{<br>${lines.join(",<br>")}<br>}`;
}

function updateDebugPanel(blocks) {
  if (!debugPanel.classList.contains("visible")) return;
  debugPanel.innerHTML = blocks.map(b => `<div class="debug-block">${fmtJson(b)}</div>`).join("");
}

// Toggle with backtick
window.addEventListener("keydown", (e) => {
  console.log(e.key)
  if (e.key === "`") {
    debugPanel.classList.toggle("visible");
    updateDebugPanel(window._lastBlocks ?? []);
  }
});

window.addEventListener("blocksUpdate", (event) => {
  const { blocks, zone } = event.detail;
  window._lastBlocks = blocks;
  audio.update(blocks);
  visual.update(blocks, zone);
  updateDebugPanel(blocks);
});

window.addEventListener("beatPulse", (event) => {
  if (!useFluid) return;
  const { x, y, color, freq } = event.detail;
  visual._pulse(x, y, color, freq);
});

// ── Unlock audio on first user interaction ────────────────────────────────────

let audioUnlocked = false;
async function unlockAudio() {
  if (audioUnlocked) return;
  audioUnlocked = true;
  canvas.removeEventListener("click", unlockAudio);
  canvas.removeEventListener("touchstart", unlockAudio);
  await audio.unlock();
  console.log("[main] Audio unlocked");
}

canvas.addEventListener("click", unlockAudio);
canvas.addEventListener("touchstart", unlockAudio);

// ── Idle detection — auto-pause/resume audio ──────────────────────────────────

const IDLE_MS  = 10_000; // silence after this many ms of unchanged block positions
const AWAKE_MS =  2_000; // re-check interval while paused (waiting for motion)

let _idleTimer    = null;
let _lastSnapshot = null; // null = not yet sampled

function _blockSnapshot(blocks) {
  // Stable string: sorted "id:x.xxx,y.xxx" pairs — cheap, deterministic
  return blocks
    .map(b => `${b.id}:${b.x.toFixed(3)},${b.y.toFixed(3)}`)
    .sort()
    .join("|");
}

function _scheduleIdleCheck(ms) {
  clearTimeout(_idleTimer);
  _idleTimer = setTimeout(_checkIdle, ms);
}

function _checkIdle() {
  const current = _blockSnapshot(window._lastBlocks ?? []);

  if (!audio.paused) {
    if (_lastSnapshot !== null && current === _lastSnapshot) {
      audio.pause();
      _scheduleIdleCheck(AWAKE_MS);
    } else {
      _lastSnapshot = current;
      _scheduleIdleCheck(IDLE_MS);
    }
  } else {
    if (current !== _lastSnapshot) {
      audio.resume();
      _lastSnapshot = current;
      _scheduleIdleCheck(IDLE_MS);
    } else {
      _scheduleIdleCheck(AWAKE_MS);
    }
  }
}

_scheduleIdleCheck(IDLE_MS);

// ── Start ─────────────────────────────────────────────────────────────────────

visual.start();
socket.connect();

console.log("[main] Shiver 2 running — click canvas to enable audio");
