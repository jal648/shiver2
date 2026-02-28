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

const canvas = document.getElementById("canvas");
const statusEl = document.getElementById("status");
const debugPanel = document.getElementById("debug-panel");

// ── Instantiate components ────────────────────────────────────────────────────
const socket_url = `ws://${window.location.hostname}:8765`;
const audio = new AudioEngine();
const visual = new Visualizer(canvas);
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

// ── Start ─────────────────────────────────────────────────────────────────────

visual.start();
socket.connect();

console.log("[main] Shiver 2 running — click canvas to enable audio");
