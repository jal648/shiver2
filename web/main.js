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

// ── Instantiate components ────────────────────────────────────────────────────

const audio = new AudioEngine();
const visual = new Visualizer(canvas);
const socket = new SocketClient("ws://localhost:8765", (status) => {
  statusEl.textContent = status;
  statusEl.className = `status-${status}`;
});

// ── Wire events ───────────────────────────────────────────────────────────────

window.addEventListener("blocksUpdate", (event) => {
  const { blocks } = event.detail;
  audio.update(blocks);
  visual.update(blocks);

  // Debug: log block count on each update
  // console.debug("[main] blocks:", blocks.length);
});

// ── Unlock audio on first user interaction ────────────────────────────────────

let audioUnlocked = false;
function unlockAudio() {
  if (audioUnlocked) return;
  audio.unlock();
  audioUnlocked = true;
  canvas.removeEventListener("click", unlockAudio);
  canvas.removeEventListener("touchstart", unlockAudio);
  console.log("[main] Audio unlocked");
}

canvas.addEventListener("click", unlockAudio);
canvas.addEventListener("touchstart", unlockAudio);

// ── Start ─────────────────────────────────────────────────────────────────────

visual.start();
socket.connect();

console.log("[main] Shiver 2 running — click canvas to enable audio");
