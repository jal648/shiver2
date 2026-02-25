/**
 * socket.js — WebSocket client
 *
 * Connects to the Python vision server and dispatches a `blocksUpdate`
 * CustomEvent on `window` whenever a new frame arrives.
 *
 * Event detail shape:
 *   { blocks: Block[], timestamp: number }
 *
 * Block shape:
 *   { id: number, x: number, y: number, w: number, h: number,
 *     vx: number, vy: number }
 */

export class SocketClient {
  /**
   * @param {string} url  WebSocket URL, e.g. "ws://localhost:8765"
   * @param {(status: string) => void} onStatus  Status change callback
   */
  constructor(url = "ws://localhost:8765", onStatus = () => {}) {
    this.url = url;
    this.onStatus = onStatus;
    this._ws = null;
    this._reconnectDelay = 2000;
  }

  connect() {
    this.onStatus("connecting");
    this._ws = new WebSocket(this.url);

    this._ws.addEventListener("open", () => {
      console.log("[socket] Connected to", this.url);
      this.onStatus("connected");
    });

    this._ws.addEventListener("message", (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch (err) {
        console.warn("[socket] Failed to parse message:", err);
        return;
      }
      window.dispatchEvent(new CustomEvent("blocksUpdate", { detail: data }));
    });

    this._ws.addEventListener("close", () => {
      console.log("[socket] Disconnected — retrying in", this._reconnectDelay, "ms");
      this.onStatus("disconnected");
      setTimeout(() => this.connect(), this._reconnectDelay);
    });

    this._ws.addEventListener("error", (err) => {
      console.warn("[socket] Error:", err);
    });
  }

  disconnect() {
    if (this._ws) {
      this._ws.close();
      this._ws = null;
    }
  }
}
