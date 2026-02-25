"""
WebSocket server — broadcasts block state to all connected browser clients.
"""

import asyncio
import json
import logging

import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


class WebSocketServer:
    """
    Async WebSocket server that keeps a set of connected clients and
    broadcasts JSON messages to all of them.
    """

    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        self._clients: set[WebSocketServerProtocol] = set()
        self._server = None

    # ------------------------------------------------------------------
    async def start(self) -> None:
        """Start the WebSocket server (call from an asyncio event loop)."""
        self._server = await websockets.serve(
            self._handle_client, self.host, self.port
        )
        logger.info("WebSocket server listening on ws://%s:%d", self.host, self.port)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()

    # ------------------------------------------------------------------
    async def _handle_client(self, ws: WebSocketServerProtocol) -> None:
        self._clients.add(ws)
        logger.info("Client connected (%d total)", len(self._clients))
        try:
            async for _ in ws:
                pass  # we don't expect messages from the browser
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(ws)
            logger.info("Client disconnected (%d total)", len(self._clients))

    # ------------------------------------------------------------------
    async def broadcast(self, state: dict) -> None:
        """Serialize state to JSON and send to all connected clients."""
        if not self._clients:
            return
        message = json.dumps(state)
        await asyncio.gather(
            *[ws.send(message) for ws in list(self._clients)],
            return_exceptions=True,
        )
