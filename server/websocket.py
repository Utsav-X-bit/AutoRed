import json
from typing import Dict, Set
from fastapi import WebSocket, WebSocketDisconnect


class WebSocketManager:
    def __init__(self):
        self._connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, run_id: str, websocket: WebSocket):
        await websocket.accept()
        if run_id not in self._connections:
            self._connections[run_id] = set()
        self._connections[run_id].add(websocket)

    def disconnect(self, run_id: str, websocket: WebSocket):
        if run_id in self._connections:
            self._connections[run_id].discard(websocket)
            if not self._connections[run_id]:
                del self._connections[run_id]

    async def send_attempt(self, run_id: str, attempt: dict):
        if run_id not in self._connections:
            return
        message = json.dumps({"type": "attempt_update", "run_id": run_id, "attempt": attempt})
        disconnected = set()
        # Iterate over a copy to avoid "Set changed size during iteration"
        for ws in set(self._connections[run_id]):
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.add(ws)
        self._connections[run_id] -= disconnected

    async def send_run_complete(self, run_id: str, run: dict):
        if run_id not in self._connections:
            return
        message = json.dumps({"type": "run_complete", "run_id": run_id, "run": run})
        for ws in set(self._connections[run_id]):
            try:
                await ws.send_text(message)
            except Exception:
                pass

    async def broadcast(self, run_id: str, data: dict):
        if run_id not in self._connections:
            return
        message = json.dumps(data)
        for ws in set(self._connections[run_id]):
            try:
                await ws.send_text(message)
            except Exception:
                pass


ws_manager = WebSocketManager()
