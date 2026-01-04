"""WebSocket log streaming broker."""
import asyncio
import json
import numpy as np
from typing import Set, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from core.log_emitter import log_emitter
from core.log_store import log_store


def serialize_for_json(obj: Any) -> Any:
    """Convert numpy types and other non-JSON-serializable types to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


async def safe_send_json(websocket: WebSocket, data: dict):
    """Send JSON data through WebSocket with numpy type handling."""
    try:
        safe_data = serialize_for_json(data)
        await websocket.send_json(safe_data)
    except Exception as e:
        print(f"Error in safe_send_json: {e}")
        raise


class ConnectionManager:
    """Manages WebSocket connections for log streaming."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.connection_queues: Dict[WebSocket, asyncio.Queue] = {}
        self.connection_tasks: Dict[WebSocket, asyncio.Task] = {}
    
    async def connect(self, websocket: WebSocket, task_id: str):
        """Accept a WebSocket connection and subscribe to task logs."""
        await websocket.accept()
        
        if task_id not in self.active_connections:
            self.active_connections[task_id] = set()
        
        self.active_connections[task_id].add(websocket)
        
        # Send historical logs
        try:
            logs = await log_store.get_logs(task_id)
            for log in logs:
                await safe_send_json(websocket, log)
        except Exception as e:
            print(f"Error sending historical logs: {e}")
        
        # Subscribe to new logs
        queue = asyncio.Queue()
        log_emitter.subscribe(task_id, queue)
        self.connection_queues[websocket] = queue
        
        # Start forwarding logs from queue to websocket
        task = asyncio.create_task(self._forward_logs(websocket, queue, task_id))
        self.connection_tasks[websocket] = task
    
    async def _forward_logs(self, websocket: WebSocket, queue: asyncio.Queue, task_id: str):
        """Forward logs from queue to WebSocket."""
        try:
            while True:
                try:
                    # Use timeout to periodically check connection state
                    log_entry = await asyncio.wait_for(queue.get(), timeout=30.0)
                    await safe_send_json(websocket, log_entry)
                except asyncio.TimeoutError:
                    # Send ping to keep connection alive
                    try:
                        await safe_send_json(websocket, {"type": "ping"})
                    except Exception:
                        break
        except (WebSocketDisconnect, RuntimeError, Exception) as e:
            print(f"WebSocket forward stopped: {type(e).__name__}")
        finally:
            log_emitter.unsubscribe(task_id, queue)
    
    def disconnect(self, websocket: WebSocket, task_id: str):
        """Handle WebSocket disconnection."""
        # Cancel the forwarding task
        if websocket in self.connection_tasks:
            self.connection_tasks[websocket].cancel()
            del self.connection_tasks[websocket]
        
        # Unsubscribe queue
        if websocket in self.connection_queues:
            queue = self.connection_queues[websocket]
            log_emitter.unsubscribe(task_id, queue)
            del self.connection_queues[websocket]
        
        # Remove from active connections
        if task_id in self.active_connections:
            self.active_connections[task_id].discard(websocket)
            
            if not self.active_connections[task_id]:
                del self.active_connections[task_id]


manager = ConnectionManager()
