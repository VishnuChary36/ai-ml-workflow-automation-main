"""Log emission and streaming utilities."""
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import defaultdict


class LogEmitter:
    """Emits structured logs to connected WebSocket clients."""
    
    def __init__(self):
        self.connections: Dict[str, List[asyncio.Queue]] = defaultdict(list)
    
    def subscribe(self, task_id: str, queue: asyncio.Queue):
        """Subscribe a queue to receive logs for a task."""
        self.connections[task_id].append(queue)
    
    def unsubscribe(self, task_id: str, queue: asyncio.Queue):
        """Unsubscribe a queue from task logs."""
        if task_id in self.connections:
            try:
                self.connections[task_id].remove(queue)
            except ValueError:
                pass
            
            if not self.connections[task_id]:
                del self.connections[task_id]
    
    async def emit(self, task_id: str, log_entry: Dict[str, Any]):
        """Emit a log entry to all subscribers of a task."""
        if task_id not in self.connections:
            return
        
        # Add timestamp if not present
        if "timestamp" not in log_entry:
            log_entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        if "task_id" not in log_entry:
            log_entry["task_id"] = task_id
        
        # Send to all subscribers
        for queue in self.connections[task_id]:
            try:
                await queue.put(log_entry)
            except Exception as e:
                print(f"Error emitting log to queue: {e}")
    
    def emit_sync(self, task_id: str, log_entry: Dict[str, Any]):
        """Synchronous wrapper for emit (for use in non-async contexts)."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.emit(task_id, log_entry))
            else:
                loop.run_until_complete(self.emit(task_id, log_entry))
        except Exception as e:
            print(f"Error in sync emit: {e}")
    
    def get_subscriber_count(self, task_id: str) -> int:
        """Get number of active subscribers for a task."""
        return len(self.connections.get(task_id, []))


# Global log emitter instance
log_emitter = LogEmitter()


def create_log_entry(
    level: str,
    message: str,
    source: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a structured log entry."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "level": level.upper(),
        "message": message,
    }
    
    if task_id:
        entry["task_id"] = task_id
    
    if source:
        entry["source"] = source
    
    if meta:
        entry["meta"] = meta
    
    return entry


def format_log_text(log_entry: Dict[str, Any]) -> str:
    """Format a log entry as human-readable text."""
    timestamp = log_entry.get("timestamp", "")
    if timestamp and "." in timestamp:
        # Truncate microseconds for readability
        timestamp = timestamp.split(".")[0] + "Z"
    
    level = log_entry.get("level", "INFO")
    source = log_entry.get("source", "")
    message = log_entry.get("message", "")
    
    if source:
        return f"{level} {timestamp} | {source} | {message}"
    else:
        return f"{level} {timestamp} | {message}"
