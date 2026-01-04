"""Log storage for persisting and retrieving task logs."""
import json
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def serialize_log_entry(log_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy types in log entry to Python native types."""
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
    return convert(log_entry)


class LogStore:
    """Stores logs to disk and memory for retrieval."""
    
    def __init__(self, storage_path: str = "./logs"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.memory_logs: Dict[str, List[Dict[str, Any]]] = {}
        self.max_memory_logs = 1000  # Per task
    
    def _get_log_file_path(self, task_id: str) -> Path:
        """Get the file path for a task's log file."""
        return self.storage_path / f"{task_id}.jsonl"
    
    async def append(self, task_id: str, log_entry: Dict[str, Any]):
        """Append a log entry to storage."""
        # Serialize to handle numpy types
        log_entry = serialize_log_entry(log_entry)
        
        # Add to memory
        if task_id not in self.memory_logs:
            self.memory_logs[task_id] = []
        
        self.memory_logs[task_id].append(log_entry)
        
        # Trim memory if needed
        if len(self.memory_logs[task_id]) > self.max_memory_logs:
            self.memory_logs[task_id] = self.memory_logs[task_id][-self.max_memory_logs:]
        
        # Append to file
        log_file = self._get_log_file_path(task_id)
        try:
            async with asyncio.Lock():
                with open(log_file, "a") as f:
                    f.write(json.dumps(log_entry, cls=NumpyEncoder) + "\n")
        except Exception as e:
            print(f"Error writing log to file: {e}")
    
    def append_sync(self, task_id: str, log_entry: Dict[str, Any]):
        """Synchronous wrapper for append."""
        # Serialize to handle numpy types
        log_entry = serialize_log_entry(log_entry)
        
        if task_id not in self.memory_logs:
            self.memory_logs[task_id] = []
        
        self.memory_logs[task_id].append(log_entry)
        
        if len(self.memory_logs[task_id]) > self.max_memory_logs:
            self.memory_logs[task_id] = self.memory_logs[task_id][-self.max_memory_logs:]
        
        log_file = self._get_log_file_path(task_id)
        try:
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry, cls=NumpyEncoder) + "\n")
        except Exception as e:
            print(f"Error writing log to file: {e}")
    
    async def get_logs(
        self,
        task_id: str,
        limit: Optional[int] = None,
        from_memory: bool = True
    ) -> List[Dict[str, Any]]:
        """Retrieve logs for a task."""
        if from_memory and task_id in self.memory_logs:
            logs = self.memory_logs[task_id]
            if limit:
                return logs[-limit:]
            return logs
        
        # Read from file
        log_file = self._get_log_file_path(task_id)
        if not log_file.exists():
            return []
        
        logs = []
        try:
            with open(log_file, "r") as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
        except Exception as e:
            print(f"Error reading log file: {e}")
        
        if limit:
            return logs[-limit:]
        return logs
    
    async def get_logs_text(self, task_id: str) -> str:
        """Retrieve logs as formatted text."""
        from .log_emitter import format_log_text
        
        logs = await self.get_logs(task_id, from_memory=False)
        return "\n".join(format_log_text(log) for log in logs)
    
    async def clear_logs(self, task_id: str):
        """Clear logs for a task."""
        if task_id in self.memory_logs:
            del self.memory_logs[task_id]
        
        log_file = self._get_log_file_path(task_id)
        if log_file.exists():
            log_file.unlink()


# Global log store instance
log_store = LogStore()
