"""Task management for tracking and controlling background jobs."""
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from sqlalchemy.orm import Session
from core.database import Task
from core.log_emitter import log_emitter, create_log_entry
from core.log_store import log_store


class TaskStatus(str, Enum):
    """Task status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Task type enum."""
    UPLOAD = "upload"
    PROFILE = "profile"
    PREPROCESS = "preprocess"
    TRAIN = "train"
    DEPLOY = "deploy"
    PREDICT = "predict"
    VISUALIZATION = "visualization"


class TaskManager:
    """Manages task lifecycle and status."""

    @staticmethod
    def create_task(
        db: Session,
        task_type: str,
        config: Optional[Dict[str, Any]] = None,
        dataset_id: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> Task:
        """Create a new task."""
        task_id = f"task-{uuid.uuid4().hex[:12]}"

        task = Task(
            id=task_id,
            task_type=task_type,
            status=TaskStatus.PENDING,
            config=config or {},
            dataset_id=dataset_id,
            model_id=model_id,
        )

        db.add(task)
        db.commit()
        db.refresh(task)

        # Emit log
        log_entry = create_log_entry(
            "INFO",
            f"Task created: {task_type}",
            source="task_manager",
            task_id=task_id,
            meta={"task_type": task_type, "dataset_id": dataset_id}
        )
        log_emitter.emit_sync(task_id, log_entry)
        log_store.append_sync(task_id, log_entry)

        return task

    @staticmethod
    def update_status(
        db: Session,
        task_id: str,
        status: TaskStatus,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ):
        """Update task status."""
        task = db.query(Task).filter(Task.id == task_id).first()
        if not task:
            return

        task.status = status

        if status == TaskStatus.RUNNING and not task.started_at:
            task.started_at = datetime.utcnow()

        if status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            task.completed_at = datetime.utcnow()

        if error:
            task.error = error

        if result:
            task.result = result

        db.commit()

        # Emit log
        log_entry = create_log_entry(
            "INFO" if status == TaskStatus.COMPLETED else "ERROR" if status == TaskStatus.FAILED else "WARN",
            f"Task status updated: {status}",
            source="task_manager",
            task_id=task_id,
            meta={"status": status, "error": error}
        )
        log_emitter.emit_sync(task_id, log_entry)
        log_store.append_sync(task_id, log_entry)

    @staticmethod
    def get_task(db: Session, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        return db.query(Task).filter(Task.id == task_id).first()

    @staticmethod
    async def emit_log(
        task_id: str,
        level: str,
        message: str,
        source: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """Emit a log for a task."""
        log_entry = create_log_entry(level, message, source, meta, task_id)
        await log_emitter.emit(task_id, log_entry)
        await log_store.append(task_id, log_entry)

    @staticmethod
    def emit_log_sync(
        task_id: str,
        level: str,
        message: str,
        source: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        """Emit a log for a task (synchronous)."""
        log_entry = create_log_entry(level, message, source, meta, task_id)
        log_emitter.emit_sync(task_id, log_entry)
        log_store.append_sync(task_id, log_entry)
