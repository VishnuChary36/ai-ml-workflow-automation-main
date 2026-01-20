"""
Pipeline Scheduler

Provides scheduling capabilities for ML pipelines using Prefect schedules
or a standalone scheduler.
"""

import os
import json
import time
import uuid
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

try:
    from prefect import serve
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule, IntervalSchedule
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False

from config import settings


class ScheduledJob:
    """Represents a scheduled job."""
    
    def __init__(
        self,
        job_id: str,
        name: str,
        flow_name: str,
        schedule: str,
        parameters: Dict[str, Any],
        enabled: bool = True,
    ):
        self.job_id = job_id
        self.name = name
        self.flow_name = flow_name
        self.schedule = schedule
        self.parameters = parameters
        self.enabled = enabled
        self.last_run: Optional[datetime] = None
        self.next_run: Optional[datetime] = None
        self.run_count: int = 0
        self.created_at = datetime.utcnow()


class PipelineScheduler:
    """
    Pipeline scheduling manager.
    
    Features:
    - Cron-based scheduling
    - Interval scheduling
    - One-time scheduled runs
    - Schedule management (create, update, delete)
    - Run history tracking
    """
    
    def __init__(self, schedules_path: Optional[str] = None):
        """
        Initialize scheduler.
        
        Args:
            schedules_path: Path to store schedule data
        """
        self.schedules_path = Path(
            schedules_path or 
            os.getenv("SCHEDULES_PATH", "./schedules")
        )
        self.schedules_path.mkdir(parents=True, exist_ok=True)
        
        self.jobs: Dict[str, ScheduledJob] = {}
        self.run_history: List[Dict[str, Any]] = []
        
        # Load existing schedules
        self._load_schedules()
        
        # Background scheduler thread
        self._scheduler_thread: Optional[threading.Thread] = None
        self._running = False
    
    def _load_schedules(self):
        """Load schedules from disk."""
        index_file = self.schedules_path / "schedules.json"
        if index_file.exists():
            with open(index_file) as f:
                data = json.load(f)
                for job_data in data.get("jobs", []):
                    job = ScheduledJob(**job_data)
                    self.jobs[job.job_id] = job
    
    def _save_schedules(self):
        """Save schedules to disk."""
        index_file = self.schedules_path / "schedules.json"
        data = {
            "jobs": [
                {
                    "job_id": job.job_id,
                    "name": job.name,
                    "flow_name": job.flow_name,
                    "schedule": job.schedule,
                    "parameters": job.parameters,
                    "enabled": job.enabled,
                }
                for job in self.jobs.values()
            ]
        }
        with open(index_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def create_schedule(
        self,
        name: str,
        flow_name: str,
        schedule: str,
        parameters: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ) -> ScheduledJob:
        """
        Create a new scheduled job.
        
        Args:
            name: Job name
            flow_name: Name of flow to run
            schedule: Cron expression or interval (e.g., "0 0 * * *" or "1h")
            parameters: Flow parameters
            enabled: Whether schedule is enabled
            
        Returns:
            Created job
        """
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        
        job = ScheduledJob(
            job_id=job_id,
            name=name,
            flow_name=flow_name,
            schedule=schedule,
            parameters=parameters or {},
            enabled=enabled,
        )
        
        # Calculate next run
        job.next_run = self._calculate_next_run(schedule)
        
        self.jobs[job_id] = job
        self._save_schedules()
        
        print(f"✓ Created schedule: {name} ({schedule})")
        return job
    
    def _calculate_next_run(self, schedule: str) -> datetime:
        """Calculate next run time from schedule."""
        now = datetime.utcnow()
        
        # Check if interval format (e.g., "1h", "30m", "1d")
        if schedule.endswith('m'):
            minutes = int(schedule[:-1])
            return now + timedelta(minutes=minutes)
        elif schedule.endswith('h'):
            hours = int(schedule[:-1])
            return now + timedelta(hours=hours)
        elif schedule.endswith('d'):
            days = int(schedule[:-1])
            return now + timedelta(days=days)
        
        # Assume cron format - simplified parsing
        # Full cron parsing would use croniter library
        parts = schedule.split()
        if len(parts) == 5:
            minute, hour = int(parts[0]), int(parts[1])
            next_run = now.replace(minute=minute, second=0, microsecond=0)
            
            if parts[1] != '*':
                next_run = next_run.replace(hour=hour)
            
            if next_run <= now:
                next_run += timedelta(days=1)
            
            return next_run
        
        # Default: run in 1 hour
        return now + timedelta(hours=1)
    
    def update_schedule(
        self,
        job_id: str,
        schedule: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        enabled: Optional[bool] = None,
    ) -> Optional[ScheduledJob]:
        """
        Update an existing schedule.
        
        Args:
            job_id: Job ID to update
            schedule: New schedule
            parameters: New parameters
            enabled: Enable/disable flag
            
        Returns:
            Updated job or None
        """
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        if schedule is not None:
            job.schedule = schedule
            job.next_run = self._calculate_next_run(schedule)
        
        if parameters is not None:
            job.parameters = parameters
        
        if enabled is not None:
            job.enabled = enabled
        
        self._save_schedules()
        return job
    
    def delete_schedule(self, job_id: str) -> bool:
        """Delete a schedule."""
        if job_id in self.jobs:
            del self.jobs[job_id]
            self._save_schedules()
            return True
        return False
    
    def get_schedule(self, job_id: str) -> Optional[ScheduledJob]:
        """Get a schedule by ID."""
        return self.jobs.get(job_id)
    
    def list_schedules(
        self,
        enabled_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        List all schedules.
        
        Args:
            enabled_only: Only return enabled schedules
            
        Returns:
            List of schedule info dicts
        """
        jobs = self.jobs.values()
        
        if enabled_only:
            jobs = [j for j in jobs if j.enabled]
        
        return [
            {
                "job_id": job.job_id,
                "name": job.name,
                "flow_name": job.flow_name,
                "schedule": job.schedule,
                "enabled": job.enabled,
                "last_run": job.last_run.isoformat() if job.last_run else None,
                "next_run": job.next_run.isoformat() if job.next_run else None,
                "run_count": job.run_count,
            }
            for job in jobs
        ]
    
    def trigger_job(self, job_id: str) -> Dict[str, Any]:
        """
        Manually trigger a job.
        
        Args:
            job_id: Job ID to trigger
            
        Returns:
            Run result
        """
        job = self.jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        return self._run_job(job)
    
    def _run_job(self, job: ScheduledJob) -> Dict[str, Any]:
        """Execute a scheduled job."""
        run_id = f"run-{uuid.uuid4().hex[:8]}"
        run_start = datetime.utcnow()
        
        result = {
            "run_id": run_id,
            "job_id": job.job_id,
            "job_name": job.name,
            "flow_name": job.flow_name,
            "started_at": run_start.isoformat(),
            "status": "running",
        }
        
        try:
            # Import and run the flow
            from . import flows
            
            flow_func = getattr(flows, job.flow_name, None)
            if not flow_func:
                raise ValueError(f"Flow not found: {job.flow_name}")
            
            # Execute flow
            flow_result = flow_func(**job.parameters)
            
            result["status"] = "completed"
            result["result"] = flow_result
            result["completed_at"] = datetime.utcnow().isoformat()
            
        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            result["failed_at"] = datetime.utcnow().isoformat()
        
        # Update job
        job.last_run = run_start
        job.run_count += 1
        job.next_run = self._calculate_next_run(job.schedule)
        
        # Add to history
        self.run_history.append(result)
        if len(self.run_history) > 1000:
            self.run_history = self.run_history[-500:]
        
        self._save_schedules()
        
        return result
    
    def get_run_history(
        self,
        job_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get run history.
        
        Args:
            job_id: Filter by job ID
            limit: Maximum results
            
        Returns:
            List of run results
        """
        history = self.run_history
        
        if job_id:
            history = [r for r in history if r.get("job_id") == job_id]
        
        return list(reversed(history[-limit:]))
    
    def start(self):
        """Start the background scheduler."""
        if self._running:
            return
        
        self._running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            daemon=True
        )
        self._scheduler_thread.start()
        print("✓ Pipeline scheduler started")
    
    def stop(self):
        """Stop the background scheduler."""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
        print("✓ Pipeline scheduler stopped")
    
    def _scheduler_loop(self):
        """Background scheduler loop."""
        while self._running:
            try:
                now = datetime.utcnow()
                
                for job in self.jobs.values():
                    if not job.enabled:
                        continue
                    
                    if job.next_run and job.next_run <= now:
                        print(f"Running scheduled job: {job.name}")
                        try:
                            self._run_job(job)
                        except Exception as e:
                            print(f"Scheduled job failed: {job.name} - {e}")
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"Scheduler error: {e}")
                time.sleep(60)
    
    def create_prefect_deployment(
        self,
        flow_name: str,
        name: str,
        schedule: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a Prefect deployment for a flow.
        
        Args:
            flow_name: Name of flow to deploy
            name: Deployment name
            schedule: Cron schedule
            parameters: Default parameters
            
        Returns:
            Deployment info
        """
        if not PREFECT_AVAILABLE:
            raise RuntimeError("Prefect is not installed")
        
        from . import flows
        
        flow_func = getattr(flows, flow_name, None)
        if not flow_func:
            raise ValueError(f"Flow not found: {flow_name}")
        
        # Create schedule
        cron_schedule = CronSchedule(cron=schedule)
        
        # Create deployment
        deployment = Deployment.build_from_flow(
            flow=flow_func,
            name=name,
            schedule=cron_schedule,
            parameters=parameters or {},
        )
        
        # Apply deployment
        deployment.apply()
        
        return {
            "flow_name": flow_name,
            "deployment_name": name,
            "schedule": schedule,
            "parameters": parameters,
            "status": "deployed",
        }


# Global scheduler instance
pipeline_scheduler = PipelineScheduler()
