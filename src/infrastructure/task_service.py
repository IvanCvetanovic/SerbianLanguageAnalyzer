import sys
import atexit
import signal
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from typing import Dict, Any, Callable, Optional


class TaskService:
    def __init__(self, max_workers: int = 1, job_ttl_seconds: int = 3600):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.progress: Dict[str, Dict[str, Any]] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.job_ttl_seconds = job_ttl_seconds

        # Ensure clean shutdown
        atexit.register(self.executor.shutdown, wait=False, cancel_futures=True)
        # Handle termination signals
        try:
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        except ValueError:
            # Signal only works in main thread
            pass

    def submit_task(self, fn: Callable, *args, **kwargs) -> str:
        """Submit a background task and return its job_id."""
        self.prune_jobs()
        job_id = str(uuid.uuid4())
        
        # Initialize progress and job metadata
        self.progress[job_id] = {"pct": 0, "stage": "Queued", "status": "running", "sections": {}}
        
        # We pass the progress dict and job_id to the function if it supports them
        # In this project, run_analysis expects (input_text, features, job_id, progress)
        future = self.executor.submit(fn, *args, **kwargs, job_id=job_id, progress=self.progress)
        
        self.jobs[job_id] = {
            "future": future,
            "status": "running",
            "result": None,
            "created_at": datetime.utcnow(),
            "error": None,
        }
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def get_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.progress.get(job_id)

    def update_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Check if job is done and update its internal result cache."""
        job = self.jobs.get(job_id)
        if not job:
            return None
        
        future: Future = job["future"]
        if future.done() and job.get("result") is None:
            try:
                job["result"] = future.result()
                job["status"] = "finished"
                
                # Update progress state to finished if not already failed
                prog = self.progress.get(job_id)
                if prog and prog.get("status") != "failed":
                    prog.update({"status": "finished", "pct": 100, "stage": "Finished"})
            except Exception as e:
                job["status"] = "failed"
                job["error"] = str(e)
                prog = self.progress.get(job_id)
                if prog:
                    prog.update({"status": "failed", "stage": "Error", "error_message": str(e)})
        
        return job

    def prune_jobs(self):
        cutoff = datetime.utcnow()
        stale = [
            jid for jid, job in list(self.jobs.items())
            if (cutoff - job["created_at"]).total_seconds() > self.job_ttl_seconds
        ]
        for jid in stale:
            self.jobs.pop(jid, None)
            self.progress.pop(jid, None)


# Singleton instance
task_service = TaskService()
