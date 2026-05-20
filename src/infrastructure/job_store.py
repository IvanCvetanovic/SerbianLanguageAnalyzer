from src.infrastructure.task_service import task_service

# Compatibility layer for existing code
jobs = task_service.jobs
progress = task_service.progress
executor = task_service.executor
prune_old_jobs = task_service.prune_jobs
