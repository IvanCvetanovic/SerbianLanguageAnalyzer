import sys
import atexit
import signal
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

jobs: dict = {}
progress: dict = {}
executor = ThreadPoolExecutor(max_workers=1)

_JOB_TTL_SECONDS = 3600

atexit.register(executor.shutdown, wait=False, cancel_futures=True)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))


def prune_old_jobs():
    cutoff = datetime.utcnow()
    stale = [
        jid for jid, job in list(jobs.items())
        if (cutoff - job["created_at"]).total_seconds() > _JOB_TTL_SECONDS
    ]
    for jid in stale:
        jobs.pop(jid, None)
        progress.pop(jid, None)
