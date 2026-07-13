from arq import create_pool
from arq.connections import RedisSettings
from arq.jobs import Job as ArqJob

from src.config import REDIS_URL


# Map job types to dedicated ARQ queues. This keeps the coding and editing
# workers from stealing each other's jobs when both register a function named
# run_job.
DEFAULT_QUEUE = "arq:queue"
JOB_TYPE_QUEUES = {
    "coding": DEFAULT_QUEUE,
    "editing": "arq:queue:editing",
}


def _redis_settings_from_url(url: str) -> RedisSettings:
    # Supports redis://host:port/db or redis://host:port/db?password=...
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6379
    database = int(parsed.path.lstrip("/") or 0)
    password = parsed.password
    return RedisSettings(host=host, port=port, database=database, password=password)


async def get_redis_pool():
    return await create_pool(_redis_settings_from_url(REDIS_URL))


async def enqueue_job(pool, job_type: str, job_id: str, payload: dict) -> ArqJob:
    queue_name = JOB_TYPE_QUEUES.get(job_type, DEFAULT_QUEUE)
    return await pool.enqueue_job(
        "run_job",
        job_type,
        job_id,
        payload,
        _job_id=job_id,
        _queue_name=queue_name,
    )
