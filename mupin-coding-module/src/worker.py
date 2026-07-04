"""ARQ worker for the Mupin Coding Module.

Consumes `coding` jobs from the Mupin API Backbone, drives the v0.2 LangGraph
pipeline, reports progress back to the backbone, and writes task workspaces
to the shared workspace volume.
"""

import asyncio
import os
from pathlib import Path

import httpx
from arq.connections import RedisSettings

from dotenv import load_dotenv

load_dotenv()

from .runner import run_swarm_task


BACKBONE_URL = os.environ.get("BACKBONE_URL", "http://mupin-api-backbone:8000")
WORKER_CONCURRENCY = int(os.environ.get("WORKER_MAX_JOBS", "4"))
WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE_ROOT", "/app/.workspaces"))


def _redis_settings() -> RedisSettings:
    import urllib.parse
    url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
    parsed = urllib.parse.urlparse(url)
    return RedisSettings(
        host=parsed.hostname or "localhost",
        port=parsed.port or 6379,
        database=int(parsed.path.lstrip("/") or 0),
        password=parsed.password,
    )


async def _post_progress(job_id: str, progress: dict):
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                f"{BACKBONE_URL}/internal/jobs/{job_id}/progress",
                json=progress,
                timeout=10,
            )
        except Exception:
            # Progress posts are best-effort; failures should not kill the task.
            pass


async def _mark_started(job_id: str) -> None:
    """Notify the backbone that this worker has begun processing the job."""
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                f"{BACKBONE_URL}/internal/jobs/{job_id}/started",
                timeout=10,
            )
        except Exception:
            # Best-effort; the job can still run if the backbone is unreachable.
            pass


async def _is_cancelled(job_id: str) -> bool:
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{BACKBONE_URL}/jobs/{job_id}", timeout=5)
            if resp.status_code == 200:
                return bool(resp.json().get("cancel_requested"))
        except Exception:
            pass
    return False


async def _finalize(job_id: str, payload: dict):
    async with httpx.AsyncClient() as client:
        try:
            await client.post(
                f"{BACKBONE_URL}/internal/jobs/{job_id}/finalize",
                params={"status": payload["status"], "error": payload.get("error")},
                json=payload.get("result"),
                timeout=10,
            )
        except Exception:
            pass


async def run_job(ctx, job_type: str, job_id: str, payload: dict) -> dict:
    """ARQ worker function invoked by the backbone queue."""
    if job_type != "coding":
        return {"status": "failed", "error": f"Unknown job type: {job_type}"}

    # Tell the backbone we have picked up this job so queue-wait metrics can
    # be separated from actual processing time.
    await _mark_started(job_id)

    # Cooperative cancellation: if cancel was requested before we started, stop now.
    if await _is_cancelled(job_id):
        await _finalize(job_id, {"status": "cancelled", "error": "Task cancelled before start"})
        return {"status": "cancelled", "error": "Task cancelled before start"}

    prompt = payload.get("prompt", "")
    profile_name = payload.get("profile_name", "python")
    workspace_dir = str(WORKSPACE_ROOT / job_id)

    progress_count = 0
    last_progress = {}

    async def progress_callback(update: dict):
        nonlocal progress_count, last_progress
        progress_count += 1
        last_progress = update
        # Throttle progress posts to avoid spamming the backbone.
        if progress_count % 3 == 0 or update.get("current_node") in (
            "sandbox_arbiter",
            "prompt_compliance_checker",
        ):
            await _post_progress(job_id, update)

    async def cancellation_check() -> bool:
        return await _is_cancelled(job_id)

    try:
        result = await run_swarm_task(
            task_id=job_id,
            prompt=prompt,
            workspace_dir=workspace_dir,
            profile_name=profile_name,
            progress_callback=progress_callback,
            is_cancelled=cancellation_check,
        )
    except asyncio.CancelledError:
        await _finalize(job_id, {"status": "cancelled", "error": "Worker job cancelled"})
        return {"status": "cancelled", "error": "Worker job cancelled"}

    await _post_progress(job_id, {
        "current_node": result.get("current_node", "FINISH"),
        "sandbox_loop_count": result.get("sandbox_loop_count", 0),
        "compliance_loop_count": result.get("compliance_loop_count", 0),
        "compliance_status": result.get("compliance_status"),
        "thoughts": result.get("thoughts", [])[-20:],
    })
    await _finalize(job_id, result)

    return {
        "status": result["status"],
        "result": result.get("result"),
        "error": result.get("error"),
    }


class WorkerSettings:
    functions = [run_job]
    redis_settings = _redis_settings()
    max_jobs = WORKER_CONCURRENCY
    keep_result = 3600  # keep ARQ result in Redis for 1 hour
    result_ttl = 3600
    # Match the per-task wall-clock deadline from llm_config.yaml (default 3600s).
    job_timeout = 3600


if __name__ == "__main__":
    from arq.worker import run_worker
    run_worker(WorkerSettings)
