"""Mupin API Backbone — v0.3 generic job queue service.

Exposes:
  POST /jobs                 submit a job
  GET  /jobs/{job_id}        status + result
  POST /jobs/{job_id}/cancel request cooperative cancellation
  GET  /jobs/{job_id}/log    return worker thought log
  POST /internal/jobs/{job_id}/progress  worker heartbeat (internal)
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import PlainTextResponse

from src.config import REDIS_URL, BACKBONE_PORT
from src.db import create_tables, async_session
from src.jobs import (
    create_job,
    get_job,
    mark_job_running,
    request_cancel,
    update_job_progress,
    finalize_job,
)
from src.models import JobResponse, JobSubmit, JobProgress
from src.queue import get_redis_pool, enqueue_job


WORKSPACE_ROOT = Path(os.environ.get("WORKSPACE_ROOT", "/shared/workspaces"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[dict, None]:
    await create_tables()
    app.state.redis_pool = await get_redis_pool()
    yield {"redis_pool": app.state.redis_pool}
    await app.state.redis_pool.close()


app = FastAPI(title="Mupin API Backbone", version="0.3.0", lifespan=lifespan)


def _job_to_response(job) -> JobResponse:
    return JobResponse(
        job_id=job.id,
        job_type=job.job_type,
        status=job.status,
        worker_id=job.worker_id,
        payload=job.payload,
        result=job.result,
        error=job.error,
        progress=job.progress,
        cancel_requested=job.cancel_requested,
        created_at=job.created_at,
        started_at=job.started_at,
        updated_at=job.updated_at,
    )


@app.post("/jobs", response_model=JobResponse)
async def submit_job(request: JobSubmit):
    async with async_session() as session:
        job = await create_job(session, request.job_type, request.payload)
        await enqueue_job(app.state.redis_pool, request.job_type, job.id, request.payload)
        return _job_to_response(job)


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: str):
    async with async_session() as session:
        job = await get_job(session, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return _job_to_response(job)


@app.post("/jobs/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(job_id: str):
    async with async_session() as session:
        job = await request_cancel(session, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        # Also ask ARQ to abort the job if it is still queued/running.
        try:
            from arq.jobs import Job as ArqJob
            arq_job = ArqJob(job_id, redis=app.state.redis_pool)
            await arq_job.abort()
        except Exception:
            pass
        return _job_to_response(job)


@app.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
async def get_job_log(job_id: str):
    log_path = WORKSPACE_ROOT / job_id / "task.log"
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log not found")
    return log_path.read_text(encoding="utf-8")


@app.post("/internal/jobs/{job_id}/started", response_model=JobResponse)
async def internal_started(job_id: str):
    async with async_session() as session:
        job = await get_job(session, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        await mark_job_running(session, job_id, worker_id="worker")
        await session.refresh(job)
        return _job_to_response(job)


@app.post("/internal/jobs/{job_id}/progress", response_model=JobResponse)
async def internal_progress(job_id: str, progress: JobProgress):
    async with async_session() as session:
        job = await get_job(session, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        await update_job_progress(
            session,
            job_id,
                {
                    "current_node": progress.current_node,
                    "sandbox_loop_count": progress.sandbox_loop_count,
                    "thoughts": progress.thoughts,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                },
        )
        await session.refresh(job)
        return _job_to_response(job)


@app.post("/internal/jobs/{job_id}/finalize", response_model=JobResponse)
async def internal_finalize(job_id: str, status: str, result: dict | None = None, error: str | None = None):
    async with async_session() as session:
        job = await get_job(session, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        await finalize_job(session, job_id, status, result, error)
        await session.refresh(job)
        return _job_to_response(job)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=BACKBONE_PORT)
