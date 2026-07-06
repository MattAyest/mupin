from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import Job
from src.models import JobProgress


def now_utc():
    return datetime.now(timezone.utc)


async def create_job(session: AsyncSession, job_type: str, payload: dict) -> Job:
    job = Job(job_type=job_type, payload=payload, status="queued")
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def get_job(session: AsyncSession, job_id: str) -> Optional[Job]:
    return await session.get(Job, job_id)


async def mark_job_running(
    session: AsyncSession,
    job_id: str,
    worker_id: str,
) -> None:
    await session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(status="running", worker_id=worker_id, started_at=now_utc(), updated_at=now_utc())
    )
    await session.commit()


async def update_job_progress(
    session: AsyncSession,
    job_id: str,
    progress: dict,
) -> None:
    await session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(progress=progress, updated_at=now_utc())
    )
    await session.commit()


async def request_cancel(session: AsyncSession, job_id: str) -> Optional[Job]:
    await session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(cancel_requested=True, updated_at=now_utc())
    )
    await session.commit()
    return await session.get(Job, job_id)


async def is_cancel_requested(session: AsyncSession, job_id: str) -> bool:
    job = await session.get(Job, job_id)
    return bool(job and job.cancel_requested)


async def finalize_job(
    session: AsyncSession,
    job_id: str,
    status: str,
    result: Optional[dict] = None,
    error: Optional[str] = None,
) -> None:
    await session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(status=status, result=result, error=error, updated_at=now_utc())
    )
    await session.commit()
