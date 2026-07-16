"""Module registry + backbone client for the Planner.

The registry describes what modules the planner can delegate to.
The backbone client submits jobs and polls for results.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

import httpx

from .config_loader import load_module_registry


BACKBONE_URL = os.environ.get("BACKBONE_URL", "http://mupin-api-backbone:8000")
POLL_INTERVAL = 5
JOB_TIMEOUT = 3600


def get_module_registry() -> list[dict]:
    reg = load_module_registry()
    return reg.get("modules", [])


def get_module_by_name(name: str) -> dict | None:
    for m in get_module_registry():
        if m["name"] == name:
            return m
    return None


def get_registry_context() -> str:
    """Render the module registry as a text block for the planner LLM."""
    lines = []
    for m in get_module_registry():
        lines.append(f"- {m['name']} (job_type: {m['job_type']})")
        lines.append(f"  Description: {m.get('description', '').strip()}")
        lines.append(f"  When to use: {m.get('when_to_use', '').strip()}")
        accepts = m.get("accepts", {})
        accepts_str = ", ".join(f"{k}: {v}" for k, v in accepts.items())
        lines.append(f"  Accepts: {accepts_str}")
        lines.append("")
    return "\n".join(lines)


async def submit_job(job_type: str, payload: dict) -> str:
    """Submit a job to the backbone. Returns the job_id."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{BACKBONE_URL}/jobs",
            json={"job_type": job_type, "payload": payload},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["job_id"]


async def get_job_status(job_id: str) -> dict:
    """Get the current status of a job from the backbone."""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BACKBONE_URL}/jobs/{job_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()


async def cancel_job(job_id: str) -> None:
    async with httpx.AsyncClient() as client:
        try:
            await client.post(f"{BACKBONE_URL}/jobs/{job_id}/cancel", timeout=10)
        except Exception:
            pass


async def poll_job_until_settled(job_id: str, timeout: int = JOB_TIMEOUT) -> dict:
    """Poll a job until it reaches a terminal status. Returns the final job dict."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = await get_job_status(job_id)
        status = job.get("status", "queued")
        if status in ("completed", "failed", "cancelled", "exhausted", "infra_exhausted"):
            return job
        await _sleep(POLL_INTERVAL)
    await cancel_job(job_id)
    return {
        "job_id": job_id,
        "status": "timeout",
        "error": f"Job did not settle within {timeout}s",
    }


async def _sleep(seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)