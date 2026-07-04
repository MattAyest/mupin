"""FastAPI layer for the v0.3 coding module.

In v0.3 this becomes a thin dev-convenience proxy. All real task execution is
offloaded to the Mupin API Backbone via ARQ. The local endpoints keep the same
contract as before so existing clients and benchmarks can opt-in gradually.
"""

import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import httpx

from .nodes import validate_config

BACKBONE_URL = os.environ.get("BACKBONE_URL", "http://mupin-api-backbone:8000")

app = FastAPI(title="Coding Module Microservice — v0.3 (dev proxy + backbone aliases)")

# Surface misconfigured LLM nodes at startup.
for _node, _err in validate_config():
    print(f"[config] {_node}: {_err}")


class TaskRequest(BaseModel):
    prompt: str
    language: str | None = "python"


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    current_node: str | None = None
    sandbox_loop_count: int = 0
    compliance_loop_count: int = 0
    workspace: str
    result: Dict[str, Any] | None = None
    error: str | None = None
    compliance_status: str | None = None
    llm_infra_exhausted: bool = False
    thoughts: list[str] = []
    node_history: list[Dict[str, Any]] = []
    llm_usage: list[Dict[str, Any]] = []
    docker_runs: list[Dict[str, Any]] = []
    classifier_history: list[Dict[str, Any]] = []


class JobSubmit(BaseModel):
    job_type: str
    payload: dict


def _map_job_to_task(job: Dict[str, Any]) -> Dict[str, Any]:
    """Backbone job shape is a superset of the legacy task response."""
    progress = job.get("progress") or {}
    return {
        "task_id": job["job_id"],
        "status": job["status"],
        "current_node": progress.get("current_node", job.get("status")),
        "sandbox_loop_count": progress.get("sandbox_loop_count", 0),
        "compliance_loop_count": progress.get("compliance_loop_count", 0),
        "workspace": f".workspaces/{job['job_id']}",
        "result": job.get("result"),
        "error": job.get("error"),
        "compliance_status": progress.get("compliance_status"),
        "llm_infra_exhausted": False,
        "thoughts": progress.get("thoughts", []),
        "node_history": [],
        "llm_usage": [],
        "docker_runs": [],
        "classifier_history": [],
    }


@app.post("/task", response_model=TaskStatusResponse)
async def generate_code(request: TaskRequest):
    """Submit a code generation prompt to the backbone and return immediately."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                f"{BACKBONE_URL}/jobs",
                json={
                    "job_type": "coding",
                    "payload": {
                        "prompt": request.prompt,
                        "profile_name": request.language or "python",
                    },
                },
                timeout=10,
            )
            resp.raise_for_status()
            return _map_job_to_task(resp.json())
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Backbone unavailable: {e}")


@app.post("/task/{task_id}/cancel", response_model=TaskStatusResponse)
async def cancel_task(task_id: str):
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f"{BACKBONE_URL}/jobs/{task_id}/cancel", timeout=10)
            resp.raise_for_status()
            return _map_job_to_task(resp.json())
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Backbone unavailable: {e}")


@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{BACKBONE_URL}/jobs/{task_id}/log", timeout=10)
            resp.raise_for_status()
            return resp.text
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise HTTPException(status_code=404, detail="Log not found")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Backbone unavailable: {e}")


# -----------------------------------------------------------------------------
# Backbone-compatible endpoints so the same client code works on either port.
# -----------------------------------------------------------------------------

@app.post("/jobs")
async def proxy_submit_job(request: JobSubmit):
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f"{BACKBONE_URL}/jobs", json=request.model_dump(), timeout=10)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Backbone unavailable: {e}")


@app.get("/jobs/{job_id}")
async def proxy_get_job(job_id: str):
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{BACKBONE_URL}/jobs/{job_id}", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise HTTPException(status_code=404, detail="Job not found")
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Backbone unavailable: {e}")


@app.post("/jobs/{job_id}/cancel")
async def proxy_cancel_job(job_id: str):
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(f"{BACKBONE_URL}/jobs/{job_id}/cancel", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=e.response.text)
        except httpx.RequestError as e:
            raise HTTPException(status_code=503, detail=f"Backbone unavailable: {e}")


@app.get("/jobs/{job_id}/log", response_class=PlainTextResponse)
async def proxy_get_job_log(job_id: str):
    log_path = Path(f".workspaces/{job_id}/task.log")
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log not found")
    return log_path.read_text(encoding="utf-8")


@app.get("/task/{task_id}/log", response_class=PlainTextResponse)
async def get_task_log(task_id: str):
    log_path = Path(f".workspaces/{task_id}/task.log")
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log not found")
    return log_path.read_text(encoding="utf-8")
