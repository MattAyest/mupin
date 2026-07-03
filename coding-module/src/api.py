"""FastAPI layer for the v0.2 coding module.

Exposes POST /task, GET /task/{id}, GET /task/{id}/log, and POST cancel.
Tasks run as tracked asyncio tasks so they can be cancelled mid-pipeline.
"""

import asyncio
import sys
import uuid
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from .graph import app as swarm_graph
from .nodes import LLMUnavailableError, SERVER_TASK_DEADLINE, validate_config

app = FastAPI(title="Coding Module Microservice — v0.2")

# Surface misconfigured LLM nodes (missing API keys, bad provider names) at
# startup rather than mid-task.
for _node, _err in validate_config():
    print(f"[config] {_node}: {_err}")

# Simple in-memory store for task status.
tasks_db: Dict[str, Dict[str, Any]] = {}

# Live asyncio handles for in-flight tasks, so a task can be cancelled.
running_tasks: Dict[str, "asyncio.Task"] = {}


class TaskRequest(BaseModel):
    prompt: str


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
    thoughts: List[str] = []
    node_history: List[Dict[str, Any]] = []
    llm_usage: List[Dict[str, Any]] = []
    docker_runs: List[Dict[str, Any]] = []
    classifier_history: List[Dict[str, Any]] = []


async def _drive_graph(task_id: str, initial_state: Dict[str, Any]) -> Dict[str, Any]:
    """Consume the v0.2 graph stream, updating live task status as nodes run."""
    final_manifest: Dict[str, Any] = {}
    stream = swarm_graph.astream(initial_state, stream_mode="updates")
    try:
        async for output in stream:
            for node_name, state_update in output.items():
                tasks_db[task_id]["current_node"] = node_name

                if "sandbox_loop_count" in state_update:
                    tasks_db[task_id]["sandbox_loop_count"] = state_update["sandbox_loop_count"]
                if "compliance_loop_count" in state_update:
                    tasks_db[task_id]["compliance_loop_count"] = state_update["compliance_loop_count"]
                if "compliance_status" in state_update:
                    tasks_db[task_id]["compliance_status"] = state_update["compliance_status"]
                if "sandbox_errors" in state_update:
                    tasks_db[task_id]["error"] = state_update["sandbox_errors"]
                if "file_manifest" in state_update:
                    final_manifest = state_update["file_manifest"]
                if "thoughts" in state_update:
                    tasks_db[task_id]["thoughts"].extend(state_update["thoughts"])
                if "node_history" in state_update:
                    tasks_db[task_id]["node_history"].extend(state_update["node_history"])
                if "llm_usage" in state_update:
                    tasks_db[task_id]["llm_usage"].extend(state_update["llm_usage"])
                if "docker_runs" in state_update:
                    tasks_db[task_id]["docker_runs"].extend(state_update["docker_runs"])
                if "classifier_history" in state_update:
                    tasks_db[task_id]["classifier_history"].extend(state_update["classifier_history"])
                if "llm_infra_exhausted" in state_update:
                    tasks_db[task_id]["llm_infra_exhausted"] = state_update["llm_infra_exhausted"]
                if "llm_infra_retries" in state_update:
                    tasks_db[task_id]["llm_infra_retries"] = state_update["llm_infra_retries"]
    finally:
        await stream.aclose()
    return final_manifest


async def run_swarm_task(task_id: str, prompt: str, workspace_dir: str, deadline: float | None = None):
    initial_state = {
        "user_prompt": prompt,
        "workspace_dir": workspace_dir,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "file_manifest": {},
        "sandbox_errors": "",
        "sandbox_diagnostics": {},
        "compliance_status": "",
        "compliance_critique": [],
        "sandbox_loop_count": 0,
        "compliance_loop_count": 0,
        "contract_loop_count": 0,
        "contract_critique": [],
        "contract_exhausted": False,
        "deadline_seconds": deadline,
        "next_node": "test_designer",
        "llm_infra_exhausted": False,
        "llm_infra_retries": {},
        "thoughts": [],
        "node_history": [],
        "llm_usage": [],
        "docker_runs": [],
        "classifier_history": [],
    }

    try:
        final_manifest = await asyncio.wait_for(
            _drive_graph(task_id, initial_state), timeout=deadline or SERVER_TASK_DEADLINE
        )

        tasks_db[task_id]["result"] = final_manifest
        final_node = tasks_db[task_id].get("current_node")

        if final_node == "prompt_compliance_checker" and tasks_db[task_id].get("compliance_status") == "PASS":
            tasks_db[task_id]["status"] = "completed"
        elif tasks_db[task_id].get("llm_infra_exhausted"):
            tasks_db[task_id]["status"] = "infra_exhausted"
            tasks_db[task_id]["error"] = (
                tasks_db[task_id].get("error")
                or "LLM infrastructure retries exhausted"
            )
        elif tasks_db[task_id].get("compliance_status") == "FAIL":
            tasks_db[task_id]["status"] = "exhausted"
            tasks_db[task_id]["error"] = (
                tasks_db[task_id].get("error")
                or "Compliance loop ceiling reached without passing"
            )
        else:
            tasks_db[task_id]["status"] = "exhausted"
            tasks_db[task_id]["error"] = (
                tasks_db[task_id].get("error")
                or "Pipeline terminated without passing compliance"
            )

    except asyncio.TimeoutError:
        tasks_db[task_id]["status"] = "exhausted"
        actual_deadline = deadline or SERVER_TASK_DEADLINE
        tasks_db[task_id]["error"] = f"Server task deadline ({actual_deadline}s) exceeded"
        tasks_db[task_id]["result"] = None
    except asyncio.CancelledError:
        tasks_db[task_id]["status"] = "cancelled"
        tasks_db[task_id]["error"] = "Task cancelled"
        tasks_db[task_id]["result"] = None
        raise
    except LLMUnavailableError as e:
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["error"] = str(e)
        tasks_db[task_id]["result"] = None
        tasks_db[task_id]["llm_usage"].extend(getattr(e, "usage_entries", []))
    except Exception as e:
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["error"] = str(e)
        tasks_db[task_id]["result"] = None
        if hasattr(e, "usage_entries"):
            tasks_db[task_id]["llm_usage"].extend(getattr(e, "usage_entries", []))
    finally:
        running_tasks.pop(task_id, None)


@app.post("/task", response_model=TaskStatusResponse)
async def generate_code(request: TaskRequest):
    """Accepts a code generation prompt, starts the swarm asynchronously,
    and returns a task_id immediately."""
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    workspace_dir = f".workspaces/{task_id}"

    tasks_db[task_id] = {
        "task_id": task_id,
        "status": "running",
        "current_node": "initializing",
        "sandbox_loop_count": 0,
        "compliance_loop_count": 0,
        "workspace": workspace_dir,
        "result": None,
        "error": None,
        "compliance_status": None,
        "llm_infra_exhausted": False,
        "llm_infra_retries": {},
        "thoughts": [],
        "node_history": [],
        "llm_usage": [],
        "docker_runs": [],
        "classifier_history": [],
    }

    running_tasks[task_id] = asyncio.create_task(
        run_swarm_task(task_id, request.prompt, workspace_dir, deadline=SERVER_TASK_DEADLINE)
    )

    return tasks_db[task_id]


@app.post("/task/{task_id}/cancel", response_model=TaskStatusResponse)
async def cancel_task(task_id: str):
    """Cancel an in-flight task. Idempotent — a settled task is returned as-is."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    handle = running_tasks.get(task_id)
    if handle is not None and not handle.done():
        handle.cancel()
        tasks_db[task_id]["status"] = "cancelled"

    return tasks_db[task_id]


@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Retrieve the status of a given task_id."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    return tasks_db[task_id]


@app.get("/task/{task_id}/log", response_class=PlainTextResponse)
async def get_task_log(task_id: str):
    """Return the thought log as plain text — one line per node action."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    return "\n".join(tasks_db[task_id].get("thoughts", []))
