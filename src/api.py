import asyncio
import uuid
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from .graph import app as swarm_graph
from .nodes import validate_config

app = FastAPI(title="Coding Module Microservice")

# Surface misconfigured LLM nodes (missing API keys, bad provider names) at
# startup rather than mid-task. Logs warnings but does not crash — a node
# that fails here will still raise its own ValueError when first invoked.
for _node, _err in validate_config():
    print(f"[config] {_node}: {_err}")

# Simple in-memory store for task status.
# For production, you might want to use Redis or a database.
tasks_db: Dict[str, Dict[str, Any]] = {}

# Live asyncio handles for in-flight tasks, so a task can be cancelled (e.g. the
# benchmark runner cancelling on timeout instead of leaving it running and
# contending with the next task). Cleared when the task settles.
running_tasks: Dict[str, "asyncio.Task"] = {}


class TaskRequest(BaseModel):
    prompt: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    current_node: str | None = None
    loop_count: int = 0
    regression_count: int = 0
    replan_count: int = 0
    workspace: str
    result: Dict[str, Any] | None = None
    error: str | None = None
    latest_verification_error: str | None = None
    thoughts: List[str] = []


async def run_swarm_task(task_id: str, prompt: str, workspace_dir: str):
    initial_state = {
        "messages": [HumanMessage(content=prompt)],
        "workspace_dir": workspace_dir,
    }

    try:
        # Run the graph and stream updates to capture intermediate state
        final_manifest = {}
        async for output in swarm_graph.astream(initial_state, stream_mode="updates"):
            for node_name, state_update in output.items():
                tasks_db[task_id]["current_node"] = node_name

                # Capture loop and error tracking metrics from the state update
                if "loop_count" in state_update:
                    tasks_db[task_id]["loop_count"] = state_update["loop_count"]
                if "regression_count" in state_update:
                    tasks_db[task_id]["regression_count"] = state_update[
                        "regression_count"
                    ]
                if "replan_count" in state_update:
                    tasks_db[task_id]["replan_count"] = state_update["replan_count"]
                if "verification_errors" in state_update:
                    tasks_db[task_id]["latest_verification_error"] = state_update[
                        "verification_errors"
                    ]
                if "file_manifest" in state_update:
                    final_manifest = state_update["file_manifest"]
                if "thoughts" in state_update:
                    tasks_db[task_id]["thoughts"].extend(state_update["thoughts"])

        # When done, update the task in the "database"
        tasks_db[task_id]["status"] = "completed"
        tasks_db[task_id]["result"] = final_manifest

    except asyncio.CancelledError:
        # Cancelled via /task/{id}/cancel. Cancellation lands at the next node
        # boundary (between astream yields), so the task stops promptly rather
        # than running on as an orphan.
        tasks_db[task_id]["status"] = "cancelled"
        tasks_db[task_id]["error"] = "Task cancelled"
        tasks_db[task_id]["result"] = None
        raise
    except Exception as e:
        tasks_db[task_id]["status"] = "failed"
        tasks_db[task_id]["error"] = str(e)
        tasks_db[task_id]["result"] = None
    finally:
        running_tasks.pop(task_id, None)


@app.post("/task", response_model=TaskStatusResponse)
async def generate_code(request: TaskRequest):
    """
    Accepts a code generation prompt, starts the swarm asynchronously,
    and returns a task_id immediately.
    """
    task_id = f"task_{uuid.uuid4().hex[:8]}"
    workspace_dir = f".workspaces/{task_id}"

    # Initialize task status
    tasks_db[task_id] = {
        "task_id": task_id,
        "status": "running",
        "current_node": "initializing",
        "loop_count": 0,
        "regression_count": 0,
        "replan_count": 0,
        "workspace": workspace_dir,
        "result": None,
        "error": None,
        "latest_verification_error": None,
        "thoughts": [],
    }

    # Trigger the LangGraph execution as a tracked asyncio task so it can be
    # cancelled (BackgroundTasks gives no handle, which let timed-out tasks run on).
    running_tasks[task_id] = asyncio.create_task(
        run_swarm_task(task_id, request.prompt, workspace_dir)
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
        # Reflect intent immediately; run_swarm_task's CancelledError handler
        # also sets this once cancellation is actually delivered.
        tasks_db[task_id]["status"] = "cancelled"

    return tasks_db[task_id]


@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Retrieves the status of a given task_id.
    If completed, the 'result' field will contain the generated files.
    """
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")

    return tasks_db[task_id]


@app.get("/task/{task_id}/log", response_class=PlainTextResponse)
async def get_task_log(task_id: str):
    """Returns the thought log as plain text — one line per node action."""
    if task_id not in tasks_db:
        raise HTTPException(status_code=404, detail="Task not found")
    return "\n".join(tasks_db[task_id].get("thoughts", []))
