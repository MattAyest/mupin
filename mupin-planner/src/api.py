"""FastAPI layer for the Planner module.

The planner is a service (not an ARQ worker). It exposes:

    POST   /plan            — submit a goal, get back workflow_id (+ first question)
    GET    /plan/{id}       — check workflow status
    POST   /plan/{id}/answer — answer a clarifying question or failure prompt
    POST   /plan/{id}/cancel — cancel a running workflow

Workflow state is stored in-memory (v0.1). Future versions may persist to Redis.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config_loader import load_env_hierarchy

load_env_hierarchy()

from .graph import app as planner_graph
from .state import PlannerState

app = FastAPI(title="Mupin Planner — v0.1")

# In-memory workflow store: workflow_id -> {state, status, task}
_workflows: dict[str, dict] = {}


class PlanRequest(BaseModel):
    goal: str


class AnswerRequest(BaseModel):
    answer: str


def _new_state(workflow_id: str, goal: str) -> PlannerState:
    return {
        "workflow_id": workflow_id,
        "goal": goal,
        "pending_question": "",
        "operator_answers": [],
        "project_structure": {},
        "steps": [],
        "current_step_index": 0,
        "current_job_id": "",
        "poll_started_at": 0.0,
        "next_node": "clarify",
        "thoughts": [],
        "llm_usage": [],
        "node_history": [],
    }


async def _run_graph(workflow_id: str) -> None:
    """Drive the planner graph until it reaches a terminal state or ASK_OPERATOR."""
    wf = _workflows.get(workflow_id)
    if not wf:
        return
    state: PlannerState = wf["state"]
    wf["status"] = "running"

    try:
        result_state: PlannerState | None = None
        async for output in planner_graph.astream(state, stream_mode="values"):
            result_state = output
            wf["state"] = output
            # Check if we paused to ask the operator.
            if output.get("next_node") == "ASK_OPERATOR":
                wf["status"] = "waiting_for_operator"
                wf["state"]["pending_question"] = output.get("pending_question", "")
                return
            # Check if we're done.
            if output.get("next_node") == "FINISH":
                wf["status"] = "completed"
                return
        # Stream ended without explicit ASK_OPERATOR or FINISH.
        if result_state and result_state.get("next_node") == "FINISH":
            wf["status"] = "completed"
        elif result_state and result_state.get("next_node") == "ASK_OPERATOR":
            wf["status"] = "waiting_for_operator"
        else:
            wf["status"] = "completed"
    except Exception as e:
        wf["status"] = "failed"
        wf["error"] = str(e)


@app.post("/plan")
async def create_plan(request: PlanRequest):
    workflow_id = f"wf_{uuid.uuid4().hex[:12]}"
    state = _new_state(workflow_id, request.goal)
    _workflows[workflow_id] = {"state": state, "status": "starting", "error": ""}

    # Start the graph in the background.
    task = asyncio.create_task(_run_graph(workflow_id))
    _workflows[workflow_id]["task"] = task

    # Wait briefly to see if it immediately asks a question.
    await asyncio.sleep(0.5)

    wf = _workflows[workflow_id]
    return {
        "workflow_id": workflow_id,
        "status": wf["status"],
        "pending_question": wf["state"].get("pending_question", ""),
        "thoughts": wf["state"].get("thoughts", [])[-10:],
    }


@app.get("/plan/{workflow_id}")
async def get_plan(workflow_id: str):
    wf = _workflows.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")

    state: PlannerState = wf["state"]
    return {
        "workflow_id": workflow_id,
        "status": wf["status"],
        "goal": state.get("goal", ""),
        "pending_question": state.get("pending_question", ""),
        "project_structure": state.get("project_structure", {}),
        "steps": [
            {
                "id": s["id"],
                "module": s["module"],
                "status": s["status"],
                "job_id": s["job_id"],
                "error": s["error"],
            }
            for s in state.get("steps", [])
        ],
        "current_step_index": state.get("current_step_index", 0),
        "thoughts": state.get("thoughts", [])[-20:],
        "error": wf.get("error", ""),
    }


@app.post("/plan/{workflow_id}/answer")
async def answer_plan(workflow_id: str, request: AnswerRequest):
    wf = _workflows.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")
    if wf["status"] != "waiting_for_operator":
        raise HTTPException(status_code=409, detail=f"Workflow is {wf['status']}, not waiting for answer")

    question = wf["state"].get("pending_question", "")
    wf["state"]["operator_answers"] = wf["state"].get("operator_answers", []) + [
        {"question": question, "answer": request.answer}
    ]
    wf["state"]["pending_question"] = ""

    # Decide where to resume:
    # - If we were in clarify and got answers, go back to clarify.
    # - If we were in decide (failure), the operator said retry/skip/abort.
    last_node = wf["state"].get("node_history", [{}])[-1].get("node", "")

    if last_node == "clarify":
        wf["state"]["next_node"] = "clarify"
    elif last_node == "decide":
        answer_lower = request.answer.lower().strip()
        steps: list = wf["state"].get("steps", [])
        idx = wf["state"].get("current_step_index", 0)
        if answer_lower == "retry" and idx < len(steps):
            steps[idx]["status"] = "pending"
            steps[idx]["error"] = ""
            steps[idx]["job_id"] = ""
            wf["state"]["steps"] = steps
            wf["state"]["next_node"] = "dispatch"
        elif answer_lower == "skip":
            wf["state"]["current_step_index"] = idx + 1
            wf["state"]["next_node"] = "dispatch"
        else:
            wf["state"]["next_node"] = "FINISH"
    else:
        wf["state"]["next_node"] = "clarify"

    # Resume the graph in the background.
    task = asyncio.create_task(_run_graph(workflow_id))
    wf["task"] = task
    wf["status"] = "running"

    await asyncio.sleep(0.5)
    return {
        "workflow_id": workflow_id,
        "status": wf["status"],
        "pending_question": wf["state"].get("pending_question", ""),
        "thoughts": wf["state"].get("thoughts", [])[-10:],
    }


@app.post("/plan/{workflow_id}/cancel")
async def cancel_plan(workflow_id: str):
    wf = _workflows.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")

    wf["status"] = "cancelled"
    task = wf.get("task")
    if task and not task.done():
        task.cancel()

    # Cancel any running backbone job.
    current_job_id = wf["state"].get("current_job_id", "")
    if current_job_id:
        from .modules import cancel_job
        await cancel_job(current_job_id)

    return {"workflow_id": workflow_id, "status": "cancelled"}


@app.get("/plan/{workflow_id}/result")
async def get_plan_result(workflow_id: str):
    wf = _workflows.get(workflow_id)
    if not wf:
        raise HTTPException(status_code=404, detail="Workflow not found")

    state: PlannerState = wf["state"]
    return {
        "workflow_id": workflow_id,
        "status": wf["status"],
        "project_structure": state.get("project_structure", {}),
        "steps": state.get("steps", []),
        "thoughts": state.get("thoughts", []),
        "llm_usage": state.get("llm_usage", []),
        "error": wf.get("error", ""),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}