"""Shared state schema for the Planner module.

The planner is a LangGraph state machine:

    clarify → plan → dispatch → inspect → decide → (next step | ask operator | FINISH)

State is kept in-memory (v0.1). Future versions may persist to Redis.
"""

from operator import add
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class TaskStep(TypedDict):
    id: str
    module: str          # "coding" | "editing"
    prompt: str          # for coding
    instruction: str     # for editing
    source_from: str     # step id to use as source (editing only)
    depends_on: List[str]
    status: str          # pending | running | completed | failed
    job_id: str          # backbone job id once dispatched
    error: str           # error message if failed
    result: Dict[str, Any]  # job result from backbone


class PlannerState(TypedDict):
    # ------------------------------------------------------------------
    # Workflow identity
    # ------------------------------------------------------------------
    workflow_id: str
    goal: str

    # ------------------------------------------------------------------
    # Operator interaction
    # ------------------------------------------------------------------
    pending_question: str           # question waiting for operator answer
    operator_answers: Annotated[List[Dict[str, str]], add]  # {question, answer}

    # ------------------------------------------------------------------
    # Plan
    # ------------------------------------------------------------------
    project_structure: Dict[str, str]   # directory -> description
    steps: List[TaskStep]
    current_step_index: int

    # ------------------------------------------------------------------
    # Execution state
    # ------------------------------------------------------------------
    current_job_id: str
    poll_started_at: float

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------
    next_node: str

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    thoughts: Annotated[List[str], add]
    llm_usage: Annotated[List[Dict[str, Any]], add]
    node_history: Annotated[List[Dict[str, Any]], add]