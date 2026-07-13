"""Shared state schema for the editing module pipeline.

The editing pipeline is intentionally small: load, analyze, plan, apply, verify.
All routing is done by writing `next_node` and reading it in conditional edges.
"""

from operator import add
from typing import Annotated, Any, Dict, List, TypedDict


class EditingState(TypedDict):
    # ------------------------------------------------------------------
    # Task inputs
    # ------------------------------------------------------------------
    source_job_id: str
    instruction: str
    workspace_dir: str
    profile_name: str

    # ------------------------------------------------------------------
    # Loaded source state
    # ------------------------------------------------------------------
    source_manifest: Dict[str, str]

    # ------------------------------------------------------------------
    # Edit artifacts
    # ------------------------------------------------------------------
    edit_plan: List[Dict[str, Any]]
    file_manifest: Dict[str, str]

    # ------------------------------------------------------------------
    # Sandbox outputs
    # ------------------------------------------------------------------
    sandbox_errors: str
    sandbox_diagnostics: Dict[str, Any]

    # ------------------------------------------------------------------
    # Loop counters / guard rails
    # ------------------------------------------------------------------
    sandbox_loop_count: int

    # ------------------------------------------------------------------
    # Routing signal
    # ------------------------------------------------------------------
    next_node: str

    # ------------------------------------------------------------------
    # Transient LLM infrastructure fault tracking
    # ------------------------------------------------------------------
    llm_infra_retries: Dict[str, int]
    llm_infra_exhausted: bool

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    thoughts: Annotated[List[str], add]
    node_history: Annotated[List[Dict[str, Any]], add]
    llm_usage: Annotated[List[Dict[str, Any]], add]
    docker_runs: Annotated[List[Dict[str, Any]], add]
