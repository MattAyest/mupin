"""Shared state schema for the editing module pipeline.

Pipeline (v0.2):
    load_source → analyze → plan → route_on_test_impact
        │ none: apply (code-only)
        │ update_existing/add_new/both: apply_tests → apply_code
        → regression_check (hard gate, max 3 loops)
        → verify (sandbox, max 5 loops)
        → FINISH

All routing is done by writing `next_node` and reading it in conditional edges.
"""

from operator import add
from typing import Annotated, Any, Dict, List, TypedDict


class EditingState(TypedDict):
    # ------------------------------------------------------------------
    # Task inputs
    # ------------------------------------------------------------------
    source_job_id: str
    source_files: Dict[str, str]
    instruction: str
    workspace_dir: str
    profile_name: str

    # ------------------------------------------------------------------
    # Loaded source state (immutable — apply nodes cannot modify this)
    # ------------------------------------------------------------------
    source_manifest: Dict[str, str]

    # ------------------------------------------------------------------
    # Analysis / planning artifacts
    # ------------------------------------------------------------------
    test_impact: str  # none | update_existing | add_new | both
    edit_plan: List[Dict[str, Any]]

    # ------------------------------------------------------------------
    # Edit artifacts
    # ------------------------------------------------------------------
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
    regression_loop_count: int
    regression_errors: str

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