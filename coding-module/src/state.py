"""Shared state schema for the v0.2 agentic coding pipeline.

The v0.2 pipeline is intentionally smaller than v0.1: a test architect, a
coder, a sandbox arbiter, and a prompt-compliance checker. All routing is
done by writing `next_node` and reading it in conditional edges.
"""

from operator import add
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class AgenticState(TypedDict):
    # ------------------------------------------------------------------
    # Task inputs
    # ------------------------------------------------------------------
    user_prompt: str
    workspace_dir: str
    profile_name: str
    python_version: Optional[str]

    # ------------------------------------------------------------------
    # Code artifacts (single-file contract)
    # ------------------------------------------------------------------
    file_manifest: Dict[str, str]

    # ------------------------------------------------------------------
    # Sandbox arbiter outputs
    # ------------------------------------------------------------------
    sandbox_errors: str
    sandbox_diagnostics: Dict[str, Any]

    # ------------------------------------------------------------------
    # Compliance checker outputs
    # ------------------------------------------------------------------
    compliance_status: str
    compliance_critique: List[str]

    # ------------------------------------------------------------------
    # Loop counters / guard rails
    # ------------------------------------------------------------------
    sandbox_loop_count: int
    compliance_loop_count: int

    # ------------------------------------------------------------------
    # Test/skeleton contract loop tracking
    # ------------------------------------------------------------------
    contract_loop_count: int
    contract_critique: List[str]
    contract_exhausted: bool

    # ------------------------------------------------------------------
    # Routing signal
    # ------------------------------------------------------------------
    next_node: str

    # ------------------------------------------------------------------
    # Transient LLM infrastructure fault tracking
    # ------------------------------------------------------------------
    # Per-node count of retries due to provider timeouts/5xx/etc. When a node
    # exhausts its budget, llm_infra_exhausted becomes True so the API can
    # report a non-code terminal status instead of "failed".
    llm_infra_retries: Dict[str, int]
    llm_infra_exhausted: bool

    # ------------------------------------------------------------------
    # Diagnostics: per-node timing, LLM usage, sandbox runs, and routing
    # decisions. Optional so old state serializes.
    # ------------------------------------------------------------------
    thoughts: Annotated[List[str], add]
    node_history: Annotated[List[Dict[str, Any]], add]
    llm_usage: Annotated[List[Dict[str, Any]], add]
    docker_runs: Annotated[List[Dict[str, Any]], add]
    classifier_history: Annotated[List[Dict[str, Any]], add]
