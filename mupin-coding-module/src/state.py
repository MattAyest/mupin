"""Shared state schema for the v0.2 agentic coding pipeline.

The v0.2 pipeline is intentionally smaller than v0.1: test_designer,
skeleton_maker, coder, and sandbox_arbiter. All routing is done by writing
`next_node` and reading it in conditional edges.
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

    # Authoritative starter signature/imports extracted from the prompt by the
    # benchmark runner. Empty for prompts without a structured contract; in that
    # case skeleton_maker falls back to test-derived inference. Used to keep the
    # generated solution's import scope and signature aligned with canonical
    # tests, which follow the prompt contract rather than the pipeline's tests.
    contract_code: Optional[str]

    # Optional project tag for a persistent dependency cache. When unset (the
    # default) each task starts and ends with a clean .deps directory. When set,
    # installed deps are kept at .deps_cache/<tag>/<hash(requirements)> on the
    # shared workspace volume and reused across tasks with identical deps.
    deps_cache_tag: Optional[str]

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
    # Loop counters / guard rails
    # ------------------------------------------------------------------
    sandbox_loop_count: int

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
