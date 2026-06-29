from operator import add
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class SwarmState(TypedDict):
    messages: List[Any]
    workspace_dir: str
    file_manifest: Dict[str, str]
    next_node: str
    loop_count: int
    regression_count: int
    replan_count: int  # increments each time error_distiller forces architect; hard ceiling prevents infinite spec loops
    verification_errors: str
    rollback_graveyard: List[str]
    architectural_plan: str
    interface_contract: (
        str  # formal interface spec written by architect, used by test_writer
    )
    contract_check_count: int  # loop guard for contract_verifier → test_writer retries
    architecture_ledger: (
        str  # accumulated across retries; written to disk by archivist_node
    )
    thoughts: Annotated[
        List[str], add
    ]  # short per-node log lines; accumulated via reducer
    # ------------------------------------------------------------------
    # Diagnostics: per-node timing, LLM token/time estimates, classifier
    # details, and Docker phase timings. Optional so old state serializes.
    # ------------------------------------------------------------------
    node_history: Annotated[
        List[Dict[str, Any]], add
    ]  # one entry per node execution (name, start/end/duration, plus node-specific metrics)
    llm_usage: Annotated[
        List[Dict[str, Any]], add
    ]  # one entry per LLM invocation (node, attempt, timing, estimated tokens, error)
    docker_runs: Annotated[
        List[Dict[str, Any]], add
    ]  # one entry per deterministic_verifier run (install/test timing, pytest summary)
    classifier_history: Annotated[
        List[Dict[str, Any]], add
    ]  # one entry per error_distiller classification (raw response, parsed fault, instruction)
    python_version: Optional[str]
    initial_prompt: Optional[str]
