"""Shared task runner for both the dev API and the ARQ worker."""

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

from .config_loader import load_env_hierarchy
from .graph import app as edit_graph
from .nodes import (
    LLMUnavailableError,
    SERVER_TASK_DEADLINE,
    cleanup_sandbox_for_task,
    cleanup_workspace_deps,
    compute_diff,
)

load_env_hierarchy()


async def drive_graph(
    task_id: str,
    initial_state: Dict[str, Any],
    progress_callback=None,
    is_cancelled=None,
    executor: ThreadPoolExecutor | None = None,
    cancel_event: threading.Event | None = None,
) -> Dict[str, Any]:
    """Consume the editing graph stream, optionally calling back with progress."""
    diagnostics = {
        "current_node": "initializing",
        "sandbox_loop_count": 0,
        "sandbox_errors": "",
        "thoughts": [],
        "node_history": [],
        "llm_usage": [],
        "docker_runs": [],
        "llm_infra_exhausted": False,
        "llm_infra_retries": {},
    }

    stream = edit_graph.astream(initial_state, stream_mode="updates")
    try:
        async for output in stream:
            if is_cancelled is not None:
                if await is_cancelled():
                    raise asyncio.CancelledError("cancel_requested via backbone")
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError("cancel_event set by worker")
            for node_name, state_update in output.items():
                diagnostics["current_node"] = node_name

                for key in ("sandbox_loop_count", "sandbox_errors"):
                    if key in state_update:
                        diagnostics[key] = state_update[key]

                if "llm_infra_exhausted" in state_update:
                    diagnostics["llm_infra_exhausted"] = state_update["llm_infra_exhausted"]
                if "llm_infra_retries" in state_update:
                    diagnostics["llm_infra_retries"] = state_update["llm_infra_retries"]

                for list_key in ("thoughts", "node_history", "llm_usage", "docker_runs"):
                    if list_key in state_update:
                        diagnostics[list_key].extend(state_update[list_key])

                if progress_callback is not None:
                    await progress_callback({
                        "current_node": diagnostics["current_node"],
                        "sandbox_loop_count": diagnostics["sandbox_loop_count"],
                        "thoughts": diagnostics["thoughts"][-20:] if diagnostics["thoughts"] else [],
                    })
    finally:
        await stream.aclose()
    return diagnostics


async def run_edit_task(
    task_id: str,
    source_job_id: str,
    instruction: str,
    workspace_dir: str,
    profile_name: str,
    source_files: dict | None = None,
    progress_callback=None,
    is_cancelled=None,
    deadline: float | None = None,
) -> Dict[str, Any]:
    """Run the full editing pipeline and return a serializable result payload."""
    os.makedirs(workspace_dir, exist_ok=True)

    cancel_event = threading.Event()
    llm_executor = ThreadPoolExecutor(
        max_workers=int(os.environ.get("LLM_EXECUTOR_THREADS", "8")),
        thread_name_prefix=f"edit_{task_id[:8]}",
    )

    initial_state = {
        "task_id": task_id,
        "source_job_id": source_job_id or "",
        "source_files": source_files or {},
        "instruction": instruction,
        "workspace_dir": workspace_dir,
        "profile_name": profile_name,
        "source_manifest": {},
        "test_impact": "none",
        "edit_plan": [],
        "file_manifest": {},
        "deleted_files": [],
        "sandbox_errors": "",
        "sandbox_diagnostics": {},
        "sandbox_loop_count": 0,
        "regression_loop_count": 0,
        "regression_errors": "",
        "test_quality_loop_count": 0,
        "test_quality_errors": "",
        "next_node": "load_source",
        "llm_infra_exhausted": False,
        "llm_infra_retries": {},
        "thoughts": [],
        "node_history": [],
        "llm_usage": [],
        "docker_runs": [],
        "llm_executor": llm_executor,
        "cancel_event": cancel_event,
    }

    result_payload: Dict[str, Any] = {
        "task_id": task_id,
        "status": "running",
        "current_node": "initializing",
        "sandbox_loop_count": 0,
        "workspace": workspace_dir,
        "result": None,
        "error": None,
        "llm_infra_exhausted": False,
        "llm_infra_retries": {},
        "thoughts": [],
        "node_history": [],
        "llm_usage": [],
        "docker_runs": [],
    }

    source_manifest: Dict[str, str] = {}
    final_manifest: Dict[str, str] = {}

    try:
        diagnostics = await asyncio.wait_for(
            drive_graph(task_id, initial_state, progress_callback, is_cancelled, llm_executor, cancel_event),
            timeout=deadline or SERVER_TASK_DEADLINE,
        )

        result_payload["current_node"] = diagnostics["current_node"]
        result_payload["sandbox_loop_count"] = diagnostics["sandbox_loop_count"]
        result_payload["sandbox_errors"] = diagnostics["sandbox_errors"]
        result_payload["thoughts"] = diagnostics["thoughts"]
        result_payload["node_history"] = diagnostics["node_history"]
        result_payload["llm_usage"] = diagnostics["llm_usage"]
        result_payload["docker_runs"] = diagnostics["docker_runs"]
        result_payload["llm_infra_exhausted"] = diagnostics["llm_infra_exhausted"]
        result_payload["llm_infra_retries"] = diagnostics["llm_infra_retries"]

        # Best-effort read final source manifest from workspace for diff calculation.
        from .profile import get_profile
        try:
            profile = get_profile(profile_name)
            from .nodes import load_workspace_manifest
            final_manifest = load_workspace_manifest(workspace_dir, profile)

            # Source manifest: if source_files were provided inline, use those
            # as the baseline. Otherwise read from the source workspace on disk.
            if source_files:
                source_manifest = dict(source_files)
            elif source_job_id:
                workspace_root = Path(workspace_dir).parent
                source_manifest = load_workspace_manifest(
                    str(workspace_root / source_job_id), profile
                )
            else:
                source_manifest = {}
        except Exception:
            pass

        result_payload["result"] = {
            "source_job_id": source_job_id,
            "file_manifest": final_manifest,
            "diff": compute_diff(source_manifest, final_manifest),
        }

        if (
            diagnostics["current_node"] == "verify"
            and not diagnostics.get("sandbox_errors")
        ):
            result_payload["status"] = "completed"
        elif diagnostics["llm_infra_exhausted"]:
            result_payload["status"] = "infra_exhausted"
            result_payload["error"] = result_payload["error"] or "LLM infrastructure retries exhausted"
        else:
            result_payload["status"] = "exhausted"
            result_payload["error"] = result_payload["error"] or "Pipeline terminated without passing verification"

    except asyncio.TimeoutError:
        result_payload["status"] = "exhausted"
        actual_deadline = deadline or SERVER_TASK_DEADLINE
        result_payload["error"] = f"Server task deadline ({actual_deadline}s) exceeded"
        result_payload["result"] = None
    except asyncio.CancelledError:
        result_payload["status"] = "cancelled"
        result_payload["error"] = "Task cancelled"
        result_payload["result"] = None
        cancel_event.set()
        cleanup_sandbox_for_task(task_id)
        raise
    except LLMUnavailableError as e:
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        result_payload["result"] = None
        if getattr(e, "usage_entries", None):
            result_payload["llm_usage"].extend(e.usage_entries)
    except Exception as e:
        result_payload["status"] = "failed"
        result_payload["error"] = str(e)
        result_payload["result"] = None
        if hasattr(e, "usage_entries"):
            result_payload["llm_usage"].extend(getattr(e, "usage_entries", []))
    finally:
        try:
            llm_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        cleanup_sandbox_for_task(task_id)
        cleanup_workspace_deps(workspace_dir)

    return result_payload
