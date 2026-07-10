"""Shared task runner for both the dev API and the ARQ worker.

Refactored from src/api.py in v0.3 so the backbone worker can drive the
LangGraph pipeline without importing the FastAPI app.
"""

import asyncio
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

from .graph import app as swarm_graph
from .nodes import (
    LLMUnavailableError,
    SERVER_TASK_DEADLINE,
    cleanup_sandbox_for_task,
    cleanup_workspace_deps,
)


async def drive_graph(
    task_id: str,
    initial_state: Dict[str, Any],
    progress_callback=None,
    is_cancelled=None,
    executor: ThreadPoolExecutor | None = None,
    cancel_event: threading.Event | None = None,
) -> Dict[str, Any]:
    """Consume the v0.2 graph stream, optionally calling back with progress.

    progress_callback receives a dict of current status whenever a node update
    arrives. It is used by the ARQ worker to POST heartbeats to the backbone.

    executor is a per-task thread pool used by LLM invocations; passing one in
    lets us shut it down cleanly on cancellation instead of leaking threads.
    cancel_event is a threading.Event that the worker can set to request
    cooperative cancellation.
    """
    final_manifest: Dict[str, Any] = {}
    tasks_db_like = {
        "current_node": "initializing",
        "sandbox_loop_count": 0,
        "sandbox_errors": "",
        "thoughts": [],
        "node_history": [],
        "llm_usage": [],
        "docker_runs": [],
        "classifier_history": [],
        "llm_infra_exhausted": False,
        "llm_infra_retries": {},
    }

    stream = swarm_graph.astream(initial_state, stream_mode="updates")
    try:
        async for output in stream:
            if is_cancelled is not None:
                if await is_cancelled():
                    raise asyncio.CancelledError("cancel_requested via backbone")
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError("cancel_event set by worker")
            for node_name, state_update in output.items():
                tasks_db_like["current_node"] = node_name

                for key in (
                    "sandbox_loop_count",
                    "sandbox_errors",
                ):
                    if key in state_update:
                        tasks_db_like[key] = state_update[key]

                if "file_manifest" in state_update:
                    final_manifest = state_update["file_manifest"]

                for list_key in ("thoughts", "node_history", "llm_usage", "docker_runs", "classifier_history"):
                    if list_key in state_update:
                        tasks_db_like[list_key].extend(state_update[list_key])

                if "llm_infra_exhausted" in state_update:
                    tasks_db_like["llm_infra_exhausted"] = state_update["llm_infra_exhausted"]
                if "llm_infra_retries" in state_update:
                    tasks_db_like["llm_infra_retries"] = state_update["llm_infra_retries"]

                if progress_callback is not None:
                    await progress_callback({
                        "current_node": tasks_db_like["current_node"],
                        "sandbox_loop_count": tasks_db_like["sandbox_loop_count"],
                        "thoughts": tasks_db_like["thoughts"][-20:] if tasks_db_like["thoughts"] else [],
                    })
    finally:
        await stream.aclose()
    return final_manifest, tasks_db_like


async def run_swarm_task(
    task_id: str,
    prompt: str,
    workspace_dir: str,
    profile_name: str,
    progress_callback=None,
    is_cancelled=None,
    deadline: float | None = None,
    contract_code: str = "",
    deps_cache_tag: str | None = None,
) -> Dict[str, Any]:
    """Run the full coding pipeline and return a serializable result payload.

    progress_callback: async callable receiving progress dicts between nodes.
    is_cancelled:      async callable returning True if the task was cancelled.
    """
    os.makedirs(workspace_dir, exist_ok=True)

    cancel_event = threading.Event()
    llm_executor = ThreadPoolExecutor(
        max_workers=int(os.environ.get("LLM_EXECUTOR_THREADS", "8")),
        thread_name_prefix=f"llm_{task_id[:8]}",
    )

    initial_state = {
        "user_prompt": prompt,
        "workspace_dir": workspace_dir,
        "profile_name": profile_name,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "contract_code": contract_code,
        "deps_cache_tag": deps_cache_tag,
        "file_manifest": {},
        "sandbox_errors": "",
        "sandbox_diagnostics": {},
        "sandbox_loop_count": 0,
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
        "classifier_history": [],
    }

    try:
        final_manifest, diagnostics = await asyncio.wait_for(
            drive_graph(task_id, initial_state, progress_callback, is_cancelled, llm_executor, cancel_event),
            timeout=deadline or SERVER_TASK_DEADLINE,
        )

        result_payload["result"] = final_manifest
        result_payload["current_node"] = diagnostics["current_node"]
        result_payload["sandbox_loop_count"] = diagnostics["sandbox_loop_count"]
        result_payload["sandbox_errors"] = diagnostics["sandbox_errors"]
        result_payload["thoughts"] = diagnostics["thoughts"]
        result_payload["node_history"] = diagnostics["node_history"]
        result_payload["llm_usage"] = diagnostics["llm_usage"]
        result_payload["docker_runs"] = diagnostics["docker_runs"]
        result_payload["classifier_history"] = diagnostics["classifier_history"]
        result_payload["llm_infra_exhausted"] = diagnostics["llm_infra_exhausted"]
        result_payload["llm_infra_retries"] = diagnostics["llm_infra_retries"]

        if (
            diagnostics["current_node"] == "sandbox_arbiter"
            and not diagnostics.get("sandbox_errors")
        ):
            result_payload["status"] = "completed"
        elif diagnostics["llm_infra_exhausted"]:
            result_payload["status"] = "infra_exhausted"
            result_payload["error"] = result_payload["error"] or "LLM infrastructure retries exhausted"
        else:
            result_payload["status"] = "exhausted"
            result_payload["error"] = result_payload["error"] or "Pipeline terminated without passing sandbox"

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
        # Shut down the per-task LLM executor.  cancel_futures=True stops any
        # in-flight httpx requests so connections and threads are released.
        try:
            llm_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        cleanup_sandbox_for_task(task_id)
        cleanup_workspace_deps(workspace_dir)

    return result_payload
