"""Node implementations for the Planner module.

The planner is a LangGraph state machine:

    clarify → plan → dispatch → inspect → decide → (next step | ask operator | FINISH)

Each node reads/updates PlannerState and sets next_node for routing.
"""

from __future__ import annotations

import json
import os
import queue
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from .config_loader import load_env_hierarchy, load_llm_config
from .modules import (
    get_registry_context,
    poll_job_until_settled,
    submit_job,
)
from .state import PlannerState, TaskStep

load_env_hierarchy()

CONFIG = load_llm_config()

OLLAMA_CLOUD_HOST = "https://ollama.com"
OLLAMA_TIMEOUT = httpx.Timeout(connect=15, read=180, write=15, pool=15)

RESILIENCE = CONFIG.get("resilience", {})
INFRA_MAX_RETRIES = int(RESILIENCE.get("infra_max_retries_per_node", 5))
INFRA_BACKOFFS = [float(x) for x in RESILIENCE.get("infra_retry_backoff_seconds", [10, 30, 60, 120, 120])]
STREAM_IDLE_DEFAULT = float(RESILIENCE.get("stream_idle_seconds", 120.0))
WORKFLOW_DEADLINE = CONFIG.get("server", {}).get("workflow_deadline_seconds", 14400)

TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504, 524}
TRANSIENT_ERROR_NAMES = {
    "ReadTimeout", "ConnectTimeout", "ConnectionError", "ConnectionResetError",
    "TimeoutError", "FutureTimeoutError", "SSLError", "SSLZeroReturnError",
    "SSLEOFError", "IncompleteReadError", "ChunkedEncodingError",
    "RemoteDisconnected", "ProtocolError",
}


# =============================================================================
# LLM factory (same pattern as coding/editing modules)
# =============================================================================
def _build_llm(node_name: str):
    if node_name not in CONFIG.get("nodes", {}):
        raise ValueError(f"No config entry for node '{node_name}'")
    cfg = CONFIG["nodes"][node_name]
    provider = cfg.get("provider")
    model = cfg.get("model")
    temperature = cfg.get("temperature", 0)
    api_key_env = cfg.get("api_key_env_var")

    def require_key(env: str | None = None):
        key_env = api_key_env or env
        key = os.getenv(key_env) if key_env else None
        if not key:
            raise ValueError(
                f"Node '{node_name}' uses provider '{provider}' but env var "
                f"'{key_env}' is not set."
            )
        return key

    if provider == "ollama-cloud":
        from langchain_ollama import ChatOllama
        key = require_key("OLLAMA_API_KEY")
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=OLLAMA_CLOUD_HOST,
            streaming=True,
            client_kwargs={
                "headers": {"Authorization": f"Bearer {key}"},
                "timeout": OLLAMA_TIMEOUT,
                "limits": httpx.Limits(max_connections=16, max_keepalive_connections=8),
            },
        )
    if provider == "google-genai":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model, temperature=temperature,
            google_api_key=require_key("GOOGLE_API_KEY"),
        )
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature, openai_api_key=require_key("OPENAI_API_KEY"))
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature, anthropic_api_key=require_key("ANTHROPIC_API_KEY"))
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        base_url = os.getenv(api_key_env or "OLLAMA_BASE_URL", "http://localhost:11434")
        return ChatOllama(
            model=model, temperature=temperature, base_url=base_url, streaming=True,
            client_kwargs={"timeout": OLLAMA_TIMEOUT},
        )
    raise ValueError(f"Unsupported provider '{provider}' for node '{node_name}'")


@lru_cache(maxsize=None)
def get_llm(node_name: str):
    return _build_llm(node_name)


# =============================================================================
# LLM invocation with retry (simplified from editing module)
# =============================================================================
def _is_transient(e: Exception) -> bool:
    if isinstance(e, FutureTimeoutError):
        return True
    name = type(e).__name__
    if name in TRANSIENT_ERROR_NAMES:
        return True
    text = str(e).lower()
    if any(tok in text for tok in ("timeout", "timed out", "rate limit", "server error", "ssl", "eof", "connection reset", "remote disconnected")):
        return True
    for m in re.finditer(r"\b(\d{3})\b", str(e)):
        if int(m.group(1)) in TRANSIENT_HTTP_STATUSES or 500 <= int(m.group(1)) < 600:
            return True
    return False


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.5)) if text else 0


def _invoke_llm(node_name: str, messages, state: dict) -> tuple[str, list[dict]]:
    cfg = CONFIG.get("nodes", {}).get(node_name, {})
    provider = cfg.get("provider", "unknown")
    model = cfg.get("model", "unknown")
    use_streaming = provider in ("ollama-cloud", "ollama")
    timeout = 600.0
    stream_idle = float(cfg.get("stream_idle_seconds", STREAM_IDLE_DEFAULT))

    input_tokens = sum(_estimate_tokens(getattr(m, "content", "")) for m in messages)
    usage_entries: list[dict] = []
    last_error: Exception | None = None

    for attempt in range(1, 4):
        started = time.perf_counter()
        try:
            llm = get_llm(node_name)
            chunk_queue: queue.Queue = queue.Queue()

            def _call():
                try:
                    if use_streaming:
                        parts: list[str] = []
                        for chunk in llm.stream(messages):
                            piece = getattr(chunk, "content", None)
                            if isinstance(piece, str) and piece:
                                parts.append(piece)
                                chunk_queue.put(("chunk", piece))
                        chunk_queue.put(("done", "".join(parts)))
                    else:
                        resp = llm.invoke(messages)
                        content = getattr(resp, "content", "")
                        chunk_queue.put(("done", content))
                except BaseException as e:
                    chunk_queue.put(("error", e))

            executor = ThreadPoolExecutor(max_workers=1)
            future = executor.submit(_call)

            content = ""
            stream_error: Exception | None = None
            done = False
            while not done:
                remaining = timeout - (time.perf_counter() - started)
                if remaining <= 0:
                    raise FutureTimeoutError(f"Wall-clock timeout after {timeout}s")
                try:
                    kind, payload = chunk_queue.get(timeout=min(stream_idle, remaining))
                except queue.Empty:
                    raise FutureTimeoutError(f"Stream idle timeout after {stream_idle:.0f}s")
                if kind == "chunk":
                    content += payload
                elif kind == "done":
                    content = payload
                    done = True
                elif kind == "error":
                    stream_error = payload
                    done = True
            executor.shutdown(wait=False, cancel_futures=True)

            if stream_error:
                raise stream_error

            elapsed = time.perf_counter() - started
            usage_entries.append({
                "node": node_name, "model": model, "provider": provider,
                "attempt": attempt, "status": "success",
                "duration_seconds": round(elapsed, 3),
                "input_tokens": input_tokens,
                "output_tokens": _estimate_tokens(content),
                "total_tokens": input_tokens + _estimate_tokens(content),
            })
            if not content.strip():
                raise Exception("Empty LLM response")
            return content, usage_entries
        except FutureTimeoutError as e:
            last_error = e
            usage_entries.append({
                "node": node_name, "model": model, "provider": provider,
                "attempt": attempt, "status": "wallclock_timeout",
                "duration_seconds": round(time.perf_counter() - started, 3),
                "input_tokens": input_tokens, "output_tokens": 0, "total_tokens": input_tokens,
            })
        except Exception as e:
            last_error = e
            usage_entries.append({
                "node": node_name, "model": model, "provider": provider,
                "attempt": attempt, "status": "error",
                "duration_seconds": round(time.perf_counter() - started, 3),
                "input_tokens": input_tokens, "output_tokens": 0, "total_tokens": input_tokens,
                "error": f"{type(e).__name__}: {e}",
            })
        if attempt < 3:
            time.sleep([2.0, 5.0, 10.0][min(attempt - 1, 2)])

    raise Exception(f"LLM for '{node_name}' failed after 3 attempts: {last_error}")


# =============================================================================
# Helpers
# =============================================================================
def _think(message: str) -> list[str]:
    ts = datetime.now().strftime("%H:%M:%S")
    return [f"[{ts}] {message}"]


def _node_entry(node_name: str) -> dict:
    now = time.time()
    return {
        "node": node_name,
        "timestamp": datetime.fromtimestamp(now).isoformat(),
    }


def _parse_json_block(content: str, tag: str) -> dict | list | None:
    match = re.search(f"<{tag}>(.*?)</{tag}>", content, re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except Exception:
        return None


# =============================================================================
# NODE: clarify
# =============================================================================
async def clarify(state: PlannerState) -> dict:
    goal = state["goal"]
    answers = state.get("operator_answers", [])
    registry = get_registry_context()

    answers_block = ""
    if answers:
        answers_block = "\n\nPrevious Q&A:\n" + "\n".join(
            f"  Q: {a['question']}\n  A: {a['answer']}" for a in answers
        )

    system = f"""You are a solutions architect planning a software project.
Your job is to read the operator's goal and decide if you need to ask
clarifying questions before decomposing it into engineering tasks.

Available modules you can delegate to:
{registry}

Rules:
- If the goal is clear enough to decompose into tasks, output no questions.
- If the goal is ambiguous, ask 1-3 focused questions.
- Do NOT ask about implementation details — those are engineering decisions
  that the modules handle.
- Keep questions short and specific.

Output format:
  <clarification>
  {{
    "questions": ["question 1", "question 2"],
    "ready_to_plan": false
  }}
  </clarification>

If no questions are needed:
  <clarification>
  {{
    "questions": [],
    "ready_to_plan": true
  }}
  </clarification>"""

    try:
        content, llm_usage = _invoke_llm(
            "clarify",
            [SystemMessage(content=system), HumanMessage(content=f"Goal: {goal}{answers_block}")],
            state=dict(state),
        )
    except Exception as e:
        return {
            "next_node": "FINISH",
            "thoughts": _think(f"clarify LLM failed: {e}"),
            "llm_usage": [],
            "node_history": [_node_entry("clarify")],
        }

    parsed = _parse_json_block(content, "clarification") or {}
    questions = parsed.get("questions", [])
    ready = parsed.get("ready_to_plan", len(questions) == 0)

    if ready or not questions:
        return {
            "next_node": "plan",
            "pending_question": "",
            "llm_usage": llm_usage,
            "thoughts": _think("Clarification complete — ready to plan"),
            "node_history": [_node_entry("clarify")],
        }

    return {
        "next_node": "ASK_OPERATOR",
        "pending_question": json.dumps(questions),
        "llm_usage": llm_usage,
        "thoughts": _think(f"Clarification: asking {len(questions)} question(s)"),
        "node_history": [_node_entry("clarify")],
    }


# =============================================================================
# NODE: plan
# =============================================================================
async def plan(state: PlannerState) -> dict:
    goal = state["goal"]
    answers = state.get("operator_answers", [])
    registry = get_registry_context()

    answers_block = ""
    if answers:
        answers_block = "\n\nOperator clarifications:\n" + "\n".join(
            f"  Q: {a['question']}\n  A: {a['answer']}" for a in answers
        )

    system = f"""You are a solutions architect decomposing a software project
into engineering tasks that will be delegated to specialist modules.

Available modules:
{registry}

Decomposition rules:
- Use CODING steps to create complete, self-contained modules from scratch.
  A coding step should produce a working, tested module — not a fragment.
- The coding module is capable of handling complex, multi-function prompts.
  Do NOT over-decompose. A REST API with auth is ONE coding job, not six.
- Only split into multiple coding steps when the components are genuinely
  independent and could be developed by separate engineers in parallel.
- Use EDITING steps only to modify or fix EXISTING code after it has been
  built — adding a feature to an existing module, fixing a bug, refactoring.
  Never use editing to incrementally build up code that should have been
  a single coding job.
- Each step MUST declare its "exports" — the function names, class names,
  and constants it makes available to other steps. This is the contract
  that lets subsequent steps reference the correct symbols. Use the format:
  "function:name", "class:Name", "constant:NAME".
- When a later step depends on an earlier step, its prompt/instruction
  should reference the exported names so the coding/editing module knows
  what to import and use.
- Each coding step's prompt MUST specify src/main.py as the target file and
  tests/test_main.py as the test file.
- Steps can depend on previous steps (depends_on). Keep it linear for now.
- Do not include setup, CI/CD, or deployment steps.

When to use ONE coding step:
- A single API, library, or service → one coding job
- A module with multiple related functions → one coding job
- "Build a REST API with auth and rate limiting" → one coding job

When to use MULTIPLE coding steps:
- A system with genuinely independent subsystems (e.g., a data pipeline +
  a web frontend + a CLI tool) → one coding step per subsystem
- A large project where components are developed independently and
  integrated later → one coding step per component

When to use EDITING steps:
- "Fix the failing tests in the API" → editing (source_from: the coding step)
- "Add rate limiting to the existing API" → editing (source_from: the coding step)
- "Refactor the auth module to use OAuth" → editing (source_from: the coding step)

Example decomposition for "Build a REST API with auth and rate limiting":
  step_1: coding — "Write a complete FastAPI application with JWT authentication,
    user registration, login, a protected /profile endpoint, and rate limiting
    middleware. Target: src/main.py, tests in tests/test_main.py."
  (ONE step — this is a single coherent module, not six incremental edits.)

Example decomposition for "Build a data pipeline with a web dashboard":
  step_1: coding — "Write a data pipeline module that reads CSV files, transforms
    data, and writes to a database. Target: src/main.py, tests in tests/test_main.py."
  step_2: coding — "Write a FastAPI web dashboard that queries the database and
    displays results. Target: src/main.py, tests in tests/test_main.py."
  (TWO steps — these are genuinely independent subsystems.)

Output format:
  <plan>
  {{
    "project_structure": {{
      "src/": "Main source directory",
      "tests/": "Test directory"
    }},
    "steps": [
      {{
        "id": "step_1",
        "module": "coding",
        "prompt": "Write a Python module with...",
        "exports": ["function:hash_password", "function:verify_password", "class:User"],
        "depends_on": []
      }},
      {{
        "id": "step_2",
        "module": "editing",
        "instruction": "Add rate limiting to the API",
        "source_from": "step_1",
        "exports": ["function:rate_limit_middleware"],
        "depends_on": ["step_1"]
      }}
    ]
  }}
  </plan>"""

    try:
        content, llm_usage = _invoke_llm(
            "plan",
            [SystemMessage(content=system), HumanMessage(content=f"Goal: {goal}{answers_block}")],
            state=dict(state),
        )
    except Exception as e:
        return {
            "next_node": "FINISH",
            "thoughts": _think(f"plan LLM failed: {e}"),
            "llm_usage": [],
            "node_history": [_node_entry("plan")],
        }

    parsed = _parse_json_block(content, "plan") or {}
    project_structure = parsed.get("project_structure", {})
    raw_steps = parsed.get("steps", [])

    steps: list[TaskStep] = []
    for s in raw_steps:
        steps.append({
            "id": s.get("id", f"step_{len(steps)+1}"),
            "module": s.get("module", "coding"),
            "prompt": s.get("prompt", ""),
            "instruction": s.get("instruction", ""),
            "source_from": s.get("source_from", ""),
            "depends_on": s.get("depends_on", []),
            "status": "pending",
            "job_id": "",
            "error": "",
            "result": {},
            "exports": s.get("exports", []),
        })

    return {
        "project_structure": project_structure,
        "steps": steps,
        "current_step_index": 0,
        "next_node": "dispatch",
        "llm_usage": llm_usage,
        "thoughts": _think(f"Planned {len(steps)} step(s) for {len(project_structure)} directories"),
        "node_history": [_node_entry("plan")],
    }


# =============================================================================
# NODE: dispatch
# =============================================================================
async def dispatch(state: PlannerState) -> dict:
    steps: list[TaskStep] = state.get("steps", [])
    idx = state.get("current_step_index", 0)

    if idx >= len(steps):
        return {
            "next_node": "FINISH",
            "thoughts": _think("All steps completed"),
            "node_history": [_node_entry("dispatch")],
        }

    step = steps[idx]
    if step["status"] != "pending":
        return {
            "next_node": "FINISH",
            "thoughts": _think(f"Step {step['id']} is {step['status']}, not pending"),
            "node_history": [_node_entry("dispatch")],
        }

    payload: dict[str, Any] = {"profile_name": "python"}

    if step["module"] == "coding":
        payload["prompt"] = step["prompt"]
        job_type = "coding"
    elif step["module"] == "editing":
        source_from = step.get("source_from", "")
        payload["instruction"] = step["instruction"]
        # Find the source job_id from a previous step.
        for prev in steps:
            if prev["id"] == source_from and prev["job_id"]:
                payload["source_job_id"] = prev["job_id"]
                break
        job_type = "editing"
    else:
        step["status"] = "failed"
        step["error"] = f"Unknown module: {step['module']}"
        return {
            "steps": steps,
            "next_node": "decide",
            "thoughts": _think(f"Unknown module: {step['module']}"),
            "node_history": [_node_entry("dispatch")],
        }

    try:
        job_id = await submit_job(job_type, payload)
    except Exception as e:
        step["status"] = "failed"
        step["error"] = f"Failed to submit job: {e}"
        return {
            "steps": steps,
            "next_node": "decide",
            "thoughts": _think(f"Failed to dispatch {step['id']}: {e}"),
            "node_history": [_node_entry("dispatch")],
        }

    step["status"] = "running"
    step["job_id"] = job_id

    return {
        "steps": steps,
        "current_job_id": job_id,
        "poll_started_at": time.time(),
        "next_node": "inspect",
        "thoughts": _think(f"Dispatched {step['id']} as {job_type} job {job_id}"),
        "node_history": [_node_entry("dispatch")],
    }


# =============================================================================
# NODE: inspect
# =============================================================================
async def inspect(state: PlannerState) -> dict:
    steps: list[TaskStep] = state.get("steps", [])
    idx = state.get("current_step_index", 0)
    job_id = state.get("current_job_id", "")

    if not job_id or idx >= len(steps):
        return {
            "next_node": "decide",
            "thoughts": _think("Nothing to inspect"),
            "node_history": [_node_entry("inspect")],
        }

    job = await poll_job_until_settled(job_id, timeout=JOB_TIMEOUT)
    step = steps[idx]
    step["status"] = job.get("status", "failed")
    step["error"] = job.get("error", "")
    step["result"] = job.get("result", {})

    return {
        "steps": steps,
        "next_node": "decide",
        "thoughts": _think(f"Step {step['id']} settled: {step['status']}"),
        "node_history": [_node_entry("inspect")],
    }


JOB_TIMEOUT = 3600


# =============================================================================
# NODE: decide
# =============================================================================
async def decide(state: PlannerState) -> dict:
    steps: list[TaskStep] = state.get("steps", [])
    idx = state.get("current_step_index", 0)

    if idx >= len(steps):
        return {
            "next_node": "FINISH",
            "thoughts": _think("All steps done — workflow complete"),
            "node_history": [_node_entry("decide")],
        }

    step = steps[idx]
    status = step["status"]

    if status == "completed":
        next_idx = idx + 1
        if next_idx >= len(steps):
            return {
                "next_node": "FINISH",
                "current_step_index": next_idx,
                "thoughts": _think(f"Step {step['id']} completed. All steps done."),
                "node_history": [_node_entry("decide")],
            }
        return {
            "next_node": "dispatch",
            "current_step_index": next_idx,
            "thoughts": _think(f"Step {step['id']} completed. Moving to step {steps[next_idx]['id']}."),
            "node_history": [_node_entry("decide")],
        }

    # Failed — ask the operator (v0.1: no auto-retry).
    question = f"Step '{step['id']}' ({step['module']}) failed with:\n{step['error'][:500]}\n\nHow would you like to proceed? Options: retry, skip, abort."
    return {
        "next_node": "ASK_OPERATOR",
        "pending_question": question,
        "thoughts": _think(f"Step {step['id']} failed — asking operator"),
        "node_history": [_node_entry("decide")],
    }