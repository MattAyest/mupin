import ast
import asyncio
import hashlib
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import httpx
import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from .state import AgenticState
from .profile import Profile, get_profile


# =============================================================================
# Custom exceptions
# =============================================================================
class LLMUnavailableError(Exception):
    """Raised when an LLM call is retried up to the configured cap and still fails.

    This is normally an infrastructure fault (timeout, 5xx, connection drop),
    not a code/test/spec fault. Transient faults are now retried at the node
    level so they do not become terminal "failed" statuses.
    """

    def __init__(
        self,
        node_name: str,
        attempts: int,
        cause: Exception,
        usage_entries: list[dict] | None = None,
        is_transient: bool | None = None,
    ):
        self.node_name = node_name
        self.attempts = attempts
        self.cause = cause
        self.usage_entries = usage_entries or []
        self.is_transient = (
            is_transient if is_transient is not None else _is_transient_llm_error(cause)
        )
        super().__init__(
            f"LLM for node '{node_name}' failed after {attempts} attempt(s): {cause}"
        )


# =============================================================================
# Configuration loading
# =============================================================================
from .config_loader import load_llm_config

CONFIG = load_llm_config()

OLLAMA_CLOUD_HOST = "https://ollama.com"
# Per-call timeout for Ollama providers.  This is intentionally generous
# because Ollama Cloud often cold-starts models; the per-node timeout in
# llm_config.yaml provides the hard budget at the retry-wrapper level.
OLLAMA_TIMEOUT = httpx.Timeout(connect=15, read=180, write=15, pool=15)

# Resilience configuration loaded from llm_config.yaml.
RESILIENCE = CONFIG.get("resilience", {})
INFRA_MAX_RETRIES = int(RESILIENCE.get("infra_max_retries_per_node", 5))
INFRA_BACKOFFS = [
    float(x) for x in RESILIENCE.get("infra_retry_backoff_seconds", [10, 30, 60, 120, 120])
]

# Streaming liveness watchdog: abort an LLM call if no new chunk arrives within
# this many seconds.  Catches mid-stream Ollama Cloud stalls (observed hanging
# for ~900s emitting zero tokens) without cutting off legitimately slow-but-
# progressing generations (max observed successful call is ~561s, but those
# stream continuously).  Tunable per-node via llm_config.yaml `stream_idle_seconds`.
STREAM_IDLE_DEFAULT = float(RESILIENCE.get("stream_idle_seconds", 120.0))

# HTTP status codes and error names that we treat as transient provider faults.
TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504, 524}
TRANSIENT_ERROR_NAMES = {
    "ReadTimeout",
    "ConnectTimeout",
    "ConnectionError",
    "ConnectionResetError",
    "TimeoutError",
    "FutureTimeoutError",
    "SSLError",
    "SSLZeroReturnError",
    "SSLEOFError",
    "IncompleteReadError",
    "ChunkedEncodingError",
    "RemoteDisconnected",
    "ProtocolError",
}

SUPPORTED_PROVIDERS = (
    "google-genai",
    "openai",
    "anthropic",
    "ollama-cloud",
    "ollama",
    "openai-compatible",
)

# Base Python tooling is now defined in profiles/python.yaml under
# sandbox.default_deps.  Each language profile owns its own test/lint/type deps.

# Loop / ceiling constants
MAX_SANDBOX_LOOPS = 5
MAX_CONTRACT_LOOPS = 2

# Server-side hard wall-clock deadline per task. A backstop that auto-cancels a
# task even if the client died without cancelling it.
SERVER_TASK_DEADLINE = CONFIG.get("server", {}).get("task_deadline_seconds", 3600)


def _profile_from_state(state: AgenticState) -> Profile:
    """Return the active language profile for this task.

    When an external contract is present (from the planner), the profile's
    file paths are overridden to match the contract's source_file and test_file.
    """
    profile = get_profile(state.get("profile_name"))
    external = state.get("external_contract")
    if external and isinstance(external, dict):
        source_file = external.get("source_file")
        test_file = external.get("test_file")
        if source_file or test_file:
            profile = profile.with_overrides(source_file, test_file)
    return profile


# =============================================================================
# Host identity propagation for sandbox containers
# =============================================================================
def _host_identity() -> Tuple[int, int]:
    """Return the (uid, gid) that sandbox containers should run as.

    When the orchestrator itself runs inside Docker as root, `os.getuid()`
    returns 0, so the sandbox would run as root too. Instead we read PUID/PGID
    from the environment, which docker-compose.yml sets from the host user.
    """
    try:
        uid = int(os.environ.get("PUID", os.getuid() if hasattr(os, "getuid") else 0))
    except (TypeError, ValueError):
        uid = 0
    try:
        gid = int(os.environ.get("PGID", os.getgid() if hasattr(os, "getgid") else 0))
    except (TypeError, ValueError):
        gid = 0
    return uid, gid


# =============================================================================
# LLM factory
# =============================================================================
def _build_llm(node_name):
    if node_name not in CONFIG.get("nodes", {}):
        raise ValueError(f"No config entry for node '{node_name}' in llm_config.yaml")

    node_config = CONFIG["nodes"][node_name]
    provider = node_config.get("provider")
    model = node_config.get("model")
    temperature = node_config.get("temperature", 0)
    api_key_env_var = node_config.get("api_key_env_var")

    def require_key(default_env=None):
        env = api_key_env_var or default_env
        key = os.getenv(env) if env else None
        if not key:
            raise ValueError(
                f"Node '{node_name}' uses provider '{provider}' but env var "
                f"'{env}' is not set. Add it to your .env file."
            )
        return key

    if provider == "google-genai":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=require_key("GOOGLE_API_KEY"),
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            openai_api_key=require_key("OPENAI_API_KEY"),
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            anthropic_api_key=require_key("ANTHROPIC_API_KEY"),
        )

    if provider in ("ollama-cloud", "ollama"):
        from langchain_ollama import ChatOllama

        if provider == "ollama-cloud":
            base_url = OLLAMA_CLOUD_HOST
            key = require_key("OLLAMA_API_KEY")
            client_kwargs = {
                "headers": {"Authorization": f"Bearer {key}"},
                "timeout": OLLAMA_TIMEOUT,
                "limits": httpx.Limits(max_connections=16, max_keepalive_connections=8),
            }
        else:
            base_url = os.getenv(
                api_key_env_var or "OLLAMA_BASE_URL", "http://localhost:11434"
            )
            client_kwargs = {
                "timeout": OLLAMA_TIMEOUT,
                "limits": httpx.Limits(max_connections=8, max_keepalive_connections=4),
            }

        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
            streaming=True,
            client_kwargs=client_kwargs,
        )

    if provider == "openai-compatible":
        from langchain_openai import ChatOpenAI

        base_url = node_config.get("base_url")
        if not base_url:
            raise ValueError(
                f"Node '{node_name}' uses provider 'openai-compatible' but no "
                f"'base_url' is set in llm_config.yaml."
            )
        key = os.getenv(api_key_env_var) if api_key_env_var else None
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=key or "not-needed",
        )

    raise ValueError(
        f"Node '{node_name}' has unsupported provider '{provider}'. "
        f"Supported providers: {', '.join(SUPPORTED_PROVIDERS)}"
    )


@lru_cache(maxsize=None)
def get_llm(node_name):
    """Lazily build (and cache) the chat client for a node."""
    return _build_llm(node_name)


def validate_config():
    """Eagerly instantiate every configured node so misconfiguration surfaces
    at setup time rather than mid-run."""
    problems = []
    for node_name in CONFIG.get("nodes", {}):
        try:
            get_llm(node_name)
        except Exception as e:  # noqa: BLE001
            problems.append((node_name, str(e)))
    return problems


# =============================================================================
# Token estimation
# =============================================================================
def _is_transient_llm_error(e: Exception) -> bool:
    """Return True if an exception looks like a transient provider fault."""
    if isinstance(e, FutureTimeoutError):
        return True
    # Walk the full __cause__ chain — langchain/httpx/requests wrap errors
    # multiple levels deep (e.g. httpx.ReadError → ssl.SSLEOFError).
    candidates = [e]
    seen = set()
    cur = e
    while True:
        inner = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
        if inner is None or id(inner) in seen:
            break
        seen.add(id(inner))
        candidates.append(inner)
        cur = inner

    for inner in candidates:
        err_name = type(inner).__name__
        err_text = f"{err_name}: {inner}"
        if err_name in TRANSIENT_ERROR_NAMES:
            return True
        # Common substring indicators (English + some provider-specific text).
        lowered = err_text.lower()
        if any(tok in lowered for tok in (
            "timeout", "timed out", "temporarily unavailable", "rate limit",
            "try again", "server error", "bad gateway", "gateway timeout",
            "origin_response_timeout",
            "ssl", "unexpected_eof", "eof occurred", "violation of protocol",
            "incomplete read", "chunked encoding", "connection reset",
            "remote disconnected", "protocol error", "empty response",
            "too many concurrent requests",
        )):
            return True
        # Extract HTTP status code from the text if present.
        import re as _re
        for m in _re.finditer(r"\b(\d{3})\b", err_text):
            status = int(m.group(1))
            if status in TRANSIENT_HTTP_STATUSES or (500 <= status < 600):
                return True
    return False


def _estimate_tokens(text: str) -> int:
    """Rough, fast token estimate for diagnostics only."""
    if not text:
        return 0
    return max(1, int(len(text) / 3.5))


def _count_message_tokens(messages) -> int:
    total = 0
    for m in messages:
        content = getattr(m, "content", "")
        if isinstance(content, str):
            total += _estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += _estimate_tokens(part.get("text", ""))
                else:
                    total += _estimate_tokens(str(part))
    return total


# =============================================================================
# LLM invocation with retry
# =============================================================================
def _invoke_with_retry(
    node_name: str,
    messages,
    workspace: str = "",
    max_attempts: int | None = None,
    timeout_seconds: float | None = None,
    state: dict | None = None,
) -> tuple[str, list[dict]]:
    """Call a node's LLM with a hard per-attempt timeout, retrying on transient
    errors (network drops, SSL EOF, incomplete reads, provider 5xx, slow streams).

    Returns (response_content, llm_usage_entries). If all attempts fail, raises
    LLMUnavailableError so the task fails fast with a clear message.
    """
    node_config = CONFIG.get("nodes", {}).get(node_name, {})
    max_attempts = max_attempts if max_attempts is not None else node_config.get("max_attempts", 3)
    timeout_seconds = timeout_seconds if timeout_seconds is not None else node_config.get("timeout_seconds", 600.0)
    provider = node_config.get("provider")
    model = node_config.get("model", "unknown")
    ollama_providers = ("ollama-cloud", "ollama")
    invoke_kwargs = {"stream": True} if provider in ollama_providers else {}

    # Cooperative cancellation: state may carry an event/executor from the runner.
    cancel_event = state.get("cancel_event") if state else None
    executor = state.get("llm_executor") if state else None

    input_tokens = _count_message_tokens(messages)
    usage_entries: list[dict] = []
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError("Task cancellation requested before LLM attempt")

        started = time.perf_counter()
        try:
            llm = get_llm(node_name)

            # Streaming liveness watchdog: consume the LLM stream chunk-by-chunk
            # in a worker thread and abort if no chunk arrives within
            # `stream_idle_seconds`.  This catches mid-stream Ollama Cloud stalls
            # (observed hanging ~900s emitting zero tokens) while letting slow-
            # but-progressing generations run up to the total `timeout_seconds`
            # wall-clock budget.  Non-streaming providers fall back to a single
            # chunk that returns the whole response, so the watchdog still
            # bounds them by `stream_idle_seconds` for the first byte and
            # `timeout_seconds` for the whole call.
            stream_idle = float(
                node_config.get("stream_idle_seconds", STREAM_IDLE_DEFAULT)
            )

            chunk_queue: "queue.Queue" = queue.Queue()
            use_streaming = bool(invoke_kwargs.get("stream"))

            def _stream_call():
                try:
                    if use_streaming:
                        # Explicit stream() returns an iterator of AIMessageChunk;
                        # consume it here so the watchdog sees each chunk.
                        metadata: dict = {}
                        parts: list[str] = []
                        for chunk in llm.stream(messages):
                            piece = getattr(chunk, "content", None)
                            if isinstance(piece, str) and piece:
                                parts.append(piece)
                                chunk_queue.put(("chunk", piece))
                            cm = getattr(chunk, "response_metadata", None)
                            if isinstance(cm, dict) and cm:
                                metadata = cm
                        content = "".join(parts)
                        chunk_queue.put(("done", (content, metadata)))
                    else:
                        response = llm.invoke(messages)
                        content = response.content if hasattr(response, "content") else ""
                        metadata = getattr(response, "response_metadata", {}) or {}
                        chunk_queue.put(("done", (content, metadata)))
                except BaseException as e:  # noqa: BLE001 - propagate to consumer
                    chunk_queue.put(("error", e))

            if executor is not None:
                future = executor.submit(_stream_call)
            else:
                local_executor = ThreadPoolExecutor(max_workers=1)
                future = local_executor.submit(_stream_call)

            content = ""
            metadata: dict = {}
            stream_error: Exception | None = None
            stream_done = False
            try:
                while not stream_done:
                    remaining = timeout_seconds - (time.perf_counter() - started)
                    if remaining <= 0:
                        raise FutureTimeoutError(
                            f"Wall-clock timeout after {timeout_seconds}s"
                        )
                    wait = min(stream_idle, remaining)
                    try:
                        kind, payload = chunk_queue.get(timeout=wait)
                    except queue.Empty:
                        raise FutureTimeoutError(
                            f"Stream idle timeout after {wait:.0f}s with no chunk"
                        )
                    if kind == "chunk":
                        content += payload
                    elif kind == "done":
                        content, metadata = payload
                        stream_done = True
                    elif kind == "error":
                        stream_error = payload
                        stream_done = True
            finally:
                if executor is None:
                    local_executor.shutdown(wait=False, cancel_futures=True)
                elif cancel_event is not None and cancel_event.is_set():
                    future.cancel()

            if stream_error is not None:
                raise stream_error

            elapsed = time.perf_counter() - started
            output_tokens = _estimate_tokens(content)
            prompt_tokens = metadata.get("prompt_eval_count") if isinstance(metadata, dict) else None
            completion_tokens = metadata.get("eval_count") if isinstance(metadata, dict) else None
            if isinstance(prompt_tokens, (int, float)) and prompt_tokens > 0:
                input_tokens = int(prompt_tokens)
            if isinstance(completion_tokens, (int, float)) and completion_tokens > 0:
                output_tokens = int(completion_tokens)
            provider_duration_ns = metadata.get("total_duration") if isinstance(metadata, dict) else None
            provider_duration = (
                round(provider_duration_ns / 1_000_000_000, 3)
                if isinstance(provider_duration_ns, (int, float)) and provider_duration_ns > 0
                else None
            )
            usage = {
                "node": node_name,
                "model": model,
                "provider": provider,
                "attempt": attempt,
                "status": "success",
                "duration_seconds": round(elapsed, 3),
                "provider_duration_seconds": provider_duration,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "error": None,
                "response_metadata": metadata,
            }
            usage_entries.append(usage)
            if workspace:
                _diag(
                    workspace,
                    node_name,
                    f"LLM success attempt {attempt}/{max_attempts}: "
                    f"{usage['duration_seconds']}s, "
                    f"~{usage['total_tokens']} tokens "
                    f"(in={usage['input_tokens']}, out={usage['output_tokens']})",
                )

            # Treat an empty successful response as a transient LLM failure so we
            # retry instead of immediately failing the whole pipeline (observed when
            # Ollama returns 0 output tokens under concurrent load).
            if not content.strip():
                raise Exception("LLM returned empty response (0 output tokens)")

            return content, usage_entries
        except FutureTimeoutError as e:
            last_error = e
            elapsed = time.perf_counter() - started
            usage_entries.append({
                "node": node_name,
                "model": model,
                "provider": provider,
                "attempt": attempt,
                "status": "wallclock_timeout",
                "duration_seconds": round(elapsed, 3),
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "total_tokens": input_tokens,
                "error": str(e) or f"Wall-clock timeout after {timeout_seconds}s",
                "response_metadata": {},
            })
            if workspace:
                _diag(
                    workspace,
                    node_name,
                    f"LLM attempt {attempt}/{max_attempts} timed out: {e}",
                )
        except Exception as e:
            last_error = e
            elapsed = time.perf_counter() - started
            usage_entries.append({
                "node": node_name,
                "model": model,
                "provider": provider,
                "attempt": attempt,
                "status": "error",
                "duration_seconds": round(elapsed, 3),
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "total_tokens": input_tokens,
                "error": f"{type(e).__name__}: {e}",
                "response_metadata": {},
            })
            if workspace:
                _diag(
                    workspace,
                    node_name,
                    f"LLM attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {e}",
                )
        if attempt < max_attempts:
            # Quicker backoff: the provider is often slow, not unavailable.
            backoff = (2.0, 5.0, 10.0)[min(attempt - 1, 2)]
            time.sleep(backoff)

    # Classify the final failure once instead of re-evaluating in callers.
    transient = _is_transient_llm_error(last_error) if last_error else False
    raise LLMUnavailableError(node_name, max_attempts, last_error, usage_entries, is_transient=transient) from last_error


# =============================================================================
# Logging helpers
# =============================================================================
def _think(workspace: str, node: str, message: str) -> list[str]:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{node}] {message}"
    log_path = os.path.join(workspace, "task.log")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        _chown_file_to_host(log_path)
    except Exception:
        pass
    return [line]


def _diag(workspace: str, node: str, detail: str) -> None:
    if not detail:
        return
    indented = "\n".join("  " + line for line in detail.strip().splitlines())
    log_path = os.path.join(workspace, "task.log")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(indented + "\n")
        _chown_file_to_host(log_path)
    except Exception as e:
        import sys
        print(f"[_diag] write failed for {workspace}: {e}", file=sys.stderr)


# =============================================================================
# Node history decorator
# =============================================================================
def _node_history_entry(
    node_name: str,
    wall_started: float,
    wall_ended: float,
    perf_duration: float | None = None,
    **extras,
) -> dict:
    return {
        "node": node_name,
        "started_at": datetime.fromtimestamp(wall_started, tz=timezone.utc).isoformat(),
        "ended_at": datetime.fromtimestamp(wall_ended, tz=timezone.utc).isoformat(),
        "duration_seconds": round(perf_duration, 3) if perf_duration is not None else round(wall_ended - wall_started, 3),
        **extras,
    }


def node_with_history(func):
    def wrapper(state: AgenticState):
        node_name = func.__name__
        wall_started = time.time()
        perf_started = time.perf_counter()
        result = func(state) or {}
        perf_ended = time.perf_counter()
        wall_ended = time.time()
        extras = result.pop("diagnostics", {}) if isinstance(result, dict) else {}
        entry = _node_history_entry(
            node_name, wall_started, wall_ended, perf_ended - perf_started, **extras
        )
        result["node_history"] = [entry]
        return result

    return wrapper


def _handle_llm_unavailable(
    node_name: str,
    state: AgenticState,
    exc: LLMUnavailableError,
    workspace: str,
):
    """Decide whether to retry the same node or mark the task infra-exhausted.

    Returns a dict update for the LangGraph state. When transient retries
    remain, the update sets next_node back to the same node and sleeps the
    configured backoff. When exhausted, it sets llm_infra_exhausted so the
    API layer can report status=infra_exhausted instead of failed.
    """
    retries = state.get("llm_infra_retries", {})
    count = retries.get(node_name, 0)
    usage_entries = list(getattr(exc, "usage_entries", []))

    if exc.is_transient and count < INFRA_MAX_RETRIES:
        new_count = count + 1
        backoff = INFRA_BACKOFFS[min(count, len(INFRA_BACKOFFS) - 1)]
        diag = (
            f"Transient LLM fault for {node_name} (attempt {new_count}/{INFRA_MAX_RETRIES}); "
            f"retrying after {backoff}s backoff"
        )
        _diag(workspace, node_name, diag)
        time.sleep(backoff)
        return {
            "llm_infra_retries": {**retries, node_name: new_count},
            "llm_usage": usage_entries,
            "sandbox_errors": "",
            "next_node": node_name,
            "thoughts": _think(workspace, node_name, diag),
        }

    # Exhausted or permanent error.
    kind = "transient" if exc.is_transient else "permanent"
    diag = (
        f"LLM infrastructure {kind} fault for {node_name} exhausted "
        f"({count} infra retries used): {exc.cause}"
    )
    _diag(workspace, node_name, diag)
    return {
        "llm_infra_retries": {**retries, node_name: count},
        "llm_infra_exhausted": True,
        "llm_usage": usage_entries,
        "sandbox_errors": diag,
        "next_node": "FINISH",
        "thoughts": _think(workspace, node_name, diag),
    }


# =============================================================================
# Workspace helpers
# =============================================================================
def resolve_host_path(container_path: str) -> str:
    """Resolve host-side path for a container-internal path."""
    abs_path = os.path.abspath(container_path)
    try:
        container_id = socket.gethostname()
        result = subprocess.run(
            ["docker", "inspect", container_id, "--format", "{{json .Mounts}}"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        mounts = json.loads(result.stdout)
        best = None
        for mount in mounts:
            dest = mount.get("Destination", "")
            if abs_path.startswith(dest) and (
                best is None or len(dest) > len(best["Destination"])
            ):
                best = mount
        if best:
            relative = abs_path[len(best["Destination"]):]
            return best["Source"] + relative
    except Exception:
        pass
    return abs_path


def cleanup_sandbox_for_task(task_id: str) -> None:
    """Remove any Docker containers spawned for a given task ID.

    Sandbox containers are named `{task_id}-install-{loop}` and
    `{task_id}-verify-{loop}`. This is called by the runner on task end,
    cancellation, or timeout to ensure nothing lingers.
    """
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        names = [n.strip() for n in result.stdout.splitlines() if n.strip().startswith(task_id) and ("-install-" in n or "-verify-" in n)]
        if names:
            subprocess.run(
                ["docker", "rm", "-f", "--volumes"] + names,
                capture_output=True,
                text=True,
                timeout=30,
            )
    except Exception:
        pass


def setup_workspace(workspace: str, profile: Profile) -> None:
    """Write the deterministic sandbox scaffolding for a task.

    These files are written once per workspace and then left alone so that
    retry loops (test_designer or coder failures) do not wipe third-party
    dependencies added by the coder node.

    The workspace is chowned to the host user identity so that sandbox
    containers running as that user can write into it without running as root.

    Scaffolding is loaded from the active profile so each language gets the
    correct test runner configuration.
    """
    os.makedirs(workspace, exist_ok=True)

    for filename, content in profile.setup_files().items():
        if not os.path.exists(os.path.join(workspace, filename)):
            _write_workspace_file(workspace, filename, content)

    deps_file = profile.sandbox_value("deps_file", "requirements.txt")
    default_deps = profile.sandbox_value("default_deps", [])
    requirements = "\n".join(default_deps) + "\n" if default_deps else ""
    if requirements and not os.path.exists(os.path.join(workspace, deps_file)):
        _write_workspace_file(workspace, deps_file, requirements)

    # Ensure the host user (not root) owns the workspace and scaffolding files
    # so the sandbox can run unprivileged. If we are already that user, chown
    # is a harmless no-op.
    _chown_to_host(workspace)


def _chown_to_host(path: str) -> None:
    """Recursively chown a path to the host user identity (PUID/PGID)."""
    host_uid, host_gid = _host_identity()
    if not host_uid and not host_gid:
        return
    try:
        for root, dirs, files in os.walk(path):
            for d in dirs:
                os.chown(os.path.join(root, d), host_uid, host_gid)
            for f in files:
                os.chown(os.path.join(root, f), host_uid, host_gid)
        os.chown(path, host_uid, host_gid)
    except Exception:
        pass


def _chown_file_to_host(path: str) -> None:
    """Chown a single file to the host user identity, if one is configured."""
    host_uid, host_gid = _host_identity()
    if not host_uid and not host_gid:
        return
    try:
        os.chown(path, host_uid, host_gid)
    except Exception:
        pass


# ------------------------------------------------------------------------------
# Dependency cache helpers
# ------------------------------------------------------------------------------
_DEPS_CACHE_INSTALLED_MARKER = ".deps_cache_installed"


def _deps_requirements_hash(workspace: str, deps_file: str = "requirements.txt") -> str:
    """Return a stable hash of the normalized requirements file content."""
    path = os.path.join(workspace, deps_file)
    if not os.path.exists(path):
        return "no_requirements"
    content = open(path, "r", encoding="utf-8").read()
    deps = sorted(line.strip() for line in content.splitlines() if line.strip())
    return hashlib.sha256("\n".join(deps).encode("utf-8")).hexdigest()[:16]


def _resolve_dep_cache_dirs(workspace: str, tag: str) -> Tuple[str, str]:
    """Return (container_path, host_path) for a tagged dependency cache entry.

    The cache lives next to the workspaces directory so it is on the same shared
    volume that is mounted into worker containers, but outside any per-task
    workspace. Path: <workspace_parent>/.deps_cache/<tag>/<requirements_hash>.

    The container path is used for all host-side file operations (makedirs,
    chown, marker checks). The host path is only for ``docker run -v`` bind mounts.
    """
    container_root = os.path.dirname(workspace)
    deps_hash = _deps_requirements_hash(workspace)
    container_cache_dir = os.path.join(container_root, ".deps_cache", tag, deps_hash)
    host_cache_dir = resolve_host_path(container_cache_dir)
    return container_cache_dir, host_cache_dir


def _dep_cache_hit(cache_dir: str) -> bool:
    """True if a valid, non-empty tagged cache entry exists."""
    marker = os.path.join(cache_dir, _DEPS_CACHE_INSTALLED_MARKER)
    if not os.path.exists(marker):
        return False
    # A non-empty dir beyond the marker means packages are present.
    for _ in os.scandir(cache_dir):
        return True
    return False


def _write_workspace_file(workspace: str, filename: str, content: str) -> None:
    filepath = os.path.join(workspace, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def _clear_directory(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)

def write_manifest_to_disk(workspace: str, manifest: Dict[str, str]) -> None:
    """Write the current manifest files to disk, clearing stale src/ and tests/ first."""
    _clear_directory(os.path.join(workspace, "src"))
    _clear_directory(os.path.join(workspace, "tests"))
    for filename, code in manifest.items():
        filepath = os.path.join(workspace, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
    # The orchestrator may be root; make sure the sandbox (host user) can read
    # and write everything we just produced.
    _chown_to_host(os.path.join(workspace, "src"))
    _chown_to_host(os.path.join(workspace, "tests"))


def cleanup_workspace_deps(workspace: str) -> None:
    """Remove ephemeral dependency trees after a task finishes.

    The generated source/tests and task.log are intentionally kept for scoring
    and inspection, but the dependency install target and transient caches are
    large and not needed once the task is done.
    """
    for subdir in (".deps", ".tmp", ".hypothesis"):
        _clear_directory(os.path.join(workspace, subdir))


def read_file_from_disk(workspace: str, filename: str) -> str:
    filepath = os.path.join(workspace, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


# =============================================================================
# File manifest parsing
# =============================================================================
def _parse_file_tags(content: str) -> Dict[str, str]:
    """Parse <file name="...">...</file> tags."""
    files: Dict[str, str] = {}
    for match in re.finditer(
        r'<file name=[\'"](.*?)[\'"]>\s*(.*?)\s*</file>',
        content,
        re.DOTALL | re.IGNORECASE,
    ):
        code = match.group(2).strip()
        code = re.sub(r"^```[a-zA-Z]*\n?", "", code).rstrip("`").strip()
        filename = match.group(1).strip()
        files[filename] = code
    return files


# =============================================================================
# NODE: test_designer
# =============================================================================
@node_with_history
def test_designer(state: AgenticState):
    prompt = state["user_prompt"]
    workspace = state.get("workspace_dir", "")
    manifest = state.get("file_manifest", {})
    contract_critique = state.get("contract_critique", [])
    sandbox_errors = state.get("sandbox_errors", "")
    contract_loop = state.get("contract_loop_count", 0)
    contract_code = state.get("contract_code") or ""

    profile = _profile_from_state(state)
    setup_workspace(workspace, profile)
    test_main_path = profile.file_path("test_main")

    system = profile.prompt("test_designer_system")

    user_prompt = f"User prompt:\n{prompt}"
    user_prompt += (
        "\n\nLanguage-specific guidance: If the prompt asks for an expression evaluator, "
        "calculator, parser, or any task that evaluates arithmetic or structured strings, "
        "generate raw expression strings and compute expected values using the target "
        "language's standard evaluator (e.g., Python's eval() with a safe scope). Do not "
        "hand-write AST renderers or string-composition logic that must preserve "
        "operator precedence; let the language parser be the source of truth."
    )

    if contract_code:
        user_prompt += (
            "\n\nContract code (authoritative signature and module-level imports "
            "extracted from the prompt; your tests MUST call task_func with these "
            "exact parameter names and defaults and rely on these import scopes. "
            "The contract defines how to CALL the function, NOT what to ASSERT about "
            "the result — behavioral-property assertions remain mandatory regardless "
            "of the contract's type hints):\n"
            f"```python\n{contract_code}\n```"
        )

    external_contract = state.get("external_contract") or {}
    if external_contract and external_contract.get("exports"):
        exports_block = "\n".join(f"  - {e}" for e in external_contract["exports"])
        user_prompt += (
            f"\n\nExternal contract from the solutions architect (AUTHORITATIVE — "
            f"your tests MUST import from {profile.file_path('source_main').replace('/', '.').removesuffix('.py')} "
            f"and test these exact symbols):\n{exports_block}"
        )

    prior_tests = manifest.get(test_main_path, "")
    if prior_tests:
        user_prompt += (
            f"\n\nYour previous {test_main_path} (contract loop {contract_loop}):\n"
            f"```python\n{prior_tests}\n```"
        )
    if contract_critique:
        user_prompt += (
            "\n\nContract compatibility critiques (the skeleton maker could not match these; "
            "revise your tests to be compatible with the prompt):\n"
            + "\n".join(f"- {c}" for c in contract_critique)
        )
    if sandbox_errors:
        user_prompt += (
            f"\n\nThe last sandbox run failed with the following errors. "
            f"If the fault is in the tests (syntax, imports, hypothesis health check, vacuity), "
            f"revise the tests:\n{sandbox_errors[:4000]}"
        )

    try:
        content, llm_usage = _invoke_with_retry(
            "test_designer",
            [SystemMessage(content=system), HumanMessage(content=user_prompt)],
            workspace,
            state=state,
        )
    except LLMUnavailableError as e:
        return _handle_llm_unavailable("test_designer", state, e, workspace)

    files = _parse_file_tags(content)

    # Validate that the required test files exist.
    test_main_path = profile.file_path("test_main")
    test_init_path = profile.file_path("test_init")
    required = {test_main_path, test_init_path}
    missing = required - set(files.keys())
    if missing:
        _diag(
            workspace,
            "test_designer",
            f"Missing required files: {missing}\nRaw response (first 2000 chars):\n{content[:2000]}",
        )
        return {
            "sandbox_errors": f"Test Designer Error: missing files {sorted(missing)}",
            "llm_usage": llm_usage,
            "next_node": "FINISH",
            "thoughts": _think(
                workspace,
                "test_designer",
                f"ERROR: missing files {sorted(missing)}",
            ),
        }

    _diag(
        workspace,
        "test_designer",
        f"Generated files:\n"
        + "\n".join(f"  {fname} ({len(code)} chars)" for fname, code in sorted(files.items())),
    )

    return {
        "file_manifest": files,
        "sandbox_errors": "",
        "sandbox_diagnostics": {},
        "sandbox_loop_count": 0,
        "llm_usage": llm_usage,
        "next_node": "skeleton_maker",
        "thoughts": _think(
            workspace,
            "test_designer",
            f"Wrote {len(files)} files: {', '.join(sorted(files.keys()))}",
        ),
    }


# =============================================================================
# NODE: skeleton_maker
# =============================================================================
@node_with_history
def skeleton_maker(state: AgenticState):
    prompt = state["user_prompt"]
    workspace = state.get("workspace_dir", "")
    manifest = state.get("file_manifest", {})
    contract_loop = state.get("contract_loop_count", 0)
    contract_code = state.get("contract_code") or ""
    external_contract = state.get("external_contract") or {}

    profile = _profile_from_state(state)
    setup_workspace(workspace, profile)

    test_main_path = profile.file_path("test_main")
    test_code = manifest.get(test_main_path, "")
    if not test_code:
        return {
            "sandbox_errors": f"Skeleton Maker Error: missing {test_main_path}",
            "next_node": "FINISH",
            "thoughts": _think(
                workspace,
                "skeleton_maker",
                f"ERROR: missing {test_main_path} before skeleton creation",
            ),
        }

    system = profile.prompt("skeleton_maker_system")

    user_prompt = f"User prompt:\n{prompt}"
    if contract_code:
        user_prompt += (
            "\n\nContract code (use VERBATIM — keep these module-level imports at the "
            "top of __SOURCE_MAIN__ and adopt the exact function signature, parameter "
            "names, defaults, and return annotation; only replace the body with pass "
            "or raise NotImplementedError. Append `  # noqa: F401` to any contract "
            "import that the function body will not reference, so linters do not strip "
            "it. The contract is the floor; also stub any additional symbols the tests "
            "require that the contract does not define):\n"
            f"```python\n{contract_code}\n```"
        )
    if external_contract and external_contract.get("exports"):
        exports_block = "\n".join(f"  - {e}" for e in external_contract["exports"])
        user_prompt += (
            f"\n\nExternal contract from the solutions architect (AUTHORITATIVE — "
            f"these are the exact symbols this module must export. Scaffold them "
            f"with the given signatures; do not rename or omit any):\n{exports_block}"
        )
    user_prompt += f"\n\nTests/{test_main_path}:\n{test_code}"

    try:
        content, llm_usage = _invoke_with_retry(
            "skeleton_maker",
            [SystemMessage(content=system), HumanMessage(content=user_prompt)],
            workspace,
            state=state,
        )
    except LLMUnavailableError as e:
        return _handle_llm_unavailable("skeleton_maker", state, e, workspace)

    files = _parse_file_tags(content)

    # Validate that the required skeleton files exist.
    source_main_path = profile.file_path("source_main")
    source_init_path = profile.file_path("source_init")
    required = {source_main_path, source_init_path}
    missing = required - set(files.keys())
    if missing:
        _diag(
            workspace,
            "skeleton_maker",
            f"Missing required files: {missing}\nRaw response (first 2000 chars):\n{content[:2000]}",
        )
        return {
            "sandbox_errors": f"Skeleton Maker Error: missing files {sorted(missing)}",
            "llm_usage": llm_usage,
            "next_node": "FINISH",
            "thoughts": _think(
                workspace,
                "skeleton_maker",
                f"ERROR: missing files {sorted(missing)}",
            ),
        }

    # Parse contract verdict.
    verdict_match = re.search(
        r"<contract_verdict>\s*(.*?)\s*</contract_verdict>",
        content,
        re.DOTALL | re.IGNORECASE,
    )
    verdict = {"compatible": True, "critique": []}
    if verdict_match:
        try:
            verdict = json.loads(verdict_match.group(1))
        except Exception as e:
            _diag(
                workspace,
                "skeleton_maker",
                f"Failed to parse contract verdict: {e}\nRaw verdict: {verdict_match.group(1)[:500]}",
            )
            verdict = {"compatible": False, "critique": ["Could not parse contract verdict"]}

    compatible = bool(verdict.get("compatible", True))
    raw_critique = verdict.get("critique", []) or []
    if not isinstance(raw_critique, list):
        raw_critique = [str(raw_critique)]
    critique = [str(c) for c in raw_critique if c]

    # When an external contract is present, the planner's contract is
    # authoritative — skip the contract loop and proceed to coder regardless.
    if external_contract:
        compatible = True
        critique = []

    # Merge skeleton into the existing manifest (tests are preserved).
    new_manifest = {**manifest, **files}

    _diag(
        workspace,
        "skeleton_maker",
        f"Generated skeleton:\n"
        + "\n".join(f"  {fname} ({len(code)} chars)" for fname, code in sorted(files.items()))
        + f"\nContract compatible: {compatible}"
        + (f"\nCritique: {critique}" if critique else ""),
    )

    if compatible:
        return {
            "file_manifest": new_manifest,
            "sandbox_errors": "",
            "sandbox_diagnostics": {},
            "contract_critique": [],
            "llm_usage": llm_usage,
            "next_node": "coder",
            "thoughts": _think(
                workspace,
                "skeleton_maker",
                f"Skeleton compatible → coder",
            ),
        }

    new_contract_loop = contract_loop + 1
    if new_contract_loop <= MAX_CONTRACT_LOOPS:
        return {
            "file_manifest": manifest,
            "sandbox_errors": "",
            "sandbox_diagnostics": {},
            "contract_loop_count": new_contract_loop,
            "contract_critique": critique,
            "llm_usage": llm_usage,
            "next_node": "test_designer",
            "thoughts": _think(
                workspace,
                "skeleton_maker",
                f"Contract incompatible (loop {new_contract_loop}/{MAX_CONTRACT_LOOPS}) → test_designer: {critique}",
            ),
        }

    # Contract loop ceiling reached: proceed to coder with a flag.
    return {
        "file_manifest": new_manifest,
        "sandbox_errors": "",
        "sandbox_diagnostics": {},
        "contract_loop_count": new_contract_loop,
        "contract_critique": critique,
        "contract_exhausted": True,
        "llm_usage": llm_usage,
        "next_node": "coder",
        "thoughts": _think(
            workspace,
            "skeleton_maker",
            f"Contract loop ceiling reached ({MAX_CONTRACT_LOOPS}); proceeding to coder with exhausted flag",
        ),
    }


# =============================================================================
# NODE: coder
# =============================================================================
@node_with_history
def coder(state: AgenticState):
    prompt = state["user_prompt"]
    workspace = state.get("workspace_dir", "")
    manifest = state.get("file_manifest", {})
    sandbox_errors = state.get("sandbox_errors", "")

    profile = _profile_from_state(state)
    test_main_path = profile.file_path("test_main")
    source_main_path = profile.file_path("source_main")
    deps_file = profile.sandbox_value("deps_file", "requirements.txt")

    test_code = manifest.get(test_main_path, "")
    stub_code = manifest.get(source_main_path, "")

    if not test_code or not stub_code:
        return {
            "sandbox_errors": f"Coder Error: missing {test_main_path} or {source_main_path} skeleton",
            "next_node": "FINISH",
            "thoughts": _think(
                workspace,
                "coder",
                "ERROR: missing frozen test or skeleton before coding",
            ),
        }

    system = profile.prompt("coder_system")

    user_prompt = f"User prompt:\n{prompt}\n\nFrozen {test_main_path}:\n{test_code}\n\nSkeleton {source_main_path}:\n{stub_code}"
    if sandbox_errors:
        user_prompt += (
            f"\n\nThe last sandbox run failed with these errors. Fix the implementation, "
            f"not the tests:\n{sandbox_errors[:4000]}"
        )

    try:
        content, llm_usage = _invoke_with_retry(
            "coder",
            [SystemMessage(content=system), HumanMessage(content=user_prompt)],
            workspace,
            state=state,
        )
    except LLMUnavailableError as e:
        return _handle_llm_unavailable("coder", state, e, workspace)

    new_files = _parse_file_tags(content)

    if source_main_path not in new_files:
        _diag(
            workspace,
            "coder",
            f"No {source_main_path} in coder output\nRaw response (first 2000 chars):\n{content[:2000]}",
        )
        return {
            "sandbox_errors": f"Coder Error: failed to produce {source_main_path}",
            "llm_usage": llm_usage,
            "next_node": "FINISH",
            "thoughts": _think(
                workspace,
                "coder",
                f"ERROR: no {source_main_path} produced",
            ),
        }

    # Preserve frozen test files and init files; only overwrite the source main file
    # and dependency file.
    source_init_path = profile.file_path("source_init")
    preserved = {
        k: v
        for k, v in manifest.items()
        if k.startswith(profile.file_path("test_dir") + "/") or k in {source_init_path}
    }
    final_manifest = {**preserved, source_main_path: new_files[source_main_path]}

    # Merge any third-party requirements from the coder on top of the base sandbox deps
    # so profile tooling is never lost on a retry.
    base_reqs = set(profile.sandbox_value("default_deps", []))
    if deps_file in new_files:
        extra_lines = [
            line.strip()
            for line in new_files[deps_file].splitlines()
            if line.strip() and line.strip() not in base_reqs
        ]
        final_manifest[deps_file] = "\n".join(sorted(base_reqs | set(extra_lines))) + "\n"
    elif deps_file in manifest:
        final_manifest[deps_file] = manifest[deps_file]

    _diag(
        workspace,
        "coder",
        f"Implementation updated:\n"
        + "\n".join(f"  {fname} ({len(code)} chars)" for fname, code in sorted(final_manifest.items())),
    )

    return {
        "file_manifest": final_manifest,
        "sandbox_errors": "",
        "sandbox_diagnostics": {},
        "llm_usage": llm_usage,
        "next_node": "sandbox_arbiter",
        "thoughts": _think(
            workspace,
            "coder",
            f"Implemented {source_main_path} ({len(new_files[source_main_path])} chars)",
        ),
    }


# =============================================================================
# NODE: sandbox_arbiter
# =============================================================================
def _docker_run_record(
    loop: int,
    status: str,
    duration: float,
    stdout: str,
    stderr: str,
) -> dict:
    return {
        "loop": loop,
        "status": status,
        "duration_seconds": round(duration, 3),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-1000:],
    }


@node_with_history
def sandbox_arbiter(state: AgenticState):
    workspace = state.get("workspace_dir", ".workspaces/default")
    manifest = state.get("file_manifest", {})
    loop = state.get("sandbox_loop_count", 0) + 1

    write_manifest_to_disk(workspace, manifest)

    profile = _profile_from_state(state)
    IMAGE = profile.sandbox_value("image", "python:3.11-slim")
    memory_limit = profile.sandbox_value("memory", "512m")
    timeout_install = profile.sandbox_value("timeout_install", 90)
    timeout_test = profile.sandbox_value("timeout_test", 120)
    timeout_total = timeout_install + timeout_test + 60  # headroom for tools + tests
    cpus = profile.sandbox_value("cpus", "1.0")

    host_uid, host_gid = _host_identity()

    # Base hardening applied to both install and verification containers.
    # The container runs as the host user, not root, so files it writes to
    # the workspace remain owned by the host user.
    task_id = state.get("task_id", workspace.split("/")[-1])

    container_hardening = [
        f"--user={host_uid}:{host_gid}",
        f"--memory={memory_limit}",
        f"--memory-swap={memory_limit}",
        f"--cpus={cpus}",
        "--pids-limit=64",
        "--ulimit=nofile=1024:1024",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--stop-timeout=10",
        "-e",
        "HOME=/tmp",
        "-e",
        "TMPDIR=/tmp",
        "-e",
        "PYTHONPYCACHEPREFIX=/tmp/pycache",
        "-e",
        f"SANDBOX_LOOP_COUNT={loop}",
    ]

    # Base workspace mount. For tagged dep caches we add a second bind that
    # overrides /workspace/.deps with the shared cache directory.
    workspace_mount = f"{resolve_host_path(workspace)}:/workspace"
    base_mounts = [
        "-v", workspace_mount,
        "-w", "/workspace",
    ]

    # Writable tmpfs for tool caches and pip temp files. Used with --read-only
    # on the verification container (and optionally on install).
    tmpfs_flags = [
        "--tmpfs", "/tmp:noexec,nosuid,size=100m",
        "--tmpfs", "/var/tmp:noexec,nosuid,size=50m",
    ]

    install_env = [
        "-e", "PIP_NO_CACHE_DIR=1",
        "-e", "PIP_DISABLE_PIP_VERSION_CHECK=1",
        "-e", "PIP_ROOT_USER_ACTION=ignore",
        "-e", "TMPDIR=/workspace/.tmp",
    ]

    deps_file = profile.sandbox_value("deps_file", "requirements.txt")

    # Decide whether to use a persistent tagged deps cache or the default
    # clean-wipe per-task .deps directory.
    deps_cache_tag = state.get("deps_cache_tag")
    cache_container_dir: Optional[str] = None
    cache_host_dir: Optional[str] = None
    cache_hit = False
    if deps_cache_tag:
        cache_container_dir, cache_host_dir = _resolve_dep_cache_dirs(workspace, deps_cache_tag)
        cache_hit = _dep_cache_hit(cache_container_dir)

    # Extra bind mounts: for tagged caches, /workspace/.deps points at the
    # shared cache directory. Otherwise pip installs into the workspace itself.
    extra_mounts: List[str] = []
    if cache_host_dir:
        extra_mounts = ["-v", f"{cache_host_dir}:/workspace/.deps"]

    ws = workspace
    _diag(ws, "sandbox_arbiter", f"Running sandbox loop {loop}/{MAX_SANDBOX_LOOPS}")

    started = time.perf_counter()
    try:
        # Phase 1: dependency install (network enabled).
        # Default is clean-slate: every attempt starts with a fresh target dir.
        # With a tagged cache, a populated cache entry skips install entirely.
        MAX_DEP_INSTALL_ATTEMPTS = 2
        install_docker_runs: List[dict] = []
        install_success = False
        last_install_error = ""

        if cache_hit:
            _diag(ws, "sandbox_arbiter", f"Tagged dep cache hit: {cache_container_dir}")
            install_success = True
        else:
            for attempt in range(1, MAX_DEP_INSTALL_ATTEMPTS + 1):
                _diag(ws, "sandbox_arbiter", f"Dependency install attempt {attempt}/{MAX_DEP_INSTALL_ATTEMPTS}")
                # Clean slate for every attempt.
                _clear_directory(os.path.join(workspace, ".deps"))
                _clear_directory(os.path.join(workspace, ".tmp"))
                if cache_container_dir:
                    _clear_directory(cache_container_dir)
                    os.makedirs(cache_container_dir, exist_ok=True)
                    _chown_to_host(cache_container_dir)

                install_script = profile.resolve(profile.sandbox_value("install_command", (
                    "mkdir -p /workspace/.tmp && "
                    f"pip install -q --target /workspace/.deps -r /workspace/{deps_file}"
                )))
                install_cmd = (
                    ["docker", "run", "--rm", "--name", f"{task_id}-install-{loop}-{attempt}"]
                    + container_hardening
                    + base_mounts
                    + extra_mounts
                    + tmpfs_flags
                    + install_env
                    + [IMAGE, "bash", "-c", install_script]
                )

                install_started = time.perf_counter()
                install_res = subprocess.run(
                    install_cmd, capture_output=True, text=True, timeout=timeout_install
                )
                install_duration = time.perf_counter() - install_started
                install_docker_runs.append(
                    _docker_run_record(loop, f"dep_install_attempt_{attempt}", install_duration, install_res.stdout, install_res.stderr)
                )

                if install_res.returncode == 0:
                    install_success = True
                    if cache_container_dir:
                        marker = os.path.join(cache_container_dir, _DEPS_CACHE_INSTALLED_MARKER)
                        open(marker, "w", encoding="utf-8").close()
                        _chown_file_to_host(marker)
                    break

                last_install_error = (
                    f"Dependency install attempt {attempt} failed (loop {loop}):\n"
                    f"STDOUT:\n{install_res.stdout}\nSTDERR:\n{install_res.stderr}"
                )
                _diag(ws, "sandbox_arbiter", last_install_error)
                if attempt < MAX_DEP_INSTALL_ATTEMPTS:
                    time.sleep(2)

        if not install_success:
            return _arbiter_failure(
                ws,
                loop,
                status="infra_fault",
                sandbox_errors=last_install_error,
                docker_runs=install_docker_runs,
            )

        # Verification container: read-only root filesystem, no network, tmpfs for
        # /tmp and /var/tmp. Only /workspace (the bind mount) is writable.
        verify_env = [
            "-e", "PYTHONPATH=/workspace/.deps",
            "-e", "RUFF_CACHE_DIR=/tmp/ruff_cache",
            "-e", "MYPY_CACHE_DIR=/tmp/mypy_cache",
        ]
        arbiter_script = profile.resolve(profile.sandbox_value("verify_command"))
        arbiter_cmd = (
            ["docker", "run", "--rm", "--network", "none", "--read-only", "--name", f"{task_id}-verify-{loop}"]
            + container_hardening
            + base_mounts
            + extra_mounts
            + tmpfs_flags
            + verify_env
            + [IMAGE, "bash", "-c", arbiter_script]
        )

        # Phase 2: arbiter run (network disabled)
        arbiter_started = time.perf_counter()
        arbiter_res = subprocess.run(
            arbiter_cmd, capture_output=True, text=True, timeout=timeout_total
        )
        arbiter_duration = time.perf_counter() - arbiter_started
        total_duration = time.perf_counter() - started

        stdout = arbiter_res.stdout
        stderr = arbiter_res.stderr
        _diag(ws, "sandbox_arbiter", f"Sandbox output:\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")

        # Re-read files in case the profile tooling mutated them.
        profile = _profile_from_state(state)
        source_main_path = profile.file_path("source_main")
        test_main_path = profile.file_path("test_main")
        new_main = read_file_from_disk(workspace, source_main_path)
        new_test = read_file_from_disk(workspace, test_main_path)
        if new_main.strip():
            manifest[source_main_path] = new_main
        if new_test.strip():
            manifest[test_main_path] = new_test

        diagnostics = _parse_sandbox_output(stdout, stderr, loop)

        docker_runs = install_docker_runs + [
            _docker_run_record(loop, diagnostics["sandbox_status"], arbiter_duration, stdout, stderr)
        ]

        if diagnostics["sandbox_status"] == "pass":
            return {
                "file_manifest": manifest,
                "sandbox_errors": "",
                "sandbox_diagnostics": diagnostics,
                "sandbox_loop_count": loop,
                "docker_runs": docker_runs,
                "next_node": "FINISH",
                "thoughts": _think(
                    ws,
                    "sandbox_arbiter",
                    f"Loop {loop}: all sandbox checks PASS ({round(total_duration, 1)}s) — finishing",
                ),
            }

        return _route_sandbox_failure(ws, loop, diagnostics, docker_runs, manifest)

    except subprocess.TimeoutExpired as e:
        total_duration = time.perf_counter() - started
        diag = f"Sandbox arbiter timed out after {round(total_duration, 1)}s"
        _diag(ws, "sandbox_arbiter", diag)
        return _arbiter_failure(
            ws,
            loop,
            status="infra_fault",
            sandbox_errors=diag + f"\n{e}",
            docker_runs=[_docker_run_record(loop, "timeout", total_duration, "", str(e))],
        )
    except Exception as e:
        total_duration = time.perf_counter() - started
        diag = f"Sandbox arbiter crashed: {type(e).__name__}: {e}"
        _diag(ws, "sandbox_arbiter", diag)
        return _arbiter_failure(
            ws,
            loop,
            status="infra_fault",
            sandbox_errors=diag,
            docker_runs=[_docker_run_record(loop, "crash", total_duration, "", str(e))],
        )


def _parse_sandbox_output(stdout: str, stderr: str, loop: int) -> Dict[str, Any]:
    """Parse marker strings from the arbiter shell output into structured diagnostics."""
    combined = stdout + "\n" + stderr
    diagnostics = {
        "sandbox_status": "pass",
        "fault_location": None,
        "raw_output": combined,
        "ruff_format_src_failed": "__RUFF_FORMAT_SRC_FAILED__" in combined,
        "ruff_format_tests_failed": "__RUFF_FORMAT_TESTS_FAILED__" in combined,
        "ruff_check_src_failed": "__RUFF_CHECK_SRC_FAILED__" in combined,
        "ruff_check_tests_failed": "__RUFF_CHECK_TESTS_FAILED__" in combined,
        "tautology_detected": "__TAUTOLOGY_DETECTED__" in combined,
        "tautology_failed": "__TAUTOLOGY_FAILED__" in combined,
        "mypy_src_failed": "__MYPY_SRC_FAILED__" in combined,
        "pytest_failed": "__PYTEST_FAILED__" in combined,
        "sandbox_loop": loop,
    }

    if "__SANDBOX_PASS__" in stdout:
        diagnostics["sandbox_status"] = "pass"
        return diagnostics

    # Classify the failure.
    if (
        diagnostics["ruff_format_tests_failed"]
        or diagnostics["ruff_check_tests_failed"]
        or diagnostics["tautology_detected"]
        or diagnostics["tautology_failed"]
    ):
        diagnostics["sandbox_status"] = "test_fault"
        diagnostics["fault_location"] = "tests"
    elif (
        diagnostics["ruff_format_src_failed"]
        or diagnostics["ruff_check_src_failed"]
        or diagnostics["mypy_src_failed"]
        or diagnostics["pytest_failed"]
    ):
        diagnostics["sandbox_status"] = "code_fault"
        diagnostics["fault_location"] = "src"
    else:
        # Unknown / infra-level failure
        diagnostics["sandbox_status"] = "infra_fault"
        diagnostics["fault_location"] = "toolchain"

    return diagnostics


def _route_sandbox_failure(
    workspace: str,
    loop: int,
    diagnostics: Dict[str, Any],
    docker_runs: List[dict],
    manifest: Dict[str, str],
) -> Dict[str, Any]:
    """Build the state update when the sandbox fails one or more checks."""
    status = diagnostics["sandbox_status"]
    # Log a concise snapshot of the diagnostics; full stdout/stderr is already
    # captured in the docker_runs record and written by _diag earlier.
    _diag(
        workspace,
        "sandbox_arbiter",
        f"Sandbox failure: status={status}, "
        + ", ".join(f"{k}={v}" for k, v in diagnostics.items() if k != "raw_output"),
    )

    if status == "test_fault":
        return {
            "file_manifest": manifest,
            "sandbox_errors": diagnostics.get("raw_output", ""),
            "sandbox_diagnostics": diagnostics,
            "sandbox_loop_count": 0,
            "docker_runs": docker_runs,
            "next_node": "test_designer",
            "thoughts": _think(
                workspace,
                "sandbox_arbiter",
                f"Loop {loop}: test-side fault → test_designer",
            ),
        }

    if status == "infra_fault":
        # Infra faults are not recoverable by the coder or architect. Terminate.
        return {
            "file_manifest": manifest,
            "sandbox_errors": diagnostics.get("raw_output", ""),
            "sandbox_diagnostics": diagnostics,
            "sandbox_loop_count": loop,
            "docker_runs": docker_runs,
            "next_node": "FINISH",
            "thoughts": _think(
                workspace,
                "sandbox_arbiter",
                f"Loop {loop}: infra fault → FINISH",
            ),
        }

    # code_fault
    if loop < MAX_SANDBOX_LOOPS:
        return {
            "file_manifest": manifest,
            "sandbox_errors": diagnostics.get("raw_output", ""),
            "sandbox_diagnostics": diagnostics,
            "sandbox_loop_count": loop,
            "docker_runs": docker_runs,
            "next_node": "coder",
            "thoughts": _think(
                workspace,
                "sandbox_arbiter",
                f"Loop {loop}: code fault → coder (sandbox_loop={loop})",
            ),
        }

    # Exhausted sandbox loops: replan from test designer.
    return {
        "file_manifest": manifest,
        "sandbox_errors": diagnostics.get("raw_output", ""),
        "sandbox_diagnostics": diagnostics,
        "sandbox_loop_count": 0,
        "docker_runs": docker_runs,
        "next_node": "test_designer",
        "thoughts": _think(
            workspace,
            "sandbox_arbiter",
            f"Loop {loop}: sandbox loop ceiling → test_designer",
        ),
    }


def _arbiter_failure(
    workspace: str,
    loop: int,
    status: str,
    sandbox_errors: str,
    docker_runs: List[dict],
) -> Dict[str, Any]:
    diagnostics = {
        "sandbox_status": status,
        "fault_location": "toolchain",
        "raw_output": sandbox_errors,
        "sandbox_loop": loop,
    }
    if status == "infra_fault":
        return {
            "sandbox_errors": sandbox_errors,
            "sandbox_diagnostics": diagnostics,
            "sandbox_loop_count": loop,
            "docker_runs": docker_runs,
            "next_node": "FINISH",
            "thoughts": _think(
                workspace,
                "sandbox_arbiter",
                f"Loop {loop}: infra fault → FINISH",
            ),
        }
    # Treat unexpected as code fault with replan.
    return {
        "sandbox_errors": sandbox_errors,
        "sandbox_diagnostics": diagnostics,
        "sandbox_loop_count": 0,
        "docker_runs": docker_runs,
        "next_node": "test_designer",
        "thoughts": _think(
            workspace,
            "sandbox_arbiter",
            f"Loop {loop}: unrecoverable sandbox fault → test_designer",
        ),
    }


# =============================================================================
# Routing functions
# =============================================================================
def route_from_sandbox(state: AgenticState) -> str:
    """Pure routing helper; kept as a function for readability and testing."""
    return state.get("next_node", "FINISH")
