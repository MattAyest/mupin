"""Node implementations for the Editing Module.

This is a deliberately smaller and simpler version of the coding module's
nodes.py. It reuses the same LLM factory, retry logic, sandbox harness, and
profile loader patterns, but implements a 4-node editing pipeline:

    load_source -> analyze -> plan -> apply -> verify -> FINISH
"""

import asyncio
import difflib
import json
import os
import queue
import re
import shutil
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx
import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from .profile import Profile, get_profile
from .state import EditingState


# =============================================================================
# Custom exceptions
# =============================================================================
class LLMUnavailableError(Exception):
    """Raised when an LLM call is retried up to the configured cap and still fails."""

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
OLLAMA_TIMEOUT = httpx.Timeout(connect=15, read=180, write=15, pool=15)

RESILIENCE = CONFIG.get("resilience", {})
INFRA_MAX_RETRIES = int(RESILIENCE.get("infra_max_retries_per_node", 5))
INFRA_BACKOFFS = [
    float(x) for x in RESILIENCE.get("infra_retry_backoff_seconds", [10, 30, 60, 120, 120])
]
STREAM_IDLE_DEFAULT = float(RESILIENCE.get("stream_idle_seconds", 120.0))

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

MAX_SANDBOX_LOOPS = int(CONFIG.get("loop_limits", {}).get("max_sandbox_loops", 5))
MAX_REGRESSION_LOOPS = int(CONFIG.get("loop_limits", {}).get("max_regression_loops", 3))
SERVER_TASK_DEADLINE = CONFIG.get("server", {}).get("task_deadline_seconds", 3600)


def _profile_from_state(state: EditingState) -> Profile:
    return get_profile(state.get("profile_name"))


# =============================================================================
# Host identity propagation for sandbox containers
# =============================================================================
def _host_identity() -> Tuple[int, int]:
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
            base_url = os.getenv(api_key_env_var or "OLLAMA_BASE_URL", "http://localhost:11434")
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
    return _build_llm(node_name)


def validate_config():
    problems = []
    for node_name in CONFIG.get("nodes", {}):
        try:
            get_llm(node_name)
        except Exception as e:
            problems.append((node_name, str(e)))
    return problems


# =============================================================================
# Token estimation and transient error classification
# =============================================================================
def _is_transient_llm_error(e: Exception) -> bool:
    if isinstance(e, FutureTimeoutError):
        return True
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
        lowered = err_text.lower()
        if any(tok in lowered for tok in (
            "timeout", "timed out", "temporarily unavailable", "rate limit",
            "try again", "server error", "bad gateway", "gateway timeout",
            "origin_response_timeout", "ssl", "unexpected_eof", "eof occurred",
            "violation of protocol", "incomplete read", "chunked encoding",
            "connection reset", "remote disconnected", "protocol error",
            "empty response", "too many concurrent requests",
        )):
            return True
        import re as _re
        for m in _re.finditer(r"\b(\d{3})\b", err_text):
            status = int(m.group(1))
            if status in TRANSIENT_HTTP_STATUSES or (500 <= status < 600):
                return True
    return False


def _estimate_tokens(text: str) -> int:
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


def _invoke_with_retry(
    node_name: str,
    messages,
    workspace: str = "",
    max_attempts: int | None = None,
    timeout_seconds: float | None = None,
    state: dict | None = None,
) -> tuple[str, list[dict]]:
    node_config = CONFIG.get("nodes", {}).get(node_name, {})
    max_attempts = max_attempts if max_attempts is not None else node_config.get("max_attempts", 3)
    timeout_seconds = timeout_seconds if timeout_seconds is not None else node_config.get("timeout_seconds", 600.0)
    provider = node_config.get("provider")
    model = node_config.get("model", "unknown")
    ollama_providers = ("ollama-cloud", "ollama")
    invoke_kwargs = {"stream": True} if provider in ollama_providers else {}

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
            stream_idle = float(node_config.get("stream_idle_seconds", STREAM_IDLE_DEFAULT))
            chunk_queue: "queue.Queue" = queue.Queue()
            use_streaming = bool(invoke_kwargs.get("stream"))

            def _stream_call():
                try:
                    if use_streaming:
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
                except BaseException as e:
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
                        raise FutureTimeoutError(f"Wall-clock timeout after {timeout_seconds}s")
                    wait = min(stream_idle, remaining)
                    try:
                        kind, payload = chunk_queue.get(timeout=wait)
                    except queue.Empty:
                        raise FutureTimeoutError(f"Stream idle timeout after {wait:.0f}s with no chunk")
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
                    f"{usage['duration_seconds']}s, ~{usage['total_tokens']} tokens",
                )
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
                _diag(workspace, node_name, f"LLM attempt {attempt}/{max_attempts} timed out: {e}")
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
                _diag(workspace, node_name, f"LLM attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {e}")
        if attempt < max_attempts:
            backoff = (2.0, 5.0, 10.0)[min(attempt - 1, 2)]
            time.sleep(backoff)

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


def _node_history_entry(node_name: str, wall_started: float, wall_ended: float, perf_duration: float | None = None, **extras) -> dict:
    return {
        "node": node_name,
        "started_at": datetime.fromtimestamp(wall_started, tz=timezone.utc).isoformat(),
        "ended_at": datetime.fromtimestamp(wall_ended, tz=timezone.utc).isoformat(),
        "duration_seconds": round(perf_duration, 3) if perf_duration is not None else round(wall_ended - wall_started, 3),
        **extras,
    }


def node_with_history(func):
    def wrapper(state: EditingState):
        node_name = func.__name__
        wall_started = time.time()
        perf_started = time.perf_counter()
        result = func(state) or {}
        perf_ended = time.perf_counter()
        wall_ended = time.time()
        extras = result.pop("diagnostics", {}) if isinstance(result, dict) else {}
        entry = _node_history_entry(node_name, wall_started, wall_ended, perf_ended - perf_started, **extras)
        result["node_history"] = [entry]
        return result
    return wrapper


def _handle_llm_unavailable(node_name: str, state: EditingState, exc: LLMUnavailableError, workspace: str):
    retries = state.get("llm_infra_retries", {})
    count = retries.get(node_name, 0)
    usage_entries = list(getattr(exc, "usage_entries", []))

    if exc.is_transient and count < INFRA_MAX_RETRIES:
        new_count = count + 1
        backoff = INFRA_BACKOFFS[min(count, len(INFRA_BACKOFFS) - 1)]
        diag = f"Transient LLM fault for {node_name} (attempt {new_count}/{INFRA_MAX_RETRIES}); retrying after {backoff}s backoff"
        _diag(workspace, node_name, diag)
        time.sleep(backoff)
        return {
            "llm_infra_retries": {**retries, node_name: new_count},
            "llm_usage": usage_entries,
            "sandbox_errors": "",
            "next_node": node_name,
            "thoughts": _think(workspace, node_name, diag),
        }

    kind = "transient" if exc.is_transient else "permanent"
    diag = f"LLM infrastructure {kind} fault for {node_name} exhausted ({count} infra retries used): {exc.cause}"
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
            if abs_path.startswith(dest) and (best is None or len(dest) > len(best["Destination"])):
                best = mount
        if best:
            relative = abs_path[len(best["Destination"]):]
            return best["Source"] + relative
    except Exception:
        pass
    return abs_path


def cleanup_sandbox_for_task(task_id: str) -> None:
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


def _chown_to_host(path: str) -> None:
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
    host_uid, host_gid = _host_identity()
    if not host_uid and not host_gid:
        return
    try:
        os.chown(path, host_uid, host_gid)
    except Exception:
        pass


def _clear_directory(path: str) -> None:
    if os.path.exists(path):
        shutil.rmtree(path)


def _write_workspace_file(workspace: str, filename: str, content: str) -> None:
    filepath = os.path.join(workspace, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def setup_workspace(workspace: str, profile: Profile) -> None:
    os.makedirs(workspace, exist_ok=True)
    for filename, content in profile.setup_files().items():
        if not os.path.exists(os.path.join(workspace, filename)):
            _write_workspace_file(workspace, filename, content)

    deps_file = profile.sandbox_value("deps_file", "requirements.txt")
    default_deps = profile.sandbox_value("default_deps", [])
    requirements = "\n".join(default_deps) + "\n" if default_deps else ""
    if requirements and not os.path.exists(os.path.join(workspace, deps_file)):
        _write_workspace_file(workspace, deps_file, requirements)
    _chown_to_host(workspace)


def write_manifest_to_disk(workspace: str, manifest: Dict[str, str], preserve_unlisted: bool = True) -> None:
    """Write manifest files to disk. If preserve_unlisted is True, any file in the
    workspace that is not in the manifest is left untouched."""
    if not preserve_unlisted:
        _clear_directory(os.path.join(workspace, "src"))
        _clear_directory(os.path.join(workspace, "tests"))

    for filename, code in manifest.items():
        filepath = os.path.join(workspace, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)

    # Always chown the directories we may have touched.
    _chown_to_host(os.path.join(workspace, "src"))
    _chown_to_host(os.path.join(workspace, "tests"))


def read_file_from_disk(workspace: str, filename: str) -> str:
    filepath = os.path.join(workspace, filename)
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def cleanup_workspace_deps(workspace: str) -> None:
    for subdir in (".deps", ".tmp", ".hypothesis"):
        _clear_directory(os.path.join(workspace, subdir))


def load_workspace_manifest(workspace: str, profile: Profile) -> Dict[str, str]:
    """Read all files in the manifest plus the source/test directories."""
    manifest: Dict[str, str] = {}
    files_to_read = set(profile.manifest_files())

    # Also read any src/ or tests/ files present.
    for dirname in (profile.file_path("source_dir"), profile.file_path("test_dir")):
        dirpath = os.path.join(workspace, dirname)
        if os.path.isdir(dirpath):
            for root, _, files in os.walk(dirpath):
                for f in files:
                    relpath = os.path.relpath(os.path.join(root, f), workspace)
                    files_to_read.add(relpath)

    for relpath in files_to_read:
        content = read_file_from_disk(workspace, relpath)
        if content or relpath in profile.manifest_files():
            manifest[relpath] = content
    return manifest


def compute_diff(source_manifest: Dict[str, str], final_manifest: Dict[str, str]) -> str:
    """Unified diff from source to final."""
    diff_lines: List[str] = []
    all_files = sorted(set(source_manifest.keys()) | set(final_manifest.keys()))
    for filename in all_files:
        old = source_manifest.get(filename, "")
        new = final_manifest.get(filename, "")
        if old == new:
            continue
        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        if old_lines and not old_lines[-1].endswith("\n"):
            old_lines[-1] += "\n"
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"
        diff_lines.extend(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{filename}",
                tofile=f"b/{filename}",
            )
        )
    return "".join(diff_lines)


# =============================================================================
# File manifest parsing
# =============================================================================
def _parse_file_tags(content: str) -> Dict[str, str]:
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


def _parse_analysis(content: str) -> Dict[str, Any]:
    match = re.search(r"<analysis>(.*?)</analysis>", content, re.DOTALL | re.IGNORECASE)
    if not match:
        return {"intent": "", "affected_files": [], "notes": []}
    try:
        return json.loads(match.group(1).strip())
    except Exception:
        return {"intent": match.group(1).strip(), "affected_files": [], "notes": []}


def _parse_plan(content: str) -> List[Dict[str, Any]]:
    match = re.search(r"<plan>(.*?)</plan>", content, re.DOTALL | re.IGNORECASE)
    if not match:
        return []
    try:
        plan = json.loads(match.group(1).strip())
        if isinstance(plan, list):
            return plan
    except Exception:
        pass
    return []


# =============================================================================
# NODE: load_source
# =============================================================================
@node_with_history
def load_source(state: EditingState):
    edit_workspace = state["workspace_dir"]
    os.makedirs(edit_workspace, exist_ok=True)

    source_job_id = state.get("source_job_id", "")
    source_files = state.get("source_files") or {}

    if source_files:
        for filename, content in source_files.items():
            _write_workspace_file(edit_workspace, filename, content)
        _diag(edit_workspace, "load_source",
              f"Loaded {len(source_files)} inline source files")
    elif source_job_id:
        workspace_root = Path(os.environ.get("WORKSPACE_ROOT", "/app/.workspaces"))
        source_workspace = str(workspace_root / source_job_id)
        if os.path.isdir(source_workspace):
            for item in os.listdir(source_workspace):
                src = os.path.join(source_workspace, item)
                dst = os.path.join(edit_workspace, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        _diag(edit_workspace, "load_source",
              f"Loaded files from source job {source_job_id}")
    else:
        _diag(edit_workspace, "load_source",
              "No source_files or source_job_id provided")

    profile = _profile_from_state(state)
    setup_workspace(edit_workspace, profile)

    source_manifest = load_workspace_manifest(edit_workspace, profile)
    _diag(edit_workspace, "load_source", f"Loaded {len(source_manifest)} files")

    return {
        "source_manifest": source_manifest,
        "file_manifest": dict(source_manifest),
        "next_node": "analyze",
        "thoughts": _think(edit_workspace, "load_source",
                           f"Loaded {len(source_manifest)} files"),
    }


# =============================================================================
# NODE: analyze
# =============================================================================
@node_with_history
def analyze(state: EditingState):
    instruction = state["instruction"]
    workspace = state["workspace_dir"]
    manifest = state.get("file_manifest", {})

    profile = _profile_from_state(state)
    system = profile.prompt("analyze_system")

    files_block = "\n\n".join(
        f"--- {fname} ---\n{content[:4000]}"
        for fname, content in sorted(manifest.items())
    )
    user_prompt = f"Instruction:\n{instruction}\n\nWorkspace files:\n{files_block}"

    try:
        content, llm_usage = _invoke_with_retry(
            "analyze",
            [SystemMessage(content=system), HumanMessage(content=user_prompt)],
            workspace,
            state=state,
        )
    except LLMUnavailableError as e:
        return _handle_llm_unavailable("analyze", state, e, workspace)

    analysis = _parse_analysis(content)
    test_impact = analysis.get("test_impact", "none")
    if test_impact not in ("none", "update_existing", "add_new", "both"):
        test_impact = "none"
    _diag(workspace, "analyze", f"Analysis: {analysis.get('intent', '')} | test_impact: {test_impact}")

    return {
        "test_impact": test_impact,
        "edit_plan": [],
        "llm_usage": llm_usage,
        "next_node": "plan",
        "thoughts": _think(workspace, "analyze", f"Intent: {analysis.get('intent', 'no analysis')} | test_impact: {test_impact} | affected: {analysis.get('affected_files', [])}"),
    }


# =============================================================================
# NODE: plan
# =============================================================================
@node_with_history
def plan(state: EditingState):
    instruction = state["instruction"]
    workspace = state["workspace_dir"]
    manifest = state.get("file_manifest", {})

    profile = _profile_from_state(state)
    system = profile.prompt("plan_system")

    files_block = "\n\n".join(
        f"--- {fname} ---\n{content[:4000]}"
        for fname, content in sorted(manifest.items())
    )
    user_prompt = f"Instruction:\n{instruction}\n\nWorkspace files:\n{files_block}"

    try:
        content, llm_usage = _invoke_with_retry(
            "plan",
            [SystemMessage(content=system), HumanMessage(content=user_prompt)],
            workspace,
            state=state,
        )
    except LLMUnavailableError as e:
        return _handle_llm_unavailable("plan", state, e, workspace)

    edit_plan = _parse_plan(content)
    test_impact = state.get("test_impact", "none")
    _diag(workspace, "plan", f"Plan: {len(edit_plan)} step(s) | test_impact: {test_impact}")

    # Route: if tests are affected, go to apply_tests (TDD: tests first).
    # Otherwise, go to apply (code-only, tests untouched).
    next_node = "apply_tests" if test_impact != "none" else "apply"

    return {
        "edit_plan": edit_plan,
        "llm_usage": llm_usage,
        "next_node": next_node,
        "thoughts": _think(workspace, "plan", f"Planned {len(edit_plan)} edit(s) → {next_node}"),
    }


# =============================================================================
# NODE: apply
# =============================================================================
@node_with_history
def apply(state: EditingState):
    """Code-only apply node (used when test_impact=none).
    Physically ignores any test file output from the LLM.
    """
    instruction = state["instruction"]
    workspace = state["workspace_dir"]
    manifest = state.get("file_manifest", {})
    edit_plan = state.get("edit_plan", [])
    sandbox_errors = state.get("sandbox_errors", "")
    regression_errors = state.get("regression_errors", "")

    profile = _profile_from_state(state)
    system = profile.prompt("apply_system")

    files_block = "\n\n".join(
        f"--- {fname} ---\n{content[:4000]}"
        for fname, content in sorted(manifest.items())
    )
    plan_block = "\n".join(f"- {step}" for step in edit_plan) if edit_plan else "Apply the instruction."
    user_prompt = f"Instruction:\n{instruction}\n\nEdit plan:\n{plan_block}\n\nWorkspace files:\n{files_block}"
    if regression_errors:
        user_prompt += f"\n\nThe last regression check failed:\n{regression_errors[:4000]}\nFix the implementation so existing tests still pass."
    elif sandbox_errors:
        user_prompt += f"\n\nThe last verification failed with these errors. Fix the implementation:\n{sandbox_errors[:4000]}"

    try:
        content, llm_usage = _invoke_with_retry(
            "apply",
            [SystemMessage(content=system), HumanMessage(content=user_prompt)],
            workspace,
            state=state,
        )
    except LLMUnavailableError as e:
        return _handle_llm_unavailable("apply", state, e, workspace)

    new_files = _parse_file_tags(content)
    if not new_files:
        _diag(workspace, "apply", f"No <file> tags in response\n{content[:2000]}")
        return {
            "sandbox_errors": "Apply Error: no file tags produced",
            "llm_usage": llm_usage,
            "next_node": "FINISH",
            "thoughts": _think(workspace, "apply", "ERROR: no file tags produced"),
        }

    # Enforce code-only: filter out any test files the LLM might output.
    test_main_path = profile.file_path("test_main")
    code_files = {k: v for k, v in new_files.items() if not k.startswith("tests/")}
    if len(code_files) < len(new_files):
        _diag(workspace, "apply", f"Filtered out test files from apply response: {set(new_files) - set(code_files)}")

    if not code_files:
        _diag(workspace, "apply", "No code files in response after filtering test files")
        return {
            "sandbox_errors": "Apply Error: no code files produced",
            "llm_usage": llm_usage,
            "next_node": "FINISH",
            "thoughts": _think(workspace, "apply", "ERROR: no code files after filtering"),
        }

    final_manifest = {**manifest, **code_files}
    _diag(workspace, "apply", f"Applied edits to: {sorted(code_files.keys())}")

    return {
        "file_manifest": final_manifest,
        "sandbox_errors": "",
        "regression_errors": "",
        "llm_usage": llm_usage,
        "next_node": "regression_check",
        "thoughts": _think(workspace, "apply", f"Edited {len(code_files)} file(s): {', '.join(sorted(code_files.keys()))}"),
    }


# =============================================================================
# NODE: apply_tests (TDD — writes tests FIRST for target behavior)
# =============================================================================
@node_with_history
def apply_tests(state: EditingState):
    """Writes/updates tests for the TARGET behavior before code is changed.
    Only outputs tests/test_main.py — cannot touch src/main.py.
    """
    instruction = state["instruction"]
    workspace = state["workspace_dir"]
    manifest = state.get("file_manifest", {})
    edit_plan = state.get("edit_plan", [])
    regression_errors = state.get("regression_errors", "")

    profile = _profile_from_state(state)
    system = profile.prompt("apply_tests_system")

    source_main_path = profile.file_path("source_main")
    test_main_path = profile.file_path("test_main")
    current_code = manifest.get(source_main_path, "")
    current_tests = manifest.get(test_main_path, "")

    plan_block = "\n".join(f"- {step}" for step in edit_plan) if edit_plan else "Apply the instruction."
    user_prompt = (
        f"Instruction:\n{instruction}\n\n"
        f"Edit plan:\n{plan_block}\n\n"
        f"Current code (src/main.py — for context, write tests for TARGET behavior):\n--- src/main.py ---\n{current_code[:4000]}\n\n"
        f"Current tests (tests/test_main.py — update these):\n--- tests/test_main.py ---\n{current_tests[:4000]}"
    )
    if regression_errors:
        user_prompt += f"\n\nThe last regression check failed:\n{regression_errors[:3000]}\nUpdate the tests to correctly reflect the target behavior."

    try:
        content, llm_usage = _invoke_with_retry(
            "apply_tests",
            [SystemMessage(content=system), HumanMessage(content=user_prompt)],
            workspace,
            state=state,
        )
    except LLMUnavailableError as e:
        return _handle_llm_unavailable("apply_tests", state, e, workspace)

    new_files = _parse_file_tags(content)
    if not new_files:
        _diag(workspace, "apply_tests", f"No <file> tags in response\n{content[:2000]}")
        return {
            "sandbox_errors": "Apply_tests Error: no file tags produced",
            "llm_usage": llm_usage,
            "next_node": "FINISH",
            "thoughts": _think(workspace, "apply_tests", "ERROR: no file tags produced"),
        }

    # Enforce tests-only: filter out any code files the LLM might output.
    test_files = {k: v for k, v in new_files.items() if k.startswith("tests/")}
    if len(test_files) < len(new_files):
        _diag(workspace, "apply_tests", f"Filtered out code files from apply_tests response: {set(new_files) - set(test_files)}")

    if not test_files:
        _diag(workspace, "apply_tests", "No test files in response after filtering")
        return {
            "sandbox_errors": "Apply_tests Error: no test files produced",
            "llm_usage": llm_usage,
            "next_node": "FINISH",
            "thoughts": _think(workspace, "apply_tests", "ERROR: no test files after filtering"),
        }

    # Merge test files into manifest — code files are preserved unchanged.
    final_manifest = {**manifest, **test_files}
    _diag(workspace, "apply_tests", f"Updated tests: {sorted(test_files.keys())}")

    return {
        "file_manifest": final_manifest,
        "sandbox_errors": "",
        "regression_errors": "",
        "llm_usage": llm_usage,
        "next_node": "apply_code",
        "thoughts": _think(workspace, "apply_tests", f"Wrote tests for target behavior: {', '.join(sorted(test_files.keys()))}"),
    }


# =============================================================================
# NODE: apply_code (TDD — writes code to pass the new tests)
# =============================================================================
@node_with_history
def apply_code(state: EditingState):
    """Writes/updates src/main.py to pass the new tests.
    Only outputs src/main.py — cannot touch tests.
    """
    instruction = state["instruction"]
    workspace = state["workspace_dir"]
    manifest = state.get("file_manifest", {})
    edit_plan = state.get("edit_plan", [])
    sandbox_errors = state.get("sandbox_errors", "")
    regression_errors = state.get("regression_errors", "")

    profile = _profile_from_state(state)
    system = profile.prompt("apply_code_system")

    source_main_path = profile.file_path("source_main")
    test_main_path = profile.file_path("test_main")
    current_code = manifest.get(source_main_path, "")
    new_tests = manifest.get(test_main_path, "")

    plan_block = "\n".join(f"- {step}" for step in edit_plan) if edit_plan else "Apply the instruction."
    user_prompt = (
        f"Instruction:\n{instruction}\n\n"
        f"Edit plan:\n{plan_block}\n\n"
        f"Current code (src/main.py):\n--- src/main.py ---\n{current_code[:4000]}\n\n"
        f"New tests (tests/test_main.py — your code MUST pass these):\n--- tests/test_main.py ---\n{new_tests[:4000]}"
    )
    if regression_errors:
        user_prompt += f"\n\nThe last regression check failed:\n{regression_errors[:3000]}\nFix the code so the regression check passes."
    elif sandbox_errors:
        user_prompt += f"\n\nThe last verification failed:\n{sandbox_errors[:3000]}\nFix the code so all tests pass."

    try:
        content, llm_usage = _invoke_with_retry(
            "apply_code",
            [SystemMessage(content=system), HumanMessage(content=user_prompt)],
            workspace,
            state=state,
        )
    except LLMUnavailableError as e:
        return _handle_llm_unavailable("apply_code", state, e, workspace)

    new_files = _parse_file_tags(content)
    if not new_files:
        _diag(workspace, "apply_code", f"No <file> tags in response\n{content[:2000]}")
        return {
            "sandbox_errors": "Apply_code Error: no file tags produced",
            "llm_usage": llm_usage,
            "next_node": "FINISH",
            "thoughts": _think(workspace, "apply_code", "ERROR: no file tags produced"),
        }

    # Enforce code-only: filter out any test files.
    code_files = {k: v for k, v in new_files.items() if not k.startswith("tests/")}
    if len(code_files) < len(new_files):
        _diag(workspace, "apply_code", f"Filtered out test files from apply_code response: {set(new_files) - set(code_files)}")

    if not code_files:
        _diag(workspace, "apply_code", "No code files in response after filtering")
        return {
            "sandbox_errors": "Apply_code Error: no code files produced",
            "llm_usage": llm_usage,
            "next_node": "FINISH",
            "thoughts": _think(workspace, "apply_code", "ERROR: no code files after filtering"),
        }

    final_manifest = {**manifest, **code_files}
    _diag(workspace, "apply_code", f"Updated code: {sorted(code_files.keys())}")

    return {
        "file_manifest": final_manifest,
        "sandbox_errors": "",
        "regression_errors": "",
        "llm_usage": llm_usage,
        "next_node": "regression_check",
        "thoughts": _think(workspace, "apply_code", f"Wrote code to pass new tests: {', '.join(sorted(code_files.keys()))}"),
    }


# =============================================================================
# NODE: regression_check (deterministic — runs ORIGINAL tests against EDITED code)
# =============================================================================
@node_with_history
def regression_check(state: EditingState):
    """Runs the original (pristine) fixture tests against the edited code.

    Non-behavioral edits: original tests MUST still pass.
    Behavioral edits: original tests MUST fail (confirming behavior changed).

    Hard gate: failures loop back to apply/apply_code (max 3 loops).
    No LLM — purely deterministic Docker sandbox run.
    """
    workspace = state["workspace_dir"]
    source_manifest = state.get("source_manifest", {})
    file_manifest = state.get("file_manifest", {})
    test_impact = state.get("test_impact", "none")
    loop = state.get("regression_loop_count", 0) + 1

    profile = _profile_from_state(state)
    test_main_path = profile.file_path("test_main")
    source_main_path = profile.file_path("source_main")

    original_tests = source_manifest.get(test_main_path, "")
    edited_code = file_manifest.get(source_main_path, "")

    if not original_tests:
        _diag(workspace, "regression_check", "No original tests in source manifest — skipping")
        return {
            "regression_loop_count": loop,
            "regression_errors": "",
            "next_node": "verify",
            "thoughts": _think(workspace, "regression_check", "No original tests — skipped"),
        }

    # Write a temporary workspace: edited code + ORIGINAL tests.
    reg_workspace = os.path.join(workspace, ".regression")
    _clear_directory(reg_workspace)
    os.makedirs(reg_workspace, exist_ok=True)

    # Copy all files from the edited manifest (code, config, etc).
    for filename, content in file_manifest.items():
        if filename == test_main_path:
            continue  # Don't write the edited tests — use original.
        _write_workspace_file(reg_workspace, filename, content)

    # Write the ORIGINAL tests.
    _write_workspace_file(reg_workspace, test_main_path, original_tests)

    # Copy setup files.
    for filename, content in profile.setup_files().items():
        _write_workspace_file(reg_workspace, filename, content)

    _chown_to_host(reg_workspace)

    # Run pytest in a sandbox.
    IMAGE = profile.sandbox_value("image", "python:3.11-slim")
    memory_limit = profile.sandbox_value("memory", "512m")
    timeout_install = profile.sandbox_value("timeout_install", 90)
    timeout_test = profile.sandbox_value("timeout_test", 120)
    timeout_total = timeout_install + timeout_test + 60
    cpus = profile.sandbox_value("cpus", "2.0")
    host_uid, host_gid = _host_identity()
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
        "-e", "HOME=/tmp",
        "-e", "TMPDIR=/tmp",
        "-e", "PYTHONPYCACHEPREFIX=/tmp/pycache",
    ]

    workspace_mount = f"{resolve_host_path(reg_workspace)}:/workspace"
    base_mounts = ["-v", workspace_mount, "-w", "/workspace"]
    tmpfs_flags = ["--tmpfs", "/tmp:noexec,nosuid,size=100m", "--tmpfs", "/var/tmp:noexec,nosuid,size=50m"]

    _diag(workspace, "regression_check", f"Running original tests against edited code (loop {loop}/{MAX_REGRESSION_LOOPS}, test_impact={test_impact})")

    started = time.perf_counter()
    try:
        # Install deps.
        _clear_directory(os.path.join(reg_workspace, ".deps"))
        _clear_directory(os.path.join(reg_workspace, ".tmp"))

        install_script = profile.resolve(profile.sandbox_value("install_command"))
        install_cmd = (
            ["docker", "run", "--rm", "--name", f"{task_id}-reg-install-{loop}"]
            + container_hardening
            + base_mounts
            + tmpfs_flags
            + ["-e", "PIP_NO_CACHE_DIR=1", "-e", "PIP_DISABLE_PIP_VERSION_CHECK=1",
               "-e", "PIP_ROOT_USER_ACTION=ignore", "-e", "TMPDIR=/workspace/.tmp"]
            + [IMAGE, "bash", "-c", install_script]
        )
        install_res = subprocess.run(install_cmd, capture_output=True, text=True, timeout=timeout_install)

        if install_res.returncode != 0:
            _diag(workspace, "regression_check", f"Install failed:\n{install_res.stderr[:1000]}")
            return _regression_route(workspace, loop, test_impact, "infra_fault",
                                     f"Regression install failed:\n{install_res.stderr[:2000]}")

        # Run only pytest (no ruff/mypy — we only care about test results).
        verify_env = ["-e", "PYTHONPATH=/workspace/.deps"]
        test_cmd = (
            ["docker", "run", "--rm", "--network", "none", "--read-only",
             "--name", f"{task_id}-reg-verify-{loop}"]
            + container_hardening
            + base_mounts
            + tmpfs_flags
            + verify_env
            + [IMAGE, "bash", "-c",
             f"export PYTHONPATH=/workspace/.deps && python -m pytest {test_main_path} -p no:cacheprovider --timeout=30 -v"]
        )
        test_res = subprocess.run(test_cmd, capture_output=True, text=True, timeout=timeout_total)
        duration = time.perf_counter() - started

        stdout = test_res.stdout
        stderr = test_res.stderr
        _diag(workspace, "regression_check", f"Pytest exit={test_res.returncode} ({duration:.1f}s)\nSTDOUT:\n{stdout[:2000]}\nSTDERR:\n{stderr[:1000]}")

        tests_passed = (test_res.returncode == 0)
        is_behavioral = test_impact in ("update_existing", "both")

        if is_behavioral:
            # Behavioral change: original tests SHOULD fail.
            if not tests_passed:
                _diag(workspace, "regression_check", "Original tests failed as expected (behavioral change confirmed)")
                return {
                    "regression_loop_count": loop,
                    "regression_errors": "",
                    "docker_runs": [_docker_run_record(loop, "regression_pass", duration, stdout, stderr)],
                    "next_node": "verify",
                    "thoughts": _think(workspace, "regression_check", f"Loop {loop}: original tests failed as expected (behavioral change) → verify"),
                }
            else:
                # Original tests still pass — the behavioral change wasn't applied.
                _diag(workspace, "regression_check", "WARNING: original tests still pass — behavioral change may not have been applied")
                return _regression_route(workspace, loop, test_impact, "behavior_not_applied",
                                         "Original tests still pass after a behavioral edit — the change may not have been applied.\n"
                                         f"Pytest output:\n{stdout[:2000]}")
        else:
            # Non-behavioral: original tests MUST still pass.
            if tests_passed:
                _diag(workspace, "regression_check", "Original tests still pass (no regression)")
                return {
                    "regression_loop_count": loop,
                    "regression_errors": "",
                    "docker_runs": [_docker_run_record(loop, "regression_pass", duration, stdout, stderr)],
                    "next_node": "verify",
                    "thoughts": _think(workspace, "regression_check", f"Loop {loop}: original tests pass (no regression) → verify"),
                }
            else:
                _diag(workspace, "regression_check", f"REGRESSION: original tests failed:\n{stdout[:2000]}")
                return _regression_route(workspace, loop, test_impact, "regression",
                                         f"Original tests failed after edit (regression):\n{stdout[:3000]}\n{stderr[:1000]}")

    except subprocess.TimeoutExpired:
        duration = time.perf_counter() - started
        _diag(workspace, "regression_check", f"Regression check timed out after {duration:.1f}s")
        return _regression_route(workspace, loop, test_impact, "timeout",
                                 f"Regression check timed out after {duration:.1f}s")
    except Exception as e:
        duration = time.perf_counter() - started
        _diag(workspace, "regression_check", f"Regression check crashed: {type(e).__name__}: {e}")
        return _regression_route(workspace, loop, test_impact, "crash",
                                 f"Regression check crashed: {type(e).__name__}: {e}")
    finally:
        _clear_directory(reg_workspace)


def _regression_route(workspace: str, loop: int, test_impact: str, fault: str, errors: str) -> dict:
    """Route after a regression check failure. Loops back to apply/apply_code
    up to MAX_REGRESSION_LOOPS, then gives up and goes to verify."""
    if loop < MAX_REGRESSION_LOOPS:
        # Route back to the appropriate apply node.
        if test_impact == "none":
            next_node = "apply"
        else:
            next_node = "apply_code"
        return {
            "regression_loop_count": loop,
            "regression_errors": errors,
            "next_node": next_node,
            "thoughts": _think(workspace, "regression_check", f"Loop {loop}: {fault} → retry {next_node} ({loop}/{MAX_REGRESSION_LOOPS})"),
        }
    # Exhausted regression loops — fall through to verify as a last resort.
    return {
        "regression_loop_count": loop,
        "regression_errors": errors,
        "next_node": "verify",
        "thoughts": _think(workspace, "regression_check", f"Loop {loop}: {fault} — regression loop ceiling reached → verify"),
    }


# =============================================================================
# NODE: verify
# =============================================================================
def _docker_run_record(loop: int, status: str, duration: float, stdout: str, stderr: str) -> dict:
    return {
        "loop": loop,
        "status": status,
        "duration_seconds": round(duration, 3),
        "stdout_tail": stdout[-2000:],
        "stderr_tail": stderr[-1000:],
    }


@node_with_history
def verify(state: EditingState):
    workspace = state.get("workspace_dir", ".workspaces/default")
    manifest = state.get("file_manifest", {})
    loop = state.get("sandbox_loop_count", 0) + 1

    write_manifest_to_disk(workspace, manifest, preserve_unlisted=True)

    profile = _profile_from_state(state)
    IMAGE = profile.sandbox_value("image", "python:3.11-slim")
    memory_limit = profile.sandbox_value("memory", "512m")
    timeout_install = profile.sandbox_value("timeout_install", 90)
    timeout_test = profile.sandbox_value("timeout_test", 120)
    timeout_total = timeout_install + timeout_test + 60
    cpus = profile.sandbox_value("cpus", "1.0")

    host_uid, host_gid = _host_identity()
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
        "-e", "HOME=/tmp",
        "-e", "TMPDIR=/tmp",
        "-e", "PYTHONPYCACHEPREFIX=/tmp/pycache",
        "-e", f"SANDBOX_LOOP_COUNT={loop}",
    ]

    workspace_mount = f"{resolve_host_path(workspace)}:/workspace"
    base_mounts = ["-v", workspace_mount, "-w", "/workspace"]
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

    _diag(workspace, "verify", f"Running verification loop {loop}/{MAX_SANDBOX_LOOPS}")

    started = time.perf_counter()
    try:
        _clear_directory(os.path.join(workspace, ".deps"))
        _clear_directory(os.path.join(workspace, ".tmp"))

        install_script = profile.resolve(profile.sandbox_value("install_command"))
        install_cmd = (
            ["docker", "run", "--rm", "--name", f"{task_id}-install-{loop}-1"]
            + container_hardening
            + base_mounts
            + tmpfs_flags
            + install_env
            + [IMAGE, "bash", "-c", install_script]
        )
        install_started = time.perf_counter()
        install_res = subprocess.run(install_cmd, capture_output=True, text=True, timeout=timeout_install)
        install_duration = time.perf_counter() - install_started
        install_docker_runs = [_docker_run_record(loop, "dep_install", install_duration, install_res.stdout, install_res.stderr)]

        if install_res.returncode != 0:
            return _arbiter_failure(
                workspace,
                loop,
                status="infra_fault",
                sandbox_errors=f"Install failed:\nSTDOUT:\n{install_res.stdout}\nSTDERR:\n{install_res.stderr}",
                docker_runs=install_docker_runs,
            )

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
            + tmpfs_flags
            + verify_env
            + [IMAGE, "bash", "-c", arbiter_script]
        )
        arbiter_started = time.perf_counter()
        arbiter_res = subprocess.run(arbiter_cmd, capture_output=True, text=True, timeout=timeout_total)
        arbiter_duration = time.perf_counter() - arbiter_started
        total_duration = time.perf_counter() - started

        stdout = arbiter_res.stdout
        stderr = arbiter_res.stderr
        _diag(workspace, "verify", f"Verification output:\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")

        # Re-read files that tooling may have mutated.
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
                "thoughts": _think(workspace, "verify", f"Loop {loop}: verification PASS ({round(total_duration, 1)}s)"),
            }

        return _route_sandbox_failure(workspace, loop, diagnostics, docker_runs, manifest, test_impact=state.get("test_impact", "none"))

    except subprocess.TimeoutExpired as e:
        total_duration = time.perf_counter() - started
        diag = f"Verification timed out after {round(total_duration, 1)}s"
        _diag(workspace, "verify", diag)
        return _arbiter_failure(
            workspace,
            loop,
            status="infra_fault",
            sandbox_errors=diag + f"\n{e}",
            docker_runs=[_docker_run_record(loop, "timeout", total_duration, "", str(e))],
        )
    except Exception as e:
        total_duration = time.perf_counter() - started
        diag = f"Verification crashed: {type(e).__name__}: {e}"
        _diag(workspace, "verify", diag)
        return _arbiter_failure(
            workspace,
            loop,
            status="infra_fault",
            sandbox_errors=diag,
            docker_runs=[_docker_run_record(loop, "crash", total_duration, "", str(e))],
        )


def _parse_sandbox_output(stdout: str, stderr: str, loop: int) -> Dict[str, Any]:
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
        diagnostics["sandbox_status"] = "infra_fault"
        diagnostics["fault_location"] = "toolchain"

    return diagnostics


def _route_sandbox_failure(
    workspace: str,
    loop: int,
    diagnostics: Dict[str, Any],
    docker_runs: List[dict],
    manifest: Dict[str, str],
    test_impact: str = "none",
) -> Dict[str, Any]:
    status = diagnostics["sandbox_status"]
    _diag(workspace, "verify", f"Verification failure: status={status} (test_impact={test_impact})")

    # Determine the correct retry target based on which apply path was used.
    uses_tdd = test_impact != "none"
    code_retry_target = "apply_code" if uses_tdd else "apply"
    test_retry_target = "apply_tests" if uses_tdd else "plan"

    if status == "test_fault":
        return {
            "file_manifest": manifest,
            "sandbox_errors": diagnostics.get("raw_output", ""),
            "sandbox_diagnostics": diagnostics,
            "sandbox_loop_count": loop,
            "docker_runs": docker_runs,
            "next_node": test_retry_target,
            "thoughts": _think(workspace, "verify", f"Loop {loop}: test-side fault → {test_retry_target}"),
        }

    if status == "infra_fault":
        return {
            "file_manifest": manifest,
            "sandbox_errors": diagnostics.get("raw_output", ""),
            "sandbox_diagnostics": diagnostics,
            "sandbox_loop_count": loop,
            "docker_runs": docker_runs,
            "next_node": "FINISH",
            "thoughts": _think(workspace, "verify", f"Loop {loop}: infra fault → FINISH"),
        }

    if loop < MAX_SANDBOX_LOOPS:
        return {
            "file_manifest": manifest,
            "sandbox_errors": diagnostics.get("raw_output", ""),
            "sandbox_diagnostics": diagnostics,
            "sandbox_loop_count": loop,
            "docker_runs": docker_runs,
            "next_node": code_retry_target,
            "thoughts": _think(workspace, "verify", f"Loop {loop}: code fault → {code_retry_target}"),
        }

    return {
        "file_manifest": manifest,
        "sandbox_errors": diagnostics.get("raw_output", ""),
        "sandbox_diagnostics": diagnostics,
        "sandbox_loop_count": loop,
        "docker_runs": docker_runs,
        "next_node": "FINISH",
        "thoughts": _think(workspace, "verify", f"Loop {loop}: sandbox loop ceiling → FINISH"),
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
            "thoughts": _think(workspace, "verify", f"Loop {loop}: infra fault → FINISH"),
        }
    return {
        "sandbox_errors": sandbox_errors,
        "sandbox_diagnostics": diagnostics,
        "sandbox_loop_count": loop,
        "docker_runs": docker_runs,
        "next_node": "plan",
        "thoughts": _think(workspace, "verify", f"Loop {loop}: unrecoverable verification fault → plan"),
    }


# =============================================================================
# Routing helpers
# =============================================================================
def route_from_verify(state: EditingState) -> str:
    return state.get("next_node", "FINISH")
