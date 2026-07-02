import ast
import json
import os
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
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "llm_config.yaml")
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

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

# HTTP status codes and error names that we treat as transient provider faults.
TRANSIENT_HTTP_STATUSES = {502, 503, 504, 524}
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

SANDBOX_DEPS = (
    "pytest",
    "hypothesis",
    "ruff",
    "mypy",
)
# Base tools are invoked inside the test container as `python -m <tool>` using
# PYTHONPATH=/workspace/.deps, because `pip install --target` does not put
# console scripts on PATH.

# Loop / ceiling constants
MAX_SANDBOX_LOOPS = 5
MAX_COMPLIANCE_LOOPS = 2

# Server-side hard wall-clock deadline per task. A backstop that auto-cancels a
# task even if the client died without cancelling it.
SERVER_TASK_DEADLINE = CONFIG.get("server", {}).get("task_deadline_seconds", 3600)


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
            }
        else:
            base_url = os.getenv(
                api_key_env_var or "OLLAMA_BASE_URL", "http://localhost:11434"
            )
            client_kwargs = {"timeout": OLLAMA_TIMEOUT}

        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=base_url,
            streaming=False,
            disable_streaming=True,
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
    # Unwrap langchain / httpx / requests wrapper exceptions.
    inner = getattr(e, "__cause__", None) or e
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
    invoke_kwargs = {"stream": False} if provider in ollama_providers else {}

    input_tokens = _count_message_tokens(messages)
    usage_entries: list[dict] = []
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        started = time.perf_counter()
        try:
            llm = get_llm(node_name)

            def _call():
                response = llm.invoke(messages, **invoke_kwargs)
                content = response.content if hasattr(response, "content") else ""
                metadata = getattr(response, "response_metadata", {}) or {}
                return content, metadata

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_call)
                content, metadata = future.result(timeout=timeout_seconds)

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
                "error": f"Wall-clock timeout after {timeout_seconds}s",
                "response_metadata": {},
            })
            if workspace:
                _diag(
                    workspace,
                    node_name,
                    f"LLM attempt {attempt}/{max_attempts} timed out after {timeout_seconds}s",
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


def setup_workspace(workspace: str) -> None:
    """Write the deterministic sandbox scaffolding for a task.

    These files are written once per workspace and then left alone so that
    retry loops (test_architect or coder failures) do not wipe third-party
    dependencies added by the coder node.

    The workspace is chowned to the host user identity so that sandbox
    containers running as that user can write into it without running as root.
    """
    os.makedirs(workspace, exist_ok=True)

    conftest = textwrap.dedent(
        """\
        from hypothesis import HealthCheck, settings

        settings.register_profile(
            "sandbox",
            max_examples=50,
            deadline=5000,
            suppress_health_check=[HealthCheck.too_slow],
        )
        settings.load_profile("sandbox")
        """
    )
    if not os.path.exists(os.path.join(workspace, "conftest.py")):
        _write_workspace_file(workspace, "conftest.py", conftest)

    pytest_ini = "[pytest]\ntestpaths = tests\npythonpath = .\n"
    if not os.path.exists(os.path.join(workspace, "pytest.ini")):
        _write_workspace_file(workspace, "pytest.ini", pytest_ini)

    requirements = "\n".join(SANDBOX_DEPS) + "\n"
    if not os.path.exists(os.path.join(workspace, "requirements.txt")):
        _write_workspace_file(workspace, "requirements.txt", requirements)

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
# NODE: test_architect
# =============================================================================
@node_with_history
def test_architect(state: AgenticState):
    prompt = state["user_prompt"]
    workspace = state.get("workspace_dir", "")
    critique = state.get("compliance_critique", [])
    sandbox_errors = state.get("sandbox_errors", "")

    setup_workspace(workspace)

    system = textwrap.dedent(
        """\
        You are the Test Architect in a strict TDD code-generation pipeline.

        Your job is to produce TWO files from the user's natural-language prompt:
          1. tests/test_main.py — a complete pytest + Hypothesis property-based test suite.
          2. src/main.py — a matching skeleton that contains only class/function signatures
             and `pass` bodies. The skeleton must use type hints and signatures that are
             EXACTLY compatible with the assertions in tests/test_main.py.

        Rules for the test suite:
          - Use Hypothesis @given for invariants and randomized domains.
          - Add one explicit @given for every rule/guarantee in the prompt.
          - Constrain strategies at the source; never use assume().
          - Use pytest.raises for raise-rule cases.
          - Write property tests that would reject a lazy, literal, or loophole-seeking
            implementation. Assume the coder will try to pass with the smallest possible
            code; close those loopholes by testing implied invariants, adversarial shapes,
            and edge cases that a robust solution must handle even if the prompt does not
            list them explicitly.
          - Free-form, tautological, or vacuous assertions are prohibited.
          - Import the implementation from src.main.
          - Every @given test must have @settings(max_examples=50).
          - Recursive Hypothesis strategies must use @st.composite and draw(...).
          - Preserve sub-expression boundaries: parenthesize any fragment inserted into a larger expression.

        Rules for the skeleton:
          - Only output signatures + pass.
          - Do not write any algorithmic logic here.
          - The signatures and type hints must be exactly what the tests expect.
          - Include src/__init__.py (empty) and tests/__init__.py (empty).

        Output format: wrap each file in exactly:
          <file name="src/main.py">...</file>
          <file name="src/__init__.py">...</file>
          <file name="tests/test_main.py">...</file>
          <file name="tests/__init__.py">...</file>

        Do not output markdown fences, commentary, or any other files.
        """
    )

    user_prompt = f"User prompt:\n{prompt}"
    user_prompt += (
        "\n\nLanguage-specific guidance: If the prompt asks for an expression evaluator, "
        "calculator, parser, or any task that evaluates arithmetic or structured strings, "
        "generate raw expression strings and compute expected values using the target "
        "language's standard evaluator (e.g., Python's eval() with a safe scope). Do not "
        "hand-write AST renderers or string-composition logic that must preserve "
        "operator precedence; let the language parser be the source of truth."
    )
    if critique:
        user_prompt += (
            "\n\nPrior compliance critiques (the previous implementation was missing these):\n"
            + "\n".join(f"- {c}" for c in critique)
        )
    if sandbox_errors:
        user_prompt += (
            f"\n\nThe last sandbox run failed with the following errors. "
            f"If the fault is in the tests (syntax, imports, hypothesis health check, vacuity), "
            f"revise the tests and the matching skeleton:\n{sandbox_errors[:4000]}"
        )

    try:
        content, llm_usage = _invoke_with_retry(
            "test_architect",
            [SystemMessage(content=system), HumanMessage(content=user_prompt)],
            workspace,
        )
    except LLMUnavailableError as e:
        return _handle_llm_unavailable("test_architect", state, e, workspace)

    files = _parse_file_tags(content)

    # Validate that the required files exist.
    required = {"src/main.py", "src/__init__.py", "tests/test_main.py", "tests/__init__.py"}
    missing = required - set(files.keys())
    if missing:
        _diag(
            workspace,
            "test_architect",
            f"Missing required files: {missing}\nRaw response (first 2000 chars):\n{content[:2000]}",
        )
        return {
            "sandbox_errors": f"Test Architect Error: missing files {sorted(missing)}",
            "llm_usage": llm_usage,
            "next_node": "FINISH",
            "thoughts": _think(
                workspace,
                "test_architect",
                f"ERROR: missing files {sorted(missing)}",
            ),
        }

    _diag(
        workspace,
        "test_architect",
        f"Generated files:\n"
        + "\n".join(f"  {fname} ({len(code)} chars)" for fname, code in sorted(files.items())),
    )

    return {
        "file_manifest": files,
        "sandbox_errors": "",
        "sandbox_diagnostics": {},
        "sandbox_loop_count": 0,
        "llm_usage": llm_usage,
        "next_node": "coder",
        "thoughts": _think(
            workspace,
            "test_architect",
            f"Wrote {len(files)} files: {', '.join(sorted(files.keys()))}",
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

    test_code = manifest.get("tests/test_main.py", "")
    stub_code = manifest.get("src/main.py", "")

    if not test_code or not stub_code:
        return {
            "sandbox_errors": "Coder Error: missing tests/test_main.py or src/main.py skeleton",
            "next_node": "FINISH",
            "thoughts": _think(
                workspace,
                "coder",
                "ERROR: missing frozen test or skeleton before coding",
            ),
        }

    system = textwrap.dedent(
        """\
        You are the Coder in a strict TDD pipeline.

        You are given:
          - The original user prompt.
          - The frozen tests/test_main.py file (you may NOT modify it).
          - The src/main.py skeleton with signatures and type hints (you may NOT change
            any signature, class name, or function name).

        Your job: replace every `pass` body in src/main.py with correct, clean algorithmic
        logic so that tests/test_main.py passes.

        Rules:
          - Only modify function/class bodies. Do not change signatures, imports, or class structure.
          - Do not output tests/test_main.py or any test file.
          - Do not hardcode outputs or special-case inputs.
          - Prefer clean, readable Python over clever one-liners.
          - If requirements.txt is needed, include it as <file name="requirements.txt">.

        Output format: wrap each file in exactly:
          <file name="src/main.py">...</file>
          <file name="requirements.txt">...</file>  (only if third-party deps are needed)

        Do not output markdown fences, commentary, or any test files.
        """
    )

    user_prompt = f"User prompt:\n{prompt}\n\nFrozen tests/test_main.py:\n{test_code}\n\nSkeleton src/main.py:\n{stub_code}"
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
        )
    except LLMUnavailableError as e:
        return _handle_llm_unavailable("coder", state, e, workspace)

    new_files = _parse_file_tags(content)

    if "src/main.py" not in new_files:
        _diag(
            workspace,
            "coder",
            f"No src/main.py in coder output\nRaw response (first 2000 chars):\n{content[:2000]}",
        )
        return {
            "sandbox_errors": "Coder Error: failed to produce src/main.py",
            "llm_usage": llm_usage,
            "next_node": "FINISH",
            "thoughts": _think(
                workspace,
                "coder",
                "ERROR: no src/main.py produced",
            ),
        }

    # Preserve frozen test files and init files; only overwrite src/main.py and requirements.txt.
    preserved = {
        k: v
        for k, v in manifest.items()
        if k.startswith("tests/") or k in {"src/__init__.py"}
    }
    final_manifest = {**preserved, "src/main.py": new_files["src/main.py"]}

    # Merge any third-party requirements from the coder on top of the base sandbox deps
    # so pytest/hypothesis/ruff/mypy are never lost on a retry.
    base_reqs = set(SANDBOX_DEPS)
    if "requirements.txt" in new_files:
        extra_lines = [
            line.strip()
            for line in new_files["requirements.txt"].splitlines()
            if line.strip() and line.strip() not in base_reqs
        ]
        final_manifest["requirements.txt"] = "\n".join(SANDBOX_DEPS + extra_lines) + "\n"
    elif "requirements.txt" in manifest:
        final_manifest["requirements.txt"] = manifest["requirements.txt"]

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
            f"Implemented src/main.py ({len(new_files['src/main.py'])} chars)",
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

    docker_cfg = CONFIG.get("docker", {})
    IMAGE = docker_cfg.get("image", "python:3.11-slim")
    memory_limit = docker_cfg.get("memory_limit", "512m")
    timeout_install = docker_cfg.get("timeout_install", 90)
    timeout_test = docker_cfg.get("timeout_test", 120)
    timeout_total = timeout_install + timeout_test + 60  # headroom for ruff/mypy/pytest

    host_uid, host_gid = _host_identity()

    # Base hardening applied to both install and verification containers.
    # The container runs as the host user, not root, so files it writes to
    # the workspace remain owned by the host user.
    hardening_flags = [
        f"--user={host_uid}:{host_gid}",
        f"--memory={memory_limit}",
        f"--memory-swap={memory_limit}",
        "--cpus=1.0",
        "--pids-limit=64",
        "--ulimit=nofile=1024:1024",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "-v",
        f"{resolve_host_path(workspace)}:/workspace",
        "-w",
        "/workspace",
        "-e",
        "HOME=/tmp",
        "-e",
        "TMPDIR=/tmp",
        "-e",
        "PYTHONPYCACHEPREFIX=/tmp/pycache",
        "-e",
        f"SANDBOX_LOOP_COUNT={loop}",
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
    install_script = (
        "mkdir -p /workspace/.tmp && "
        "pip install -q --target /workspace/.deps -r /workspace/requirements.txt"
    )
    install_cmd = (
        ["docker", "run", "--rm"]
        + hardening_flags
        + tmpfs_flags
        + install_env
        + [IMAGE, "bash", "-c", install_script]
    )

    # Verification container: read-only root filesystem, no network, tmpfs for
    # /tmp and /var/tmp. Only /workspace (the bind mount) is writable.
    # All Python tools are invoked as `python -m` because pip's `--target` flag
    # installs console scripts under .deps/bin, which is not on PATH.
    verify_env = [
        "-e", "PYTHONPATH=/workspace/.deps",
        "-e", "RUFF_CACHE_DIR=/tmp/ruff_cache",
        "-e", "MYPY_CACHE_DIR=/tmp/mypy_cache",
    ]
    arbiter_script = textwrap.dedent(
        """\
        set -euo pipefail
        export PYTHONPATH=/workspace/.deps
        export RUFF_CACHE_DIR=/tmp/ruff_cache
        export MYPY_CACHE_DIR=/tmp/mypy_cache

        echo "__RUFF_FORMAT_SRC_START__"
        python -m ruff format src/main.py || { echo "__RUFF_FORMAT_SRC_FAILED__"; exit 1; }
        echo "__RUFF_FORMAT_SRC_OK__"

        echo "__RUFF_FORMAT_TESTS_START__"
        python -m ruff format tests/test_main.py || { echo "__RUFF_FORMAT_TESTS_FAILED__"; exit 1; }
        echo "__RUFF_FORMAT_TESTS_OK__"

        echo "__RUFF_CHECK_SRC_START__"
        python -m ruff check --fix src/main.py || { echo "__RUFF_CHECK_SRC_FAILED__"; exit 1; }
        echo "__RUFF_CHECK_SRC_OK__"

        echo "__RUFF_CHECK_TESTS_START__"
        python -m ruff check --fix tests/test_main.py || { echo "__RUFF_CHECK_TESTS_FAILED__"; exit 1; }
        echo "__RUFF_CHECK_TESTS_OK__"

        echo "__TAUTOLOGY_START__"
        python -c "
import ast, sys
tree = ast.parse(open('tests/test_main.py').read())
for n in ast.walk(tree):
    if isinstance(n, ast.Compare) and len(n.comparators) == 1:
        if ast.dump(n.left) == ast.dump(n.comparators[0]):
            print('__TAUTOLOGY_DETECTED__')
            sys.exit(1)
" || { echo "__TAUTOLOGY_FAILED__"; exit 1; }
        echo "__TAUTOLOGY_OK__"

        echo "__MYPY_START__"
        python -m mypy --ignore-missing-imports src/main.py || { echo "__MYPY_SRC_FAILED__"; exit 1; }
        echo "__MYPY_OK__"

        echo "__PYTEST_START__"
        python -m pytest tests/test_main.py \
            -p no:cacheprovider \
            --hypothesis-seed=42 \
            --hypothesis-profile=sandbox \
            || { echo "__PYTEST_FAILED__"; exit 1; }
        echo "__PYTEST_OK__"

        echo "__SANDBOX_PASS__"
        """
    )
    arbiter_cmd = (
        ["docker", "run", "--rm", "--network", "none", "--read-only"]
        + hardening_flags
        + tmpfs_flags
        + verify_env
        + [IMAGE, "bash", "-c", arbiter_script]
    )

    ws = workspace
    _diag(ws, "sandbox_arbiter", f"Running sandbox loop {loop}/{MAX_SANDBOX_LOOPS}")

    started = time.perf_counter()
    try:
        # Phase 1: dependency install (network enabled)
        install_started = time.perf_counter()
        install_res = subprocess.run(
            install_cmd, capture_output=True, text=True, timeout=timeout_install
        )
        install_duration = time.perf_counter() - install_started
        if install_res.returncode != 0:
            diag = (
                f"Dependency install failed (loop {loop}):\n"
                f"STDOUT:\n{install_res.stdout}\nSTDERR:\n{install_res.stderr}"
            )
            _diag(ws, "sandbox_arbiter", diag)
            return _arbiter_failure(
                ws,
                loop,
                status="infra_fault",
                sandbox_errors=diag,
                docker_runs=[_docker_run_record(loop, "dep_install_failed", install_duration, install_res.stdout, install_res.stderr)],
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

        # Re-read files in case ruff mutated them.
        new_main = read_file_from_disk(workspace, "src/main.py")
        new_test = read_file_from_disk(workspace, "tests/test_main.py")
        if new_main:
            manifest["src/main.py"] = new_main
        if new_test:
            manifest["tests/test_main.py"] = new_test

        diagnostics = _parse_sandbox_output(stdout, stderr, loop)

        docker_runs = [_docker_run_record(loop, diagnostics["sandbox_status"], arbiter_duration, stdout, stderr)]

        if diagnostics["sandbox_status"] == "pass":
            return {
                "file_manifest": manifest,
                "sandbox_errors": "",
                "sandbox_diagnostics": diagnostics,
                "sandbox_loop_count": loop,
                "docker_runs": docker_runs,
                "next_node": "prompt_compliance_checker",
                "thoughts": _think(
                    ws,
                    "sandbox_arbiter",
                    f"Loop {loop}: all sandbox checks PASS ({round(total_duration, 1)}s)",
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
            "next_node": "test_architect",
            "thoughts": _think(
                workspace,
                "sandbox_arbiter",
                f"Loop {loop}: test-side fault → test_architect",
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

    # Exhausted sandbox loops: replan from test architect.
    return {
        "file_manifest": manifest,
        "sandbox_errors": diagnostics.get("raw_output", ""),
        "sandbox_diagnostics": diagnostics,
        "sandbox_loop_count": 0,
        "docker_runs": docker_runs,
        "next_node": "test_architect",
        "thoughts": _think(
            workspace,
            "sandbox_arbiter",
            f"Loop {loop}: sandbox loop ceiling → test_architect",
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
        "next_node": "test_architect",
        "thoughts": _think(
            workspace,
            "sandbox_arbiter",
            f"Loop {loop}: unrecoverable sandbox fault → test_architect",
        ),
    }


# =============================================================================
# NODE: prompt_compliance_checker
# =============================================================================
@node_with_history
def prompt_compliance_checker(state: AgenticState):
    prompt = state["user_prompt"]
    workspace = state.get("workspace_dir", "")
    manifest = state.get("file_manifest", {})
    compliance_loop = state.get("compliance_loop_count", 0)
    prior_critique = state.get("compliance_critique", [])

    src_code = manifest.get("src/main.py", "")
    test_code = manifest.get("tests/test_main.py", "")

    system = textwrap.dedent(
        """\
        You are the Prompt Compliance Checker in a TDD code-generation pipeline.

        You are given:
          - The original natural-language user prompt.
          - The passing src/main.py implementation.
          - The passing tests/test_main.py suite.

        Evaluate whether src/main.py structurally satisfies the user prompt.
        Only judge completeness against the prompt's explicit functional requirements.
        Style, naming conventions, and performance optimization are explicitly out of scope.
        However, if the prompt names a standard algorithmic construct (e.g., "graph",
        "parser", "heap", "queue", "evaluator"), the implementation must be structurally
        sound for that construct: correct semantics, idiomatic data structures, and no
        obvious anti-patterns that would make it fail under normal adversarial use. Reject
        only clear functional or structural defects, not cosmetic preferences.
        Do NOT require features the prompt does not mention.

        Output exactly a JSON object and nothing else:
          {"compliance_status": "PASS" | "FAIL", "missing_features": ["..."]}

        Use "PASS" only if the implementation demonstrably covers every functional
        requirement in the prompt. Use "FAIL" if a required behavior or feature is
        missing or incomplete; list the missing features concisely, one string per item.
        """
    )

    user_prompt = f"User prompt:\n{prompt}\n\nsrc/main.py:\n{src_code}\n\ntests/test_main.py:\n{test_code}"
    if prior_critique:
        user_prompt += (
            "\n\nPrior compliance critiques:\n"
            + "\n".join(f"- {c}" for c in prior_critique)
        )

    try:
        content, llm_usage = _invoke_with_retry(
            "prompt_compliance_checker",
            [SystemMessage(content=system), HumanMessage(content=user_prompt)],
            workspace,
        )
    except LLMUnavailableError as e:
        return _handle_llm_unavailable("prompt_compliance_checker", state, e, workspace)

    # Extract JSON from the response (allow surrounding markdown fences).
    json_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```|\{.*\"compliance_status\".*\}",
        content,
        re.DOTALL,
    )
    raw_json = json_match.group(1) if json_match and json_match.group(1) else content
    # Try to find any JSON object if the above failed.
    if not raw_json.strip().startswith("{"):
        obj_match = re.search(r"\{.*\}", content, re.DOTALL)
        raw_json = obj_match.group(0) if obj_match else content

    try:
        verdict = json.loads(raw_json)
    except Exception as e:
        _diag(
            workspace,
            "prompt_compliance_checker",
            f"Failed to parse compliance JSON. Raw content:\n{content[:2000]}\nParse error: {e}",
        )
        # Default to FAIL with the raw content as a critique.
        verdict = {
            "compliance_status": "FAIL",
            "missing_features": [f"Compliance checker returned unparseable JSON: {str(e)[:200]}"],
        }

    status = verdict.get("compliance_status", "FAIL").upper()
    missing = verdict.get("missing_features", []) or []
    if not isinstance(missing, list):
        missing = [str(missing)]
    missing = [str(m) for m in missing if m]

    _diag(
        workspace,
        "prompt_compliance_checker",
        f"Compliance status: {status}\nMissing features: {missing}",
    )

    new_critique = list(prior_critique) + missing

    if status == "PASS":
        return {
            "compliance_status": "PASS",
            "compliance_critique": new_critique,
            "llm_usage": llm_usage,
            "next_node": "FINISH",
            "thoughts": _think(
                workspace,
                "prompt_compliance_checker",
                "Compliance PASS — finishing",
            ),
        }

    new_compliance_loop = compliance_loop + 1
    if new_compliance_loop <= MAX_COMPLIANCE_LOOPS:
        return {
            "compliance_status": "FAIL",
            "compliance_critique": new_critique,
            "compliance_loop_count": new_compliance_loop,
            "sandbox_errors": "",
            "sandbox_diagnostics": {},
            "sandbox_loop_count": 0,
            "llm_usage": llm_usage,
            "next_node": "test_architect",
            "thoughts": _think(
                workspace,
                "prompt_compliance_checker",
                f"Compliance FAIL ({new_compliance_loop}/{MAX_COMPLIANCE_LOOPS}) → test_architect: {missing}",
            ),
        }

    # Compliance loop ceiling reached.
    return {
        "compliance_status": "FAIL",
        "compliance_critique": new_critique,
        "compliance_loop_count": new_compliance_loop,
        "llm_usage": llm_usage,
        "next_node": "FINISH",
        "thoughts": _think(
            workspace,
            "prompt_compliance_checker",
            f"Compliance FAIL — loop ceiling reached, finishing",
        ),
    }


# =============================================================================
# Routing functions
# =============================================================================
def route_from_sandbox(state: AgenticState) -> str:
    """Pure routing helper; kept as a function for readability and testing."""
    return state.get("next_node", "FINISH")


def route_from_compliance(state: AgenticState) -> str:
    """Pure routing helper; kept as a function for readability and testing."""
    return state.get("next_node", "FINISH")
