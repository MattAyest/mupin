import ast
import json
import os
import re
import shutil
import socket
import subprocess
from datetime import datetime
from functools import lru_cache

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from .state import SwarmState

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "llm_config.yaml")
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

# Ollama Cloud is reached over the public API host with bearer auth.
OLLAMA_CLOUD_HOST = "https://ollama.com"

SUPPORTED_PROVIDERS = (
    "google-genai",
    "openai",
    "anthropic",
    "ollama-cloud",
    "ollama",
    "openai-compatible",
)


# ---------------------------------------------------------
# LLM FACTORY
# Builds a LangChain chat client for a single node from llm_config.yaml.
# Providers map to suppliers; "ollama-cloud" targets the hosted Ollama
# service (ollama.com) so most nodes can run off an Ollama Pro account,
# while the heaviest nodes can be pointed at a premium API instead.
#
# Heavy provider imports are deferred until a node actually needs them so
# that an unused/uninstalled provider never breaks startup.
# ---------------------------------------------------------
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

    if provider == "ollama-cloud":
        from langchain_ollama import ChatOllama

        key = require_key("OLLAMA_API_KEY")
        return ChatOllama(
            model=model,
            temperature=temperature,
            base_url=OLLAMA_CLOUD_HOST,
            client_kwargs={"headers": {"Authorization": f"Bearer {key}"}},
            request_timeout=300,
        )

    if provider == "ollama":
        # Local / self-hosted Ollama daemon.
        # api_key_env_var (if any) names the base-URL env var.
        from langchain_ollama import ChatOllama

        base_url = os.getenv(
            api_key_env_var or "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        return ChatOllama(model=model, temperature=temperature, base_url=base_url)

    if provider == "openai-compatible":
        # Any self-hosted server speaking the OpenAI API (vLLM, LocalAI,
        # LM Studio, llama.cpp, text-generation-webui, ...).
        # Requires a `base_url` in the node config; api_key is optional
        # (many self-hosted servers ignore it — default to a placeholder).
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
    at setup time rather than mid-run. Returns a list of (node, error) tuples."""
    problems = []
    for node_name in CONFIG.get("nodes", {}):
        try:
            get_llm(node_name)
        except Exception as e:  # noqa: BLE001 - report, don't crash
            problems.append((node_name, str(e)))
    return problems


# Load loop limits
LOOP_CEILING = CONFIG["loop_limits"]["max_verification_loops"]
REGRESSION_CEILING = CONFIG["loop_limits"]["max_regression_count"]
REPLAN_CEILING = CONFIG["loop_limits"]["max_replan_count"]


# ---------------------------------------------------------
# HELPER: Unified thought logger.
# Each node calls this with a short one-line summary of what it
# did or decided. The thought is appended to the state's `thoughts`
# list (via the Annotated reducer) AND written to task.log on disk
# so progress is visible even without polling the API.
# ---------------------------------------------------------
def _think(workspace: str, node: str, message: str) -> list[str]:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] [{node}] {message}"
    try:
        with open(os.path.join(workspace, "task.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    return [line]


def _diag(workspace: str, node: str, detail: str) -> None:
    """Write a multi-line diagnostic block to task.log only (not to state).
    Indented under the preceding _think one-liner for readability."""
    if not detail:
        return
    indented = "\n".join("  " + line for line in detail.strip().splitlines())
    try:
        with open(os.path.join(workspace, "task.log"), "a", encoding="utf-8") as f:
            f.write(indented + "\n")
    except Exception as e:
        import sys
        print(f"[_diag] write failed for {workspace}: {e}", file=sys.stderr)


# ---------------------------------------------------------
# HELPER: Resolve host-side path for a container-internal path.
# Docker sets the container hostname to the container ID, so we can
# inspect our own mounts via the socket to find the host path.
# ---------------------------------------------------------
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
            if abs_path.startswith(dest) and (
                best is None or len(dest) > len(best["Destination"])
            ):
                best = mount
        if best:
            relative = abs_path[len(best["Destination"]) :]
            return best["Source"] + relative
    except Exception:
        pass
    return abs_path


# ---------------------------------------------------------
# NODE: WORKSPACE LOADER
# ---------------------------------------------------------
def workspace_loader(state: SwarmState):
    workspace = state.get("workspace_dir", ".workspaces/default")
    os.makedirs(workspace, exist_ok=True)
    return {
        "next_node": "architect_node",
        "thoughts": _think(
            workspace, "workspace_loader", f"Workspace ready at {workspace}"
        ),
    }


# ---------------------------------------------------------
# NODE: ARCHITECT
# Always runs. Produces two outputs stored separately in state:
#   <plan>     — architectural shape + quality properties for the code writer
#   <contract> — precise behavioral spec for the test writer (signatures,
#                returns, raise-rules, correctness guarantees)
# On retry from error_distiller (fault=spec), receives prior context
# and revises the contract.
# ---------------------------------------------------------
def architect_node(state: SwarmState):
    prompt = state.get("messages", [])[-1].content
    feedback = state.get("verification_errors", "")
    existing_contract = state.get("interface_contract", "")

    system = (
        "You are the systems architect in a TDD code-generation pipeline.\n"
        "Think like a data engineer: you care about the shape of the data, how it flows,\n"
        "and the contract at each boundary — not the tiny implementation details.\n"
        "You define the what and the why; the code writer handles the how.\n\n"
        "Your contract is the sole source of truth for the test writer, who never sees\n"
        "the original task. The code writer reads both sections to build the implementation.\n\n"
        "Output exactly two XML sections:\n\n"
        "<plan>\n"
        "The architectural shape that fits this problem: the kind of structure it needs\n"
        "and any quality properties that matter for correctness or performance. Scale this\n"
        "to the problem — a single sentence for a simple function, more depth for a system.\n"
        "</plan>\n\n"
        "<contract>\n"
        "A precise behavioral specification. For every public function or method, define:\n"
        "  - The exact signature with full type hints\n"
        "  - What it returns and what that value represents\n"
        "  - When it raises, stated as a rule rather than a list of cases\n"
        "  - The guarantees that define correctness, shown with a few concrete\n"
        "    input-to-output pairs at the meaningful boundaries\n\n"
        "State conditions as rules. A rule covers every case at once; a list always\n"
        "misses one and invites endless extension. Show the difference:\n\n"
        '    As a rule:  "Raises TypeError unless n is an int."\n'
        '    As a list:  "Raises TypeError for float, str, None, and similar."\n\n'
        "The rule is complete and closed. The list is neither. Always prefer the rule.\n"
        "</contract>"
    )

    user_prompt = prompt
    if feedback and existing_contract:
        user_prompt += (
            f"\n\nPrevious contract:\n{existing_contract}"
            f"\n\nRun failed with spec fault — revise the contract:\n{feedback}"
        )

    try:
        res = get_llm("architect_node").invoke(
            [SystemMessage(content=system), HumanMessage(content=user_prompt)]
        )
        content = res.content

        plan_match = re.search(
            r"<plan>(.*?)</plan>", content, re.DOTALL | re.IGNORECASE
        )
        contract_match = re.search(
            r"<contract>(.*?)</contract>", content, re.DOTALL | re.IGNORECASE
        )

        plan = plan_match.group(1).strip() if plan_match else content
        contract = contract_match.group(1).strip() if contract_match else content

    except Exception as e:
        plan = f"Fallback plan: {str(e)}"
        contract = prompt

    is_replan = bool(feedback and existing_contract)
    ws = state.get("workspace_dir", "")
    thought = (
        f"Replanning — {feedback[:80]}"
        if is_replan
        else f"Designed plan ({len(plan)} chars), contract ({len(contract)} chars)"
    )
    _diag(ws, "architect_node", f"PLAN:\n{plan}\n\nCONTRACT:\n{contract}")

    return {
        "architectural_plan": plan,
        "interface_contract": contract,
        "contract_check_count": 0,
        "next_node": "test_writer",
        "thoughts": _think(ws, "architect_node", thought),
    }


# ---------------------------------------------------------
# NODE: TEST WRITER
# Receives ONLY the interface contract — never sees the implementation.
# Tests are written as a spec, not a mirror of the code.
# On retry from error_distiller (fault=tests), receives the prior
# verification error as context to revise specific tests.
# ---------------------------------------------------------
def test_writer(state: SwarmState):
    contract = state.get("interface_contract", "")
    errors = state.get("verification_errors", "")
    manifest = state.get("file_manifest", {})

    system = (
        "You are the test writer node in a TDD pipeline.\n"
        "Tests are written before any implementation exists. They verify that the contract holds —\n"
        "they are the executable check on correctness, not a cage for the implementer.\n\n"
        "Write a pytest test suite that covers exactly what the interface contract specifies —\n"
        "every behavior, return guarantee, and raise-rule it defines, and nothing it does not.\n\n"
        "Output each test file wrapped in an XML tag:\n"
        '<file name="tests/test_main.py">\n'
        "# file content here\n"
        "</file>\n\n"
        "RULES:\n"
        "1. ALL test files go in a tests/ subdirectory and must be named test_*.py.\n"
        "2. Always include a tests/__init__.py (can be empty).\n"
        "3. Import the implementation from src.main (the public API lives there).\n"
        "4. Reach for hypothesis where a property holds across a whole input domain and "
        "property-based testing earns its keep. For fixed behaviors and specific cases, a plain "
        "assertion is clearer. When you do use hypothesis:\n"
        "   - Constrain the strategy at the source rather than filtering with assume().\n"
        "     A bounded strategy like st.integers(min_value=1) is better than st.integers() with a guard.\n"
        "   - Generate inputs already in the required shape — sorted data via .map(sorted), for example.\n"
        "   - Add @settings(max_examples=50) to every @given test.\n"
        "5. Output raw Python inside the XML tags — no markdown code fences.\n"
        "6. Output ONLY test files — no src/ files, no requirements.txt."
    )

    user_prompt = f"Interface Contract:\n{contract}"
    if errors:
        user_prompt += (
            f"\n\nPrevious failures — revise tests to match the contract:\n{errors}"
        )

    try:
        response = get_llm("test_writer").invoke(
            [SystemMessage(content=system), HumanMessage(content=user_prompt)]
        )
        content = response.content if hasattr(response, "content") else ""

        test_files = {}
        matches = re.finditer(
            r'<file name=[\'"](.*?)[\'"]>\s*(.*?)\s*</file>',
            content,
            re.DOTALL | re.IGNORECASE,
        )
        for match in matches:
            code = match.group(2).strip()
            code = re.sub(r"^```[a-zA-Z]*\n?", "", code).rstrip("`").strip()
            filename = match.group(1).strip()
            if filename.startswith("tests/"):
                test_files[filename] = code

        if not test_files:
            raise ValueError("No test files generated.")

        ws = state.get("workspace_dir", "")
        test_summary = "\n".join(
            f"  {fname} ({len(code)} chars, {code.count('def test_')} tests)"
            for fname, code in test_files.items()
        )
        _diag(ws, "test_writer", f"Test files:\n{test_summary}")

        # Preserve any existing src/ files; replace all test files with new ones
        src_files = {
            k: v
            for k, v in manifest.items()
            if k.startswith("src/") or k == "requirements.txt"
        }
        return {
            "file_manifest": {**src_files, **test_files},
            "next_node": "contract_verifier",
            "thoughts": _think(
                ws,
                "test_writer",
                f"Wrote {len(test_files)} test files"
                + (f" — revised: {errors[:80]}" if errors else ""),
            ),
        }

    except Exception as e:
        return {
            "verification_errors": f"Test Writer Error: {str(e)}",
            "next_node": "error_distiller",
            "thoughts": _think(
                state.get("workspace_dir", ""), "test_writer", f"ERROR: {str(e)[:100]}"
            ),
        }


# ---------------------------------------------------------
# NODE: CONTRACT VERIFIER
# Lightweight flash-model check: do the generated tests correctly and
# completely reflect the interface contract? Runs before any code is
# written so bad tests are caught cheaply.
# Loop-guarded: after 3 failed checks it proceeds anyway to avoid
# an infinite test_writer loop.
# ---------------------------------------------------------
def contract_verifier(state: SwarmState):
    contract = state.get("interface_contract", "")
    manifest = state.get("file_manifest", {})
    check_count = state.get("contract_check_count", 0) + 1

    test_context = "\n\n".join(
        f"# {filename}\n{code}"
        for filename, code in manifest.items()
        if filename.startswith("tests/") and filename.endswith(".py")
    )

    system = (
        "You are the contract compliance checker in a TDD pipeline.\n"
        "Tests were written from an interface contract before any implementation existed.\n"
        "Your job: confirm the tests faithfully reflect what the contract states.\n\n"
        "Check two things only:\n"
        "  - Do any tests contradict the contract — wrong signature, wrong return, "
        "an exception the contract does not specify, or behavior the contract does not define?\n"
        "  - Does any behavior, return guarantee, or raise-rule the contract explicitly states "
        "go untested?\n\n"
        "Judge against what the contract actually says. Do not require tests for inputs or cases "
        "the contract does not mention — absence from the contract is not a gap.\n\n"
        "If the tests faithfully reflect the contract, respond with exactly:\n"
        "PASS\n\n"
        "Otherwise list each real issue, one sentence per line:\n"
        "FAIL: <contradiction or untested contract requirement>\n"
        "FAIL: <contradiction or untested contract requirement>"
    )

    user_prompt = f"Contract:\n{contract}\n\nTest Suite:\n{test_context}"

    try:
        res = get_llm("contract_verifier").invoke(
            [SystemMessage(content=system), HumanMessage(content=user_prompt)]
        )
        verdict = res.content.strip()
        ws = state.get("workspace_dir", "")
        _diag(ws, "contract_verifier", f"Verdict (attempt {check_count}/3):\n{verdict}")
    except Exception as e:
        # LLM call failed — surface the error but proceed rather than blocking the pipeline
        return {
            "contract_check_count": check_count,
            "verification_errors": f"Contract verifier unavailable: {str(e)}",
            "next_node": "code_writer",
            "thoughts": _think(
                state.get("workspace_dir", ""),
                "contract_verifier",
                f"Validator unavailable, proceeding — {str(e)[:80]}",
            ),
        }

    if verdict.upper().startswith("PASS"):
        return {
            "contract_check_count": check_count,
            "verification_errors": "",
            "next_node": "code_writer",
            "thoughts": _think(
                ws,
                "contract_verifier",
                "Tests match contract — PASS",
            ),
        }

    # Tests don't match the contract
    if check_count >= 3:
        # Force proceed after 3 attempts rather than loop forever
        return {
            "contract_check_count": check_count,
            "verification_errors": "",
            "next_node": "code_writer",
            "thoughts": _think(
                ws,
                "contract_verifier",
                f"Tests mismatch (attempt {check_count}/3) — forcing proceed",
            ),
        }

    return {
        "contract_check_count": check_count,
        "verification_errors": verdict,
        "next_node": "test_writer",
        "thoughts": _think(
            ws,
            "contract_verifier",
            f"Tests mismatch (attempt {check_count}/3) — {verdict[:80]}",
        ),
    }


# ---------------------------------------------------------
# NODE: CODE WRITER
# TDD mode: receives the architectural plan AND the pre-written tests.
# Writes only src/ files and requirements.txt — never test files.
# On retry the tests are frozen; only the implementation changes.
# ---------------------------------------------------------
def code_writer(state: SwarmState):
    prompt = state.get("messages", [])[-1].content
    plan = state.get("architectural_plan", "")
    contract = state.get("interface_contract", "")
    errors = state.get("verification_errors", "")
    graveyard = state.get("rollback_graveyard", [])
    manifest = state.get("file_manifest", {})

    system = (
        "You are the implementation node in a TDD pipeline.\n"
        "Your job is to correctly implement the interface contract.\n"
        "The contract defines the expected inputs, outputs, exceptions, and edge cases — "
        "that is what you are writing to.\n"
        "A test suite will verify your implementation against the contract. "
        "Do not optimise for the tests — implement the contract correctly and the tests will pass.\n\n"
        "Output each source file wrapped in an XML tag:\n"
        '<file name="src/main.py">\n'
        "# file content here\n"
        "</file>\n\n"
        "RULES:\n"
        "1. The public implementation lives in src/main.py. The test suite imports from src.main, "
        "so the contract's public functions and classes must be defined or importable there.\n"
        "2. Always include src/__init__.py. Additional helper modules may go in src/ if useful.\n"
        '3. Always output a <file name="requirements.txt"> listing every third-party pip dependency.\n'
        "   If no third-party packages are needed, output an empty requirements.txt.\n"
        "4. Do NOT output any test files.\n"
        "5. Output raw Python inside the XML tags — no markdown code fences.\n"
        "6. Do not hardcode outputs or special-case inputs — implement the actual logic.\n"
        "   A semantic validator checks for this and will send you back if found.\n"
    )

    user_prompt = prompt
    if contract:
        user_prompt += f"\n\nInterface Contract (primary spec — implement this correctly, do not find loopholes):\n{contract}"
    if plan:
        user_prompt += f"\n\nArchitecture Plan:\n{plan}"
    if errors:
        user_prompt += f"\n\nPrevious failures — fix the implementation:\n{errors}"
    if graveyard:
        user_prompt += "\n\nAVOID these failed approaches:\n" + "\n".join(
            graveyard[-2:]
        )

    ws = state.get("workspace_dir", "")
    max_attempts = 3
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            if attempt > 1:
                _think(ws, "code_writer", f"Retry {attempt - 1}/{max_attempts - 1} after: {str(last_error)[:80]}")

            response = get_llm("code_writer").invoke(
                [SystemMessage(content=system), HumanMessage(content=user_prompt)]
            )
            content = response.content if hasattr(response, "content") else ""

            new_files = {}
            matches = re.finditer(
                r'<file name=[\'"](.*?)[\'"]>\s*(.*?)\s*</file>',
                content,
                re.DOTALL | re.IGNORECASE,
            )
            for match in matches:
                code = match.group(2).strip()
                code = re.sub(r"^```[a-zA-Z]*\n?", "", code).rstrip("`").strip()
                filename = match.group(1).strip()
                if filename.startswith("src/") or filename == "requirements.txt":
                    new_files[filename] = code

            if not new_files:
                raise ValueError("Failed XML formatting.")

            file_summary = "\n".join(
                f"  {fname} ({len(code)} chars)" for fname, code in new_files.items()
            )
            _diag(ws, "code_writer", f"Source files:\n{file_summary}")

            # Keep the frozen tests; replace only src/ and requirements.txt
            test_files = {k: v for k, v in manifest.items() if k.startswith("tests/")}
            return {
                "file_manifest": {**test_files, **new_files},
                "next_node": "static_analyzer",
                "thoughts": _think(
                    ws,
                    "code_writer",
                    f"Wrote {len(new_files)} source files"
                    + (f" — fixing: {errors[:80]}" if errors else ""),
                ),
            }

        except Exception as e:
            last_error = e
            _diag(ws, "code_writer", f"Attempt {attempt}/{max_attempts} failed: {e}")

    return {
        "verification_errors": f"Code Writer Error: {str(last_error)}",
        "next_node": "error_distiller",
        "thoughts": _think(ws, "code_writer", f"ERROR after {max_attempts} attempts: {str(last_error)[:80]}"),
    }


# ---------------------------------------------------------
# NODE: STATIC ANALYZER
# ---------------------------------------------------------
def static_analyzer(state: SwarmState):
    manifest = state.get("file_manifest", {})
    errors = []

    for filename, code in manifest.items():
        if filename.endswith(".py"):
            try:
                ast.parse(code)
            except SyntaxError as e:
                errors.append(f"SyntaxError in {filename}: {e.msg} at line {e.lineno}")

    if errors:
        ws = state.get("workspace_dir", "")
        _diag(ws, "static_analyzer", "\n".join(errors))
        return {
            "verification_errors": "STATIC ANALYSIS FAILED:\n" + "\n".join(errors),
            "next_node": "error_distiller",
            "thoughts": _think(
                ws,
                "static_analyzer",
                f"Syntax errors in {len(errors)} files",
            ),
        }

    py_count = sum(1 for f in manifest if f.endswith(".py"))
    return {
        "next_node": "deterministic_verifier",
        "thoughts": _think(
            state.get("workspace_dir", ""),
            "static_analyzer",
            f"All {py_count} files parse cleanly",
        ),
    }


# ---------------------------------------------------------
# NODE: DETERMINISTIC VERIFIER
# Two-phase hardened Docker execution:
#   Phase 1 — install deps (network on, no user code runs)
#   Phase 2 — run pytest (network off, user code executes)
# ---------------------------------------------------------
def deterministic_verifier(state: SwarmState):
    workspace = state.get("workspace_dir", ".workspaces/default")
    manifest = state.get("file_manifest", {})
    loops = state.get("loop_count", 0) + 1

    pytest_ini = "[pytest]\ntestpaths = tests\npythonpath = .\n"
    with open(os.path.join(workspace, "pytest.ini"), "w") as f:
        f.write(pytest_ini)

    conftest = (
        "from hypothesis import HealthCheck, settings\n"
        "settings.register_profile(\n"
        "    'default', max_examples=50, deadline=5000,\n"
        "    suppress_health_check=[HealthCheck.too_slow]\n"
        ")\n"
        "settings.load_profile('default')\n"
    )
    with open(os.path.join(workspace, "conftest.py"), "w") as f:
        f.write(conftest)

    for filename, code in manifest.items():
        filepath = os.path.join(workspace, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(code)

    # Clear stale src/ so orphan files from a previous loop don't persist on disk.
    src_dir = os.path.join(workspace, "src")
    if os.path.exists(src_dir):
        shutil.rmtree(src_dir)

    # Clear stale deps so a changed requirements.txt on retry gets a clean install.
    # Pre-create so the container user (running as host UID) can write into it.
    deps_dir = os.path.join(workspace, ".deps")
    if os.path.exists(deps_dir):
        shutil.rmtree(deps_dir)
    os.makedirs(deps_dir)

    host_workspace = resolve_host_path(workspace)

    docker_cfg = CONFIG.get("docker", {})
    IMAGE = docker_cfg.get("image", "python:3.11-slim")
    memory_limit = docker_cfg.get("memory_limit", "512m")
    timeout_install = docker_cfg.get("timeout_install", 90)
    timeout_test = docker_cfg.get("timeout_test", 120)

    # docker run [OPTIONS] IMAGE [COMMAND] — all -e flags must come before the image name.
    # Run as host user so files written to the volume remain owned by that user,
    # allowing shutil.rmtree on retry without permission errors.
    # os.getuid/os.getgid don't exist on Windows; Docker Desktop maps the
    # container user to the host user automatically, so we skip the flag there.
    hardening_flags = [
        "--memory",
        memory_limit,
        "--memory-swap",
        memory_limit,
        "--cpus",
        "1.0",
        "--pids-limit",
        "64",
        "--ulimit",
        "nofile=1024:1024",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "-v",
        f"{host_workspace}:/workspace",
        "-w",
        "/workspace",
        "-e",
        "HOME=/tmp",
    ]
    if hasattr(os, "getuid"):
        hardening_flags.insert(0, f"--user={os.getuid()}:{os.getgid()}")

    install_script = (
        "pip install -q --target /workspace/.deps pytest hypothesis && "
        "if [ -f requirements.txt ]; then "
        "pip install -q --target /workspace/.deps -r requirements.txt; "
        "fi"
    )
    install_cmd = (
        ["docker", "run", "--rm"]
        + hardening_flags
        + ["-e", "PIP_ROOT_USER_ACTION=ignore", "-e", "PIP_DISABLE_PIP_VERSION_CHECK=1"]
        + [IMAGE, "bash", "-c", install_script]
    )

    test_cmd = (
        ["docker", "run", "--rm", "--network", "none"]
        + hardening_flags
        + ["-e", "PYTHONPATH=/workspace/.deps"]
        + [IMAGE, "python", "-m", "pytest", "-p", "no:cacheprovider"]
    )

    try:
        install_res = subprocess.run(
            install_cmd, capture_output=True, text=True, timeout=timeout_install
        )
        if install_res.returncode != 0:
            install_error = f"DEPENDENCY INSTALL FAILED:\nSTDOUT:\n{install_res.stdout}\nSTDERR:\n{install_res.stderr}"
            _diag(workspace, "deterministic_verifier", install_error)
            return {
                "verification_errors": install_error,
                "loop_count": loops,
                "next_node": "error_distiller" if loops < LOOP_CEILING else "FINISH",
                "thoughts": _think(
                    workspace,
                    "deterministic_verifier",
                    f"Loop {loops}: dep install FAILED",
                ),
            }

        res = subprocess.run(
            test_cmd, capture_output=True, text=True, timeout=timeout_test
        )

        if res.returncode == 0:
            # Tests passed — route through error_distiller for semantic contract validation before archiving
            _diag(workspace, "deterministic_verifier", f"pytest output:\n{res.stdout}")
            return {
                "loop_count": loops,
                "verification_errors": "",
                "regression_count": 0,
                "next_node": "error_distiller",
                "thoughts": _think(
                    workspace, "deterministic_verifier", f"Loop {loops}: tests PASSED"
                ),
            }

        # Extract a one-line summary from stderr for the thought log
        stderr_lines = res.stderr.strip().splitlines()
        fail_summary = next(
            (l for l in stderr_lines if "FAILED" in l or "ERROR" in l or "failed" in l),
            stderr_lines[-1] if stderr_lines else "unknown error",
        )
        _diag(
            workspace,
            "deterministic_verifier",
            f"STDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}",
        )
        return {
            "verification_errors": f"STDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}",
            "loop_count": loops,
            "next_node": "error_distiller" if loops < LOOP_CEILING else "FINISH",
            "thoughts": _think(
                workspace,
                "deterministic_verifier",
                f"Loop {loops}: tests FAILED — {fail_summary[:100]}",
            ),
        }

    except subprocess.TimeoutExpired:
        return {
            "verification_errors": f"Execution Error: Test suite timed out ({timeout_test}s). Infinite loop or heavy fuzzing detected.",
            "loop_count": loops,
            "next_node": "error_distiller" if loops < LOOP_CEILING else "FINISH",
            "thoughts": _think(
                workspace,
                "deterministic_verifier",
                f"Loop {loops}: TIMEOUT ({timeout_test}s)",
            ),
        }


# ---------------------------------------------------------
# NODE: ERROR DISTILLER
# Dual-mode node — always sits between the verifier and the archivist.
#
# Mode 1 — Semantic validation (verification_errors is empty):
#   Tests passed; now check whether the implementation actually satisfies
#   the contract rather than merely gaming the test suite.
#   PASS → archivist_node
#   FAIL → code_writer with the semantic gap as the instruction
#
# Mode 2 — Fault classification (verification_errors is set):
#   Classifies the failure and routes:
#     implementation → code_writer
#     tests          → test_writer
#     spec           → architect_node
#
# Shared safety valves:
#   regression_count >= REGRESSION_CEILING → force architect replan
#   replan_count     >= REPLAN_CEILING     → terminate
# ---------------------------------------------------------
def error_distiller(state: SwarmState):
    raw_error = state.get("verification_errors", "")
    graveyard = list(state.get("rollback_graveyard", []))
    regression_count = state.get("regression_count", 0)
    replan_count = state.get("replan_count", 0)
    contract = state.get("interface_contract", "")
    manifest = state.get("file_manifest", {})
    workspace = state.get("workspace_dir", "")

    # ------------------------------------------------------------------
    # MODE 1: semantic contract validation — tests passed, no error set
    # ------------------------------------------------------------------
    if not raw_error:
        impl_context = "\n\n".join(
            f"# {filename}\n{code}"
            for filename, code in manifest.items()
            if filename.startswith("src/") and filename.endswith(".py")
        )

        system = (
            "You are the semantic contract validator in a TDD pipeline.\n"
            "Review the implementation against the contract as an independent check on whether\n"
            "it genuinely implements the specified behavior, not just whether it runs.\n\n"
            "Look specifically for:\n"
            "- Hardcoded outputs: returning a known value instead of computing the result\n"
            "- Special-casing: branching on particular inputs to fake correctness\n"
            "- Missing algorithm: satisfying assertions without implementing the real logic\n"
            "- Missing behavior: a guarantee the contract states that the code silently skips\n\n"
            "Respond with EXACTLY one of these two formats — no other text:\n"
            "PASS\n"
            "FAIL: <one sentence describing the specific violation or loophole>"
        )
        user_prompt = f"Contract:\n{contract}\n\nImplementation:\n{impl_context}"

        try:
            res = get_llm("error_distiller").invoke(
                [SystemMessage(content=system), HumanMessage(content=user_prompt)]
            )
            verdict = res.content.strip()
            _diag(workspace, "error_distiller", f"Semantic verdict:\n{verdict}")
        except Exception as e:
            # Validator unavailable — proceed rather than block the pipeline
            return {
                "verification_errors": f"Semantic validator unavailable: {str(e)}",
                "next_node": "archivist_node",
                "thoughts": _think(
                    workspace,
                    "error_distiller",
                    f"Semantic validator unavailable — {str(e)[:80]}",
                ),
            }

        if verdict.upper().startswith("PASS"):
            return {
                "verification_errors": "",
                "next_node": "archivist_node",
                "thoughts": _think(
                    workspace, "error_distiller", "Semantic validation PASS — archiving"
                ),
            }

        # Semantic loophole found — treat as an implementation fault
        instruction = (
            verdict.split("FAIL:", 1)[-1].strip() if ":" in verdict else verdict
        )
        instruction = (
            f"Implementation passes tests but violates contract: {instruction}"
        )
        graveyard.append(verdict[-500:])
        new_regression_count = regression_count + 1

        if new_regression_count >= REGRESSION_CEILING:
            if replan_count >= REPLAN_CEILING:
                return {
                    "verification_errors": f"Exhausted {REPLAN_CEILING} architect replans without success.",
                    "rollback_graveyard": graveyard,
                    "regression_count": new_regression_count,
                    "replan_count": replan_count,
                    "next_node": "FINISH",
                    "thoughts": _think(
                        workspace,
                        "error_distiller",
                        f"Semantic FAIL #{new_regression_count} — exhausted {REPLAN_CEILING} replans, giving up",
                    ),
                }
            return {
                "verification_errors": "Persistent semantic failures. Architect must revise the spec.",
                "rollback_graveyard": graveyard,
                "regression_count": 0,
                "replan_count": replan_count + 1,
                "next_node": "architect_node",
                "thoughts": _think(
                    workspace,
                    "error_distiller",
                    f"Semantic FAIL #{new_regression_count} — forcing replan {replan_count + 1}/{REPLAN_CEILING}",
                ),
            }

        return {
            "verification_errors": instruction,
            "rollback_graveyard": graveyard,
            "regression_count": new_regression_count,
            "next_node": "code_writer",
            "thoughts": _think(
                workspace,
                "error_distiller",
                f"Semantic FAIL #{new_regression_count} — {instruction[:80]}",
            ),
        }

    # ------------------------------------------------------------------
    # MODE 2: fault classification — an error trace is present
    # ------------------------------------------------------------------
    graveyard.append(raw_error[-500:])
    regression_count += 1
    _diag(
        workspace,
        "error_distiller",
        f"Regression {regression_count}/{REGRESSION_CEILING}, replan {replan_count}/{REPLAN_CEILING}\n"
        f"Error trace (last 500 chars):\n{raw_error[-500:]}",
    )

    if regression_count >= REGRESSION_CEILING:
        if replan_count >= REPLAN_CEILING:
            return {
                "verification_errors": f"Exhausted {REPLAN_CEILING} architect replans without success.",
                "rollback_graveyard": graveyard,
                "regression_count": regression_count,
                "replan_count": replan_count,
                "next_node": "FINISH",
                "thoughts": _think(
                    workspace,
                    "error_distiller",
                    f"Regression #{regression_count} — exhausted {REPLAN_CEILING} replans, giving up",
                ),
            }
        return {
            "verification_errors": "Persistent failures across regressions. Architect must revise the spec.",
            "rollback_graveyard": graveyard,
            "regression_count": 0,
            "replan_count": replan_count + 1,
            "next_node": "architect_node",
            "thoughts": _think(
                workspace,
                "error_distiller",
                f"Regression #{regression_count} — forcing replan {replan_count + 1}/{REPLAN_CEILING}",
            ),
        }

    # Static syntax errors are always implementation faults — skip the LLM call
    if raw_error.startswith("STATIC ANALYSIS FAILED:"):
        fault_type = "syntax"
        instruction = raw_error[:300]
        next_node = "code_writer"
    else:
        system = (
            "You are the fault classifier in a TDD pipeline.\n"
            "A test run failed. Classify where the fault lies and write a one-sentence fix instruction.\n\n"
            "Respond in EXACTLY this format — two lines, no other text:\n"
            "FAULT: implementation|tests|spec\n"
            "INSTRUCTION: <one actionable sentence for the responsible node>\n\n"
            "FAULT TYPE DEFINITIONS:\n"
            "- implementation: code logic is wrong or incomplete; tests and contract are correct\n"
            "  Example: tests expect sorted output but the sort step is missing from the implementation\n"
            "- tests: tests contradict or misrepresent the interface contract\n"
            "  Example: tests import a function name that differs from what the contract specifies\n"
            "- spec: the contract is ambiguous, contradictory, or missing required detail\n"
            "  Example: the contract omits the return type and the tests disagree on what it should be\n\n"
            "Default to 'implementation' when the error is a runtime exception or assertion failure\n"
            "with no evidence of a test or contract problem."
        )

        user_prompt = f"Contract:\n{contract}\n\nFailure Trace:\n{raw_error}"

        try:
            res = get_llm("error_distiller").invoke(
                [SystemMessage(content=system), HumanMessage(content=user_prompt)]
            )
            lines = res.content.strip().splitlines()

            fault_line = next(
                (l for l in lines if l.upper().startswith("FAULT:")),
                "FAULT: implementation",
            )
            instr_line = next(
                (l for l in lines if l.upper().startswith("INSTRUCTION:")),
                f"INSTRUCTION: {raw_error[:200]}",
            )

            fault_type = fault_line.split(":", 1)[1].strip().lower()
            instruction = instr_line.split(":", 1)[1].strip()

            next_node = {"tests": "test_writer", "spec": "architect_node"}.get(
                fault_type, "code_writer"
            )

        except Exception:
            instruction = raw_error[:300]
            next_node = "code_writer"
            fault_type = "unknown"

    _diag(
        workspace,
        "error_distiller",
        f"Fault classification:\n"
        f"  type: {fault_type}\n"
        f"  route: {next_node}\n"
        f"  regression: {regression_count}/{REGRESSION_CEILING}\n"
        f"  replan: {replan_count}/{REPLAN_CEILING}\n"
        f"  instruction: {instruction}",
    )

    new_replan_count = (
        replan_count + 1 if next_node == "architect_node" else replan_count
    )
    if next_node == "architect_node" and replan_count >= REPLAN_CEILING:
        return {
            "verification_errors": f"Exhausted {REPLAN_CEILING} architect replans without success.",
            "rollback_graveyard": graveyard,
            "regression_count": regression_count,
            "replan_count": new_replan_count,
            "next_node": "FINISH",
            "thoughts": _think(
                workspace,
                "error_distiller",
                f"FAULT: {fault_type} — replan ceiling hit, giving up",
            ),
        }

    return {
        "verification_errors": instruction,
        "rollback_graveyard": graveyard,
        "regression_count": regression_count,
        "replan_count": new_replan_count,
        "next_node": next_node,
        "thoughts": _think(
            workspace,
            "error_distiller",
            f"FAULT: {fault_type} → {next_node} — {instruction[:80]}",
        ),
    }


# ---------------------------------------------------------
# NODE: ARCHIVIST
# On success, summarises the plan and contract into a ledger entry,
# appends it to the in-state ledger, and writes it to disk.
# ---------------------------------------------------------
def archivist_node(state: SwarmState):
    workspace = state.get("workspace_dir", "")
    plan = state.get("architectural_plan", "")
    contract = state.get("interface_contract", "")
    ledger = state.get("architecture_ledger", "")

    if not plan and not contract:
        return {
            "next_node": "FINISH",
            "thoughts": _think(
                workspace, "archivist_node", "No plan/contract to archive"
            ),
        }

    system = (
        "You are the archivist node in a TDD pipeline. A coding task just completed successfully.\n"
        "Distil the contract into exactly 3 core constraints — the key invariants or non-obvious\n"
        "decisions that would be easy to get wrong on a second pass.\n\n"
        "Output a single markdown section in this format:\n\n"
        "## Task: <one-line description inferred from the contract>\n"
        "1. <constraint>\n"
        "2. <constraint>\n"
        "3. <constraint>\n\n"
        "Output ONLY the markdown section — no preamble, no explanation."
    )
    prompt = f"Current Ledger:\n{ledger}\n\nContract:\n{contract}"

    try:
        res = get_llm("archivist_node").invoke(
            [SystemMessage(content=system), HumanMessage(content=prompt)]
        )
        updated_ledger = (ledger + "\n\n" + res.content).strip()
        _diag(workspace, "archivist_node", f"Ledger entry:\n{res.content}")
        if workspace:
            with open(os.path.join(workspace, ".architecture.md"), "w") as f:
                f.write(updated_ledger)
        return {
            "architecture_ledger": updated_ledger,
            "next_node": "FINISH",
            "thoughts": _think(
                workspace, "archivist_node", "Archived plan to .architecture.md"
            ),
        }
    except Exception as e:
        import sys
        print(f"[archivist_node] archive failed: {e}", file=sys.stderr)
        return {
            "next_node": "FINISH",
            "thoughts": _think(
                workspace, "archivist_node", f"Archive failed — {str(e)[:80]}"
            ),
        }
