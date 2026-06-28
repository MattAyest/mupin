import ast
import json
import os
import re
import shutil
import socket
import subprocess
import time
from datetime import datetime
from functools import lru_cache

import yaml
from langchain_core.messages import HumanMessage, SystemMessage

from .state import SwarmState


class LLMUnavailableError(Exception):
    """Raised when an LLM call is retried up to the configured cap and still fails.

    This is an infrastructure fault, not a code/test/spec fault. It is surfaced
    as a terminal failure so the run is not silently scored as a code-quality
    issue and the operator sees a clear message.
    """

    def __init__(self, node_name: str, attempts: int, cause: Exception):
        self.node_name = node_name
        self.attempts = attempts
        self.cause = cause
        super().__init__(
            f"LLM for node '{node_name}' failed after {attempts} attempt(s): {cause}"
        )

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

# Server-side hard wall-clock deadline per task. A backstop that auto-cancels a
# task even if the client died without cancelling it (issue #19). Independent of
# and deliberately more generous than the client-side benchmark --timeout; this
# only catches orphans the loop ceilings haven't already stopped.
SERVER_TASK_DEADLINE = CONFIG.get("server", {}).get("task_deadline_seconds", 3600)


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


def _invoke_with_retry(
    node_name: str,
    messages,
    workspace: str = "",
    max_attempts: int = 3,
    backoff_seconds: float = 2.0,
):
    """Call a node's LLM, retrying on transient errors (network drops, SSL EOF,
    incomplete reads, provider 5xx).

    These are infrastructure faults, not problems with the prompt or the code, so
    retrying at the source keeps them from cascading into the fault classifier as
    a false 'implementation' fault.

    Returns the response content string. If all attempts fail, raises
    LLMUnavailableError so the task fails fast with a clear message.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = get_llm(node_name).invoke(messages)
            return response.content if hasattr(response, "content") else ""
        except Exception as e:
            last_error = e
            if workspace:
                _diag(
                    workspace,
                    node_name,
                    f"LLM attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {e}",
                )
            if attempt < max_attempts:
                time.sleep(backoff_seconds)

    raise LLMUnavailableError(node_name, max_attempts, last_error) from last_error


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
    python_version = state.get("python_version", "3.11")

    system = (
        "You are the systems architect in a TDD code-generation pipeline.\n"
        "Define the public contract and architectural shape; the code writer decides the implementation.\n\n"
        "Output exactly two XML sections:\n\n"
        "<plan>\n"
        "Describe the architectural shape and quality properties (correctness, performance, failure modes).\n"
        "Keep this high-level — mention structure and invariants, not algorithms or data layouts.\n"
        "</plan>\n\n"
        "<contract>\n"
        "A precise behavioral specification. Define what each operation means and what guarantees hold;\n"
        "do not prescribe the internal algorithm, data layout, or implementation steps.\n"
        "For every public function or method, define:\n"
        "  - Exact signature with type hints valid for Python {python_version}\n"
        "  - What it returns and what that value represents\n"
        "  - When it raises, stated as a closed rule (not a list of examples)\n"
        "  - The guarantees that define correctness, with concrete boundary examples\n\n"
        "State conditions as rules. Rules are complete; lists always miss a case.\n"
        "Example of form (not content):\n"
        '  Rule:  "Raises TypeError unless n is an int."\n'
        '  List:  "Raises TypeError for float, str, None, and similar."  ← never write this\n'
        "</contract>"
    ).format(python_version=python_version)

    user_prompt = prompt
    if feedback and existing_contract:
        user_prompt += (
            f"\n\nPrevious contract:\n{existing_contract}"
            f"\n\nRun failed with spec fault — revise the contract:\n{feedback}"
        )

    content = _invoke_with_retry(
        "architect_node",
        [SystemMessage(content=system), HumanMessage(content=user_prompt)],
        state.get("workspace_dir", ""),
    )

    plan_match = re.search(
        r"<plan>(.*?)</plan>", content, re.DOTALL | re.IGNORECASE
    )
    contract_match = re.search(
        r"<contract>(.*?)</contract>", content, re.DOTALL | re.IGNORECASE
    )

    plan = plan_match.group(1).strip() if plan_match else content
    contract = contract_match.group(1).strip() if contract_match else content

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
        # A new contract starts a fresh implementation cycle: drop the now-obsolete
        # tests/code so the writers regenerate against the new spec (rather than
        # minimally patching artifacts built for the old one), and clear the stale
        # error that triggered the replan. The graveyard is intentionally kept so
        # failed approaches are still carried forward as "avoid" guidance.
        "file_manifest": {},
        "verification_errors": "",
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
        "Write tests that prove the contract holds. Never see the implementation.\n\n"
        "QUALITY OVER COUNT. One test per rule; one property test per invariant.\n"
        "Do not enumerate variants of the same behavior (e.g. n=1,2,3).\n\n"
        "Build the suite from these techniques, choosing whichever fits each rule:\n"
        "  - RULE: one test for each behavior, return guarantee, and raise-rule.\n"
        "  - PROPERTY: hypothesis tests for invariants that hold across a domain.\n"
        "  - BOUNDARY: empty, single, zero, maximum, precision limits, ordering forks.\n"
        "  - METAMORPHIC / ROUND-TRIP: relations like decode(encode(x)) == x.\n\n"
        "Derive every expected value from the contract's RULES.\n"
        "If the contract is ambiguous, do not invent stricter behavior — test only what it explicitly states.\n"
        "Keep generated inputs inside the contract's valid success domain; use pytest.raises for raise cases.\n\n"
        "Output each test file wrapped in an XML tag:\n"
        '<file name="tests/test_main.py">\n'
        "# file content here\n"
        "</file>\n\n"
        "RULES:\n"
        "1. The primary test file must be named exactly tests/test_main.py.\n"
        "   (Not tests/test_<problem>.py — the public API is always src.main.)\n"
        "2. Always include a tests/__init__.py (can be empty).\n"
        "3. Import the implementation from src.main.\n"
        "4. Constrain hypothesis strategies at the source; never use assume().\n"
        "5. Add @settings(max_examples=50) to every @given test.\n"
        "6. Output raw Python inside the XML tags — no markdown fences.\n"
        "7. Output ONLY test files.\n\n"
        "HYPOTHESIS COMPOSITE STRATEGIES:\n"
        "If you generate recursive structured data (e.g., arithmetic expressions, trees, nested formats),\n"
        "each recursive helper MUST be its own @st.composite strategy and MUST be called with draw(...).\n"
        "Never pass a plain str/list to draw(); only pass hypothesis Strategy objects.\n"
        "Correct recursive pattern (note nested @st.composite and draw(...) on every recursive call):\n"
        "  @st.composite\n"
        "  def expressions(draw, max_depth=4):\n"
        "      number = st.integers(min_value=1, max_value=99).map(str)\n"
        "      @st.composite\n"
        "      def factor(draw, depth):\n"
        "          if depth <= 0: return draw(number)\n"
        "          choice = draw(st.integers(0, 2))\n"
        "          if choice == 0: return draw(number)\n"
        "          if choice == 1:\n"
        "              inner = draw(expr(depth - 1))\n"
        "              return f'({inner})'\n"
        "          op = draw(st.sampled_from('+-'))\n"
        "          inner = draw(factor(depth - 1))\n"
        "          return f'{op}{inner}'\n"
        "      @st.composite\n"
        "      def term(draw, depth):\n"
        "          left = draw(factor(depth))\n"
        "          n = draw(st.integers(0, 2))\n"
        "          parts = [left]\n"
        "          for _ in range(n):\n"
        "              op = draw(st.sampled_from('*/'))\n"
        "              parts.append(op)\n"
        "              parts.append(draw(factor(depth)))\n"
        "          return ''.join(parts)\n"
        "      @st.composite\n"
        "      def expr(draw, depth):\n"
        "          left = draw(term(depth))\n"
        "          n = draw(st.integers(0, 2))\n"
        "          parts = [left]\n"
        "          for _ in range(n):\n"
        "              op = draw(st.sampled_from('+-'))\n"
        "              parts.append(op)\n"
        "              parts.append(draw(term(depth)))\n"
        "          return ''.join(parts)\n"
        "      return draw(expr(max_depth))\n"
    )

    prior_tests = {
        k: v
        for k, v in manifest.items()
        if k.startswith("tests/") and k.endswith(".py")
    }

    user_prompt = f"Interface Contract:\n{contract}"
    if errors and prior_tests:
        # Revise the existing suite — do NOT regenerate from scratch. Rewriting
        # drops tests that were already correct and makes the suite churn.
        current_tests = "\n\n".join(
            f'<file name="{fname}">\n{code}\n</file>'
            for fname, code in prior_tests.items()
        )
        user_prompt += (
            f"\n\nYour current test suite:\n{current_tests}"
            f"\n\nIt was flagged with:\n{errors}"
            f"\n\nMake the MINIMAL change needed to fix the flagged test(s). Keep every other "
            f"test exactly as it is — do not drop, rename, or rewrite tests that already match "
            f"the contract. Output the complete revised file(s)."
        )
    elif errors:
        user_prompt += (
            f"\n\nPrevious failures — revise tests to match the contract:\n{errors}"
        )

    try:
        content = _invoke_with_retry(
            "test_writer",
            [SystemMessage(content=system), HumanMessage(content=user_prompt)],
            state.get("workspace_dir", ""),
        )

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

    # If an implementation already exists, this is a re-verification after a
    # tests-only revision: re-run the EXISTING code against the new tests instead
    # of regenerating it (regeneration discards a working solution and regresses).
    # Only the first pass, with no code yet, needs the implementation written.
    proceed_node = "static_analyzer" if "src/main.py" in manifest else "code_writer"

    test_context = "\n\n".join(
        f"# {filename}\n{code}"
        for filename, code in manifest.items()
        if filename.startswith("tests/") and filename.endswith(".py")
    )

    system = (
        "You are the contract compliance checker in a TDD pipeline.\n"
        "Tests were written from an interface contract before any implementation existed.\n"
        "Your job: confirm the tests faithfully reflect what the contract states.\n\n"
        "Check these things:\n"
        "  - Do any tests contradict the contract — wrong signature, wrong return, "
        "an exception the contract does not specify, or behavior the contract does not define?\n"
        "  - Does any assertion expect a value that contradicts the contract's stated rules or "
        "guarantees? Derive the correct expected value from the rules (precedence, associativity, "
        "raise-rules) and flag any assertion that disagrees.\n"
        "  - Does any hypothesis strategy generate inputs outside the contract's valid input "
        "domain and then assert success? An input the contract says must raise is not a valid "
        "input for a success assertion — the strategy must exclude it at the source.\n"
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

    ws = state.get("workspace_dir", "")
    verdict = _invoke_with_retry(
        "contract_verifier",
        [SystemMessage(content=system), HumanMessage(content=user_prompt)],
        ws,
    ).strip()
    _diag(ws, "contract_verifier", f"Verdict (attempt {check_count}/3):\n{verdict}")

    if verdict.upper().startswith("PASS"):
        return {
            "contract_check_count": check_count,
            "verification_errors": "",
            "next_node": proceed_node,
            "thoughts": _think(
                ws,
                "contract_verifier",
                "Tests match contract — PASS"
                + (" (re-verifying existing code)" if proceed_node == "static_analyzer" else ""),
            ),
        }

    # Tests don't match the contract
    if check_count >= 3:
        # Force proceed after 3 attempts rather than loop forever
        return {
            "contract_check_count": check_count,
            "verification_errors": "",
            "next_node": proceed_node,
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
    python_version = state.get("python_version", "3.11")

    system = (
        "You are the implementation node in a TDD pipeline.\n"
        "Implement the interface contract correctly. Do not optimise for the tests;\n"
        "correct behaviour makes the tests pass as a consequence.\n\n"
        "The <plan> suggests architecture but is non-binding.\n"
        "The <contract> is the binding specification. If they conflict, follow the contract.\n\n"
        "Output each source file wrapped in an XML tag:\n"
        '<file name="src/main.py">\n'
        "# file content here\n"
        "</file>\n\n"
        "RULES:\n"
        "1. Public implementation lives in src/main.py; tests import from src.main.\n"
        "2. Include src/__init__.py. Additional helper modules may go in src/.\n"
        "3. Use only syntax and types available in Python {python_version}.\n"
        '4. Always output a <file name="requirements.txt"> (empty if no third-party deps).\n'
        "5. Do NOT output test files.\n"
        "6. Output raw Python inside the XML tags — no markdown fences.\n"
        "7. Do not hardcode outputs or special-case inputs — implement the actual logic."
    ).format(python_version=python_version)

    prior_src = {
        k: v
        for k, v in manifest.items()
        if k.startswith("src/") and k.endswith(".py")
    }

    user_prompt = prompt
    if contract:
        user_prompt += f"\n\nInterface Contract (primary spec — implement this correctly, do not find loopholes):\n{contract}"
    if plan:
        user_prompt += f"\n\nArchitecture Plan:\n{plan}"
    if errors and prior_src:
        # Revise the existing implementation — do NOT rewrite from scratch.
        # Regenerating discards working code and regresses behavior that already
        # passed. Give the writer its own prior source to patch minimally.
        current_impl = "\n\n".join(
            f'<file name="{fname}">\n{code}\n</file>'
            for fname, code in prior_src.items()
        )
        user_prompt += (
            f"\n\nYour current implementation (already passing most tests):\n{current_impl}"
            f"\n\nIt failed with:\n{errors}"
            f"\n\nMake the MINIMAL change needed to fix this specific failure. Preserve all "
            f"behavior that already works — do not rewrite or restructure. Output the complete "
            f"revised file(s)."
        )
    elif errors:
        user_prompt += f"\n\nPrevious failures — fix the implementation:\n{errors}"
    if graveyard:
        user_prompt += "\n\nAVOID these failed approaches:\n" + "\n".join(
            graveyard[-2:]
        )

    ws = state.get("workspace_dir", "")

    content = _invoke_with_retry(
        "code_writer",
        [SystemMessage(content=system), HumanMessage(content=user_prompt)],
        ws,
    )

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
        # This is a formatting/content problem, not an LLM outage — route to
        # error_distiller so the loop can recover.
        err = "Code Writer Error: Failed XML formatting."
        return {
            "verification_errors": err,
            "next_node": "error_distiller",
            "thoughts": _think(ws, "code_writer", err),
        }

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

    # Clear stale src/ BEFORE writing the manifest, so orphan files from a previous
    # loop don't persist while the fresh manifest files are written in clean.
    src_dir = os.path.join(workspace, "src")
    if os.path.exists(src_dir):
        shutil.rmtree(src_dir)

    for filename, code in manifest.items():
        filepath = os.path.join(workspace, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(code)

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

        # Extract a one-line summary for the thought log. pytest writes its
        # result to STDOUT (not STDERR), so parse stdout: grab the final banner
        # line ("==== N failed, M passed in 0.59s ===="), then fall back to the
        # first FAILED/ERROR line, then stderr. (#22)
        out_lines = res.stdout.strip().splitlines()
        summary_line = next(
            (
                l.strip().strip("=").strip()
                for l in reversed(out_lines)
                if l.strip().startswith("=")
                and any(w in l for w in ("passed", "failed", "error", "no tests"))
            ),
            "",
        )
        first_fail = next(
            (l.strip() for l in out_lines if l.startswith(("FAILED", "ERROR"))), ""
        )
        stderr_lines = res.stderr.strip().splitlines()
        fail_summary = (
            summary_line
            or first_fail
            or (stderr_lines[-1] if stderr_lines else "unknown error")
        )

        _diag(
            workspace,
            "deterministic_verifier",
            f"STDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}",
        )

        # A suite that fails to COLLECT (import error, bad hypothesis API, syntax
        # in a test, empty suite) is a test-harness defect, not a logic fault.
        # pytest signals this with exit code 2/5 or an "error" summary with no
        # "failed". Tag it so error_distiller routes it straight to test_writer
        # with no LLM classify (mirrors STATIC ANALYSIS FAILED). A MIXED run
        # (both failed and errors) takes the normal path — real assertion
        # failures still need classifying. (#20)
        is_collection_error = res.returncode in (2, 5) or (
            "error" in summary_line.lower() and "failed" not in summary_line.lower()
        )
        error_prefix = "TEST COLLECTION FAILED:\n" if is_collection_error else ""
        verdict_word = "ERRORED (collection)" if is_collection_error else "FAILED"

        return {
            "verification_errors": f"{error_prefix}STDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}",
            "loop_count": loops,
            "next_node": "error_distiller" if loops < LOOP_CEILING else "FINISH",
            "thoughts": _think(
                workspace,
                "deterministic_verifier",
                f"Loop {loops}: tests {verdict_word} — {fail_summary[:100]}",
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

        verdict = _invoke_with_retry(
            "error_distiller",
            [SystemMessage(content=system), HumanMessage(content=user_prompt)],
            workspace,
        ).strip()
        _diag(workspace, "error_distiller", f"Semantic verdict:\n{verdict}")

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
    # Collection failures (pytest failing to import/load test modules) can be
    # caused by implementation bugs just as often as by test-harness bugs,
    # because importing the tests transitively imports the implementation.
    # Route them through the normal classifier so the traceback determines
    # the responsible node, rather than hard-coding them to test_writer.
    # The only deterministic short-circuit kept is STATIC ANALYSIS FAILED,
    # where the error source is unambiguously the implementation.

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
            "A test run failed. Decide whether the fault is in the implementation, the tests,\n"
            "or the contract, then write a one-sentence fix instruction.\n\n"
            "Think step by step in a short REASONING paragraph. Then output exactly:\n"
            "REASONING: <one sentence explaining which evidence in the traceback points to the fault>\n"
            "FAULT: <exactly one of: implementation, tests, spec>\n"
            "INSTRUCTION: <one actionable sentence for the responsible node>\n\n"
            "FAULT TYPES:\n"
            "- implementation: code logic is wrong or incomplete; contract/tests are correct.\n"
            "- tests: tests contradict the contract or are malformed (imports, hypothesis API, syntax).\n"
            "- spec: the contract is ambiguous, contradictory, or missing required detail.\n\n"
            "GUIDELINES:\n"
            "- Default to 'implementation' for runtime exceptions/assertion failures unless the\n"
            "  traceback clearly points to a test or contract problem.\n"
            "- For collection failures (pytest failing to load test modules), inspect the traceback.\n"
            "  If the error originates in src/main.py or its imports, it's 'implementation'.\n"
            "  If it originates in tests/ or from pytest/hypothesis API misuse, it's 'tests'.\n"
            "- If the classifier output is unusable, the system defaults to 'implementation'."
        )

        user_prompt = f"Contract:\n{contract}\n\nFailure Trace:\n{raw_error}"

        VALID_FAULTS = ("implementation", "tests", "spec")

        verdict = _invoke_with_retry(
            "error_distiller",
            [SystemMessage(content=system), HumanMessage(content=user_prompt)],
            workspace,
        ).strip()
        _diag(workspace, "error_distiller", f"Raw classifier response:\n{verdict}")
        lines = verdict.strip().splitlines()

        reasoning_line = next(
            (l for l in lines if l.upper().startswith("REASONING:")), ""
        )
        fault_line = next(
            (l for l in lines if l.upper().startswith("FAULT:")), ""
        )
        instr_line = next(
            (l for l in lines if l.upper().startswith("INSTRUCTION:")), ""
        )

        reasoning = reasoning_line.split(":", 1)[1].strip() if ":" in reasoning_line else ""
        raw_fault = fault_line.split(":", 1)[1].strip().lower() if ":" in fault_line else ""
        instruction = instr_line.split(":", 1)[1].strip() if ":" in instr_line else ""

        # Accept an exact match, or a value wrapped in extra words (exactly one
        # valid word present). An echoed template ('implementation|tests|spec')
        # names all three, so it resolves to None — fall back loudly instead of
        # silently misrouting to the default.
        if raw_fault in VALID_FAULTS:
            fault_type = raw_fault
        else:
            present = [v for v in VALID_FAULTS if re.search(rf"\b{v}\b", raw_fault)]
            fault_type = present[0] if len(present) == 1 else None

        if fault_type is None:
            fault_type = "implementation"
            _diag(
                workspace,
                "error_distiller",
                f"WARNING: unusable fault classification — defaulting to 'implementation'. "
                f"Raw FAULT line: {fault_line!r}",
            )
        if not instruction:
            instruction = raw_error[:200]

        _diag(workspace, "error_distiller", f"Classifier reasoning: {reasoning}")

        next_node = {"tests": "test_writer", "spec": "architect_node"}.get(
            fault_type, "code_writer"
        )

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

    res = _invoke_with_retry(
        "archivist_node",
        [SystemMessage(content=system), HumanMessage(content=prompt)],
        workspace,
    )
    updated_ledger = (ledger + "\n\n" + res).strip()
    _diag(workspace, "archivist_node", f"Ledger entry:\n{res}")
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
