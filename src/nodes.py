import os
import ast
import re
import json
import socket
import shutil
import subprocess
import yaml

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
# from langchain_community.chat_models import ChatOllama # Uncomment if using Ollama

from .state import SwarmState

# Load configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'llm_config.yaml')
with open(CONFIG_PATH, 'r') as f:
    CONFIG = yaml.safe_load(f)

# Factory function to create LLM instances based on config
def create_llm(node_name):
    node_config = CONFIG['nodes'][node_name]
    provider = node_config['provider']
    model = node_config['model']
    temperature = node_config.get('temperature', 0)
    api_key_env_var = node_config.get('api_key_env_var')

    api_key = os.getenv(api_key_env_var) if api_key_env_var else None

    if provider == "google-genai":
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)
    elif provider == "openai":
        return ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)
    elif provider == "anthropic":
        return ChatAnthropic(model=model, temperature=temperature, anthropic_api_key=api_key)
    # elif provider == "ollama":
    #     return ChatOllama(model=model, base_url=os.getenv(api_key_env_var, "http://localhost:11434"))
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# Create LLM instances for each node
llm_architect = create_llm("architect_node")
llm_test_writer = create_llm("test_writer")
llm_contract_verifier = create_llm("contract_verifier")
llm_code_writer = create_llm("code_writer")
llm_error_distiller = create_llm("error_distiller")
llm_archivist = create_llm("archivist_node")

# Load loop limits
LOOP_CEILING = CONFIG['loop_limits']['max_verification_loops']
REGRESSION_CEILING = CONFIG['loop_limits']['max_regression_count']
REPLAN_CEILING = CONFIG['loop_limits']['max_replan_count']
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
            capture_output=True, text=True, timeout=5
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


# ---------------------------------------------------------
# NODE: WORKSPACE LOADER
# ---------------------------------------------------------
def workspace_loader(state: SwarmState):
    workspace = state.get("workspace_dir", ".workspaces/default")
    os.makedirs(workspace, exist_ok=True)
    return {"next_node": "architect_node"}


# ---------------------------------------------------------
# NODE: ARCHITECT
# Always runs. Produces two outputs stored separately in state:
#   <plan>     — architectural decisions for the code writer
#   <contract> — formal interface spec for the test writer (function sigs,
#                behaviours, errors raised, edge cases that MUST be covered)
# On retry from error_distiller (fault=spec), receives prior context
# and revises the contract.
# ---------------------------------------------------------
def architect_node(state: SwarmState):
    prompt = state.get("messages", [])[-1].content
    feedback = state.get("verification_errors", "")
    existing_contract = state.get("interface_contract", "")

    system = (
        "You are a software architect. Output your response in exactly two XML sections:\n\n"
        "<plan>\n"
        "High-level design: module structure, algorithm choices, data flow.\n"
        "</plan>\n\n"
        "<contract>\n"
        "Formal interface specification. For every public function include:\n"
        "- Signature with type hints\n"
        "- Return type and value description\n"
        "- Every exception that MUST be raised and under what condition\n"
        "- Edge cases that tests MUST cover\n"
        "Be precise — this contract is handed directly to a test writer "
        "who will write tests BEFORE any code is written.\n"
        "</contract>"
    )

    user_prompt = prompt
    if feedback and existing_contract:
        user_prompt += (
            f"\n\nPrevious contract:\n{existing_contract}"
            f"\n\nRun failed with spec fault — revise the contract:\n{feedback}"
        )

    try:
        res = llm_architect.invoke([SystemMessage(content=system), HumanMessage(content=user_prompt)])
        content = res.content

        plan_match = re.search(r'<plan>(.*?)</plan>', content, re.DOTALL | re.IGNORECASE)
        contract_match = re.search(r'<contract>(.*?)</contract>', content, re.DOTALL | re.IGNORECASE)

        plan = plan_match.group(1).strip() if plan_match else content
        contract = contract_match.group(1).strip() if contract_match else content

    except Exception as e:
        plan = f"Fallback plan: {str(e)}"
        contract = prompt

    return {
        "architectural_plan": plan,
        "interface_contract": contract,
        "contract_check_count": 0,
        "next_node": "test_writer",
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
        "You are a Python test engineer. Write a pytest test suite based SOLELY on "
        "the interface contract provided. Do NOT invent behaviour beyond what the contract specifies.\n"
        "Output test files wrapped in XML tags:\n"
        "<file name=\"tests/test_main.py\">\n...\n</file>\n"
        "RULES:\n"
        "1. ALL test files go in a tests/ subdirectory named test_*.py.\n"
        "2. Always include a tests/__init__.py file.\n"
        "3. Import from src (e.g. from src.module import thing).\n"
        "4. Use pytest and hypothesis for property-based testing where appropriate.\n"
        "5. Output raw Python inside the XML tags — no markdown fences.\n"
        "6. Only output test files — no source or requirements files."
    )

    user_prompt = f"Interface Contract:\n{contract}"
    if errors:
        user_prompt += f"\n\nPrevious failures — revise tests to match the contract:\n{errors}"

    try:
        response = llm_test_writer.invoke([SystemMessage(content=system), HumanMessage(content=user_prompt)])
        content = response.content if hasattr(response, 'content') else ""

        test_files = {}
        matches = re.finditer(r'<file name=[\'"](.*?)[\'"]>\s*(.*?)\s*</file>', content, re.DOTALL | re.IGNORECASE)
        for match in matches:
            code = match.group(2).strip()
            code = re.sub(r'^```[a-zA-Z]*\n?', '', code).rstrip('`').strip()
            filename = match.group(1).strip()
            if filename.startswith("tests/"):
                test_files[filename] = code

        if not test_files:
            raise ValueError("No test files generated.")

        # Preserve any existing src/ files; replace all test files with new ones
        src_files = {k: v for k, v in manifest.items() if k.startswith("src/") or k == "requirements.txt"}
        return {"file_manifest": {**src_files, **test_files}, "next_node": "contract_verifier"}

    except Exception as e:
        return {"verification_errors": f"Test Writer Error: {str(e)}", "next_node": "error_distiller"}


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
        "You are a contract compliance checker for a TDD pipeline.\n"
        "Given an interface contract and a test suite, determine whether the tests "
        "correctly and completely reflect the contract.\n"
        "Respond with EXACTLY one of:\n"
        "PASS\n"
        "FAIL: <one sentence describing the specific gap or mismatch>"
    )

    user_prompt = f"Contract:\n{contract}\n\nTest Suite:\n{test_context}"

    try:
        res = llm_contract_verifier.invoke([SystemMessage(content=system), HumanMessage(content=user_prompt)])
        verdict = res.content.strip()
    except Exception as e:
        # LLM call failed — surface the error but proceed rather than blocking the pipeline
        return {
            "contract_check_count": check_count,
            "verification_errors": f"Contract verifier unavailable: {str(e)}",
            "next_node": "code_writer",
        }

    if verdict.upper().startswith("PASS"):
        return {"contract_check_count": check_count, "verification_errors": "", "next_node": "code_writer"}

    # Tests don't match the contract
    if check_count >= 3:
        # Force proceed after 3 attempts rather than loop forever
        return {"contract_check_count": check_count, "verification_errors": "", "next_node": "code_writer"}

    return {
        "contract_check_count": check_count,
        "verification_errors": verdict,
        "next_node": "test_writer",
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

    test_context = "\n\n".join(
        f"# {filename}\n{code}"
        for filename, code in manifest.items()
        if filename.startswith("tests/") and filename.endswith(".py")
    )

    system = (
        "You are a master Python programmer working in TDD mode. "
        "A test suite has already been written — your job is to write an implementation that passes it.\n"
        "Output implementation files wrapped in XML tags:\n"
        "<file name=\"src/main.py\">\n...\n</file>\n"
        "RULES:\n"
        "1. All source files go in src/. Always include src/__init__.py.\n"
        "2. Always output a <file name=\"requirements.txt\"> listing every third-party dependency.\n"
        "3. Do NOT output any test files — tests are frozen.\n"
        "4. Output raw Python inside the XML tags — no markdown fences.\n"
        "5. The contract is your primary spec — implement it correctly. Tests verify the contract; they do not replace it."
    )

    user_prompt = prompt
    if contract:
        user_prompt += f"\n\nInterface Contract (primary spec — implement this correctly, do not find loopholes):\n{contract}"
    if plan:
        user_prompt += f"\n\nArchitecture Plan:\n{plan}"
    if test_context:
        user_prompt += f"\n\nTests to pass:\n{test_context}"
    if errors:
        user_prompt += f"\n\nPrevious failures — fix the implementation:\n{errors}"
    if graveyard:
        user_prompt += "\n\nAVOID these failed approaches:\n" + "\n".join(graveyard[-2:])

    try:
        response = llm_code_writer.invoke([SystemMessage(content=system), HumanMessage(content=user_prompt)])
        content = response.content if hasattr(response, 'content') else ""

        new_files = {}
        matches = re.finditer(r'<file name=[\'"](.*?)[\'"]>\s*(.*?)\s*</file>', content, re.DOTALL | re.IGNORECASE)
        for match in matches:
            code = match.group(2).strip()
            code = re.sub(r'^```[a-zA-Z]*\n?', '', code).rstrip('`').strip()
            filename = match.group(1).strip()
            if filename.startswith("src/") or filename == "requirements.txt":
                new_files[filename] = code

        if not new_files:
            raise ValueError("Failed XML formatting.")

        # Keep the frozen tests; replace only src/ and requirements.txt
        test_files = {k: v for k, v in manifest.items() if k.startswith("tests/")}
        return {"file_manifest": {**test_files, **new_files}, "next_node": "static_analyzer"}

    except Exception as e:
        return {"verification_errors": f"Code Writer Error: {str(e)}", "next_node": "error_distiller"}


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
        return {"verification_errors": "STATIC ANALYSIS FAILED:\n" + "\n".join(errors), "next_node": "error_distiller"}

    return {"next_node": "deterministic_verifier"}


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

    # docker run [OPTIONS] IMAGE [COMMAND] — all -e flags must come before the image name.
    # Run as host user so files written to the volume remain owned by that user,
    # allowing shutil.rmtree on retry without permission errors.
    hardening_flags = [
        f"--user={os.getuid()}:{os.getgid()}",
        "--memory", "512m",
        "--memory-swap", "512m",
        "--cpus", "1.0",
        "--pids-limit", "64",
        "--ulimit", "nofile=1024:1024",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "-v", f"{host_workspace}:/workspace",
        "-w", "/workspace",
        "-e", "HOME=/tmp",
    ]

    IMAGE = "python:3.11-slim"

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
        install_res = subprocess.run(install_cmd, capture_output=True, text=True, timeout=90)
        if install_res.returncode != 0:
            install_error = f"DEPENDENCY INSTALL FAILED:\nSTDOUT:\n{install_res.stdout}\nSTDERR:\n{install_res.stderr}"
            return {
                "verification_errors": install_error,
                "loop_count": loops,
                "next_node": "error_distiller" if loops < LOOP_CEILING else "FINISH",
            }

        res = subprocess.run(test_cmd, capture_output=True, text=True, timeout=120)

        if res.returncode == 0:
            # Tests passed — route through error_distiller for semantic contract validation before archiving
            return {"loop_count": loops, "verification_errors": "", "regression_count": 0, "next_node": "error_distiller"}

        return {
            "verification_errors": f"STDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}",
            "loop_count": loops,
            "next_node": "error_distiller" if loops < LOOP_CEILING else "FINISH",
        }

    except subprocess.TimeoutExpired:
        return {
            "verification_errors": "Execution Error: Test suite timed out (120s). Infinite loop or heavy fuzzing detected.",
            "loop_count": loops,
            "next_node": "error_distiller" if loops < LOOP_CEILING else "FINISH",
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
            "You are a semantic contract validator for a TDD code generation pipeline.\n"
            "Given an interface contract and a Python implementation, determine whether "
            "the implementation correctly and completely fulfils the contract — not just "
            "whether it passes specific test cases.\n"
            "Look for: hardcoded outputs, missing edge-case handling, loopholes that satisfy "
            "test assertions without implementing the actual algorithm.\n"
            "Respond with EXACTLY one of:\n"
            "PASS\n"
            "FAIL: <one sentence describing the specific violation or loophole>"
        )
        user_prompt = f"Contract:\n{contract}\n\nImplementation:\n{impl_context}"

        try:
            res = llm_error_distiller.invoke([SystemMessage(content=system), HumanMessage(content=user_prompt)])
            verdict = res.content.strip()
        except Exception as e:
            # Validator unavailable — proceed rather than block the pipeline
            return {"verification_errors": f"Semantic validator unavailable: {str(e)}", "next_node": "archivist_node"}

        if verdict.upper().startswith("PASS"):
            return {"verification_errors": "", "next_node": "archivist_node"}

        # Semantic loophole found — treat as an implementation fault
        instruction = verdict.split("FAIL:", 1)[-1].strip() if ":" in verdict else verdict
        instruction = f"Implementation passes tests but violates contract: {instruction}"
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
                }
            return {
                "verification_errors": "Persistent semantic failures. Architect must revise the spec.",
                "rollback_graveyard": graveyard,
                "regression_count": 0,
                "replan_count": replan_count + 1,
                "next_node": "architect_node",
            }

        return {
            "verification_errors": instruction,
            "rollback_graveyard": graveyard,
            "regression_count": new_regression_count,
            "next_node": "code_writer",
        }

    # ------------------------------------------------------------------
    # MODE 2: fault classification — an error trace is present
    # ------------------------------------------------------------------
    graveyard.append(raw_error[-500:])
    regression_count += 1

    if regression_count >= REGRESSION_CEILING:
        if replan_count >= REPLAN_CEILING:
            return {
                "verification_errors": f"Exhausted {REPLAN_CEILING} architect replans without success.",
                "rollback_graveyard": graveyard,
                "regression_count": regression_count,
                "replan_count": replan_count,
                "next_node": "FINISH",
            }
        return {
            "verification_errors": "Persistent failures across regressions. Architect must revise the spec.",
            "rollback_graveyard": graveyard,
            "regression_count": 0,
            "replan_count": replan_count + 1,
            "next_node": "architect_node",
        }

    system = (
        "You are a fault classifier for a TDD code generation pipeline.\n"
        "Given a contract and a failure trace, classify the fault and provide a fix instruction.\n"
        "Respond in EXACTLY this format (two lines, no extra text):\n"
        "FAULT: implementation|tests|spec\n"
        "INSTRUCTION: <one actionable sentence for the responsible node>\n\n"
        "FAULT TYPES:\n"
        "- implementation: the code does not correctly satisfy what the tests expect\n"
        "- tests: the tests do not correctly reflect the interface contract\n"
        "- spec: the contract itself is ambiguous, incomplete, or incorrect"
    )

    user_prompt = f"Contract:\n{contract}\n\nFailure Trace:\n{raw_error}"

    try:
        res = llm_error_distiller.invoke([SystemMessage(content=system), HumanMessage(content=user_prompt)])
        lines = res.content.strip().splitlines()

        fault_line = next((l for l in lines if l.upper().startswith("FAULT:")), "FAULT: implementation")
        instr_line = next((l for l in lines if l.upper().startswith("INSTRUCTION:")), f"INSTRUCTION: {raw_error[:200]}")

        fault_type = fault_line.split(":", 1)[1].strip().lower()
        instruction = instr_line.split(":", 1)[1].strip()

        next_node = {"tests": "test_writer", "spec": "architect_node"}.get(fault_type, "code_writer")

    except Exception:
        instruction = raw_error[:300]
        next_node = "code_writer"

    new_replan_count = replan_count + 1 if next_node == "architect_node" else replan_count
    if next_node == "architect_node" and new_replan_count > REPLAN_CEILING:
        return {
            "verification_errors": f"Exhausted {REPLAN_CEILING} architect replans without success.",
            "rollback_graveyard": graveyard,
            "regression_count": regression_count,
            "replan_count": new_replan_count,
            "next_node": "FINISH",
        }

    return {
        "verification_errors": instruction,
        "rollback_graveyard": graveyard,
        "regression_count": regression_count,
        "replan_count": new_replan_count,
        "next_node": next_node,
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
        return {"next_node": "FINISH"}

    system = "Summarise the successful architectural plan and contract into 3 core constraints. Output ONLY markdown."
    prompt = f"Current Ledger:\n{ledger}\n\nPlan:\n{plan}\n\nContract:\n{contract}"

    try:
        res = llm_archivist.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        updated_ledger = (ledger + "\n\n" + res.content).strip()
        if workspace:
            with open(os.path.join(workspace, ".architecture.md"), "w") as f:
                f.write(updated_ledger)
        return {"architecture_ledger": updated_ledger, "next_node": "FINISH"}
    except Exception:
        return {"next_node": "FINISH"}



