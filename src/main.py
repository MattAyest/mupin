import os
import ast
import re
import json
import socket
import subprocess
from typing import TypedDict, List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Use ChatOpenAI or ChatOllama based on availability
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    pass

# Initialize LLMs (Requires GOOGLE_API_KEY environment variable)
llm_fast = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0) # Fast model
llm_heavy = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0) # Heavy model

class SwarmState(TypedDict):
    messages: List[Any]
    workspace_dir: str
    file_manifest: Dict[str, str]
    next_node: str
    loop_count: int
    regression_count: int
    verification_errors: str
    rollback_graveyard: List[str]
    architectural_plan: str
    architecture_ledger: str

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
        # Find the mount whose Destination is the longest prefix of our path
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
    return {"next_node": "speculative_router"}

# ---------------------------------------------------------
# NODE: SPECULATIVE ROUTER
# ---------------------------------------------------------
def speculative_router(state: SwarmState):
    prompt = state.get("messages", [])[-1].content
    
    # Simple heuristic to decide if we need the architect
    if len(prompt) > 200 or "complex" in prompt.lower() or "architecture" in prompt.lower():
        return {"next_node": "architect_node"}
    
    return {"next_node": "local_synthesizer"}

# ---------------------------------------------------------
# NODE: ARCHITECT
# ---------------------------------------------------------
def architect_node(state: SwarmState):
    prompt = state.get("messages", [])[-1].content
    system = "You are a software architect. Create a brief plan for the following request."
    
    try:
        res = llm_heavy.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        plan = res.content
    except Exception as e:
        plan = f"Fallback plan due to error: {str(e)}"
        
    return {"architectural_plan": plan, "next_node": "environment_node"}

# ---------------------------------------------------------
# NODE: ENVIRONMENT CONFIG
# ---------------------------------------------------------
def environment_node(state: SwarmState):
    plan = state.get("architectural_plan", "")
    system = "List Python dependencies for this plan as a requirements.txt file. Output ONLY valid requirements.txt content."
    
    try:
        res = llm_fast.invoke([SystemMessage(content=system), HumanMessage(content=plan)])
        reqs = res.content.strip()
    except Exception:
        reqs = "pytest\nhypothesis"
        
    manifest = state.get("file_manifest", {})
    if reqs and "```" not in reqs: # rudimentary safety check
        manifest["requirements.txt"] = reqs
        
    return {"file_manifest": manifest, "next_node": "cloud_synthesizer"}

# ---------------------------------------------------------
# NODE: LOCAL SYNTHESIZER
# ---------------------------------------------------------
def local_synthesizer(state: SwarmState):
    # Fallback to cloud if local isn't implemented or complex
    return {"next_node": "cloud_synthesizer"}

# ---------------------------------------------------------
# NODE: CLOUD SYNTHESIZER
# ---------------------------------------------------------
def cloud_synthesizer(state: SwarmState):
    prompt = state.get("messages", [])[-1].content
    plan = state.get("architectural_plan", "")
    errors = state.get("verification_errors", "")
    graveyard = state.get("rollback_graveyard", [])
    
    system = (
        "You are a master Python programmer. Output your code wrapped in XML tags like this:\n"
        "<file name=\"main.py\">\nprint('hello')\n</file>\n"
        "RULES:\n"
        "1. All source files go in a src/ subdirectory. Always include a src/__init__.py.\n"
        "2. ALL test files must be placed under a tests/ subdirectory and named test_*.py.\n"
        "3. Always include a tests/__init__.py file.\n"
        "4. Do not place any source or test files in the root directory.\n"
        "5. In test files, import from src (e.g. from src.module import thing).\n"
        "6. Output raw Python code inside the XML tags — no markdown fences."
    )
    
    if plan:
        prompt += f"\n\nArchitecture Plan to follow:\n{plan}"

    if errors:
        prompt += f"Previous Test Trace:\n{errors}\n\nFix the logic."
    if graveyard:
        prompt += f"AVOID these failed approaches:\n" + "\n".join(graveyard[-2:]) # Show last 2 failures
        
    try:
        response = llm_heavy.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        content = response.content if hasattr(response, 'content') else ""
        
        new_files = {}
        matches = re.finditer(r'<file name=[\'"](.*?)[\'"]>\s*(.*?)\s*</file>', content, re.DOTALL | re.IGNORECASE)
        for match in matches:
            code = match.group(2).strip()
            code = re.sub(r'^```[a-zA-Z]*\n?', '', code).rstrip('`').strip()
            new_files[match.group(1).strip()] = code
            
        if not new_files:
            raise ValueError("Failed XML formatting.")
            
        return {"file_manifest": new_files, "next_node": "static_analyzer"}
    except Exception as e:
        return {"verification_errors": f"Synthesis Error: {str(e)}", "next_node": "error_distiller"}

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
    
    docker_cmd = [
        "docker", "run", "--rm",
        "--memory", "512m", "--cpus", "1.0",
        "-e", "PIP_ROOT_USER_ACTION=ignore",
        "-e", "PIP_DISABLE_PIP_VERSION_CHECK=1",
        "-v", f"{resolve_host_path(workspace)}:/workspace",
        "-w", "/workspace",
        "python:3.11-slim",
        "bash", "-c", "if [ -f requirements.txt ]; then pip install -q -r requirements.txt 2>/dev/null; else pip install -q hypothesis pytest 2>/dev/null; fi && pytest"
    ]
    
    try:
        # Hard 120 second timeout for complex fuzzing matrices
        res = subprocess.run(docker_cmd, capture_output=True, text=True, timeout=120)
        
        if res.returncode == 0:
            return {"loop_count": loops, "verification_errors": "", "regression_count": 0, "next_node": "archivist_node"}
            
        error_output = f"STDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}"
        return {
            "verification_errors": error_output, 
            "loop_count": loops, 
            "next_node": "error_distiller" if loops < 10 else "FINISH"
        }
    except subprocess.TimeoutExpired:
        return {
            "verification_errors": "Execution Error: Test suite timed out (120s limit reached). Infinite loop or heavy fuzzing detected.", 
            "loop_count": loops, 
            "next_node": "error_distiller" if loops < 10 else "FINISH"
        }

# ---------------------------------------------------------
# NODE: ERROR DISTILLER (DIAGNOSTICS & REGRESSION TRACKING)
# ---------------------------------------------------------
def error_distiller(state: SwarmState):
    raw_error = state.get("verification_errors", "")
    graveyard = state.get("rollback_graveyard", [])
    regression_count = state.get("regression_count", 0) + 1
    
    # Track the failure 
    graveyard.append(raw_error[-500:]) 
    
    # If we hit 2 regressions, force the Architect to write a new plan
    if regression_count >= 2:
        return {
            "verification_errors": "Multiple regressions. Algorithmic flaw detected. Rewrite plan.",
            "rollback_graveyard": graveyard,
            "regression_count": 0,
            "next_node": "architect_node"
        }
    
    prompt = [
        SystemMessage(content=(
            "You are a strict Python diagnostic tool. Filter the trace into a direct instruction.\n"
            "RULES:\n"
            "1. SyntaxError: Point out exact missing syntax.\n"
            "2. ModuleNotFoundError: Add missing standard library import.\n"
            "3. Division by Zero/Overflow: Add explicit guard clauses.\n"
            "4. Hypothesis float precision boundary failure (e.g., equilibrium tests): Instruct system to loosen `pytest.approx()` to `abs=1e-4` or scale dynamically based on mass/inputs."
        )), 
        HumanMessage(content=f"Error Trace:\n{raw_error}")
    ]
    
    try:
        res = llm_fast.invoke(prompt)
        brief = res.content
    except Exception:
        brief = raw_error[:300]
        
    return {
        "verification_errors": brief, 
        "rollback_graveyard": graveyard,
        "regression_count": regression_count,
        "next_node": "cloud_synthesizer"
    }

# ---------------------------------------------------------
# NODE: ARCHIVIST (MEMORY PERSISTENCE)
# ---------------------------------------------------------
def archivist_node(state: SwarmState):
    workspace = state.get("workspace_dir", "")
    plan = state.get("architectural_plan", "")
    ledger = state.get("architecture_ledger", "")
    
    system = "Summarize the successful architectural plan into 3 core constraints. Output ONLY markdown."
    prompt = f"Current Ledger:\n{ledger}\n\nSuccessful Plan:\n{plan}"
    
    try:
        res = llm_fast.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
        new_ledger = res.content
        
        with open(os.path.join(workspace, ".architecture.md"), "w") as f:
            f.write(new_ledger)
    except Exception:
        pass
        
    return {"next_node": "FINISH"}

# ---------------------------------------------------------
# GRAPH COMPILATION
# ---------------------------------------------------------
workflow = StateGraph(SwarmState)

workflow.add_node("workspace_loader", workspace_loader)
workflow.add_node("speculative_router", speculative_router)
workflow.add_node("architect_node", architect_node)
workflow.add_node("environment_node", environment_node)
workflow.add_node("local_synthesizer", local_synthesizer)
workflow.add_node("cloud_synthesizer", cloud_synthesizer)
workflow.add_node("static_analyzer", static_analyzer)
workflow.add_node("deterministic_verifier", deterministic_verifier)
workflow.add_node("error_distiller", error_distiller)
workflow.add_node("archivist_node", archivist_node)

workflow.set_entry_point("workspace_loader")

workflow.add_conditional_edges("workspace_loader", lambda x: x["next_node"], {"speculative_router": "speculative_router"})
workflow.add_conditional_edges("speculative_router", lambda x: x["next_node"], {"local_synthesizer": "local_synthesizer", "architect_node": "architect_node"})
workflow.add_conditional_edges("architect_node", lambda x: x["next_node"], {"environment_node": "environment_node"})
workflow.add_conditional_edges("environment_node", lambda x: x["next_node"], {"cloud_synthesizer": "cloud_synthesizer"})
workflow.add_conditional_edges("local_synthesizer", lambda x: x["next_node"], {"static_analyzer": "static_analyzer", "cloud_synthesizer": "cloud_synthesizer"})
workflow.add_conditional_edges("cloud_synthesizer", lambda x: x["next_node"], {"static_analyzer": "static_analyzer", "error_distiller": "error_distiller"})
workflow.add_conditional_edges("static_analyzer", lambda x: x["next_node"], {"deterministic_verifier": "deterministic_verifier", "error_distiller": "error_distiller"})
workflow.add_conditional_edges("deterministic_verifier", lambda x: x["next_node"], {"error_distiller": "error_distiller", "archivist_node": "archivist_node", "FINISH": END})
workflow.add_conditional_edges("error_distiller", lambda x: x["next_node"], {"cloud_synthesizer": "cloud_synthesizer", "architect_node": "architect_node", "FINISH": END})
workflow.add_conditional_edges("archivist_node", lambda x: x["next_node"], {"FINISH": END})

app = workflow.compile()
