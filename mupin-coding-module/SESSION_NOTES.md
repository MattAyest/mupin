# Coding Module — Agent Onboarding (v0.2)

> **Purpose:** This document gives a new AI agent everything needed to work on the Coding Module project without reading every source file. Read this first.

---

## 1. What This Project Is (v0.2)

A self-healing Python code-generation microservice. A user posts a natural-language prompt to a FastAPI endpoint. A LangGraph state machine then:

1. `test_designer` — designs a lean pytest + Hypothesis test suite from the prompt.
2. `skeleton_maker` — derives a matching `src/main.py` skeleton from the tests and checks prompt/test/skeleton compatibility.
3. `coder` — replaces the skeleton bodies with real logic so the tests pass.
4. `sandbox_arbiter` — runs ruff, mypy, and pytest inside a hardened Docker container.
5. `prompt_compliance_checker` — verifies the passing implementation actually covers every functional requirement in the user's prompt.

If a node fails, it routes back to the node that can fix it:

- test-side or contract faults → `test_designer`
- implementation/runtime faults → `coder`
- prompt-coverage gaps → `test_designer` with a critique
- infrastructure faults → `FINISH`

The service runs **inside Docker** (Docker-in-Docker) so the arbiter can spawn sibling `python:3.11-slim` containers via the host Docker socket.

**Tech stack:** Python 3.11, LangGraph, LangChain, FastAPI, Docker, pytest + Hypothesis.

> **v0.1 legacy:** the older eight-node pipeline (architect, test_writer, contract_verifier, code_writer, static_analyzer, deterministic_verifier, error_distiller, archivist) is preserved only in git history. Do not refer to it as current.

---

## 2. Repository Layout

```
Coding-Module/
├── Dockerfile              # Orchestrator image (python:3.11-slim + docker CLI)
├── docker-compose.yml      # Mounts docker.sock + .workspaces + llm_config.yaml
├── llm_config.yaml         # PER-NODE LLM routing (providers, models, loop limits)
├── requirements.txt        # Orchestrator deps (langgraph, langchain-*, fastapi...)
├── README.md               # Public docs (GitHub)
├── SESSION_NOTES.md        # THIS FILE — private onboarding guide for agents
├── AGENTS.md               # Short project rules for OpenCode agents
├── .env                    # API keys (not committed)
├── .env.example            # Key template (committed)
└── src/
    ├── __init__.py
    ├── main.py             # Local-dev re-export of the graph app
    ├── api.py              # FastAPI endpoints: POST /task, GET /task/{id}, GET /task/{id}/log
    ├── graph.py            # LangGraph StateGraph wiring (5 nodes)
    ├── state.py            # AgenticState TypedDict
    └── nodes.py            # All node implementations + LLM factory + helpers
```

**Per-task output** lands in `.workspaces/<task_id>/` (host-mounted). Each workspace gets `src/`, `tests/`, `conftest.py`, `pytest.ini`, `requirements.txt`, and `task.log`.

---

## 3. The State Graph (`src/graph.py`)

```
test_designer
      │
      ▼
skeleton_maker
      │
      ▼
coder
      │
      ▼
sandbox_arbiter ───────┐
      │                │ (up to max_sandbox_loops)
      ▼                │
prompt_compliance_checker
      │                │
      ▼                │
FINISH ◄───────────────┘
```

Every node returns a dict update with at least `next_node`. Conditional edges read `state["next_node"]` and map it to either a node name or `END`. Valid `next_node` values are exactly the keys in each node's edge mapping.

---

## 4. State Schema (`src/state.py`)

```python
class AgenticState(TypedDict):
    user_prompt: str                 # original prompt
    workspace_dir: str               # ".workspaces/task_xxxxxxxx"
    python_version: Optional[str]    # orchestrator Python version

    file_manifest: Dict[str, str]    # {"src/main.py": "...", "tests/test_main.py": "..."}

    sandbox_errors: str              # last arbiter error / stdout+stderr
    sandbox_diagnostics: Dict[str, Any]   # structured fault classification

    compliance_status: str           # "PASS" | "FAIL" | ""
    compliance_critique: List[str]   # accumulated missing-feature notes

    sandbox_loop_count: int          # incremented each arbiter run
    compliance_loop_count: int       # incremented each compliance retry
    contract_loop_count: int         # incremented each contract retry
    contract_critique: List[str]     # accumulated contract mismatch notes
    contract_exhausted: bool         # True if contract loop ceiling was reached

    next_node: str                   # routing signal

    # Reducers — LangGraph appends each update across the pipeline.
    thoughts: Annotated[List[str], add]
    node_history: Annotated[List[Dict[str, Any]], add]
    llm_usage: Annotated[List[Dict[str, Any]], add]
    docker_runs: Annotated[List[Dict[str, Any]], add]
    classifier_history: Annotated[List[Dict[str, Any]], add]
```

`thoughts` uses `Annotated[List[str], add]` so each node's `_think()` output is appended rather than replaced.

---

## 5. Node-by-Node Reference (`src/nodes.py`)

### LLM Factory — `_build_llm` / `get_llm`

Every node calls `get_llm(node_name)` to get a cached LangChain chat client. Config comes from `llm_config.yaml`. Heavy provider imports are deferred. `get_llm` is `lru_cache`d.

Supported providers: `ollama-cloud`, `ollama` (local), `google-genai`, `openai`, `anthropic`, `openai-compatible`.

`validate_config()` in `api.py` eagerly tests all configured nodes at startup.

### Logging Helpers — `_think` / `_diag`

```python
def _think(workspace, node, message) -> list[str]:
    # Writes "[HH:MM:SS] [node] message" to task.log and returns it for the state reducer

def _diag(workspace, node, detail) -> None:
    # Writes an indented multi-line block to task.log only (not API response)
```

Both use `encoding="utf-8"`.

### `setup_workspace`

Writes `conftest.py`, `pytest.ini`, and `requirements.txt` **only if they do not already exist**. This is critical: retry loops must not wipe third-party dependencies added by the `coder` node.

Base sandbox deps (`SANDBOX_DEPS`): `pytest`, `hypothesis`, `ruff`, `mypy`. No `radon` in v0.2.

### `test_designer`

Outputs exactly:

```
<file name="tests/test_main.py">...</file>
<file name="tests/__init__.py">...</file>
```

Rules it follows:
- Uses Hypothesis `@given` for invariants and randomized domains.
- Constrains strategies at the source; never `assume()`.
- Uses `pytest.raises` for raise-rule cases.
- Imports the implementation from `src.main`.
- Every `@given` test has `@settings(max_examples=50)`.
- Generates only a lean baseline covering the prompt's explicit requirements.
- Does not output `src/main.py`, `requirements.txt`, or any skeleton.

On a contract or compliance retry it receives `contract_critique`, `compliance_critique`, and `sandbox_errors` and revises.
Routes to `skeleton_maker` on success, `FINISH` on error.

### `skeleton_maker`

Receives the current `tests/test_main.py` and produces:

```
<file name="src/main.py">...</file>
<file name="src/__init__.py">...</file>
<contract_verdict>{"compatible": true|false, "critique": ["..."]}</contract_verdict>
```

Rules it follows:
- Derives signatures and type hints directly from the tests.
- Bodies are only `pass` or `raise NotImplementedError`.
- Checks whether the tests are compatible with the prompt.
- Routes to `coder` if compatible.
- Routes back to `test_designer` with `contract_critique` if incompatible (up to `MAX_CONTRACT_LOOPS`).
- After the contract loop ceiling it proceeds to `coder` with `contract_exhausted=True`.

### `coder`

Given the frozen `tests/test_main.py` and skeleton `src/main.py`, replaces every `pass` body with correct logic.

- Only modifies function/class bodies.
- Preserves all `tests/*` files and `src/__init__.py`.
- If it emits `requirements.txt`, the node **merges** it with the base `SANDBOX_DEPS` so pytest/hypothesis/ruff/mypy are never lost.
- Routes to `sandbox_arbiter` on success, `FINISH` on error.

### `sandbox_arbiter`

Two Docker runs:

1. **Install** (network on): `pip install -q --target /workspace/.deps -r /workspace/requirements.txt`.
2. **Verify** (network off): a single shell block that runs:
   - `python -m ruff format src/main.py` and `tests/test_main.py` (separately, with distinct markers).
   - `python -m ruff check --fix src/main.py` and `tests/test_main.py` (separately).
   - A tiny AST check for tautological `assert x == x` in tests.
   - `python -m mypy --ignore-missing-imports src/main.py`.
   - `python -m pytest tests/test_main.py`.

All tools are invoked as `python -m` with `PYTHONPATH=/workspace/.deps` because `pip install --target` does not put console scripts on `PATH`.

Failure classification:
- `test_fault` (→ `test_designer`): ruff failure on tests, tautology.
- `code_fault` (→ `coder`): ruff failure on src, mypy src failure, pytest failure.
- `infra_fault` (→ `FINISH`): dep install failure, timeout, unexpected crash.

### `prompt_compliance_checker`

Reads `user_prompt`, `src/main.py`, and `tests/test_main.py`. Outputs JSON:

```json
{"compliance_status": "PASS" | "FAIL", "missing_features": ["..."]}
```

- `PASS` → `FINISH`.
- `FAIL` with loops remaining → `test_designer` with the critique appended to `compliance_critique`.
- `FAIL` at ceiling → `FINISH`.

---

## 6. API Layer (`src/api.py`)

In-memory `tasks_db` dict (not persistent across restarts). Endpoints:

- `POST /task` — starts a task, returns `task_id` and `status="running"`.
- `GET /task/{task_id}` — full status including thoughts, node_history, llm_usage, docker_runs.
- `GET /task/{task_id}/log` — plain-text thought log.
- `POST /task/{task_id}/cancel` — cancels an in-flight task.

Tasks run as tracked `asyncio.Task` handles, so cancellation works at node boundaries.

A task is `completed` only when the final node is `prompt_compliance_checker` and `compliance_status == "PASS"`.

---

## 7. Configuration (`llm_config.yaml`)

Per-node provider/model/temperature. Loop limits and Docker settings are also here.

```yaml
loop_limits:
    max_sandbox_loops: 5
    max_compliance_loops: 2

docker:
    image: "python:3.11-slim"
    timeout_install: 90
    timeout_test: 120
    memory_limit: "512m"
```

`docker-compose.yml` mounts it read-only — model changes don't need a rebuild. Changes to `src/` still require `docker compose up --build -d` because source is copied into the image.

---

## 8. Running It

### Production / Docker

```bash
docker compose up --build -d
curl -X POST http://localhost:8000/task -H "Content-Type: application/json" -d '{"prompt": "..."}'
```

### Local dev

```bash
uvicorn src.api:app --reload
```

The verifier still spawns sibling containers against the host Docker socket.

### Benchmarks

```bash
python benchmarks/runner.py
python benchmarks/runner.py --ids fibonacci stack
python benchmarks/runner.py --summary
```

`requests` is now in `requirements.txt` so the runner works inside the venv.

---

## 9. Common Gotchas

- **Do not invoke sandbox tools as bare commands.** Use `python -m` with `PYTHONPATH=/workspace/.deps`.
- **Do not overwrite `requirements.txt` on every `test_designer` run.** `setup_workspace` is idempotent.
- **Always merge base sandbox deps into any coder-provided `requirements.txt`.**
- **Mypy is run only on `src/main.py`.** Tests are too likely to trigger stub/import noise.
- **Radon was removed in v0.2.** It added fragility without enough value.
- **`.architecture.md` is no longer produced.** The archivist node was removed.
- **Sandbox containers must run as the host user, not root.** Use `PUID`/`PGID` (set in `docker-compose.yml` / `.env`); `_host_identity()` reads them and `setup_workspace` chowns the workspace accordingly.
- **Verification container uses `--read-only`** with writable tmpfs for `/tmp` and `/var/tmp`; only `/workspace` is writable.
- **Prompt changes in `src/nodes.py` should be noted in this file** under the latest session section (per `AGENTS.md`).

---

## 10. Prompt Change Log

### 2026-07-03 — system-prompt rewrite across v0.2 nodes (Anthropic best practices)

**Problem:** The system prompts in `src/nodes.py` relied on negative rule lists
("Do NOT...", "never...", "prohibited..."), had no in-context examples, and lacked
XML structure and self-check steps. This contributed to weak or brittle test
generation. For example, `merge_sorted` tests only checked `result == sorted(result)`
(missing black-box coverage of input preservation and completeness), while earlier
`test_architect` tests tried to monkeypatch `list.sort`, which crashes on Python 3.11.
Hard questions like `csv_parse` also hit Hypothesis API mismatches (mixing deprecated
`whitelist/blacklist` with newer `include/exclude` arguments).

**Change:** Rewrote the system prompts for `test_designer`, `skeleton_maker`, `coder`,
and `prompt_compliance_checker` in `coding-module/src/nodes.py`:
- XML-structured sections: `<role>`, `<goal>`, `<inputs>`, `<rules>`, `<examples>`,
  `<output_format>`, `<quality_check>`.
- Positive framing of rules, with explanations for necessary prohibitions.
- Added 2–3 in-context examples per prompt.
- Added a self-check step before finishing.
- Added a rule forbidding mixing of deprecated and current Hypothesis API arguments.
- Kept rules language-agnostic where possible while preserving Python output format.

**Example of the new `test_designer` structure:**

```xml
<role>
You are the Test Designer in a strict test-driven code-generation pipeline.
Your job is to produce a focused baseline test suite in tests/test_main.py.
</role>

<goal>
Write tests that:
1. Cover every explicit functional requirement in the user prompt with at least one test.
2. Exercise invariants and randomized domains using Hypothesis @given where appropriate.
3. Use pytest.raises for raise-rule cases.
4. Output only tests/test_main.py and tests/__init__.py.
</goal>

<rules>
- Use @settings(max_examples=50) on every @given test.
- Constrain Hypothesis strategies at the source; never use assume().
- Import the implementation from src.main.
- Write assertions directly: assert x == y, not assert (x) == (y).
- Use black-box property tests only. Do not monkeypatch, mock, or subclass
  built-in types or methods to enforce implementation details.
- Do not use pytest fixtures inside @given tests.
- For expression evaluators, calculators, parsers, or similar tasks, generate raw
  expression strings and compute expected values using the target language's
  standard evaluator in a safe scope.
- Do not output src/main.py, requirements.txt, or any skeleton.
</rules>
```

Full prompts are in `coding-module/src/nodes.py`.

**Verification planned:** Run the medium suite first, then the full 12-question suite
three times. Compare pass rate, runtime, and test quality against the partial
full-12-r1 baseline and the old `test_architect` runs.

**Files changed:** `coding-module/src/nodes.py`, `coding-module/SESSION_NOTES.md`.

### 2026-07-01 — anti-lazy test contract + resilience tuning

- Updated `test_designer` system prompt in `src/nodes.py`:
  replaced the restrictive "concrete boundary tests only when explicitly mentioned" rule with
  a lean baseline principle: tests must cover every explicit prompt requirement but should not
  generate adversarial or edge-case tests beyond what the prompt states.
  This reduces first-pass generation time on hard questions.
- Updated `prompt_compliance_checker` system prompt: style/performance remain out of scope, but
  standard named constructs (graph, parser, heap, queue, evaluator) must be structurally sound
  and free of obvious anti-patterns that would fail under normal adversarial use.
- Added per-node `timeout_seconds` and `max_attempts` to `llm_config.yaml` (defaults picked up
  by `_invoke_with_retry`). Reduced default per-attempt timeout from 900s to 600s and tightened
  the third retry backoff to 10s.
- Added `deadline_seconds` plumbing in `src/api.py` so the server task deadline propagates
  to `asyncio.wait_for` and is reported correctly on timeout.
- Added per-question `timeout_seconds` support to `benchmarks/runner.py` and `questions.json`,
  giving `csv_parse`, `graph`, and `evaluator` 30-40 min budgets while keeping the default
  runner timeout at 20 min. Added failure-mode bucketing to diagnostic summary.
- Considered (and reverted) a `test_designer` prompt edit that would have required tests to
  use only `st.composite` from `hypothesis.strategies` and prohibited importing the top-level
  `composite` decorator from `hypothesis`. The goal was to prevent `ImportError: cannot import name 'composite'`
  failures seen on strategy-heavy questions like `csv_parse`. Reverted because it is a prompt change
  the user did not request; kept as a note in case future runs continue to show this failure mode.



### 2026-07-03 — split `test_architect` into `test_designer` + `skeleton_maker`

**Problem identified:** The `test_architect` node was overloaded. Its prompt mixed test design,
skeleton derivation, and adversarial reasoning in one LLM call. Hard questions frequently hit
runner timeouts, and the generated tests sometimes contained internal inconsistencies (e.g.,
`csv_parse` blank-line expectations contradicted the expected output).

**Change implemented:** Replaced `test_architect` with two focused nodes:
1. `test_designer` — writes only a lean baseline `tests/test_main.py` from the prompt.
2. `skeleton_maker` — derives the `src/main.py` skeleton from those tests and emits a
   `<contract_verdict>` declaring whether the tests are compatible with the prompt.

**Routing:**
- `test_designer` → `skeleton_maker`.
- If `skeleton_maker` reports an incompatible contract, it routes back to `test_designer` with a
  critique (up to `MAX_CONTRACT_LOOPS = 2`). After the ceiling it proceeds to `coder` with
  `contract_exhausted=True`.
- Sandbox test-side faults and `prompt_compliance_checker` failures route back to `test_designer`.

**Files changed:** `src/nodes.py`, `src/graph.py`, `src/state.py`, `llm_config.yaml`, `README.md`.

**No hardening loop was added.** Adversarial/edge-case test generation remains out of scope for
now; the focus is on reducing first-pass generation time and improving test/skeleton consistency.

### 2026-06-30 — calculator expression-generation guidance

- Updated `test_designer` user prompt in `src/nodes.py` with language-specific guidance:
  for expression evaluators / calculators / parsers, generate raw expression strings and
  compute expected values using the target language's standard evaluator (e.g., Python's
  `eval()` with a safe scope). Avoid hand-written AST renderers or string-composition
  logic that must preserve precedence; let the language parser be the source of truth.
  This prevents renderer/associativity bugs where the generated test's AST and rendered
  string evaluated differently.

### 2026-06-30 — calculator sub-expression boundary fix

- Updated `test_designer` system prompt in `src/nodes.py` with the rule:
  "Preserve sub-expression boundaries: parenthesize any fragment inserted into a larger expression."
  This prevents generated Hypothesis strategies from accidentally changing the intended
  value when composing recursive expressions (e.g., division-by-zero denominators that
  evaluated to non-zero due to missing parentheses).

### 2026-07-01 — Ollama Cloud host fix

- Changed `OLLAMA_CLOUD_HOST` in `src/nodes.py` from `https://ollama.com` to
  `https://api.ollama.com` because `api.ollama.cloud`/`ollama.com` were not resolving on
  the local network and causing `ReadTimeout` failures for `test_designer`.
- **Note:** a later LiteLLM proxy integration was built and then reverted; the orchestrator
  currently calls Ollama Cloud directly again.

### 2026-07-01 — transient LLM infrastructure fault retry

- Added per-node transient LLM fault retry logic in `src/nodes.py`.
  `_invoke_with_retry` now tags the final `LLMUnavailableError` as transient or permanent
  based on error type/status codes (timeouts, 5xx/524, connection errors are transient;
  4xx/auth/config errors are permanent). Nodes catch `LLMUnavailableError` and, for
  transient faults, return `next_node: <self>` up to `infra_max_retries_per_node` times
  with configurable backoff. When exhausted, they set `llm_infra_exhausted=True` and route
  to `FINISH`.
- Added `llm_infra_retries` and `llm_infra_exhausted` to `AgenticState` in `src/state.py`.
- Added `resilience` block to `llm_config.yaml` with `infra_max_retries_per_node: 5` and
  backoff `[10, 30, 60, 120, 120]`.
- Updated `src/api.py` to emit `status="infra_exhausted"` when `llm_infra_exhausted` is set,
  so Ollama-induced failures are not scored as code failures.
- Updated `benchmarks/runner.py` diagnostics to bucket `infra_exhausted` separately.
- Permanent LLM/config errors still surface as `status="failed"`.

### 2026-07-02 — Ollama Cloud non-streaming timeout hypothesis

- Investigated intermittent Ollama Cloud failures (`ReadTimeout`, `RemoteProtocolError`) affecting heavy questions (`csv_parse`, `token_bucket`, `rle`, etc.).
- Root-cause hypothesis: `src/nodes.py` configures `ChatOllama` with `streaming=False` and `disable_streaming=True`, holding a single long HTTP request open to `api.ollama.com`. When `kimi-k2.7-code:cloud` needs 60–180+ s to emit 14K–20K tokens, the client-side read timeout expires before the response finishes.
- External evidence:
  - Ollama issue #3995 reports identical behavior for large models with `stream=false`; streaming resolved it.
  - Project `berthmc/presentations` fixed 300 s false timeouts by switching Ollama chat to `stream:true` so the HTTP read timeout resets between chunks.
- Proposed experiment: enable streaming for the `ollama-cloud` provider (`streaming=True`, drop `disable_streaming=True`) and re-run heavy benchmark questions. Fallback: raise per-attempt read timeout or use a streaming-aware HTTP client.

### 2026-07-04 — streaming liveness watchdog for Ollama Cloud stalls

- Follow-up to the 2026-07-02 streaming hypothesis.  Streaming *was* enabled, but
  the 3x benchmark run (`bench3x.log`, runs starting 19:05, 20:06, 20:50) and the
  per-LLM-call metrics (`benchmarks/metrics.jsonl`) showed the deeper problem:
  intermittent `kimi-k2.7-code:cloud` stalls emit **zero tokens for ~900s** before
  the per-attempt wall-clock cap kills the call.  Successful calls for the same
  nodes (`test_designer` p90=96s max=458s; `coder` p90=124s max=561s) stream
  continuously, so the ~900s timeouts are mid-stream hangs, not slow generations.
- Worst case observed: `evaluator` run 060105 burned 3 consecutive 900s
  `test_designer` timeouts (2700s) before the 4th attempt succeeded in 59s.
  The retry stack (`_invoke_with_retry` 3 attempts × `_handle_llm_unavailable`
  `INFRA_MAX_RETRIES=5`) multiplied one bad call into ~30+ min of dead time.
- Fix: added a **streaming liveness watchdog** in `_invoke_with_retry`
  (`src/nodes.py`).  The worker thread consumes `llm.stream(messages)` chunk-
  by-chunk into a `queue.Queue`; the main thread pops with a per-chunk idle
  timeout (`stream_idle_seconds`, default 120s, tunable via `llm_config.yaml`
  `resilience.stream_idle_seconds` or per-node `stream_idle_seconds`).  A hung
  stream now aborts in ~120s instead of ~900s; a slow-but-progressing stream
  still runs up to the total `timeout_seconds` wall-clock budget.  Non-streaming
  providers use `llm.invoke()` and are bounded by the total wall-clock only.
- Added `resilience` block to `llm_config.yaml` with `stream_idle_seconds: 120`
  made explicit (previously only code defaults).
- Verified with an isolated mock-LLM test: ok-stream completes; hang-at-start
  and hang-mid both abort at the idle timeout (3 attempts → ~13s with backoff
  instead of ~2700s); provider errors surface as `error` not `wallclock_timeout`.
- Did **not** change the malformed-file `FINISH` behaviour in `test_designer` /
  `skeleton_maker` (separate reliability issue causing the run-2 `calculator`
  and `largest_prime_digit_sum` failures — see open issues).  Left for a
  follow-up commit to keep this change focused and revertable.

### 2026-07-01 — self-loop conditional edge fix

- Fixed `KeyError('test_architect')` crash that killed tasks during infra retry.
  The nodes return `next_node: <self>` on transient LLM faults, but the conditional edge
  mappings in `src/graph.py` did not include those self-loop targets, so LangGraph raised
  `KeyError` and the runner recorded `status="failed" error='test_architect'`.
- Updated `src/graph.py` to add `test_architect`, `coder`, and `prompt_compliance_checker`
  as valid self-loop targets in their respective conditional edge mappings.
  (After the 2026-07-03 split, `test_designer` and `skeleton_maker` also need self-loop targets.)
- Updated `src/api.py` to propagate `llm_infra_exhausted` and `llm_infra_retries` from the
  graph stream into `tasks_db` / `TaskStatusResponse`, so the API correctly reports
  `infra_exhausted` instead of `failed`.
- Verified the fix by re-running the 12-question benchmark (`run_20260701_215012` / label
  `run_12_002`): the crash disappeared and the run completed all 12 questions.
  Result: **9/12 passed**. Remaining failures:
  - `merge_sorted`: `test_architect` hallucinated Hypothesis API `st.lists(..., sorted=True)`;
    the generated tests fail collection and the sandbox routes to `coder` as a code fault.
    (This failure mode now routes to `test_designer`.)
  - `calculator`: runner per-question timeout (20 min) exceeded.
  - `evaluator`: runner per-question timeout (40 min) exceeded.

### 2026-07-03 — SESSION_NOTES cleanup

- Updated the v0.2 architecture overview and state schema to reflect the
  `test_designer`/`skeleton_maker` split.
- Replaced remaining `test_architect` references in the onboarding sections
  with `test_designer` and `skeleton_maker`.
- Removed the detailed LiteLLM proxy subsections because that integration was
  reverted and is no longer part of the active architecture.

### v0.2 refactor

- Rewrote `test_designer` (formerly `test_architect`), `coder`, `sandbox_arbiter`, `prompt_compliance_checker` system prompts.
- Dropped `contract_verifier`, `error_distiller`, `archivist_node`, `static_analyzer`, `deterministic_verifier`, `architect_node` prompts.
- Simplified prompts to produce a single-file contract: `src/main.py` + `tests/test_main.py`.
- Sandbox arbiter now uses `python -m ruff/mypy/pytest` instead of bare commands.
- Hardened sandbox: test containers run as host `PUID`/`PGID`, verification container uses `--read-only` + tmpfs, caches isolated to `/tmp`.

---

## 11. Language Profile Refactor (Step 1 Implemented)

### Goal
Make the Coding Module language-agnostic by extracting all language-specific
conventions (file paths, sandbox image, test runner, linter, prompts) into a
pluggable profile system. **Step 1 implements Python only;** no second language is
added yet. The second language remains in planning/session notes.

### Why now
All current hard-coded assumptions are Python-specific:
- File paths: `src/main.py`, `tests/test_main.py`, `src/__init__.py`
- Import path: `from src.main import ...`
- Sandbox image: `python:3.11-slim`
- Tools: `pytest`, `ruff`, `mypy`, `hypothesis`
- Dependency file: `requirements.txt`
- Workspace setup: `conftest.py`, `pytest.ini`
- Prompt examples and rules reference Python syntax

A profile system lets us add a second language later without scattering
language-specific conditionals across `nodes.py`.

### Step 1 design

#### 1. New file: `coding-module/profiles/python.yaml`
Contains the complete Python profile:

```yaml
name: python
display_name: Python 3.11
default: true

sandbox:
  image: python:3.11-slim
  deps_file: requirements.txt
  default_deps:
    - pytest
    - hypothesis
    - ruff
    - mypy
  setup_command: |
    pip install -q --target /workspace/.deps -r /workspace/requirements.txt
  run_command: |
    python -m ruff format {{source_main}}
    python -m ruff format {{test_main}}
    python -m ruff check --fix {{source_main}}
    python -m ruff check --fix {{test_main}}
    python -m mypy --ignore-missing-imports {{source_main}}
    python -m pytest {{test_main}} -p no:cacheprovider --hypothesis-seed=42 --hypothesis-profile=sandbox

files:
  source_dir: src
  test_dir: tests
  source_main: src/main.py
  test_main: tests/test_main.py
  source_init: src/__init__.py
  test_init: tests/__init__.py
  manifest:
    - src/main.py
    - src/__init__.py
    - tests/test_main.py
    - tests/__init__.py
    - requirements.txt

prompts:
  import_path: src.main
  source_file: src/main.py
  test_file: tests/test_main.py
  test_framework: pytest
  property_library: hypothesis
  linter: ruff
  type_checker: mypy
  deps_file: requirements.txt

  test_designer_system: |
    <role>
    You are the Test Designer in a strict test-driven Python code-generation pipeline.
    ... (full prompt) ...
    </role>
    ...

  skeleton_maker_system: |
    ...

  coder_system: |
    ...

  compliance_checker_system: |
    ...
```

#### 2. New file: `coding-module/src/profile.py`
A small loader that:
- Reads `profiles/<name>.yaml`
- Validates required keys
- Exposes profile values as typed attributes
- Falls back to `python` if the requested profile is missing

```python
from pathlib import Path
import yaml

PROFILE_DIR = Path(__file__).with_suffix("").parent.parent / "profiles"

class Profile:
    ...

def get_profile(name: str = "python") -> Profile:
    ...
```

#### 3. Refactor `coding-module/src/nodes.py`
- Replace every hard-coded file path (`src/main.py`, `tests/test_main.py`, etc.)
  with lookups from the active profile.
- Replace the hard-coded sandbox command block with the profile's
  `run_command` template.
- Load system prompts from the profile instead of inline strings.
- Keep node logic (manifest handling, loop routing, verdict parsing) unchanged.

#### 4. Refactor `coding-module/src/api.py`
- Accept optional `language` field in `POST /task` request body.
- Default to `python` if not provided.
- Store `profile_name` in the initial task state.

#### 5. Refactor `coding-module/src/state.py`
- Add `profile_name: str` to `AgenticState`.

### What Step 1 does NOT do
- Does not add a second language profile.
- Does not change the Python sandbox image or tools.
- Does not change Python prompt behavior.

### Step 2 (planned, not implemented)
Choose one second language (e.g., JavaScript/TypeScript or Go) and:
- Create `profiles/<lang>.yaml`.
- Add a small benchmark set for that language.
- Extend `sandbox_arbiter` if the profile's `verify_command` isn't enough on its own.
- Document any language-specific edge cases in this section.

### Verification for Step 1 (completed)
After the refactor:
1. ✅ `python -m py_compile` on all `src/*.py` files.
2. ✅ `docker compose up --build` confirms the API starts.
3. ✅ Smoke test: `fibonacci` task passes end-to-end in ~30s.
4. ✅ `bounded_queue` task passes end-to-end in ~80s with the new profile.

### Actual changes made
- Removed `docker:` runtime section from `coding-module/llm_config.yaml`;
  runtime settings now live in `profiles/python.yaml`.
- Updated `coding-module/Dockerfile` to `COPY profiles/ ./profiles/`.
- Updated `coding-module/docker-compose.yml` to mount `./profiles:/app/profiles:ro`
  so prompt/tooling tweaks do not require a rebuild.
- Added `pytest-timeout` to Python profile default deps and `--timeout=30 -v` to the
  pytest invocation in `verify_command`.
- Set Python profile `cpus: "2.0"` to reduce GIL starvation in concurrency tests.
- Added a concurrency rule to `test_designer_system` in the profile.

### Files changed in Step 1
- `coding-module/src/nodes.py` (refactored)
- `coding-module/src/api.py` (language parameter)
- `coding-module/src/state.py` (profile_name field)
- `coding-module/src/profile.py` (new)
- `coding-module/profiles/python.yaml` (new)
- `coding-module/Dockerfile` (copy profiles)
- `coding-module/docker-compose.yml` (mount profiles)
- `coding-module/llm_config.yaml` (remove docker runtime section)
- `coding-module/SESSION_NOTES.md` (this entry)

## 12. Production-Readiness Benchmark Goal (Initial)

To declare the Coding Module production-ready, the system must demonstrate
consistent performance across four benchmark tiers.

### Tier 1 — Smoke regression
- Run 5 easy/medium questions: `fibonacci`, `stack`, `merge_sorted`, `calculator`,
  `word_frequency`.
- Target: 5/5 pass in under 15 minutes total.
- Run after every prompt, architecture, or dependency change.

### Tier 2 — Custom regression suite
- Run the full custom suite three times.
- Initial target while the suite is 12 questions: ≥ 90% pass rate (32/36) with
  no question failing in all 3 runs.
- The suite should be expanded to **20 questions** as we approach 100% on the
  current 12; the same ≥ 90% target applies to the larger set.
- Latency target: average round ≤ 45 minutes; easy/medium questions ≤ 10
  minutes each.

### Tier 3 — Standard Python benchmark
- Run HumanEval+ (or BigCodeBench if dependency cost is acceptable).
- Initial target for the current model (`kimi-k2.7-code:cloud`):
  - HumanEval+ pass@1 ≥ 60%.
  - BigCodeBench pass@1 ≥ 40% (optional, higher cost).
- Purpose: compare against published baselines and catch distribution shift.

### Tier 4 — Stress / edge-case suite
- Curated 10 questions covering:
  - Concurrency: `bounded_queue`, rate limiters.
  - Parsing: `csv_parse`, expression evaluators, JSON parsers.
  - Library usage: optional DS-1000 or BigCodeBench subset if data science is a
    target domain.
- Target: ≥ 80% pass.

### Quality gates
Beyond pass rate, production-readiness also requires:
1. No sandbox flakiness: each stress/concurrency question passes in 2/3
   consecutive runs.
2. One-loop success: ≥ 70% of tasks pass on the first sandbox loop.
3. Test quality: generated tests catch lazy implementations (e.g. `merge_sorted`
   returning `sorted(a+b)`).
4. Dependency hygiene: no manual `requirements.txt` fixes after generation.
5. Determinism: same prompt yields the same pass/fail outcome ≥ 80% of the time.

### Current status
- Tier 2 custom suite: expanded from 12 to **20 questions**. Smoke tests on the
  8 new HumanEval-inspired questions passed end-to-end.
- Full 20-question v4 reliability run (10×) was started on 2026-07-04 and stopped
  during round 8 to begin the v0.3 backbone refactor. Completed rounds:
  - R1: 19/20 passed (avg 302.7s)
  - R2: 20/20 passed (avg 296.9s)
  - R3: 20/20 passed (avg 300.8s)
  - R4: 19/20 passed (avg 219.4s)
  - R5: 19/20 passed (avg 383.3s)
  - R6: 19/20 passed (avg 293.5s)
  - R7: 20/20 passed (avg 197.1s)
  - Aggregate over completed rounds: **136/140 (97.1%)**
- Tier 1 smoke suite: already passes consistently.
- Tiers 3 and 4: not yet implemented or measured.

### Expanded question set (20 total)
The 8 new questions added to `coding-module/benchmarks/questions.json` are:
- `make_palindrome` (hard) — shortest palindrome prefix extension.
- `decode_cyclic` (medium) — encode/decode string cycling groups of three.
- `triples_sum_to_zero` (medium) — 3-sum decision problem.
- `smallest_change` (medium) — min changes to make list palindromic.
- `closest_integer` (medium) — round string number, ties away from zero.
- `get_row` (medium) — 2D irregular-grid coordinate search.
- `sort_array` (medium) — sort ascending/descending based on first+last parity.
- `largest_prime_digit_sum` (hard) — largest prime digit sum from positive ints.

### Next steps
1. Lock in Tier 2 performance by completing the current 3× full-suite run on
   the expanded 20-question set.
2. Add a HumanEval+ runner to `coding-module/benchmarks/` for Tier 3.
3. Build the 10-question stress suite and measure baseline for Tier 4.
4. Iterate prompts/profiles until all targets are met.

## 13. v0.3 Backbone Architecture Decision

Decided to evolve the Coding Module into a true Mupin backbone by introducing a
shared job infrastructure service.

### Module boundaries

- `mupin-api-backbone/` — shared job submission, queue, persistence, and dispatch.
  - Inter-module API only (not a public consumer API).
  - Direct submission accepted during initial setup/testing.
  - Owns Redis (ARQ), Postgres, and the REST API.
  - Writes all job state and results; workers post progress/finalize via internal endpoints.
- `mupin-coding-module/` — renamed from `coding-module/`. Becomes a pure worker.
  - Consumes `coding` jobs from the backbone queue.
  - Runs the existing LangGraph pipeline.
  - Returns results via the ARQ result callback and posts terminal status to the backbone.
  - Keeps a dev-only `POST /task` convenience endpoint that proxies to the backbone.

### Implementation choices

| Decision | Choice |
|---|---|
| Queue | ARQ on Redis |
| Persistence | SQLAlchemy async + asyncpg on Postgres |
| Worker concurrency | 4 jobs per coding worker (env `WORKER_MAX_JOBS`) |
| Cancellation | Cooperative (worker checks `cancel_requested` between graph nodes) |
| Worker-to-backbone result path | ARQ result callback + internal finalize endpoint |
| Local deployment | Single root `docker-compose.yml` for backbone + worker |
| Benchmark runner batch size | 20 jobs in flight at a time |
| Authentication | Deferred; noted as future addition |

### Auth future addition

Add API authentication to `mupin-api-backbone` before exposing it beyond internal
Mupin modules. Options to evaluate: API keys, mTLS, or short-lived signed JWTs
issued by the Mupin orchestrator. Record the chosen scheme here once designed.

## 14. Benchmark runner now splits queue wait from processing time

Added `started_at` to the backbone job model so the benchmark runner can
distinguish time spent waiting for a worker slot from actual pipeline work.

### Motivation
The full 20-question suite showed per-question totals of 800–1300s while the
actual LLM + sandbox work was often only 40–60s. The difference was queue wait
caused by 4 concurrent worker slots and a few long-running questions such as
`csv_parse`.

### Changes
- `mupin-api-backbone/src/db.py` — added nullable `started_at` column to `jobs`.
- `mupin-api-backbone/src/jobs.py` — `mark_job_running` sets `started_at`.
- `mupin-api-backbone/src/models.py` — `JobResponse` exposes `started_at`.
- `mupin-api-backbone/src/main.py` — new `POST /internal/jobs/{job_id}/started`
  endpoint; `_job_to_response` includes `started_at`.
- `mupin-coding-module/src/worker.py` — worker calls `_mark_started(job_id)`
  immediately when ARQ invokes `run_job`.
- `mupin-coding-module/benchmarks/runner.py` — computes
  `queue_wait_seconds` and `processing_seconds`, writes them to `results.jsonl`,
  and prints them in the run summary.

### Result schema additions
Each row in `results.jsonl` now contains:
- `started_at` — ISO-8601 timestamp when the worker began execution.
- `queue_wait_seconds` — time from runner submission to worker start.
- `processing_seconds` — time from worker start to completion.
- `elapsed_seconds` — unchanged total time from submission to completion.

### Next steps
- Re-run the full 20-question suite and compare `processing_seconds` across runs
  instead of `elapsed_seconds`.
- Decide whether to reduce queue wait by increasing `WORKER_MAX_JOBS`, adding
  priority queues, or running hard questions in a separate batch.

## 15. Open Issues / Next Steps for v0.3

- Add automated tests for the backbone API contract (submit, poll, cancel, finalize).
- Verify benchmark runner throughput with 20 concurrent jobs through the backbone.
- Implement authentication on `mupin-api-backbone` before exposing beyond internal modules.
- Decide whether to remove the dev `POST /task` proxy once the backbone UI is ready.
- Add observability: worker heartbeat, queue depth metrics, job duration histograms.
- HumanEval+ runner: `PER_Q_TIMEOUT` (1800s) is too short — pipeline median
  wall time per task is ~38 min, max 46 min. 17/164 tasks in the `full164`
  run were cancelled before the worker ever started them (queue saturation at
  `WORKER_MAX_JOBS=6`). Either raise the per-question timeout, raise
  `WORKER_MAX_JOBS`, or run the suite in smaller batches.

## 16. HumanEval+ scorer fix + re-score (2026-07-05)

The `full164` HumanEval+ run reported 0/164 pass@1. Root cause was a scorer
bug, not the model.

### Scorer bug
`benchmarks/humaneval_runner.py:180` wrapped the generated module in a
free-standing `{ ... }` block before exec'ing it as a script, which is a
`SyntaxError` at module level in Python. Every `completed` task failed with
`unexpected output / SyntaxError: invalid syntax` pointing at the first line
of the generated source. Fix: emit `generated` directly (no `{ ... }` wrapper).

### New runner flag
`humaneval_runner.py --rescore [--rescore-run RUN_ID] [--rescore-out FILE]`
re-runs `score_problem` against the existing on-disk
`.workspaces/<job_id>/src/main.py` for every `completed` row, without
re-running the pipeline. Output goes to `humaneval_rescore.jsonl` by default.

### Re-score results (full164)
- Completed-and-scored subset (80): **78/80 = 97.5% pass@1**. The 2 failures
  (HumanEval/116, HumanEval/145) are real assertion failures.
- Timed-out-but-code-on-disk subset (66): **66/66 = 100% pass@1**. These jobs
  finished in the DB (status `completed`, node `prompt_compliance_checker`)
  ~31-46 min after submission, but the runner's `PER_Q_TIMEOUT=1800s` cap
  fired first, so the runner cancelled/abandoned them and never scored them.
- 17 tasks: cancelled before the worker started them (`Task cancelled before
  start`) — no code generated. Caused by queue saturation: 164 tasks submitted
  in batches of 6 against `WORKER_MAX_JOBS=6`.
- 1 task: `exhausted` (sandbox loop limit hit).
- **Combined pass@1: 144/164 = 87.8%** of the full dataset, with 18 tasks
  unscored due to queue saturation (not a correctness failure).

### Files
- `benchmarks/humaneval_runner.py` — scorer fix (line ~180) + `--rescore` mode.
- `benchmarks/humaneval_rescore.jsonl` — re-scored completed jobs.
- `benchmarks/humaneval_rescore_timeouts.jsonl` — re-scored timed-out jobs
  that had `src/main.py` on disk.

## 17. HumanEval+ runner timeout-semantics fix (2026-07-05)

The `full164` run's 83 "timeouts" were not real hangs — the DB showed 63 of
them completed 31–46 min after submission, and 17 were cancelled before the
worker ever started them. Two root causes in the runner:

1. **Per-job clock started at submission, not worker start.** `PER_Q_TIMEOUT`
   (1800s) was measured from the POST `/jobs` return, so a task that queued
   25 min then ran 10 min got killed at the 1800s mark having only had 10 min
   of real compute. With 164 tasks submitted 6-wide against
   `WORKER_MAX_JOBS=6`, later tasks burned their whole budget in the queue.
2. **Shared `total_timeout_hit` event killed all jobs when one hit the total
   cap.** A single job noticing `elapsed_run >= TOTAL_TIMEOUT` set a shared
   event that made every other poll loop abandon its job on the next
   iteration — a whole-system kill switch misfiring as a per-job one.

### Changes to `benchmarks/humaneval_runner.py`
- **Slot-based submission.** Replaced `_submit_all` (submit-all-then-poll)
  with a `ThreadPoolExecutor` where each worker thread loops
  submit → poll → score → next problem. At most `--batch-size` jobs are in
  flight at once, so queue wait is near zero. No more 164-thread fan-out.
- **Per-job deadline from `started_at`.** The per-job timeout clock starts
  when the worker picks the job up (the `started_at` field on the poll
  response), not at submission. A short `QUEUE_GRACE = 120s` covers
  submission acceptance only — with slot-based submission this should never
  fire; if it does, it's a backbone health issue (status `queue_timeout`).
- **Removed the whole-run kill switch.** Deleted `TOTAL_TIMEOUT`,
  `total_timeout_hit`, `cancel_grace_deadline`, and the `--total-timeout`
  CLI flag. Each job has its own deadline; there is no shared event.
- **`PER_Q_TIMEOUT` 1800s → 3600s.** DB showed real pipeline work up to
  46 min/task; 1h gives headroom. Exposed as `--per-q-timeout`.
- **`started_at` parsing** handles the `Z` suffix from the backbone
  (`2026-07-04T17:59:10.016159Z`) for Python <3.11 compatibility.

### New CLI
```
python benchmarks/humaneval_runner.py --per-q-timeout 3600 --batch-size 6
```
`--total-timeout` is gone. The `--rescore` mode (added in §16) is unchanged.

## 18. Benchmark roadmap — next steps after HumanEval+ (2026-07-05)

After the `full164_v2` HumanEval+ re-run completes, the next two benchmark
phases are locked in. Order: Tier 2 re-run first (quick regression check on
the v0.3 backbone), then BigCodeBench Instruct-Hard (the harder standard
benchmark from SESSION_NOTES §12 Tier 3).

### Phase 1 — Re-run Tier 2 (custom 20-question suite) on v0.3 backbone

**Why:** The 97.1% Tier 2 number in §12 was measured on the pre-v0.3
backbone. The refactor changed worker semantics (ARQ queue,
`WORKER_MAX_JOBS`, `started_at`), so this confirms no regression. Closes
out Tier 2 on the new infra.

**Code change:** `benchmarks/runner.py` has the **same timeout bugs** that
were just fixed in `humaneval_runner.py`:
- Per-job clock from submission time (`deadline = q_start + per_q_timeout`).
- Shared `total_timeout_hit` event that cancels all jobs when one hits the
  total cap.
- Submit-all-then-poll (`_submit_all`) instead of slot-based submission.

Port the fix from `humaneval_runner.py`:
- Slot-based `ThreadPoolExecutor` (each worker loops submit→poll→next).
- Per-job deadline from `started_at`.
- Delete `TOTAL_TIMEOUT`, `TOTAL_TIMEOUT_ACTION`, `total_timeout_hit`,
  `--total-timeout`, `--total-timeout-action`.
- Raise default `TIMEOUT` to 3600s, expose as `--per-q-timeout`.
- Keep `--sequential` mode but apply the same `started_at`-based deadline.

**Scoring:** unchanged — `runner.py` checks `files_generated > 0` and
`status == "completed"` (the pipeline's sandbox/compliance is the test).

**Run:** `python benchmarks/runner.py --label tier2_v0.3_r{1,2,3} --batch-size 6 --per-q-timeout 3600`, 3 runs to match the Tier 2 protocol (≥90% pass over 3 runs, no question failing all 3, avg round ≤45 min).

**Estimated:** ~30 min code + ~1h for 3 runs.

### Phase 2 — BigCodeBench Instruct-Hard (148 tasks)

**What:** BigCodeBench is a 1140-task software-engineering benchmark; the
`Instruct-Hard` split is 148 real-world tasks with complex instructions and
diverse library function calls. Harder than HumanEval+ — frontier models
report ~40–60% pass@1 on Instruct-Hard. Target from §12: ≥40% pass@1.

**Dataset:** `pip install bigcodebench`; `get_bigcodebench(split="instruct")`
provides `{task_id, instruction, function_signature, test, ...}`. We use the
Hard subset (148 tasks).

**New runner:** `benchmarks/bigcodebench_runner.py` — clone the fixed
`humaneval_runner.py` structure:
- `load_problems(split="instruct", subset="hard")`.
- `wrap_prompt()` adapts the BigCodeBench instruction + signature to the
  Mupin pipeline's prompt format ("Implement the following as a Python
  module with the exact signature given…").
- `score_problem()` extracts the generated function from `src/main.py`,
  concatenates with BigCodeBench's canonical `test` field, execs in docker,
  checks for `__PASS__`. Same pattern as the (fixed) HumanEval scorer.
- Inherits slot-based submission, `started_at` deadline, `--per-q-timeout`,
  `--rescore`.

**Scoring image:** `python:3.11-slim` lacks the libraries BigCodeBench
tasks require. Build a `bigcodebench-eval` image from `python:3.11-slim` +
`numpy pandas requests beautifulsoup4 lxml` (expand after seeing
`ModuleNotFoundError` failures).

**Run:** `python benchmarks/bigcodebench_runner.py --label bcb_hard --batch-size 6 --per-q-timeout 3600`. Expected ~2–4h wall-clock.

**Estimated:** ~2h code + image build + ~3h run time.

### Execution order
Phase 1 first (quick, closes Tier 2). Phase 2 after Phase 1 completes. The
`full164_v2` HumanEval+ run finishes independently in the background — its
final number gets reported whenever it completes, regardless of these
phases. Do not run Phase 1 concurrently with `full164_v2` (would push 12
concurrent jobs against `WORKER_MAX_JOBS=6` and saturate the worker).

## 19. Legacy: v0.1 Pipeline

The previous system used eight nodes: `workspace_loader` → `architect_node` → `test_writer` ↔ `contract_verifier` → `code_writer` ↔ `static_analyzer` → `deterministic_verifier` → `error_distiller` → `archivist_node` / `FINISH`.

It was over-complicated for the current target reliability and is preserved in git history only. Revert via git if you need to resurrect it.

## 20. Session 2026-07-05 — BigCodeBench Instruct-Hard R1 + R2

### Tier 2 (custom 20-question suite) — partial
- **r1: 20/20 = 100%** ✅
- **r2: 16/20 = 80%** (3 `infra_exhausted` LLM blip: evaluator, word_frequency, min_heap; 1 cancelled: make_palindrome)
- **r3: not run** — stopped to start BigCodeBench Phase 2
- `benchmarks/runner.py` updated: ported slot-based + `started_at`-deadline fix from `humaneval_runner.py`, added `infra_exhausted` to terminal status set, raised `QUEUE_GRACE` 120→600s, renamed `--timeout`→`--per-q-timeout` (default 3600s)

### BigCodeBench Instruct-Hard R1 — 15.5% (23/148)
Run completed. **50% of failures were a disk-space infrastructure problem**, not code quality:
- `exhausted` (disk full → `pip install` failed → sandbox_arbiter gave up): **63 tasks**
- `assertion_failure`: 42 tasks (test-design misalignment + exact-value checks)
- `network_mock_mismatch`: 14 tasks (coder used unmockable import patterns)
- `scorer __future__ bug`: 3 tasks (scorer prepended imports before `from __future__`)

### Fixes applied between R1 and R2
1. **Disk cleanup** — freed 112GB (78GB old workspaces + 34GB docker prune). Host went from 100% → 53% usage.
2. **Scorer `__future__` bug** — `benchmarks/bigcodebench_runner.py`: detect `from __future__ import` at top of generated code, prepend scoring imports *after* them.
3. **Mock-idiom prompt changes** (all in `profiles/python.yaml`):
   - Lifted blanket mock ban in test_designer — now allows `unittest.mock.patch` for external I/O (requests, urllib, open, subprocess). Ban stays on built-in types and the function under test.
   - Added mock-idiom example to test_designer examples.
   - Added coder rules: use `import module` style (not `from module import function`) so calls are patchable; don't call `.raise_for_status()`/`.json()` unless prompt requires them.
4. **Infra:** `docker-compose.yml` (root) — added `benchmarks/` rw mount into worker container. `bigcodebench` installed in worker. Runner executes inside worker via `docker exec`.

### BigCodeBench Instruct-Hard R2 — 29.1% (43/148)
Run completed. Nearly doubled from R1.

**R1 → R2 comparison:**
| | R1 | R2 |
|---|---|---|
| pass@1 | 15.5% | 29.1% |
| exhausted (disk) | 63 | 4 |
| completed | 84 | 139 |
| assertion_failure | 42 | 75 |
| network_mock | 14 | 18 |

**R2 failure breakdown (105 failures):**
- `assertion_failure`: **75 (71%)** — the dominant failure mode
- `network_mock_mismatch`: 18 (17%)
- `pipeline_timeout`: 4 (3%)
- `pipeline_exhausted`: 4 (3%)
- `scorer_issue`: 3 (2%)
- `pipeline_infra_exhausted`: 1 (0%)

### Identified general weaknesses (from benchmark diagnostics, not benchmark-specific optimizations)
These are *general pipeline quality* improvements surfaced by the benchmark, not targeted fixes to score higher. All are internal — zero added user friction.

1. **Wrong interpretation of ambiguous prompts** (~40-50 tasks) — test_designer picks one valid reading of an ambiguous phrase; canonical tests expect a different reading. Pipeline's tests pass, canonical tests fail. *Proposed fix (not yet implemented): multi-interpretation test generation — test_designer writes tests for multiple plausible interpretations; compliance checker resolves which is correct.*

2. **Tests don't reflect true intent** (~75 tasks) — test_designer writes structural tests (isinstance, len, dtype) instead of behavioral tests (values, bounds, monotonicity). Coder optimizes against weak tests, produces right-type-wrong-value code. *Proposed fix (IMPLEMENTED this session): strengthen test_designer goal #1 to require behavioral properties, not just type/shape. See §21.*

3. **Ignores user-provided contracts** (~10-15 tasks) — BigCodeBench prompts include a function signature with specific parameter names/defaults/imports; skeleton_maker derives its own from the tests, which may differ. *Proposed fix (not yet implemented): runner extracts starter code from prompt, passes as `contract_code` in payload; skeleton_maker uses it verbatim if present.*

4. **Gives up on transient sandbox failures** (~4-5 tasks) — `dep_install_failed` is classified as `infra_fault` → immediate FINISH, no retry. *Proposed fix (not yet implemented): progressive retry — (1) retry same install, (2) `pip install --no-deps`, (3) route to coder with "reduce deps" directive, (4) only then FINISH.*

5. **Hard-to-test code** (~5-10 tasks) — coder used `from requests import get` (not mockable) or called `.raise_for_status()` (breaks Mock). *Fix already applied in R2 — coder prompt now requires `import module` style and defensive response handling.*

### Files changed this session
- `benchmarks/runner.py` — ported slot-based + `started_at`-deadline fix, `infra_exhausted` terminal status, `QUEUE_GRACE` 120→600, `--per-q-timeout` rename
- `benchmarks/bigcodebench_runner.py` — new runner; fixed `__future__` import ordering in scorer; `--entrypoint python` override for official image
- `docker-compose.yml` (root) — added `benchmarks/` rw mount into worker
- `profiles/python.yaml` — lifted mock ban (scoped), added mock-idiom example, added coder mock-compatible rules, **strengthened test_designer goal #1 to require behavioral properties** (§21)
- `benchmarks/logs/tier2_v0.3_r*.out` — Tier 2 results
- `benchmarks/logs/bcb_hard.out` / `bcb_hard_r2.out` — BigCodeBench R1/R2 logs
- `benchmarks/bigcodebench_results.jsonl` — R2 results

## 21. Behavioral test generation (test_designer prompt strengthening)

**Isolated change applied 2026-07-05.** Only change in this round — to isolate the effect on the next benchmark run.

**What changed** (all in `profiles/python.yaml`, test_designer_system):
1. **Goal #1** strengthened: "Cover every explicit functional requirement with at least one test that constrains the BEHAVIORAL PROPERTIES of the output — not just type or shape, but actual values, bounds, monotonicity, membership, or input-output relationships. Type-only checks (isinstance, len, dtype) are necessary but never sufficient. Where you can compute an exact expected value reliably, assert it. For complex transformations, assert behavioral properties: bounds, monotonicity, membership, sign, or input-output relationships."
2. **Fetch_table example** fixed: added `assert df.iloc[0]["A"] == "1"` — models the behavioral-property standard (checks actual cell value, not just column name + row count).
3. **Quality_check** updated: "Every test for an explicit requirement verifies a behavioral property (value, bound, monotonicity, membership, or input-output relationship), not just type/shape. If any test only checks isinstance/len/dtype, add a behavioral assertion or remove the test."

**Why behavioral properties, not exact values:** AI is unreliable at computing exact expected values on complex multi-step transformations (it'll get `fibonacci(5) == [0,1,1,2,3]` right but miscompute a filtered+sorted DataFrame). Behavioral properties (bounds, monotonicity, membership) are easier for the AI to reason about correctly, tighter than structural checks, and catch the dominant failure mode (right type, wrong values/direction/order). The existing Hypothesis `@given` infrastructure is already designed for this — the change just raises the bar on what "cover a requirement" means.

**Not yet implemented (waiting to isolate this change's effect):**
- Weakness #1 (multi-interpretation test generation)
- Weakness #3 (contract fidelity in skeleton)
- Weakness #4 (sandbox resilience / retry on dep_install_failed)

**Next step:** Re-run BigCodeBench Instruct-Hard R3 with only this change. Compare R3 vs R2 to measure the effect of behavioral test generation. Do NOT combine with other improvements until this one's effect is measured.
