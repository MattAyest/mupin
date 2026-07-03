# Coding Module — Agent Onboarding (v0.2)

> **Purpose:** This document gives a new AI agent everything needed to work on the Coding Module project without reading every source file. Read this first.

---

## 1. What This Project Is (v0.2)

A self-healing Python code-generation microservice. A user posts a natural-language prompt to a FastAPI endpoint. A LangGraph state machine then:

1. `test_architect` — designs a pytest + Hypothesis test suite and a matching `src/main.py` skeleton from the prompt.
2. `coder` — replaces the skeleton bodies with real logic so the tests pass.
3. `sandbox_arbiter` — runs ruff, mypy, and pytest inside a hardened Docker container.
4. `prompt_compliance_checker` — verifies the passing implementation actually covers every functional requirement in the user's prompt.

If a node fails, it routes back to the node that can fix it:

- test-side faults → `test_architect`
- implementation/runtime faults → `coder`
- prompt-coverage gaps → `test_architect` with a critique
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
    ├── graph.py            # LangGraph StateGraph wiring (4 nodes)
    ├── state.py            # AgenticState TypedDict
    └── nodes.py            # All node implementations + LLM factory + helpers
```

**Per-task output** lands in `.workspaces/<task_id>/` (host-mounted). Each workspace gets `src/`, `tests/`, `conftest.py`, `pytest.ini`, `requirements.txt`, and `task.log`.

---

## 3. The State Graph (`src/graph.py`)

```
test_architect
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

### `test_architect`

Outputs exactly:

```
<file name="src/main.py">...</file>
<file name="src/__init__.py">...</file>
<file name="tests/test_main.py">...</file>
<file name="tests/__init__.py">...</file>
```

Rules it follows:
- Uses Hypothesis `@given` for invariants and randomized domains.
- Constrains strategies at the source; never `assume()`.
- Uses `pytest.raises` for raise-rule cases.
- Imports the implementation from `src.main`.
- Every `@given` test has `@settings(max_examples=50)`.

On a compliance retry it receives `compliance_critique` and `sandbox_errors` and revises.
Routes to `coder` on success, `FINISH` on error.

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
- `test_fault` (→ `test_architect`): ruff failure on tests, tautology.
- `code_fault` (→ `coder`): ruff failure on src, mypy src failure, pytest failure.
- `infra_fault` (→ `FINISH`): dep install failure, timeout, unexpected crash.

### `prompt_compliance_checker`

Reads `user_prompt`, `src/main.py`, and `tests/test_main.py`. Outputs JSON:

```json
{"compliance_status": "PASS" | "FAIL", "missing_features": ["..."]}
```

- `PASS` → `FINISH`.
- `FAIL` with loops remaining → `test_architect` with the critique appended to `compliance_critique`.
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
- **Do not overwrite `requirements.txt` on every `test_architect` run.** `setup_workspace` is idempotent.
- **Always merge base sandbox deps into any coder-provided `requirements.txt`.**
- **Mypy is run only on `src/main.py`.** Tests are too likely to trigger stub/import noise.
- **Radon was removed in v0.2.** It added fragility without enough value.
- **`.architecture.md` is no longer produced.** The archivist node was removed.
- **Sandbox containers must run as the host user, not root.** Use `PUID`/`PGID` (set in `docker-compose.yml` / `.env`); `_host_identity()` reads them and `setup_workspace` chowns the workspace accordingly.
- **Verification container uses `--read-only`** with writable tmpfs for `/tmp` and `/var/tmp`; only `/workspace` is writable.
- **Prompt changes in `src/nodes.py` should be noted in this file** under the latest session section (per `AGENTS.md`).

---

## 10. Prompt Change Log

### 2026-07-01 — anti-lazy test contract + resilience tuning

- Updated `test_architect` system prompt in `src/nodes.py`:
  replaced the restrictive "concrete boundary tests only when explicitly mentioned" rule with
  a general adversarial principle: tests must reject lazy, literal, or loophole-seeking
  implementations by covering implied invariants and edge cases the prompt does not enumerate.
  This closes the root cause behind shallow `csv_parse`/`graph`/`evaluator` implementations.
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
- Considered (and reverted) a `test_architect` prompt edit that would have required tests to
  use only `st.composite` from `hypothesis.strategies` and prohibited importing the top-level
  `composite` decorator from `hypothesis`. The goal was to prevent `ImportError: cannot import name 'composite'`
  failures seen on strategy-heavy questions like `csv_parse`. Reverted because it is a prompt change
  the user did not request; kept as a note in case future runs continue to show this failure mode.

### 2026-06-30 — calculator expression-generation guidance

- Updated `test_architect` user prompt in `src/nodes.py` with language-specific guidance:
  for expression evaluators / calculators / parsers, generate raw expression strings and
  compute expected values using the target language's standard evaluator (e.g., Python's
  `eval()` with a safe scope). Avoid hand-written AST renderers or string-composition
  logic that must preserve precedence; let the language parser be the source of truth.
  This prevents renderer/associativity bugs where the generated test's AST and rendered
  string evaluated differently.

### 2026-06-30 — calculator sub-expression boundary fix

- Updated `test_architect` system prompt in `src/nodes.py` with the rule:
  "Preserve sub-expression boundaries: parenthesize any fragment inserted into a larger expression."
  This prevents generated Hypothesis strategies from accidentally changing the intended
  value when composing recursive expressions (e.g., division-by-zero denominators that
  evaluated to non-zero due to missing parentheses).

### 2026-07-01 — Ollama Cloud host fix

- Changed `OLLAMA_CLOUD_HOST` in `src/nodes.py` from `https://ollama.com` to
  `https://api.ollama.com` because `api.ollama.cloud`/`ollama.com` were not resolving on
  the local network and causing `ReadTimeout` failures for `test_architect`.
- **Note:** a later LiteLLM proxy integration was built and then reverted; the orchestrator
  currently calls Ollama Cloud directly again.

### 2026-07-01 — LiteLLM Ollama Cloud proxy integration (reverted)

> **Status:** This integration was built and verified, but the user later removed it and
> returned to direct Ollama Cloud routing. The proxy directory may still exist at
> `~/DockerContainers/litellm-ollama-proxy`, but it is not part of the active architecture.

- Added a local LiteLLM proxy in `~/DockerContainers/litellm-ollama-proxy` that fronts two
  Ollama Cloud accounts, round-robins requests, and retries/cooldowns failing backends.
  - Fixed `config.yaml` to use the `ollama_chat/*` provider with `api_base: https://ollama.com`,
    which is the correct LiteLLM mapping for Ollama Cloud's native `/api/chat` endpoint.
    The previous `openai/*` + `https://api.ollama.com/v1` configuration caused 405 errors
    because Ollama Cloud's `/v1` OpenAI-compatible passthrough does not accept those model names.
  - Fixed `LITELLM_MASTER_KEY` to start with `sk-` as required by LiteLLM.
  - Put the proxy and the Coding Module on a shared external Docker network `llm-proxy` so
    the orchestrator could reach the proxy by container name.
- Updated `llm_config.yaml` to route all nodes through the proxy:
  - provider: `openai-compatible`
  - base_url: `http://litellm-ollama-proxy:4000/v1`
  - api_key_env_var: `LITELLM_MASTER_KEY`
  - model names: `kimi-k2.7-code:cloud` for `test_architect`/`coder`, and
    `nemotron-3-nano:30b:cloud` (correct Ollama Cloud name) for `prompt_compliance_checker`.
- Updated `.env` and `.env.example` to expose `LITELLM_MASTER_KEY` and retired the direct
  `OLLAMA_API_KEY` from active use.
- Updated `docker-compose.yml` to attach the Coding Module API to the `llm-proxy` network.
- Verified end-to-end with a `fibonacci` benchmark run that passed in ~55s.

### 2026-07-01 — LiteLLM proxy: expose all Ollama Cloud models (reverted)

> **Status:** This proxy work was reverted with the rest of the LiteLLM integration.

- Replaced the explicit 4-model proxy config with the full Ollama Cloud catalog.
  The proxy queried `https://api.ollama.com/v1/models` and registered every model
  (35 base + `:cloud` variants) across both backends, for a total of 70 model names
  and 140 endpoints.
- Fixed the Coding Module `prompt_compliance_checker` model from the incorrect
  `nemotron-3-nano:30b-cloud` to the actual Ollama Cloud name `nemotron-3-nano:30b:cloud`.
- Re-tested the Coding Module with a `fibonacci` run; it passed end-to-end.

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

### 2026-07-01 — self-loop conditional edge fix

- Fixed `KeyError('test_architect')` crash that killed tasks during infra retry.
  The nodes return `next_node: <self>` on transient LLM faults, but the conditional edge
  mappings in `src/graph.py` did not include those self-loop targets, so LangGraph raised
  `KeyError` and the runner recorded `status="failed" error='test_architect'`.
- Updated `src/graph.py` to add `test_architect`, `coder`, and `prompt_compliance_checker`
  as valid self-loop targets in their respective conditional edge mappings.
- Updated `src/api.py` to propagate `llm_infra_exhausted` and `llm_infra_retries` from the
  graph stream into `tasks_db` / `TaskStatusResponse`, so the API correctly reports
  `infra_exhausted` instead of `failed`.
- Verified the fix by re-running the 12-question benchmark (`run_20260701_215012` / label
  `run_12_002`): the crash disappeared and the run completed all 12 questions.
  Result: **9/12 passed**. Remaining failures:
  - `merge_sorted`: `test_architect` hallucinated Hypothesis API `st.lists(..., sorted=True)`;
    the generated tests fail collection and the sandbox routes to `coder` as a code fault.
  - `calculator`: runner per-question timeout (20 min) exceeded.
  - `evaluator`: runner per-question timeout (40 min) exceeded.

### v0.2 refactor

- Rewrote `test_architect`, `coder`, `sandbox_arbiter`, `prompt_compliance_checker` system prompts.
- Dropped `contract_verifier`, `error_distiller`, `archivist_node`, `static_analyzer`, `deterministic_verifier`, `architect_node` prompts.
- Simplified prompts to produce a single-file contract: `src/main.py` + `tests/test_main.py`.
- Sandbox arbiter now uses `python -m ruff/mypy/pytest` instead of bare commands.
- Hardened sandbox: test containers run as host `PUID`/`PGID`, verification container uses `--read-only` + tmpfs, caches isolated to `/tmp`.

---

## 11. Legacy: v0.1 Pipeline

The previous system used eight nodes: `workspace_loader` → `architect_node` → `test_writer` ↔ `contract_verifier` → `code_writer` ↔ `static_analyzer` → `deterministic_verifier` → `error_distiller` → `archivist_node` / `FINISH`.

It was over-complicated for the current target reliability and is preserved in git history only. Revert via git if you need to resurrect it.
