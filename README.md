# Coding Module — v0.2

A self-healing Python code-generation microservice. Submit a natural-language prompt and a LangGraph agent:

1. Designs the test suite and a matching skeleton from the prompt.
2. Implements `src/main.py` against those tests.
3. Runs ruff, mypy, and pytest inside a hardened Docker sandbox.
4. Checks that the implementation actually satisfies the original prompt.

If a step fails, the agent routes back to the node that can fix it and tries again. Nodes can also retry themselves when a transient LLM infrastructure fault occurs (timeout, 5xx/524, connection drop), up to the configured `infra_max_retries_per_node`.

> This is **v0.2**. The older eight-node pipeline is preserved only in git history.

---

## Architecture

```
         ┌─────────────────────────────┐
         │  (up to infra_max_retries)  │
         ▼                             │
test_architect ───────────────────────┤
      │                               │
      ▼                               │
coder ────────────────────────────────┤
      │                               │
      ▼                               │
sandbox_arbiter ───────┐  (up to 5 loops)
      │                │
      ▼                │
prompt_compliance_checker
      │                │
      ▼                │
FINISH ◄───────────────┘
```

### How it works

| Node | Responsibility | Routes on failure |
|---|---|---|
| `test_architect` | Writes `tests/test_main.py`, `src/main.py` skeleton, and init files from the prompt. | `coder` on success; `test_architect` on transient LLM fault; `FINISH` if it cannot emit the required files or after exhausting infra retries. |
| `coder` | Replaces skeleton bodies with real logic; preserves tests. | `sandbox_arbiter` on success; `coder` on transient LLM fault; `FINISH` if no `src/main.py` is produced. |
| `sandbox_arbiter` | Installs deps, formats/lints code, runs `mypy` on `src/main.py`, runs pytest. | `coder` for implementation/test/runtime faults; `test_architect` for test-side faults; `FINISH` for infrastructure faults. |
| `prompt_compliance_checker` | Reads the passing implementation and tests and judges whether every functional requirement in the prompt is covered. | `test_architect` with a critique, up to 2 times; `prompt_compliance_checker` on transient LLM fault; then `FINISH`. |

### Per-node LLM routing

Each node has its own provider/model entry in `llm_config.yaml`. The default config runs everything on Ollama Cloud with a single `OLLAMA_API_KEY`.

| Node | Default model | Role |
|---|---|---|
| `test_architect` | `kimi-k2.7-code:cloud` | Designs tests + skeleton |
| `coder` | `kimi-k2.7-code:cloud` | Implements `src/main.py` |
| `prompt_compliance_checker` | `nemotron-3-nano:30b-cloud` | Semantic prompt check |

Swap models without rebuilding — `llm_config.yaml` is mounted read-only at runtime.

---

## Prerequisites

- Docker on the host (the orchestrator spawns sibling test containers via the Docker socket).
- An Ollama Cloud API key (`OLLAMA_API_KEY`) — or point any node to another provider in `llm_config.yaml`.

## Getting started

1. Copy `.env.example` to `.env` and set `OLLAMA_API_KEY`.
2. Build and start:
   ```bash
   docker compose up --build -d
   ```
3. Submit a task:
   ```bash
   curl -X POST http://localhost:8000/task \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Write a Python module with a single function fibonacci(n: int) -> list[int] that returns the first n Fibonacci numbers."}'
   ```
4. Poll status:
   ```bash
   curl http://localhost:8000/task/<task_id>
   ```
5. Tail the thought log:
   ```bash
   curl http://localhost:8000/task/<task_id>/log
   ```

### Status values

- `running` — active.
- `completed` — sandbox passed and prompt compliance checker returned `PASS`.
- `exhausted` — loop/replan ceiling, server deadline, or an infrastructure fault.
- `cancelled` — cancelled via `POST /task/<task_id>/cancel`.
- `infra_exhausted` — transient LLM infrastructure faults (timeouts, 5xx/524, connection drops) exhausted the per-node retry budget.
- `failed` — unhandled exception, including permanent `LLMUnavailableError` after retries.

### Cancel a task

```bash
curl -X POST http://localhost:8000/task/<task_id>/cancel
```

---

## Observability

`GET /task/{task_id}` returns:

- `task_id`, `status`, `current_node`
- `sandbox_loop_count`, `compliance_loop_count`
- `compliance_status` (`PASS`, `FAIL`, or empty)
- `llm_infra_exhausted` — `true` if the retry budget for transient LLM faults was exhausted
- `error` — last `sandbox_errors` value or terminal reason
- `thoughts` — one-liner per node
- `node_history` — wall-clock timings
- `llm_usage` — per-invocation model, duration, token estimates
- `docker_runs` — per-sandbox stdout/stderr tails and durations
- `result` — final `file_manifest`

Detailed diagnostic output is also written to `.workspaces/<task_id>/task.log` on the host.

---

## Generated workspace layout

```
.workspaces/<task_id>/
├── src/
│   ├── __init__.py
│   └── main.py          # generated implementation
├── tests/
│   ├── __init__.py
│   └── test_main.py     # generated tests
├── conftest.py          # hypothesis profile
├── pytest.ini
├── requirements.txt     # base sandbox deps + any extras from coder
└── task.log             # full diagnostic log
```

No `.architecture.md` ledger is produced in v0.2.

---

## Configuration

`llm_config.yaml` controls per-node models, loop limits, and Docker settings.

```yaml
loop_limits:
    max_sandbox_loops: 5
    max_compliance_loops: 2

docker:
    image: "python:3.11-slim"
    timeout_install: 90
    timeout_test: 120
    memory_limit: "512m"

server:
    task_deadline_seconds: 3600

resilience:
    infra_max_retries_per_node: 5
    infra_retry_backoff_seconds: [10, 30, 60, 120, 120]
```

Supported providers: `ollama-cloud`, `ollama` (local), `openai`, `anthropic`, `google-genai`, `openai-compatible`.

## Docker sandbox hardening

The test container runs with:

- `--user` set to the host user (`PUID`/`PGID`, default `1000:1000`) — **never root**
- `--read-only` root filesystem during verification; only `/workspace` is writable
- `--tmpfs` mounts for `/tmp` and `/var/tmp` so caches stay in memory
- `--network none` during verification (install phase needs network for packages)
- `--cap-drop ALL`
- `--security-opt no-new-privileges`
- `--memory 512m` / `--memory-swap 512m`
- `--pids-limit 64`
- `--ulimit nofile=1024:1024`
- Tool caches forced to `/tmp` via `RUFF_CACHE_DIR` and `MYPY_CACHE_DIR`

The orchestrator itself still runs as root in its container (it needs the Docker socket), but sandbox test containers are unprivileged and run as the host user. `PUID`/`PGID` can be set in `.env` if your host user is not `1000:1000`.

---

## Benchmarks

```bash
python benchmarks/runner.py              # all questions
python benchmarks/runner.py --ids fibonacci stack
python benchmarks/runner.py --summary    # historical runs
python benchmarks/runner.py --diagnostics
```

Results append to `benchmarks/results.jsonl` and `benchmarks/metrics.jsonl`.

---

## Local development (without Docker)

```bash
uvicorn src.api:app --reload
```

The verifier still spawns sibling test containers against the host Docker socket.

---

## Version history

- **v0.2** (current) — simplified four-node pipeline: `test_architect`, `coder`, `sandbox_arbiter`, `prompt_compliance_checker`. Added per-node transient LLM fault retry and self-loop conditional edges.
- **v0.1** — eight-node pipeline with separate architect, test writer, contract verifier, code writer, static analyzer, deterministic verifier, error distiller, and archivist. Recoverable from git history if needed.

### Latest benchmark snapshot

Run `run_20260701_215012` (12 questions, direct Ollama Cloud routing): **9/12 passed**.

- Passed: `fibonacci`, `stack`, `word_frequency`, `csv_parse`, `graph`, `token_bucket`, `min_heap`, `bounded_queue`, `rle`.
- Failed: `merge_sorted` (Hypothesis API hallucination: `st.lists(..., sorted=True)`), `calculator` (runner timeout), `evaluator` (runner timeout).
