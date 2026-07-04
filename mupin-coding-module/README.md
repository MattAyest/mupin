# Coding Module — v0.3

A self-healing Python code-generation worker. Submit a natural-language prompt via the Mupin API Backbone and a LangGraph agent:

1. Designs a lean baseline test suite from the prompt.
2. Derives a matching skeleton from those tests and checks contract compatibility.
3. Implements `src/main.py` against the tests.
4. Runs ruff, mypy, and pytest inside a hardened Docker sandbox.
5. Checks that the implementation actually satisfies the original prompt.

If a step fails, the agent routes back to the node that can fix it and tries again. Nodes can also retry themselves when a transient LLM infrastructure fault occurs (timeout, 5xx/524, connection drop), up to the configured `infra_max_retries_per_node`.

> This is **v0.3**. The module is now a pure ARQ worker consuming `coding` jobs from `mupin-api-backbone`. A dev convenience `POST /task` endpoint still proxies to the backbone for local use. The older eight-node v0.1 pipeline is preserved only in git history.

---

## Architecture

```
          ┌─────────────────────────────┐
          │  (up to infra_max_retries)  │
          ▼                             │
 test_designer ───────────────────────┤
       │                               │
       ▼                               │
 skeleton_maker ──────────────────────┤
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
| `test_designer` | Writes a lean `tests/test_main.py` covering the prompt's explicit requirements. | `skeleton_maker` on success; `test_designer` on transient LLM fault; `FINISH` if it cannot emit the required test files. |
| `skeleton_maker` | Derives `src/main.py` skeleton from the tests and checks prompt/skeleton/test compatibility. | `coder` on success; `test_designer` with a critique if the contract is incompatible (up to 2 loops); `FINISH` if it cannot emit the skeleton. |
| `coder` | Replaces skeleton bodies with real logic; preserves tests. | `sandbox_arbiter` on success; `coder` on transient LLM fault; `FINISH` if no `src/main.py` is produced. |
| `sandbox_arbiter` | Installs deps, formats/lints code, runs `mypy` on `src/main.py`, runs pytest. | `coder` for implementation/test/runtime faults; `test_designer` for test-side faults; `FINISH` for infrastructure faults. |
| `prompt_compliance_checker` | Reads the passing implementation and tests and judges whether every functional requirement in the prompt is covered. | `test_designer` with a critique, up to 2 times; `prompt_compliance_checker` on transient LLM fault; then `FINISH`. |

### Per-node LLM routing

Each node has its own provider/model entry in `llm_config.yaml`. The default config runs everything on Ollama Cloud with a single `OLLAMA_API_KEY`.

| Node | Default model | Role |
|---|---|---|
| `test_designer` | `kimi-k2.7-code:cloud` | Designs baseline tests |
| `skeleton_maker` | `kimi-k2.7-code:cloud` | Derives skeleton + contract check |
| `coder` | `kimi-k2.7-code:cloud` | Implements `src/main.py` |
| `prompt_compliance_checker` | `nemotron-3-nano:30b-cloud` | Semantic prompt check |

Swap models without rebuilding — `llm_config.yaml` is mounted read-only at runtime.

---

## Prerequisites

- Docker on the host (the orchestrator spawns sibling test containers via the Docker socket).
- An Ollama Cloud API key (`OLLAMA_API_KEY`) — or point any node to another provider in `llm_config.yaml`.

## Getting started

1. Copy `.env.example` to `.env` and set `OLLAMA_API_KEY`.
2. Build and start the full stack from the repo root:
   ```bash
   cd ..
   docker compose up --build -d
   ```
   This starts Postgres, Redis, `mupin-api-backbone` (port 8001), the coding worker, and a dev proxy (port 8000).
3. Submit a job through the backbone:
   ```bash
   curl -X POST http://localhost:8001/jobs \
     -H "Content-Type: application/json" \
     -d '{"job_type": "coding", "payload": {"prompt": "Write a Python module with a single function fibonacci(n: int) -> list[int] that returns the first n Fibonacci numbers."}}'
   ```
4. Poll status:
   ```bash
   curl http://localhost:8001/jobs/<job_id>
   ```
5. Tail the thought log:
   ```bash
   curl http://localhost:8001/jobs/<job_id>/log
   ```

The legacy dev endpoint still works and proxies to the backbone:
```bash
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"prompt": "..."}'
```

### Benchmark runner

```bash
# Run all questions concurrently (default)
python3 benchmarks/runner.py

# Specific questions, concurrent
python3 benchmarks/runner.py --ids fibonacci stack word_frequency

# Sequential mode (one at a time)
python3 benchmarks/runner.py --sequential

# Adjust total-run cap (default 3600s = 1 hour)
python3 benchmarks/runner.py --total-timeout 7200

# On total timeout: finish in-progress jobs instead of cancelling them
python3 benchmarks/runner.py --total-timeout-action finish_in_progress
```

The runner also respects environment variables:
- `MUPIN_TOTAL_TIMEOUT` — total run timeout in seconds (default `3600`)
- `MUPIN_TOTAL_TIMEOUT_ACTION` — `cancel` or `finish_in_progress` (default `cancel`)

Per-question timeout defaults to `2400`s (40 min) and can be overridden with `--timeout`.

### Docker maintenance

The sandbox can leave exited containers and build cache behind. Run the prune script occasionally:

```bash
./scripts/docker-prune.sh --dry-run   # preview
./scripts/docker-prune.sh            # clean
```

This removes exited containers (except persistent services), dangling build cache, and unused images older than 24h.

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

Each result row records three durations:
- `elapsed_seconds` — total wall-clock time from submission to completion.
- `queue_wait_seconds` — time spent waiting for a worker slot (`started_at - start_time`).
- `processing_seconds` — actual pipeline execution time (`end_time - started_at`).

This split makes it easy to tell whether a slow question is slow because of
provider/worker work or because it sat behind other jobs in the ARQ queue.

---

## Local development (without Docker)

```bash
uvicorn src.api:app --reload
```

The verifier still spawns sibling test containers against the host Docker socket.

---

## Version history

- **v0.3** (current) — worker consumes `coding` jobs from `mupin-api-backbone` via ARQ/Redis. Backbone persists job state in Postgres. Dev API at `localhost:8000` proxies to backbone at `localhost:8001`.
- **v0.2** — five-node pipeline: `test_designer`, `skeleton_maker`, `coder`, `sandbox_arbiter`, `prompt_compliance_checker`. Added per-node transient LLM fault retry and self-loop conditional edges.
- **v0.1** — eight-node pipeline with separate architect, test writer, contract verifier, code writer, static analyzer, deterministic verifier, error distiller, and archivist. Recoverable from git history if needed.

### Latest benchmark snapshot

Run `run_20260701_215012` (12 questions, direct Ollama Cloud routing): **9/12 passed**.

- Passed: `fibonacci`, `stack`, `word_frequency`, `csv_parse`, `graph`, `token_bucket`, `min_heap`, `bounded_queue`, `rle`.
- Failed: `merge_sorted` (Hypothesis API hallucination: `st.lists(..., sorted=True)`), `calculator` (runner timeout), `evaluator` (runner timeout).
