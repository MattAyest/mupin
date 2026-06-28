# Coding Module — Self-Healing Code Generation Microservice

An autonomous code generation microservice built with LangGraph and FastAPI. Submit a natural language prompt and the agent architects a solution, writes tests first (TDD), generates an implementation, and iteratively verifies it inside an isolated Docker sandbox. Failures are classified by fault type and routed to the appropriate node for repair — the agent fixes itself until tests pass.

## Architecture

```
workspace_loader
      │
      ▼
architect_node ──────────────────────────────┐
      │                                      │
      ▼                                      │
test_writer ◄──────────┐                    │
      │                │                    │
      ▼                │                    │
contract_verifier ─────┘ (up to 3 retries)  │
      │                                      │
      ▼                                      │
code_writer ◄──────────┐                    │
      │                │                    │
      ▼                │                    │
static_analyzer ───────┘                    │
      │                                      │
      ▼                                      │
deterministic_verifier                      │
      │                                      │
      ▼                                      │
error_distiller ──► code_writer  (impl)     │
                ──► test_writer  (tests)     │
                ──► architect_node (spec) ───┘
                ──► archivist_node (success)
                ──► FINISH (ceiling hit)
```

### How it works

1. **Architect** acts as a systems/data engineer: defines the architectural shape and a precise behavioral contract (signatures, return guarantees, raise-rules stated as rules rather than enumerated lists).
2. **Test writer** writes a pytest suite that verifies the contract — never sees the implementation. Imports from `src.main`.
3. **Contract verifier** checks (cheaply) that the tests faithfully reflect what the contract explicitly states; retries up to 3 times. All LLM nodes share a small retry wrapper (`_invoke_with_retry`) for transient provider errors; if retries are exhausted, the task fails fast with a clear `failed` status instead of silently proceeding with bad output.
4. **Code writer** implements the contract (not the tests) in `src/main.py` — correct behavior makes the tests pass as a consequence.
5. **Static analyzer** catches syntax errors deterministically before running Docker.
6. **Deterministic verifier** installs deps and runs pytest inside a hardened `python:3.11-slim` sibling container (network disabled during test run).
7. **Error distiller** classifies failures as `implementation`, `tests`, or `spec` faults and routes accordingly. On success, it does a semantic contract check to catch hardcoded outputs or loopholes.
8. **Archivist** on success, summarises the architecture into a `.architecture.md` ledger.

### Per-node LLM routing

Every node picks its own provider and model via `llm_config.yaml` — change models without rebuilding the image (the file is hot-mounted). The default config runs the entire pipeline on **Ollama Cloud** with a single `OLLAMA_API_KEY`.

| Node | Model | Tier |
|---|---|---|
| architect_node | kimi-k2.7-code:cloud | premium (deep design reasoning) |
| test_writer | minimax-m3:cloud | heavy (predictable-latency coder) |
| contract_verifier | nemotron-3-nano:30b-cloud | medium-light |
| code_writer | minimax-m3:cloud | heavy (predictable-latency coder) |
| error_distiller | nemotron-3-nano:30b-cloud | medium |
| archivist_node | nemotron-3-nano:30b-cloud | light |

The writer nodes use **minimax-m3** rather than a reasoning model: implementing an
already-precise contract needs consistent latency, not extended chain-of-thought.
The architect keeps a reasoning model where designing the contract pays for it.

To escalate a node to a frontier API, swap the provider/model in `llm_config.yaml` — no rebuild needed.

## Prerequisites

- Docker on the host (the verifier spawns sibling containers via the Docker socket)
- An Ollama Cloud API key (`OLLAMA_API_KEY`) — or swap any node to another provider in `llm_config.yaml`

## Getting Started

1. Clone the repo:
   ```bash
   git clone <repo-url>
   cd Coding-Module
   ```

2. Copy `.env.example` to `.env` and fill in your key:
   ```bash
   cp .env.example .env
   # edit .env and set OLLAMA_API_KEY=<your key>
   ```

3. Build and start:
   ```bash
   docker compose up --build -d
   ```

## Usage

**Submit a task:**
```bash
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python module that implements an LRU cache."}'
```

Returns a `task_id` immediately. The pipeline runs in the background.

**Poll for status:**
```bash
curl http://localhost:8000/task/<task_id>
```

Response includes `status`, `current_node`, `loop_count`, `regression_count`, `replan_count`, `thoughts` (one-line per-node log), and on completion `result` (the full file manifest).

Status values:
- `running` — task is active
- `completed` — tests passed, semantic validation passed, and the ledger was archived (`archivist_node` reached)
- `exhausted` — loop/replan ceiling reached or server deadline exceeded without success
- `cancelled` — task was cancelled via `POST /task/<task_id>/cancel`
- `failed` — the graph raised an unhandled exception, including `LLMUnavailableError` when an LLM provider fails after retries

**Cancel an in-flight task:**
```bash
curl -X POST http://localhost:8000/task/<task_id>/cancel
```

Useful for benchmark harnesses that time out: cancelling stops the pipeline promptly instead of leaving an orphan task running.

**Watch the thought log:**
```bash
curl http://localhost:8000/task/<task_id>/log
```

Plain-text one-liner per node action — good for tailing progress. Detailed diagnostic content (LLM responses, pytest output) is written to `.workspaces/<task_id>/task.log`.

**Generated files on disk:**
```
.workspaces/<task_id>/
├── src/           # generated implementation
├── tests/         # generated test suite
├── conftest.py    # hypothesis settings (max_examples=50)
├── pytest.ini
├── requirements.txt
├── task.log       # full diagnostic log
└── .architecture.md   # written on success
```

## Configuration

`llm_config.yaml` controls per-node models, loop limits, and Docker sandbox settings. It is mounted read-only — changes take effect on the next task without a rebuild.

```yaml
loop_limits:
    max_verification_loops: 10   # hard ceiling on verifier runs per task
    max_regression_count: 4      # failures before forcing architect replan
    max_replan_count: 3          # replans before giving up entirely

docker:
    image: "python:3.11-slim"
    timeout_install: 90
    timeout_test: 120
    memory_limit: "512m"
```

Supported providers: `ollama-cloud`, `ollama` (local), `openai`, `anthropic`, `google-genai`, `openai-compatible`.

## Docker sandbox hardening

The test container runs with:
- `--network none` during the test phase
- `--cap-drop ALL`
- `--security-opt no-new-privileges`
- `--memory 512m` / `--memory-swap 512m`
- `--pids-limit 64`
- Runs as the host user UID so workspace files stay writable after the run
