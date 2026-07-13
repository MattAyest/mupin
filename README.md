# Mupin

Mupin is a modular system of AI agents. Each agent lives in its own module under this repository.

## Modules

- [`mupin-api-backbone/`](./mupin-api-backbone/) — Shared job submission, queue, persistence, and dispatch service. All workers register with it via ARQ/Redis and report results back to it.
- [`mupin-coding-module/`](./mupin-coding-module/) — A self-healing code-generation worker. Given a natural-language prompt, it designs tests, derives a skeleton, implements the code, and verifies it in a hardened Docker sandbox.
- [`mupin-editing-module/`](./mupin-editing-module/) — An editing worker that takes an existing workspace (typically from the coding module) and applies a natural-language edit instruction, then re-verifies the result and returns a unified diff.

## Getting started

From the repository root:

```bash
# Copy the unified env template and set your keys
cp .env.example .env
# edit .env with your provider keys

docker compose up --build -d

# Generate code
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python module with a single function fibonacci(n: int) -> list[int] that returns the first n Fibonacci numbers."}'

# Edit the generated code (replace <job_id> with the coding job id)
curl -X POST http://localhost:8003/edit \
  -H "Content-Type: application/json" \
  -d '{"source_job_id": "<job_id>", "instruction": "Add docstrings to all public functions"}'
```

## Configuration hierarchy

- The root `.env.example` and `llm_config.yaml.example` provide global defaults.
- Each module can override globals with its own `.env` and `llm_config.yaml`.
- When running via `docker compose`, the root `.env` is loaded first and module `.env` files override it.

## Benchmarks

- Coding: `mupin-coding-module/benchmarks/runner.py`
- Editing: `mupin-editing-module/benchmarks/runner.py`

## Repository guide

See [`AGENTS.md`](./AGENTS.md) for agent-specific conventions and build rules.
