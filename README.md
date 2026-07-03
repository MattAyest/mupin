# Mupin

Mupin is a modular system of AI agents. Each agent lives in its own module under this repository.

## Modules

- [`coding-module/`](./coding-module/) — A self-healing Python code-generation microservice. Given a natural-language prompt, it designs tests, derives a skeleton, implements the code, and verifies it in a hardened Docker sandbox.

## Getting started

From the repository root:

```bash
docker compose up --build -d
curl -X POST http://localhost:8000/task \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a Python module with a single function fibonacci(n: int) -> list[int] that returns the first n Fibonacci numbers."}'
```

This starts the Coding Module service. Additional modules will be added to the top-level compose file as they are introduced.

## Repository guide

See [`AGENTS.md`](./AGENTS.md) for agent-specific conventions and build rules.
