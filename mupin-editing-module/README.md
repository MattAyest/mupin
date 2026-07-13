# Editing Module — v0.1

A Mupin worker that edits existing code. Unlike the Coding Module, which generates a workspace from a prompt, the Editing Module takes an existing workspace produced by another job (typically the Coding Module) and applies a natural-language editing instruction.

The module follows the same backbone integration pattern as the Coding Module:

- Consumes `editing` jobs from `mupin-api-backbone` via ARQ/Redis.
- Reports progress via internal backbone endpoints.
- Runs the same language-profile sandbox verification as the Coding Module.
- Returns the updated workspace plus a unified diff against the source.

## Pipeline

```
load_source → analyze → plan → apply → verify → FINISH
                 ↑_________| (on code/test/runtime fault)
```

| Node | Responsibility |
|---|---|
| `load_source` | Copy the source workspace into a new editing workspace. |
| `analyze` | Read files + instruction and summarize what must change. |
| `plan` | Emit a concrete per-file edit plan. |
| `apply` | Rewrite the affected files while preserving the rest. |
| `verify` | Run the language profile's sandbox checks (ruff, mypy, pytest). |

## Getting started

The Editing Module is started automatically by the root `docker-compose.yml` using the unified configuration hierarchy (root `.env` + optional module-local `.env`). If you need to start it in isolation:

```bash
cd mupin-editing-module
cp .env.example .env
# add OLLAMA_API_KEY and any other provider keys, or rely on the root ../.env

docker compose up --build -d
```

The dev API proxy listens on port `8003` by default (override with `EDITING_API_PORT` in the root `.env`).

## Submit an edit

```bash
curl -X POST http://localhost:8003/edit \
  -H "Content-Type: application/json" \
  -d '{"source_job_id": "task_abc123", "instruction": "Add type hints and a test for negative inputs", "language": "python"}'
```

Or through the backbone directly:

```bash
curl -X POST http://localhost:8001/jobs \
  -H "Content-Type: application/json" \
  -d '{"job_type": "editing", "payload": {"source_job_id": "task_abc123", "instruction": "Add type hints and a test for negative inputs", "profile_name": "python"}}'
```

## Result shape

A completed editing job returns:

- `file_manifest` — all files in the final workspace.
- `diff` — unified diff from the source workspace.
- `workspace` — path to the new editing workspace on the shared volume.

## Benchmark

Run the 20-question editing benchmark from the module directory:

```bash
cd mupin-editing-module/benchmarks
python runner.py --batch-size 4

# Reuse cached base coding jobs after the first run:
python runner.py --batch-size 4 --reuse-base
```

## Language profiles

Profiles live in `profiles/`. The default `python.yaml` reuses the same sandbox image and verification commands as the Coding Module, so edits are held to the same quality bar.

Future languages can be added by creating a new `<lang>.yaml` profile with the same `sandbox`, `files`, and `prompts` sections.
