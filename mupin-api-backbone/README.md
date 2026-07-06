# Mupin API Backbone

Generic job submission, queue, persistence, and dispatch service for Mupin modules.

## API

- `POST /jobs` — submit a job
  ```json
  {
    "job_type": "coding",
    "payload": {"prompt": "Write a fibonacci function", "profile_name": "python"}
  }
  ```
- `GET /jobs/{job_id}` — status, progress, and result
- `POST /jobs/{job_id}/cancel` — request cooperative cancellation
- `GET /jobs/{job_id}/log` — worker thought log (reads from shared workspace)

## Internal endpoints (used by workers)

- `POST /internal/jobs/{job_id}/progress` — heartbeat between graph nodes
- `POST /internal/jobs/{job_id}/finalize` — terminal status + result

## Stack

- FastAPI + Uvicorn
- ARQ on Redis
- SQLAlchemy async + asyncpg on Postgres

## Local development

```bash
# Requires redis and postgres to be running (see root docker-compose.yml)
python -m src.main
```
