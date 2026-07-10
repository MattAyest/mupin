#!/usr/bin/env python3
"""
Benchmark runner for the Coding Module (Tier 2 custom 20-question suite).

Usage:
    python benchmarks/runner.py                        # run all questions (slot-based, 6 wide)
    python benchmarks/runner.py --ids fibonacci stack  # run specific questions
    python benchmarks/runner.py --sequential           # run one question at a time
    python benchmarks/runner.py --batch-size 6         # cap concurrent submissions (default 20)
    python benchmarks/runner.py --per-q-timeout 3600   # per-job cap, measured from worker start
    python benchmarks/runner.py --summary              # print history of past runs
    python benchmarks/runner.py --diagnostics          # per-node token/time breakdown

Results are appended to benchmarks/results.jsonl — one JSON line per question.
Per-node metrics are appended to benchmarks/metrics.jsonl.

Execution model: slot-based. At most `--batch-size` jobs are in flight at once;
each worker thread loops submit -> poll -> record -> next problem. Queue wait is
therefore near zero, and the per-job timeout (--per-q-timeout) measures real
pipeline work, measured from the worker's `started_at` timestamp -- NOT from
submission time. There is no whole-run kill switch.
"""

import argparse
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = os.environ.get("MUPIN_BACKBONE_URL", os.environ.get("CODING_MODULE_URL", "http://localhost:8001"))
QUESTIONS_FILE = Path(__file__).parent / "questions.json"
RESULTS_FILE = Path(__file__).parent / "results.jsonl"
METRICS_FILE = Path(__file__).parent / "metrics.jsonl"

POLL_INTERVAL = 5          # seconds between status polls
PER_Q_TIMEOUT = 3600      # 1h per question, measured from worker start (started_at).
                           # The full pipeline can take up to ~46 min for hard questions;
                           # 1h gives comfortable headroom.
QUEUE_GRACE = 600          # seconds to wait for the worker to pick up a submitted
                           # job (i.e. for `started_at` to appear). With slot-based
                           # submission a job is only submitted when a slot is free, so
                           # pickup is normally within seconds. The grace is a safety
                           # net for a dead/sick backbone, NOT a queue-wait cap -- if
                           # it fires, something is genuinely wrong. 600s gives wide
                           # headroom for ARQ polling + Redis round-trips.
DEFAULT_BATCH_SIZE = 20    # Tier 2 suite is only 20 questions; default opens all slots.


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_questions(ids=None):
    with open(QUESTIONS_FILE) as f:
        questions = json.load(f)
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    return questions


_global_deps_cache_tag: str | None = None


def set_deps_cache_tag(tag: str | None) -> None:
    global _global_deps_cache_tag
    _global_deps_cache_tag = tag


def submit_task(prompt: str) -> str:
    payload = {"job_type": "coding", "payload": {"prompt": prompt, "profile_name": "python"}}
    if _global_deps_cache_tag:
        payload["payload"]["deps_cache_tag"] = _global_deps_cache_tag
    resp = requests.post(
        f"{BASE_URL}/jobs",
        json=payload,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["job_id"]


def _normalize_job(data: dict) -> dict:
    """Flatten backbone job response into the legacy task shape."""
    progress = data.get("progress") or {}
    normalized = dict(data)
    normalized.setdefault("task_id", data.get("job_id"))
    normalized.setdefault("current_node", progress.get("current_node", data.get("status")))
    normalized.setdefault("sandbox_loop_count", progress.get("sandbox_loop_count", 0))
    normalized.setdefault("thoughts", progress.get("thoughts", []))
    return normalized


def poll_task(task_id: str):
    resp = requests.get(f"{BASE_URL}/jobs/{task_id}", timeout=10)
    resp.raise_for_status()
    return _normalize_job(resp.json())


def cancel_task(task_id: str):
    """Ask the backbone to cancel a job so a timeout doesn't leave it running."""
    try:
        requests.post(f"{BASE_URL}/jobs/{task_id}/cancel", timeout=10)
    except Exception as e:
        print(f"  CANCEL ERROR: {e}")


def _diagnostic_fields(data: dict):
    return {
        "node_history": data.get("node_history", []),
        "llm_usage": data.get("llm_usage", []),
        "docker_runs": data.get("docker_runs", []),
        "classifier_history": data.get("classifier_history", []),
        "latest_verification_error": data.get("error"),
    }


def _started_at_from_job(data: dict) -> datetime | None:
    """Return the moment the worker actually began processing the job.

    The backbone exposes this as `started_at` in ISO-8601 format.  If it is
    missing (older workers/backbones), we fall back to the first node-history
    timestamp so that queue-wait estimates remain useful.
    """
    started = data.get("started_at")
    if started:
        try:
            return datetime.fromisoformat(started.replace("Z", "+00:00"))
        except Exception:
            pass
    for entry in data.get("node_history", []) or []:
        started = entry.get("started_at")
        if started:
            try:
                return datetime.fromisoformat(started.replace("Z", "+00:00"))
            except Exception:
                pass
    return None


def _metrics_rows(run_id: str, qid: str, result: dict, diag: dict) -> list[dict]:
    rows = []
    for entry in diag.get("node_history", []):
        rows.append({
            "run_id": run_id,
            "question_id": qid,
            "task_id": result.get("task_id"),
            "metric_type": "node",
            "metric": entry,
        })
    for entry in diag.get("llm_usage", []):
        rows.append({
            "run_id": run_id,
            "question_id": qid,
            "task_id": result.get("task_id"),
            "metric_type": "llm",
            "metric": entry,
        })
    for entry in diag.get("docker_runs", []):
        rows.append({
            "run_id": run_id,
            "question_id": qid,
            "task_id": result.get("task_id"),
            "metric_type": "docker",
            "metric": entry,
        })
    for entry in diag.get("classifier_history", []):
        rows.append({
            "run_id": run_id,
            "question_id": qid,
            "task_id": result.get("task_id"),
            "metric_type": "classifier",
            "metric": entry,
        })
    return rows


def _build_result(
    run_id: str,
    question: dict,
    task_id: str | None,
    start_time: float,
    start_iso: str,
    status: str,
    data: dict | None,
    error: str | None = None,
) -> dict:
    data = data or {}
    now = time.time()
    now_iso = datetime.now(timezone.utc).isoformat()
    elapsed = round(now - start_time, 1)

    started_at = _started_at_from_job(data)
    if started_at:
        started_ts = started_at.timestamp()
        queue_wait_seconds = round(max(0.0, started_ts - start_time), 1)
        processing_seconds = round(max(0.0, now - started_ts), 1)
    else:
        queue_wait_seconds = 0.0
        processing_seconds = elapsed

    diagnostics = _diagnostic_fields(data)
    result = {
        "run_id": run_id,
        "question_id": question["id"],
        "difficulty": question["difficulty"],
        "task_id": task_id,
        "start_time": start_iso,
        "end_time": now_iso,
        "started_at": started_at.isoformat() if started_at else None,
        "elapsed_seconds": elapsed,
        "queue_wait_seconds": queue_wait_seconds,
        "processing_seconds": processing_seconds,
        "status": status,
        "error": error or data.get("error"),
        "sandbox_loop_count": data.get("sandbox_loop_count", 0),
        "files_generated": len(data.get("result") or {}),
        "thoughts_count": len(data.get("thoughts", [])),
        **diagnostics,
    }
    append_metrics(_metrics_rows(run_id, question["id"], result, diagnostics))
    return result


# ---------------------------------------------------------------------------
# Per-question processing (shared by sequential and slot-based modes)
# ---------------------------------------------------------------------------

def _process_one(
    question: dict,
    run_id: str,
    per_q_timeout: int,
    label_lines_prefix: bool = True,
) -> dict:
    """Submit one question, poll until settled, and return the result row.

    Slot-based execution: each worker thread calls this in a loop, so at most
    `batch_size` jobs are ever in flight at once. Queue wait is therefore near
    zero, and the per-job timeout measures real pipeline work, not queue time.

    The per-job deadline is computed from `started_at` (set by the worker when
    it picks the job up), not from submission time. A short QUEUE_GRACE window
    covers submission acceptance only.
    """
    qid = question["id"]
    submit_time = time.time()
    submit_iso = datetime.now(timezone.utc).isoformat()

    if label_lines_prefix:
        print(f"\n{'─' * 60}")
        print(f"  [{qid}]  difficulty={question['difficulty']}")
        print(f"  {question['prompt'][:100]}...")
        print(f"{'─' * 60}")

    # Submit (with a short grace for the POST to return).
    try:
        task_id = submit_task(question["prompt"])
    except Exception as e:
        print(f"  [{qid}] SUBMIT ERROR: {e}")
        return _build_result(run_id, question, None, submit_time, submit_iso,
                             "submit_error", error=str(e))

    print(f"  [{qid:<20}] submitted {task_id}")

    # Poll until settled or per-job timeout (clock starts at worker start).
    deadline = None              # set once started_at appears
    submit_deadline = submit_time + QUEUE_GRACE
    last_data = {}
    last_node = None

    while True:
        # If the worker hasn't started yet, only the submission grace applies.
        if deadline is None and time.time() > submit_deadline:
            # Worker hasn't picked the job up within QUEUE_GRACE -- with
            # slot-based submission this should never happen; treat as a
            # backbone health issue and fail fast.
            cancel_task(task_id)
            result = _build_result(run_id, question, task_id, submit_time, submit_iso,
                                   "queue_timeout", last_data,
                                   error=f"Worker did not start within {QUEUE_GRACE}s "
                                         f"(backbone health issue)")
            print(f"  [{qid:<20}] QUEUE_TIMEOUT (no worker start in {QUEUE_GRACE}s)")
            return result

        if deadline is not None and time.time() > deadline:
            cancel_task(task_id)
            result = _build_result(run_id, question, task_id, submit_time, submit_iso,
                                   "timeout", last_data,
                                   error=f"Timed out after {per_q_timeout}s of worker time")
            print(f"  [{qid:<20}] TIMEOUT after {result['elapsed_seconds']}s "
                  f"(worker budget {per_q_timeout}s exceeded; "
                  f"last node={last_data.get('current_node')} "
                  f"sbox={last_data.get('sandbox_loop_count', 0)})")
            return result

        try:
            data = poll_task(task_id)
            last_data = data
        except Exception as e:
            print(f"  [{qid:<20}] poll_error: {e} — retrying in {POLL_INTERVAL}s")
            time.sleep(POLL_INTERVAL)
            continue

        status = data.get("status")
        node = data.get("current_node")

        # Lock in the per-job deadline once the worker has started.
        if deadline is None and data.get("started_at"):
            try:
                started_iso = data["started_at"].replace("Z", "+00:00")
                started_dt = datetime.fromisoformat(started_iso)
                deadline = started_dt.timestamp() + per_q_timeout
                print(f"  [{qid:<20}] worker started; deadline set "
                      f"(+{per_q_timeout}s from {data['started_at']})")
            except Exception:
                # If we can't parse started_at, fall back to submission time so
                # we don't lose the timeout entirely.
                deadline = submit_time + per_q_timeout

        if node != last_node:
            last_node = node
            print(f"  [{qid:<20}] {time.time() - submit_time:>7.1f}s -> {node}  "
                  f"(sbox={data.get('sandbox_loop_count', 0)})")

        if status in ("completed", "failed", "cancelled", "exhausted", "infra_exhausted"):
            result = _build_result(run_id, question, task_id, submit_time, submit_iso, status, data)
            verdict = "PASS" if status == "completed" and result["files_generated"] else "FAIL"
            print(
                f"\n  [{qid:<20}] {verdict} in {result['elapsed_seconds']}s  |  "
                f"queue={result.get('queue_wait_seconds', 0.0):.1f}s  "
                f"process={result.get('processing_seconds', result['elapsed_seconds']):.1f}s  |  "
                f"sbox={result['sandbox_loop_count']}  "
                f"files={result['files_generated']}"
            )
            return result

        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Sequential execution
# ---------------------------------------------------------------------------

def run_questions_sequential(questions: list[dict], run_id: str, per_q_timeout: int) -> list[dict]:
    print(f"\nBenchmark run (sequential): {run_id}")
    print(f"  Questions: {len(questions)}")
    print(f"  API:       {BASE_URL}")
    print(f"  Per-job cap: {per_q_timeout}s (from worker start)")
    results = []
    for q in questions:
        results.append(_process_one(q, run_id, per_q_timeout, label_lines_prefix=True))
    return results


# ---------------------------------------------------------------------------
# Slot-based concurrent execution
# ---------------------------------------------------------------------------

def run_questions_concurrent(
    questions: list[dict],
    run_id: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    per_q_timeout: int = PER_Q_TIMEOUT,
) -> list[dict]:
    """Run questions slot-based: at most `batch_size` jobs in flight at once.

    Each worker thread loops: submit -> poll -> record -> pick up the next
    question. This bounds queue wait to near zero (a job is only submitted when
    a worker slot is free), so the per-job timeout measures real pipeline work,
    not queue time. There is no whole-run kill switch -- each job has its own
    deadline computed from `started_at`.
    """
    run_start = time.time()
    print(f"\nBenchmark run: {run_id}")
    print(f"  Questions:    {len(questions)}")
    print(f"  API:          {BASE_URL}")
    print(f"  Concurrency:  {batch_size} (slot-based)")
    print(f"  Per-job cap:  {per_q_timeout}s (from worker start)")

    # Each worker pops the next question off a shared iterator.
    question_iter = iter(questions)
    counter_lock = threading.Lock()
    results: list[dict] = []
    results_lock = threading.Lock()
    done_count = [0]

    def worker_loop():
        while True:
            with counter_lock:
                try:
                    question = next(question_iter)
                except StopIteration:
                    return
            result = _process_one(question, run_id, per_q_timeout, label_lines_prefix=False)
            with results_lock:
                results.append(result)
                done_count[0] += 1
                idx = done_count[0]
            print(f"  ---- [{idx}/{len(questions)}] done ----")

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(worker_loop) for _ in range(batch_size)]
        for f in futures:
            f.result()

    print(f"\n  Total wall-clock time: {time.time() - run_start:.1f}s")

    # Preserve question order for the summary/diagnostics.
    results_by_id = {r["question_id"]: r for r in results}
    return [results_by_id[q["id"]] for q in questions if q["id"] in results_by_id]


# ---------------------------------------------------------------------------
# Results / diagnostics / history
# ---------------------------------------------------------------------------

def append_result(result: dict):
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


def append_metrics(metrics: list[dict]):
    if not metrics:
        return
    with open(METRICS_FILE, "a") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")


def print_summary(results: list[dict]):
    if not results:
        return
    print(f"\n{'═' * 70}")
    print(f"  RUN SUMMARY  ({results[0]['run_id']})")
    print(f"{'═' * 70}")
    print(
        f"  {'ID':<20} {'DIFF':<8} {'STATUS':<10} {'TOTAL':>8}  "
        f"{'QUEUE':>8}  {'PROCESS':>8}  {'SBOX':>4}  {'FILES':>5}"
    )
    print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*8}  {'─'*8}  {'─'*8}  {'─'*4}  {'─'*5}")
    total_elapsed = 0
    total_queue = 0.0
    total_process = 0.0
    passed = 0
    for r in results:
        verdict = "PASS" if r["status"] == "completed" and r["files_generated"] > 0 else "FAIL"
        if verdict == "PASS":
            passed += 1
        total_elapsed += r["elapsed_seconds"]
        total_queue += r.get("queue_wait_seconds", 0.0)
        total_process += r.get("processing_seconds", r["elapsed_seconds"])
        print(
            f"  {r['question_id']:<20} {r['difficulty']:<8} {verdict:<10} "
            f"{r['elapsed_seconds']:>7.1f}s  "
            f"{r.get('queue_wait_seconds', 0.0):>7.1f}s  "
            f"{r.get('processing_seconds', r['elapsed_seconds']):>7.1f}s  "
            f"{r['sandbox_loop_count']:>4}  "
            f"{r['files_generated']:>5}"
        )
    n = len(results)
    print(f"{'─' * 86}")
    print(
        f"  {passed}/{n} passed   total: {total_elapsed:.1f}s   "
        f"queue: {total_queue:.1f}s ({100*total_queue/total_elapsed:.1f}%)   "
        f"process: {total_process:.1f}s ({100*total_process/total_elapsed:.1f}%)   "
        f"avg: {total_elapsed/n:.1f}s"
    )
    print(f"{'═' * 86}\n")


def print_diagnostics(results: list[dict]):
    """Print a per-node token/time summary from the captured diagnostics."""
    if not results:
        return
    print(f"\n{'#' * 70}")
    print(f"  DIAGNOSTIC SUMMARY  ({results[0]['run_id']})")
    print(f"{'#' * 70}")

    failure_modes: dict[str, int] = {}
    for r in results:
        status = r.get("status")
        if status == "infra_exhausted":
            failure_modes["infra_exhausted"] = failure_modes.get("infra_exhausted", 0) + 1
        elif status != "completed":
            err = r.get("error") or "unknown"
            if "timed out" in err.lower() or "deadline" in err.lower():
                key = "timeout"
            elif "LLM for node" in err:
                key = f"llm_fault:{err.split('node')[1].split()[0] if 'node' in err else 'unknown'}"
            else:
                key = err[:60]
            failure_modes[key] = failure_modes.get(key, 0) + 1
    if failure_modes:
        print("\n  Failure mode counts:")
        for k, v in sorted(failure_modes.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v}")

    node_totals: dict[str, dict] = {}
    for r in results:
        for entry in r.get("llm_usage", []):
            node = entry["node"]
            if node not in node_totals:
                node_totals[node] = {
                    "calls": 0,
                    "errors": 0,
                    "duration": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                }
            node_totals[node]["calls"] += 1
            if entry.get("status") != "success":
                node_totals[node]["errors"] += 1
            node_totals[node]["duration"] += entry.get("duration_seconds", 0) or 0
            node_totals[node]["input_tokens"] += entry.get("input_tokens", 0) or 0
            node_totals[node]["output_tokens"] += entry.get("output_tokens", 0) or 0
            node_totals[node]["total_tokens"] += entry.get("total_tokens", 0) or 0

    if node_totals:
        print(
            f"\n  {'NODE':<26} {'CALLS':>5} {'ERRORS':>6} {'TIME':>8} "
            f"{'IN_TOK':>8} {'OUT_TOK':>8} {'TOT_TOK':>8}"
        )
        print(
            f"  {'─'*26} {'─'*5} {'─'*6} {'─'*8} "
            f"{'─'*8} {'─'*8} {'─'*8}"
        )
        for node, t in sorted(node_totals.items()):
            print(
                f"  {node:<26} {t['calls']:>5} {t['errors']:>6} {t['duration']:>7.1f}s"
                f"  {t['input_tokens']:>8}  {t['output_tokens']:>8}  {t['total_tokens']:>8}"
            )

    docker_total = 0.0
    docker_count = 0
    for r in results:
        for run in r.get("docker_runs", []):
            val = run.get("duration_seconds")
            if isinstance(val, (int, float)):
                docker_total += val
                docker_count += 1
    if docker_count:
        print(
            f"\n  Docker runs: {docker_count}   "
            f"total sandbox time: {docker_total:.1f}s   "
            f"avg: {docker_total/docker_count:.1f}s"
        )

    fault_counts: dict[str, int] = {}
    for r in results:
        for ch in r.get("classifier_history", []):
            fault = ch.get("fault") or "unknown"
            fault_counts[fault] = fault_counts.get(fault, 0) + 1
    if fault_counts:
        print(
            f"\n  Classifier routes: "
            f"{', '.join(f'{k}: {v}' for k, v in sorted(fault_counts.items()))}"
        )

    print(f"{'#' * 70}\n")


def print_historical_summary():
    if not RESULTS_FILE.exists():
        print("No results yet.")
        return

    runs: dict[str, list] = {}
    with open(RESULTS_FILE) as f:
        for line in f:
            r = json.loads(line.strip())
            runs.setdefault(r["run_id"], []).append(r)

    for run_id, results in sorted(runs.items()):
        ts = results[0]["start_time"][:19].replace("T", " ")
        passed = sum(1 for r in results if r["status"] == "completed" and r["files_generated"] > 0)
        total = len(results)
        avg_time = sum(r["elapsed_seconds"] for r in results) / total
        label = results[0].get("label") or ""
        label_str = f"  [{label}]" if label else ""
        print(
            f"  {run_id}  {ts}  {passed}/{total} passed  "
            f"avg {avg_time:.1f}s{label_str}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global BASE_URL, PER_Q_TIMEOUT
    parser = argparse.ArgumentParser(description="Coding Module benchmark runner (Tier 2)")
    parser.add_argument("--ids", nargs="+", help="Run only these question IDs")
    parser.add_argument("--summary", action="store_true", help="Print history of past runs and exit")
    parser.add_argument("--url", default=None, help="Override API base URL")
    parser.add_argument("--per-q-timeout", type=int, default=None,
                        help=f"Per-job timeout in seconds, measured from worker "
                             f"start (default {PER_Q_TIMEOUT})")
    parser.add_argument("--label", default="",
                        help="Short tag stored on every result row")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Print token/time diagnostic summary after the run")
    parser.add_argument("--sequential", action="store_true",
                        help="Run questions one at a time instead of concurrently")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Maximum jobs submitted in one batch (default {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--deps-cache-tag", default=None,
                        help="Persist installed deps under a shared tag for project-long work")
    parser.add_argument("--clear-deps-cache", action="store_true",
                        help="Wipe the shared .deps_cache directory and exit")
    args = parser.parse_args()

    if args.clear_deps_cache:
        # The tier-2 runner talks to the dev API on 8001 by default; the cache
        # sits alongside the workspace volume in the project root.
        cache_root = Path(__file__).resolve().parent.parent / ".deps_cache"
        if cache_root.exists():
            shutil.rmtree(cache_root)
            print(f"Cleared dependency cache: {cache_root}")
        else:
            print(f"No dependency cache to clear: {cache_root}")
        return

    if args.url:
        BASE_URL = args.url

    if args.per_q_timeout is not None:
        PER_Q_TIMEOUT = args.per_q_timeout

    if args.deps_cache_tag:
        set_deps_cache_tag(args.deps_cache_tag)

    if args.summary:
        print_historical_summary()
        return

    questions = load_questions(args.ids)
    if not questions:
        print("No questions matched.")
        sys.exit(1)

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.sequential:
        results = run_questions_sequential(questions, run_id, PER_Q_TIMEOUT)
    else:
        results = run_questions_concurrent(
            questions, run_id,
            batch_size=args.batch_size,
            per_q_timeout=PER_Q_TIMEOUT,
        )

    for r in results:
        r["label"] = args.label
        append_result(r)

    print_summary(results)
    if args.diagnostics:
        print_diagnostics(results)


if __name__ == "__main__":
    main()