#!/usr/bin/env python3
"""Benchmark runner for the Editing Module (20-question suite).

Two-phase execution:
  1. Generate a base coding workspace for every source_id (or reuse cached base jobs).
  2. Submit an editing job against each base workspace with the specified instruction.

Usage:
    python benchmarks/runner.py                          # run all 20 edits
    python benchmarks/runner.py --ids fibonacci_docstring stack_type_hints
    python benchmarks/runner.py --reuse-base             # reuse already-generated base jobs from previous run
    python benchmarks/runner.py --batch-size 4           # cap concurrent submissions
    python benchmarks/runner.py --base-batch-size 6      # cap concurrent base coding jobs
    python benchmarks/runner.py --per-q-timeout 1800     # per-edit timeout
    python benchmarks/runner.py --summary                # print history

Results are appended to benchmarks/results.jsonl (one row per edit).
"""

import argparse
import json
import os
import sys
import time
import threading
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = os.environ.get("MUPIN_BACKBONE_URL", os.environ.get("EDIT_MODULE_URL", "http://localhost:8001"))
CODING_QUESTIONS_FILE = Path(__file__).parent.parent.parent / "mupin-coding-module" / "benchmarks" / "questions.json"
EDIT_QUESTIONS_FILE = Path(__file__).parent / "questions.json"
RESULTS_FILE = Path(__file__).parent / "results.jsonl"
METRICS_FILE = Path(__file__).parent / "metrics.jsonl"
BASE_JOBS_FILE = Path(__file__).parent / "base_jobs.json"

POLL_INTERVAL = 5
PER_Q_TIMEOUT = 1800      # per edit job, measured from worker start
BASE_PER_Q_TIMEOUT = 3600  # per base coding job
QUEUE_GRACE = 600
DEFAULT_BATCH_SIZE = 6
DEFAULT_BASE_BATCH_SIZE = 6


def load_edit_questions(ids=None):
    with open(EDIT_QUESTIONS_FILE) as f:
        questions = json.load(f)
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    return questions


def load_coding_questions():
    with open(CODING_QUESTIONS_FILE) as f:
        return json.load(f)


def submit_coding_job(prompt: str) -> str:
    resp = requests.post(
        f"{BASE_URL}/jobs",
        json={"job_type": "coding", "payload": {"prompt": prompt, "profile_name": "python"}},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["job_id"]


def submit_edit_job(source_job_id: str, instruction: str) -> str:
    resp = requests.post(
        f"{BASE_URL}/jobs",
        json={"job_type": "editing", "payload": {"source_job_id": source_job_id, "instruction": instruction, "profile_name": "python"}},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["job_id"]


def poll_job(job_id: str) -> dict:
    resp = requests.get(f"{BASE_URL}/jobs/{job_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def cancel_job(job_id: str) -> None:
    try:
        requests.post(f"{BASE_URL}/jobs/{job_id}/cancel", timeout=10)
    except Exception as e:
        print(f"  CANCEL ERROR: {e}")


def _started_at_from_job(data: dict) -> datetime | None:
    started = data.get("started_at")
    if started:
        try:
            return datetime.fromisoformat(started.replace("Z", "+00:00"))
        except Exception:
            pass
    return None


def _wait_for_job(job_id: str, qid: str, per_q_timeout: int, prefix: str) -> dict:
    submit_time = time.time()
    submit_iso = datetime.now(timezone.utc).isoformat()
    deadline = None
    submit_deadline = submit_time + QUEUE_GRACE
    last_data = {}
    last_node = None

    while True:
        if deadline is None and time.time() > submit_deadline:
            cancel_job(job_id)
            return {
                "task_id": job_id,
                "status": "queue_timeout",
                "start_time": submit_iso,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.time() - submit_time, 1),
                "error": f"Worker did not start within {QUEUE_GRACE}s",
            }

        if deadline is not None and time.time() > deadline:
            cancel_job(job_id)
            return {
                "task_id": job_id,
                "status": "timeout",
                "start_time": submit_iso,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.time() - submit_time, 1),
                "error": f"Timed out after {per_q_timeout}s of worker time",
            }

        try:
            data = poll_job(job_id)
            last_data = data
        except Exception as e:
            print(f"  [{prefix}{qid}] poll_error: {e} — retrying in {POLL_INTERVAL}s")
            time.sleep(POLL_INTERVAL)
            continue

        status = data.get("status")
        node = data.get("current_node")

        if deadline is None and data.get("started_at"):
            try:
                started_dt = _started_at_from_job(data)
                if started_dt:
                    deadline = started_dt.timestamp() + per_q_timeout
            except Exception:
                deadline = submit_time + per_q_timeout

        if node != last_node:
            last_node = node
            print(f"  [{prefix}{qid}] {time.time() - submit_time:>7.1f}s -> {node}  (sbox={data.get('sandbox_loop_count', 0)})")

        if status in ("completed", "failed", "cancelled", "exhausted", "infra_exhausted"):
            started_at = _started_at_from_job(data)
            queue_wait = round(max(0.0, started_at.timestamp() - submit_time), 1) if started_at else 0.0
            process = round(max(0.0, time.time() - (started_at.timestamp() if started_at else submit_time)), 1)
            print(
                f"\n  [{prefix}{qid}] {status.upper()} in {round(time.time() - submit_time, 1)}s  |  "
                f"queue={queue_wait:.1f}s  process={process:.1f}s  |  "
                f"sbox={data.get('sandbox_loop_count', 0)}"
            )
            return {
                "task_id": job_id,
                "status": status,
                "start_time": submit_iso,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.time() - submit_time, 1),
                "queue_wait_seconds": queue_wait,
                "processing_seconds": process,
                "sandbox_loop_count": data.get("sandbox_loop_count", 0),
                "error": data.get("error"),
                "result": data.get("result"),
                "progress": data.get("progress") or {},
                "node_history": data.get("node_history", []),
                "llm_usage": data.get("llm_usage", []),
                "docker_runs": data.get("docker_runs", []),
            }

        time.sleep(POLL_INTERVAL)


def _generate_base_jobs(questions: list[dict], batch_size: int, per_q_timeout: int) -> dict[str, str]:
    """Generate one base coding job per unique source_id. Returns source_id->job_id."""
    coding_by_id = {q["id"]: q for q in load_coding_questions()}
    source_ids = sorted({q["source_id"] for q in questions})
    base_jobs: dict[str, str] = {}
    lock = threading.Lock()

    def worker_loop():
        while True:
            with lock:
                try:
                    sid = next(iter_source_ids)
                except StopIteration:
                    return
            cq = coding_by_id.get(sid)
            if not cq:
                print(f"  [base {sid}] missing source coding question")
                with lock:
                    base_jobs[sid] = None
                return
            job_id = submit_coding_job(cq["prompt"])
            print(f"  [base {sid}] submitted {job_id}")
            result = _wait_for_job(job_id, sid, per_q_timeout, "base ")
            with lock:
                base_jobs[sid] = job_id if result["status"] == "completed" else None

    iter_source_ids = iter(source_ids)
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(worker_loop) for _ in range(batch_size)]
        for f in as_completed(futures):
            f.result()

    return base_jobs


def _load_cached_base_jobs() -> dict[str, str]:
    if BASE_JOBS_FILE.exists():
        with open(BASE_JOBS_FILE) as f:
            return json.load(f)
    return {}


def _save_base_jobs(base_jobs: dict[str, str]) -> None:
    with open(BASE_JOBS_FILE, "w") as f:
        json.dump(base_jobs, f, indent=2)


def run_edit_benchmark(
    questions: list[dict],
    run_id: str,
    batch_size: int,
    base_batch_size: int,
    per_q_timeout: int,
    base_per_q_timeout: int,
    reuse_base: bool,
) -> list[dict]:
    run_start = time.time()
    print(f"\nEditing benchmark run: {run_id}")
    print(f"  Edit questions:   {len(questions)}")
    print(f"  API:              {BASE_URL}")
    print(f"  Edit concurrency: {batch_size} (slot-based)")
    print(f"  Base concurrency: {base_batch_size} (slot-based)")

    # Phase 1: ensure base jobs exist.
    if reuse_base:
        base_jobs = _load_cached_base_jobs()
        print(f"  Reusing {len(base_jobs)} cached base jobs")
    else:
        base_jobs = _generate_base_jobs(questions, base_batch_size, base_per_q_timeout)
        _save_base_jobs(base_jobs)
        print(f"  Generated base jobs for {len(base_jobs)} source questions")

    missing_base = sorted({q["source_id"] for q in questions} - set(base_jobs.keys()))
    if missing_base:
        print(f"  Missing base jobs for: {missing_base}")
        sys.exit(1)

    failed_base = [sid for sid, jid in base_jobs.items() if jid is None]
    if failed_base:
        print(f"  Base generation failed for: {failed_base}")
        # Allow continuing; edits for these will fail fast.

    # Phase 2: submit edits.
    results: list[dict] = []
    results_lock = threading.Lock()
    done_count = [0]
    counter_lock = threading.Lock()

    def worker_loop():
        while True:
            with counter_lock:
                try:
                    question = next(iter_questions)
                except StopIteration:
                    return
            qid = question["id"]
            sid = question["source_id"]
            base_id = base_jobs.get(sid)
            if not base_id:
                result = {
                    "run_id": run_id,
                    "question_id": qid,
                    "source_id": sid,
                    "difficulty": question["difficulty"],
                    "instruction": question["instruction"],
                    "base_job_id": base_id,
                    "edit_job_id": None,
                    "status": "base_failed",
                    "error": "Base coding job failed or missing",
                    "start_time": datetime.now(timezone.utc).isoformat(),
                    "end_time": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": 0.0,
                    "sandbox_loop_count": 0,
                    "diff_present": False,
                }
            else:
                print(f"\n  [{qid}] source={sid} base={base_id}")
                edit_id = submit_edit_job(base_id, question["instruction"])
                print(f"  [{qid}] edit submitted {edit_id}")
                edit_result = _wait_for_job(edit_id, qid, per_q_timeout, "")
                result = {
                    "run_id": run_id,
                    "question_id": qid,
                    "source_id": sid,
                    "difficulty": question["difficulty"],
                    "instruction": question["instruction"],
                    "base_job_id": base_id,
                    "edit_job_id": edit_id,
                    "status": edit_result["status"],
                    "error": edit_result.get("error"),
                    "start_time": edit_result["start_time"],
                    "end_time": edit_result["end_time"],
                    "elapsed_seconds": edit_result["elapsed_seconds"],
                    "queue_wait_seconds": edit_result.get("queue_wait_seconds", 0.0),
                    "processing_seconds": edit_result.get("processing_seconds", edit_result["elapsed_seconds"]),
                    "sandbox_loop_count": edit_result.get("sandbox_loop_count", 0),
                    "diff_present": bool(
                        (edit_result.get("result") or {}).get("diff", "").strip()
                    ),
                    "node_history": edit_result.get("node_history", []),
                    "llm_usage": edit_result.get("llm_usage", []),
                    "docker_runs": edit_result.get("docker_runs", []),
                }

            with results_lock:
                results.append(result)
                done_count[0] += 1
                idx = done_count[0]
            print(f"  ---- [{idx}/{len(questions)}] edit done ----")

    iter_questions = iter(questions)
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(worker_loop) for _ in range(batch_size)]
        for f in as_completed(futures):
            f.result()

    print(f"\n  Total wall-clock time: {time.time() - run_start:.1f}s")

    results_by_id = {r["question_id"]: r for r in results}
    return [results_by_id[q["id"]] for q in questions if q["id"] in results_by_id]


def append_result(result: dict) -> None:
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


def append_metrics(metrics: list[dict]) -> None:
    if not metrics:
        return
    with open(METRICS_FILE, "a") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")


def _metrics_rows(run_id: str, qid: str, result: dict) -> list[dict]:
    rows = []
    for entry in result.get("node_history", []):
        rows.append({"run_id": run_id, "question_id": qid, "metric_type": "node", "metric": entry})
    for entry in result.get("llm_usage", []):
        rows.append({"run_id": run_id, "question_id": qid, "metric_type": "llm", "metric": entry})
    for entry in result.get("docker_runs", []):
        rows.append({"run_id": run_id, "question_id": qid, "metric_type": "docker", "metric": entry})
    return rows


def print_summary(results: list[dict]):
    if not results:
        return
    print(f"\n{'═' * 90}")
    print(f"  EDIT RUN SUMMARY  ({results[0]['run_id']})")
    print(f"{'═' * 90}")
    print(
        f"  {'ID':<28} {'DIFF':<8} {'STATUS':<10} {'TOTAL':>8}  "
        f"{'QUEUE':>8}  {'PROCESS':>8}  {'SBOX':>4}  {'DIFF':>5}"
    )
    print(f"  {'─'*28} {'─'*8} {'─'*10} {'─'*8}  {'─'*8}  {'─'*8}  {'─'*4}  {'─'*5}")
    total_elapsed = 0
    total_queue = 0.0
    total_process = 0.0
    passed = 0
    for r in results:
        verdict = "PASS" if r["status"] == "completed" else "FAIL"
        if verdict == "PASS":
            passed += 1
        total_elapsed += r["elapsed_seconds"]
        total_queue += r.get("queue_wait_seconds", 0.0)
        total_process += r.get("processing_seconds", r["elapsed_seconds"])
        print(
            f"  {r['question_id']:<28} {r['difficulty']:<8} {verdict:<10} "
            f"{r['elapsed_seconds']:>7.1f}s  "
            f"{r.get('queue_wait_seconds', 0.0):>7.1f}s  "
            f"{r.get('processing_seconds', r['elapsed_seconds']):>7.1f}s  "
            f"{r['sandbox_loop_count']:>4}  "
            f"{'yes' if r.get('diff_present') else 'no':>5}"
        )
    n = len(results)
    print(f"{'─' * 90}")
    print(
        f"  {passed}/{n} passed   total: {total_elapsed:.1f}s   "
        f"queue: {total_queue:.1f}s ({100*total_queue/total_elapsed:.1f}%)   "
        f"process: {total_process:.1f}s ({100*total_process/total_elapsed:.1f}%)   "
        f"avg: {total_elapsed/n:.1f}s"
    )
    print(f"{'═' * 90}\n")


def print_diagnostics(results: list[dict]):
    if not results:
        return
    print(f"\n{'#' * 70}")
    print(f"  EDIT DIAGNOSTIC SUMMARY  ({results[0]['run_id']})")
    print(f"{'#' * 70}")

    failure_modes: dict[str, int] = {}
    for r in results:
        status = r.get("status")
        if status != "completed":
            err = r.get("error") or "unknown"
            if "timed out" in err.lower() or "deadline" in err.lower():
                key = "timeout"
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
                node_totals[node] = {"calls": 0, "errors": 0, "duration": 0.0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
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
        passed = sum(1 for r in results if r["status"] == "completed")
        total = len(results)
        avg_time = sum(r["elapsed_seconds"] for r in results) / total
        label = results[0].get("label") or ""
        label_str = f"  [{label}]" if label else ""
        print(
            f"  {run_id}  {ts}  {passed}/{total} passed  "
            f"avg {avg_time:.1f}s{label_str}"
        )


def main():
    global BASE_URL, PER_Q_TIMEOUT, BASE_PER_Q_TIMEOUT
    parser = argparse.ArgumentParser(description="Editing Module benchmark runner")
    parser.add_argument("--ids", nargs="+", help="Run only these edit question IDs")
    parser.add_argument("--summary", action="store_true", help="Print history of past runs and exit")
    parser.add_argument("--url", default=None, help="Override API base URL")
    parser.add_argument("--per-q-timeout", type=int, default=None, help=f"Per-edit timeout (default {PER_Q_TIMEOUT})")
    parser.add_argument("--base-per-q-timeout", type=int, default=None, help=f"Per-base-coding timeout (default {BASE_PER_Q_TIMEOUT})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Max concurrent edit jobs (default {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--base-batch-size", type=int, default=DEFAULT_BASE_BATCH_SIZE, help=f"Max concurrent base coding jobs (default {DEFAULT_BASE_BATCH_SIZE})")
    parser.add_argument("--reuse-base", action="store_true", help="Reuse cached base job IDs instead of regenerating them")
    parser.add_argument("--diagnostics", action="store_true", help="Print token/time diagnostic summary")
    parser.add_argument("--label", default="", help="Short tag stored on every result row")
    args = parser.parse_args()

    if args.url:
        BASE_URL = args.url
    if args.per_q_timeout is not None:
        PER_Q_TIMEOUT = args.per_q_timeout
    if args.base_per_q_timeout is not None:
        BASE_PER_Q_TIMEOUT = args.base_per_q_timeout

    if args.summary:
        print_historical_summary()
        return

    questions = load_edit_questions(args.ids)
    if not questions:
        print("No edit questions matched.")
        sys.exit(1)

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = run_edit_benchmark(
        questions,
        run_id,
        batch_size=args.batch_size,
        base_batch_size=args.base_batch_size,
        per_q_timeout=PER_Q_TIMEOUT,
        base_per_q_timeout=BASE_PER_Q_TIMEOUT,
        reuse_base=args.reuse_base,
    )

    for r in results:
        r["label"] = args.label
        append_result(r)
        append_metrics(_metrics_rows(run_id, r["question_id"], r))

    print_summary(results)
    if args.diagnostics:
        print_diagnostics(results)


if __name__ == "__main__":
    main()
