#!/usr/bin/env python3
"""
Benchmark runner for the Coding Module.

Usage:
    python benchmarks/runner.py                        # run all questions concurrently
    python benchmarks/runner.py --ids fibonacci stack  # run specific questions
    python benchmarks/runner.py --sequential           # run one question at a time
    python benchmarks/runner.py --batch-size 20        # cap concurrent submissions
    python benchmarks/runner.py --summary              # print history of past runs

Results are appended to benchmarks/results.jsonl — one JSON line per question.
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

POLL_INTERVAL = 5   # seconds between status polls
TIMEOUT = 2400      # seconds before we give up on a single task (40 min)
                    # Expert/hard questions may override this via questions.json
DEFAULT_BATCH_SIZE = 20  # per SESSION_NOTES v0.3 decision

# Total-run timeout: protects long benchmark suites from running forever.
# Each question still has its own per-question TIMEOUT above.
TOTAL_TIMEOUT = int(os.environ.get("MUPIN_TOTAL_TIMEOUT", "3600"))  # 1 hour default
TOTAL_TIMEOUT_ACTION = os.environ.get("MUPIN_TOTAL_TIMEOUT_ACTION", "cancel")  # 'cancel' or 'finish_in_progress'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_questions(ids=None):
    with open(QUESTIONS_FILE) as f:
        questions = json.load(f)
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    return questions


def submit_task(prompt: str) -> str:
    resp = requests.post(
        f"{BASE_URL}/jobs",
        json={"job_type": "coding", "payload": {"prompt": prompt, "profile_name": "python"}},
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
    normalized.setdefault("compliance_loop_count", progress.get("compliance_loop_count", 0))
    normalized.setdefault("compliance_status", progress.get("compliance_status"))
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
            return datetime.fromisoformat(started)
        except Exception:
            pass
    for entry in data.get("node_history", []) or []:
        started = entry.get("started_at")
        if started:
            try:
                return datetime.fromisoformat(started)
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
        "compliance_loop_count": data.get("compliance_loop_count", 0),
        "files_generated": len(data.get("result") or {}),
        "thoughts_count": len(data.get("thoughts", [])),
        "compliance_status": data.get("compliance_status"),
        **diagnostics,
    }
    append_metrics(_metrics_rows(run_id, question["id"], result, diagnostics))
    return result


def run_question_sequential(
    question: dict,
    run_id: str,
    run_start: float,
    total_timeout: int = TOTAL_TIMEOUT,
    total_timeout_action: str = TOTAL_TIMEOUT_ACTION,
) -> dict:
    """Run a single question end-to-end, polling until it settles."""
    qid = question["id"]

    # Check total-run budget before even starting this question.
    elapsed_run = time.time() - run_start
    if elapsed_run >= total_timeout:
        print(f"\n  [{qid}] SKIPPED — total run timeout ({total_timeout}s) already exceeded")
        now_iso = datetime.now(timezone.utc).isoformat()
        return _build_result(
            run_id, question, None, run_start, now_iso, "total_timeout",
            error=f"Skipped: total run timeout ({total_timeout}s) exceeded before start",
        )

    print(f"\n{'─' * 60}")
    print(f"  [{qid}]  difficulty={question['difficulty']}")
    print(f"  {question['prompt'][:100]}...")
    print(f"{'─' * 60}")

    start_time = time.time()
    start_iso = datetime.now(timezone.utc).isoformat()

    try:
        task_id = submit_task(question["prompt"])
    except Exception as e:
        print(f"  SUBMIT ERROR: {e}")
        return _build_result(run_id, question, None, start_time, start_iso, "submit_error", error=str(e))

    print(f"  task_id={task_id}")
    last_node = None
    last_data = {}
    per_q_timeout = question.get("timeout_seconds", TIMEOUT)
    deadline = start_time + per_q_timeout
    cancel_grace_deadline: float | None = None

    while time.time() < deadline:
        # Check total-run budget during the question too.
        if time.time() - run_start >= total_timeout:
            if total_timeout_action == "cancel":
                if cancel_grace_deadline is None:
                    cancel_task(task_id)
                    cancel_grace_deadline = time.time() + 60
                    print(f"\n  TOTAL RUN TIMEOUT ({total_timeout}s) — cancelling [{qid}], waiting 60s grace")
                elif time.time() > cancel_grace_deadline:
                    print(f"\n  [{qid}] FORCE total_timeout after cancellation grace period")
                    return _build_result(run_id, question, task_id, start_time, start_iso, "total_timeout", last_data,
                                            error=f"Total run timeout ({total_timeout}s) exceeded; cancellation grace period elapsed")
            else:
                print(f"\n  TOTAL RUN TIMEOUT ({total_timeout}s) — letting [{qid}] finish, but run will stop after it")

        try:
            data = poll_task(task_id)
            last_data = data
        except Exception as e:
            print(f"  POLL ERROR: {e} — retrying in {POLL_INTERVAL}s")
            time.sleep(POLL_INTERVAL)
            continue

        status = data.get("status")
        node = data.get("current_node")
        elapsed = round(time.time() - start_time, 1)

        if node != last_node:
            print(
                f"  {elapsed:>7.1f}s  → {node}  "
                f"(sandbox_loops={data.get('sandbox_loop_count', 0)} "
                f"compliance_loops={data.get('compliance_loop_count', 0)})"
            )
            last_node = node

        if status in ("completed", "failed", "cancelled", "exhausted", "total_timeout"):
            result = _build_result(run_id, question, task_id, start_time, start_iso, status, data)
            verdict = "PASS" if status == "completed" and result["files_generated"] else "FAIL"
            print(
            f"\n  {verdict}  in {result['elapsed_seconds']}s  |  "
            f"queue={result.get('queue_wait_seconds', 0.0):.1f}s  "
            f"process={result.get('processing_seconds', result['elapsed_seconds']):.1f}s  |  "
            f"sbox_loops={result['sandbox_loop_count']}  "
            f"compliance_loops={result['compliance_loop_count']}  "
            f"files={result['files_generated']}"
        )
            return result

        time.sleep(POLL_INTERVAL)

    cancel_task(task_id)
    result = _build_result(run_id, question, task_id, start_time, start_iso, "timeout", last_data,
                            error=f"Timed out after {per_q_timeout}s")
    print(f"\n  TIMEOUT after {result['elapsed_seconds']}s — cancelled "
          f"(per-question limit={per_q_timeout}s, "
          f"last node={last_data.get('current_node')} "
          f"sandbox_loops={last_data.get('sandbox_loop_count', 0)})")
    return result


# ---------------------------------------------------------------------------
# Concurrent execution
# ---------------------------------------------------------------------------

def _submit_all(questions: list[dict], batch_size: int) -> list[tuple[dict, str | None, str | None, float, str]]:
    """Submit questions in batches and return (question, task_id, error, start_time, start_iso)."""
    submitted = []
    print(f"  Submitting {len(questions)} questions in batches of {batch_size}...")

    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(submit_task, q["prompt"]): q
                for q in batch
            }
            for future in as_completed(futures):
                q = futures[future]
                start_time = time.time()
                start_iso = datetime.now(timezone.utc).isoformat()
                try:
                    task_id = future.result()
                    submitted.append((q, task_id, None, start_time, start_iso))
                    print(f"    [{q['id']}] submitted {task_id}")
                except Exception as e:
                    submitted.append((q, None, str(e), start_time, start_iso))
                    print(f"    [{q['id']}] SUBMIT ERROR: {e}")

    return submitted


def run_questions_concurrent(
    questions: list[dict],
    run_id: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    total_timeout: int = TOTAL_TIMEOUT,
    total_timeout_action: str = TOTAL_TIMEOUT_ACTION,
) -> list[dict]:
    """Submit all questions concurrently (in batches), then poll until all settle."""
    run_start = time.time()
    print(f"\nConcurrent benchmark run: {run_id}")
    print(f"  Questions: {len(questions)}")
    print(f"  API:       {BASE_URL}")
    print(f"  Batch:     {batch_size}")
    print(f"  Total cap: {total_timeout}s  (action={total_timeout_action})")

    pending = _submit_all(questions, batch_size)
    results_map: dict[str, dict] = {}
    live_nodes: dict[str, str | None] = {q["id"]: None for q in questions}
    locks = {q["id"]: threading.Lock() for q in questions}
    total_timeout_hit = threading.Event()

    def poll_loop(question: dict, task_id: str | None, submit_error: str | None, q_start: float, q_start_iso: str):
        qid = question["id"]
        if submit_error:
            results_map[qid] = _build_result(run_id, question, None, q_start, q_start_iso,
                                              "submit_error", error=submit_error)
            return

        per_q_timeout = question.get("timeout_seconds", TIMEOUT)
        deadline = q_start + per_q_timeout
        last_data = {}
        cancel_grace_deadline: float | None = None

        while time.time() < deadline:
            # Total-run budget check.
            elapsed_run = time.time() - run_start
            if elapsed_run >= total_timeout:
                if not total_timeout_hit.is_set():
                    total_timeout_hit.set()
                    if total_timeout_action == "cancel":
                        cancel_task(task_id)
                        cancel_grace_deadline = time.time() + 60
                        print(f"\n  TOTAL RUN TIMEOUT ({total_timeout}s) — cancelling [{qid}], waiting 60s grace")
                    else:
                        print(f"\n  TOTAL RUN TIMEOUT ({total_timeout}s) — letting in-progress jobs finish")

                if total_timeout_action == "cancel":
                    if cancel_grace_deadline is not None and time.time() > cancel_grace_deadline:
                        print(f"  [{qid:<20}] FORCE total_timeout after grace period")
                        result = _build_result(run_id, question, task_id, q_start, q_start_iso, "total_timeout", last_data,
                                                error=f"Total run timeout ({total_timeout}s) exceeded; cancellation grace period elapsed")
                        results_map[qid] = result
                        return

            try:
                data = poll_task(task_id)
                last_data = data
            except Exception as e:
                with locks[qid]:
                    live_nodes[qid] = f"poll_error ({e})"
                time.sleep(POLL_INTERVAL)
                continue

            status = data.get("status")
            node = data.get("current_node")

            with locks[qid]:
                if node != live_nodes[qid]:
                    live_nodes[qid] = node
                    print(f"  [{qid:<20}] {time.time() - q_start:>7.1f}s → {node}  "
                          f"(sbox={data.get('sandbox_loop_count', 0)} comp={data.get('compliance_loop_count', 0)})")

            if status in ("completed", "failed", "cancelled", "exhausted", "total_timeout"):
                result = _build_result(run_id, question, task_id, q_start, q_start_iso, status, data)
                verdict = "PASS" if status == "completed" and result["files_generated"] else "FAIL"
                print(f"  [{qid:<20}] {verdict} in {result['elapsed_seconds']}s  |  "
                      f"sbox={result['sandbox_loop_count']} comp={result['compliance_loop_count']} files={result['files_generated']}")
                results_map[qid] = result
                return

            time.sleep(POLL_INTERVAL)

        cancel_task(task_id)
        result = _build_result(run_id, question, task_id, q_start, q_start_iso, "timeout", last_data,
                                error=f"Timed out after {per_q_timeout}s")
        print(f"  [{qid:<20}] TIMEOUT after {result['elapsed_seconds']}s")
        results_map[qid] = result

    with ThreadPoolExecutor(max_workers=len(questions)) as executor:
        futures = [
            executor.submit(poll_loop, q, tid, err, qs, qiso)
            for q, tid, err, qs, qiso in pending
        ]
        for future in as_completed(futures):
            future.result()

    total_elapsed = round(time.time() - run_start, 1)
    print(f"\n  Total wall-clock time: {total_elapsed}s")
    if total_timeout_hit.is_set():
        print(f"  WARNING: total run timeout ({total_timeout}s) fired during this run")
    return [results_map[q["id"]] for q in questions]


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
        f"{'QUEUE':>8}  {'PROCESS':>8}  {'SBOX':>4}  {'COMP':>4}  {'FILES':>5}"
    )
    print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*8}  {'─'*8}  {'─'*8}  {'─'*4}  {'─'*4}  {'─'*5}")
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
            f"{r['compliance_loop_count']:>4}  "
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
    global BASE_URL, TIMEOUT, TOTAL_TIMEOUT, TOTAL_TIMEOUT_ACTION
    parser = argparse.ArgumentParser(description="Coding Module benchmark runner")
    parser.add_argument("--ids", nargs="+", help="Run only these question IDs")
    parser.add_argument("--summary", action="store_true", help="Print history of past runs and exit")
    parser.add_argument("--url", default=None, help="Override API base URL")
    parser.add_argument("--timeout", type=int, default=None,
                        help=f"Per-task timeout in seconds (default {TIMEOUT})")
    parser.add_argument("--label", default="",
                        help="Short tag stored on every result row")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Print token/time diagnostic summary after the run")
    parser.add_argument("--sequential", action="store_true",
                        help="Run questions one at a time instead of concurrently")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Maximum jobs submitted in one batch (default {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--total-timeout", type=int, default=None,
                        help=f"Total run timeout in seconds (default {TOTAL_TIMEOUT})")
    parser.add_argument("--total-timeout-action", choices=["cancel", "finish_in_progress"],
                        default=None,
                        help=f"Action when total timeout fires (default {TOTAL_TIMEOUT_ACTION})")
    args = parser.parse_args()

    if args.url:
        BASE_URL = args.url

    if args.timeout:
        TIMEOUT = args.timeout

    if args.total_timeout is not None:
        TOTAL_TIMEOUT = args.total_timeout

    if args.total_timeout_action is not None:
        TOTAL_TIMEOUT_ACTION = args.total_timeout_action

    if args.summary:
        print_historical_summary()
        return

    questions = load_questions(args.ids)
    if not questions:
        print("No questions matched.")
        sys.exit(1)

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_start = time.time()

    if args.sequential:
        print(f"\nBenchmark run: {run_id}")
        print(f"Questions:     {len(questions)}")
        print(f"API:           {BASE_URL}")
        print("Mode:          sequential")
        print(f"Total cap:     {TOTAL_TIMEOUT}s  (action={TOTAL_TIMEOUT_ACTION})")
        if args.label:
            print(f"Label:         {args.label}")
        results = [run_question_sequential(q, run_id, run_start, TOTAL_TIMEOUT, TOTAL_TIMEOUT_ACTION) for q in questions]
    else:
        results = run_questions_concurrent(
            questions, run_id,
            batch_size=args.batch_size,
            total_timeout=TOTAL_TIMEOUT,
            total_timeout_action=TOTAL_TIMEOUT_ACTION,
        )

    for r in results:
        r["label"] = args.label
        append_result(r)

    print_summary(results)
    if args.diagnostics:
        print_diagnostics(results)


if __name__ == "__main__":
    main()
