#!/usr/bin/env python3
"""Benchmark runner for the Editing Module (100-question suite, fixture-based).

Loads static fixture workspaces from benchmarks/fixtures/<source>/ and submits
editing jobs with inline source_files — fully decoupled from the coding module.

Usage:
    python benchmarks/runner.py                          # run all 100 edits
    python benchmarks/runner.py --ids fibonacci_docstring stack_docstring
    python benchmarks/runner.py --category documentation  # run one category
    python benchmarks/runner.py --batch-size 4            # cap concurrent submissions
    python benchmarks/runner.py --per-q-timeout 1800      # per-edit timeout
    python benchmarks/runner.py --summary                  # print history
    python benchmarks/runner.py --diagnostics             # token/time breakdown

Results are appended to benchmarks/results.jsonl (one row per edit).
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

BASE_URL = os.environ.get("MUPIN_BACKBONE_URL", os.environ.get("EDIT_MODULE_URL", "http://localhost:8001"))
QUESTIONS_FILE = Path(__file__).parent / "questions_100.json"
FIXTURES_DIR = Path(__file__).parent / "fixtures"
RESULTS_FILE = Path(__file__).parent / "results.jsonl"
METRICS_FILE = Path(__file__).parent / "metrics.jsonl"

POLL_INTERVAL = 5
PER_Q_TIMEOUT = 1800
QUEUE_GRACE = 600
DEFAULT_BATCH_SIZE = 6

FIXTURE_FILES = {"src/main.py", "src/__init__.py", "tests/test_main.py", "tests/__init__.py"}


def load_questions(ids=None, category=None):
    with open(QUESTIONS_FILE) as f:
        questions = json.load(f)
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    if category:
        questions = [q for q in questions if q["category"] == category]
    return questions


def load_fixture_files(source: str) -> dict[str, str]:
    fixture_path = FIXTURES_DIR / source
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")
    files = {}
    for relpath in FIXTURE_FILES:
        filepath = fixture_path / relpath
        if filepath.exists():
            files[relpath] = filepath.read_text(encoding="utf-8")
    if "src/main.py" not in files:
        raise FileNotFoundError(f"Fixture {source} missing src/main.py")
    return files


def submit_edit_job(source_files: dict, instruction: str) -> str:
    resp = requests.post(
        f"{BASE_URL}/jobs",
        json={
            "job_type": "editing",
            "payload": {
                "source_files": source_files,
                "instruction": instruction,
                "profile_name": "python",
            },
        },
        timeout=30,
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


def _wait_for_job(job_id: str, qid: str, per_q_timeout: int) -> dict:
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
            print(f"  [{qid}] poll_error: {e} — retrying in {POLL_INTERVAL}s")
            time.sleep(POLL_INTERVAL)
            continue

        status = data.get("status")
        node = data.get("current_node")

        if deadline is None and data.get("started_at"):
            started_dt = _started_at_from_job(data)
            if started_dt:
                deadline = started_dt.timestamp() + per_q_timeout

        if node != last_node:
            last_node = node
            print(f"  [{qid}] {time.time() - submit_time:>7.1f}s -> {node}  (sbox={data.get('sandbox_loop_count', 0)})")

        if status in ("completed", "failed", "cancelled", "exhausted", "infra_exhausted"):
            started_at = _started_at_from_job(data)
            queue_wait = round(max(0.0, started_at.timestamp() - submit_time), 1) if started_at else 0.0
            process = round(max(0.0, time.time() - (started_at.timestamp() if started_at else submit_time)), 1)
            print(
                f"\n  [{qid}] {status.upper()} in {round(time.time() - submit_time, 1)}s  |  "
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


def run_edit_benchmark(
    questions: list[dict],
    run_id: str,
    batch_size: int,
    per_q_timeout: int,
) -> list[dict]:
    run_start = time.time()
    print(f"\nEditing benchmark run: {run_id}")
    print(f"  Questions:    {len(questions)}")
    print(f"  API:          {BASE_URL}")
    print(f"  Concurrency:  {batch_size} (slot-based)")
    print(f"  Per-job cap:  {per_q_timeout}s (from worker start)")
    print(f"  Fixtures:     {FIXTURES_DIR}")

    # Pre-load all fixture files to avoid repeated disk reads.
    fixture_cache: dict[str, dict[str, str]] = {}
    for q in questions:
        src = q["source"]
        if src not in fixture_cache:
            fixture_cache[src] = load_fixture_files(src)
    print(f"  Loaded {len(fixture_cache)} fixtures")

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
            src = question["source"]
            source_files = fixture_cache.get(src, {})

            if not source_files:
                result = {
                    "run_id": run_id,
                    "question_id": qid,
                    "source": src,
                    "category": question["category"],
                    "difficulty": question["difficulty"],
                    "instruction": question["instruction"],
                    "edit_job_id": None,
                    "status": "fixture_missing",
                    "error": f"Fixture not found for {src}",
                    "start_time": datetime.now(timezone.utc).isoformat(),
                    "end_time": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": 0.0,
                    "queue_wait_seconds": 0.0,
                    "processing_seconds": 0.0,
                    "sandbox_loop_count": 0,
                    "diff_present": False,
                    "node_history": [],
                    "llm_usage": [],
                    "docker_runs": [],
                }
            else:
                print(f"\n  [{qid}] source={src} category={question['category']}")
                edit_id = submit_edit_job(source_files, question["instruction"])
                print(f"  [{qid}] edit submitted {edit_id}")
                edit_result = _wait_for_job(edit_id, qid, per_q_timeout)
                result = {
                    "run_id": run_id,
                    "question_id": qid,
                    "source": src,
                    "category": question["category"],
                    "difficulty": question["difficulty"],
                    "instruction": question["instruction"],
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
    print(f"\n{'═' * 100}")
    print(f"  EDIT RUN SUMMARY  ({results[0]['run_id']})")
    print(f"{'═' * 100}")
    print(
        f"  {'ID':<28} {'CAT':<14} {'STATUS':<10} {'TOTAL':>8}  "
        f"{'QUEUE':>8}  {'PROCESS':>8}  {'SBOX':>4}  {'DIFF':>5}"
    )
    print(f"  {'─'*28} {'─'*14} {'─'*10} {'─'*8}  {'─'*8}  {'─'*8}  {'─'*4}  {'─'*5}")
    total_elapsed = 0
    total_queue = 0.0
    total_process = 0.0
    passed = 0
    diff_count = 0
    for r in results:
        verdict = "PASS" if r["status"] == "completed" else "FAIL"
        if verdict == "PASS":
            passed += 1
        if r.get("diff_present"):
            diff_count += 1
        total_elapsed += r["elapsed_seconds"]
        total_queue += r.get("queue_wait_seconds", 0.0)
        total_process += r.get("processing_seconds", r["elapsed_seconds"])
        print(
            f"  {r['question_id']:<28} {r['category']:<14} {verdict:<10} "
            f"{r['elapsed_seconds']:>7.1f}s  "
            f"{r.get('queue_wait_seconds', 0.0):>7.1f}s  "
            f"{r.get('processing_seconds', r['elapsed_seconds']):>7.1f}s  "
            f"{r['sandbox_loop_count']:>4}  "
            f"{'yes' if r.get('diff_present') else 'no':>5}"
        )
    n = len(results)
    print(f"{'─' * 100}")
    print(
        f"  {passed}/{n} passed   diff_present: {diff_count}/{n}   "
        f"total: {total_elapsed:.1f}s   "
        f"queue: {total_queue:.1f}s ({100*total_queue/total_elapsed:.1f}%)   "
        f"process: {total_process:.1f}s ({100*total_process/total_elapsed:.1f}%)   "
        f"avg: {total_elapsed/n:.1f}s"
    )

    # Per-category breakdown
    from collections import defaultdict
    cat_stats = defaultdict(lambda: {"passed": 0, "total": 0, "time": 0.0})
    for r in results:
        cat = r["category"]
        cat_stats[cat]["total"] += 1
        cat_stats[cat]["time"] += r["elapsed_seconds"]
        if r["status"] == "completed":
            cat_stats[cat]["passed"] += 1

    print(f"\n  Per-category:")
    print(f"  {'CATEGORY':<14} {'PASS':>8} {'AVG_TIME':>8}")
    print(f"  {'─'*14} {'─'*8} {'─'*8}")
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        avg = s["time"] / s["total"] if s["total"] else 0
        print(f"  {cat:<14} {s['passed']}/{s['total']:<7} {avg:>7.1f}s")

    print(f"{'═' * 100}\n")


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
                node_totals[node] = {"calls": 0, "errors": 0, "duration": 0.0,
                                     "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
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
        print(f"  {'─'*26} {'─'*5} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
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
        print(f"\n  Docker runs: {docker_count}   total sandbox time: {docker_total:.1f}s   avg: {docker_total/docker_count:.1f}s")

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
        print(f"  {run_id}  {ts}  {passed}/{total} passed  avg {avg_time:.1f}s{label_str}")


def main():
    global BASE_URL, PER_Q_TIMEOUT
    parser = argparse.ArgumentParser(description="Editing Module 100-question benchmark runner")
    parser.add_argument("--ids", nargs="+", help="Run only these question IDs")
    parser.add_argument("--category", default=None, help="Run only one category (documentation, robustness, feature, refactor, behavior)")
    parser.add_argument("--summary", action="store_true", help="Print history of past runs and exit")
    parser.add_argument("--url", default=None, help="Override API base URL")
    parser.add_argument("--per-q-timeout", type=int, default=None, help=f"Per-edit timeout (default {PER_Q_TIMEOUT})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help=f"Max concurrent edit jobs (default {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--diagnostics", action="store_true", help="Print token/time diagnostic summary")
    parser.add_argument("--label", default="", help="Short tag stored on every result row")
    args = parser.parse_args()

    if args.url:
        BASE_URL = args.url
    if args.per_q_timeout is not None:
        PER_Q_TIMEOUT = args.per_q_timeout

    if args.summary:
        print_historical_summary()
        return

    questions = load_questions(args.ids, args.category)
    if not questions:
        print("No questions matched.")
        sys.exit(1)

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results = run_edit_benchmark(
        questions,
        run_id,
        batch_size=args.batch_size,
        per_q_timeout=PER_Q_TIMEOUT,
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