#!/usr/bin/env python3
"""
HumanEval+ benchmark runner for the Coding Module.

Submits each HumanEval+ problem prompt through the full Mupin pipeline
(test_designer -> skeleton_maker -> coder -> sandbox) as a
normal coding job, then scores the generated src/main.py against the canonical
HumanEval+ test suite.

Usage:
    python benchmarks/humaneval_runner.py                       # full 164-problem run
    python benchmarks/humaneval_runner.py --limit 10             # first 10 problems (smoke test)
    python benchmarks/humaneval_runner.py --ids HumanEval/0 HumanEval/1  # specific tasks
    python benchmarks/humaneval_runner.py --batch-size 6        # concurrency (default 6)
    python benchmarks/humaneval_runner.py --per-q-timeout 3600  # per-job cap (default 3600s)
    python benchmarks/humaneval_runner.py --summary             # print pass@1 from last run
    python benchmarks/humaneval_runner.py --rescore             # re-score existing completed jobs in place

Results are appended to benchmarks/humaneval_results.jsonl - one JSON line per
problem.  pass@1 = passing / total.

Execution model: slot-based. At most `--batch-size` jobs are in flight at once;
each worker thread loops submit -> poll -> score -> next problem. Queue wait is
therefore near zero, and the per-job timeout (`--per-q-timeout`) measures real
pipeline work, measured from the worker's `started_at` timestamp -- NOT from
submission time. There is no whole-run kill switch.

Requires: pip install evalplus requests
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
RESULTS_FILE = Path(__file__).parent / "humaneval_results.jsonl"
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent / ".workspaces"

POLL_INTERVAL = 5          # seconds between status polls
PER_Q_TIMEOUT = 3600       # 1h per question, measured from worker start (started_at).
                            # The full pipeline can take up to ~46 min for the slowest
                            # HumanEval+ tasks; 1h gives comfortable headroom.
QUEUE_GRACE = 120          # seconds to wait for the backbone to accept a submission
                            # (i.e. for the POST /jobs to return). NOT a queue-wait
                            # cap -- slot-based submission keeps queue wait near zero.
DEFAULT_BATCH_SIZE = 6      # matches WORKER_MAX_JOBS after the 4->6 bump

# Scoring container matches the python profile sandbox image.
SCORE_IMAGE = "python:3.11-slim"
SCORE_TIMEOUT = 30           # seconds for the canonical test run


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_problems(ids=None, limit=None):
    """Load HumanEval+ problems from the evalplus package.

    Returns a list of dicts with: task_id, prompt, entry_point, test,
    base_input, plus_input, atol, contract, canonical_solution.
    """
    from evalplus.data import get_human_eval_plus

    ds = get_human_eval_plus()
    problems = []
    for tid in sorted(ds.keys()):
        if ids and tid not in ids:
            continue
        problems.append(ds[tid])
    if limit:
        problems = problems[:limit]
    return problems


# ---------------------------------------------------------------------------
# Prompt wrapping
# ---------------------------------------------------------------------------

def wrap_prompt(human_eval_prompt: str) -> str:
    """Wrap a HumanEval prompt (signature + docstring) for the Mupin pipeline.

    The pipeline's test_designer expects natural-language requirements; a bare
    function signature + docstring is enough for the coder but can confuse
    test generation.  The light wrapper signals 'implement this as a module'
    while preserving the exact signature and docstring.
    """
    return (
        "Implement the following Python function as a module with the exact "
        "signature given. The module must define the function at the top level "
        "so it can be imported.\n\n"
        f"{human_eval_prompt}"
    )


# ---------------------------------------------------------------------------
# Backbone API helpers
# ---------------------------------------------------------------------------

def submit_task(prompt: str) -> str:
    resp = requests.post(
        f"{BASE_URL}/jobs",
        json={"job_type": "coding", "payload": {"prompt": prompt, "profile_name": "python"}},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["job_id"]


def poll_task(task_id: str):
    resp = requests.get(f"{BASE_URL}/jobs/{task_id}", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    progress = data.get("progress") or {}
    normalized = dict(data)
    normalized.setdefault("current_node", progress.get("current_node", data.get("status")))
    normalized.setdefault("sandbox_loop_count", progress.get("sandbox_loop_count", 0))
    return normalized


def cancel_task(task_id: str):
    try:
        requests.post(f"{BASE_URL}/jobs/{task_id}/cancel", timeout=10)
    except Exception as e:
        print(f"  CANCEL ERROR: {e}")


# ---------------------------------------------------------------------------
# Scoring against canonical HumanEval+ tests
# ---------------------------------------------------------------------------

def _extract_function(source: str, entry_point: str) -> str:
    """Extract the source lines defining entry_point from a module.

    Returns the function definition and body as a string.  If the entry_point
    is not found, returns the whole source (let the test harness fail with a
    NameError, which we record as a failure).
    """
    import ast

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) \
                and node.name == entry_point:
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") and node.end_lineno else len(lines)
            return "".join(lines[start:end])
    return source


def score_problem(job_id: str, problem: dict) -> dict:
    """Score a completed job against the canonical HumanEval+ tests.

    Returns {"pass": bool, "error": str | None, "extracted": bool}.
    """
    entry_point = problem["entry_point"]
    test_src = problem["test"]
    atol = problem.get("atol", 0)

    # Read the generated implementation from the workspace.
    main_path = WORKSPACE_ROOT / job_id / "src" / "main.py"
    if not main_path.exists():
        return {"pass": False, "error": f"src/main.py not found at {main_path}",
                "extracted": False}

    try:
        generated = main_path.read_text()
    except Exception as e:
        return {"pass": False, "error": f"read error: {e}", "extracted": False}

    extracted = _extract_function(generated, entry_point)

    # Build the scoring script: exec the generated source to define
    # entry_point at module level, then run the canonical check().
    # The generated source must NOT be wrapped in a block ({ ... }) —
    # that is a SyntaxError at module level in Python.
    scoring_script = (
        "import sys, traceback\n"
        "sys.setrecursionlimit(2000)\n"
        f"{generated}\n"
        f"\n\nMETADATA = {{}}\n\n{test_src}\n\n"
        f"try:\n"
        f"    check({entry_point})\n"
        f"    print('__PASS__')\n"
        f"except AssertionError as e:\n"
        f"    print('__FAIL_ASSERT__')\n"
        f"    traceback.print_exc()\n"
        f"except Exception as e:\n"
        f"    print('__FAIL_ERROR__')\n"
        f"    traceback.print_exc()\n"
    )

    # Run in a docker container matching the sandbox profile.
    import subprocess
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "-i", SCORE_IMAGE, "python"],
            input=scoring_script,
            capture_output=True,
            text=True,
            timeout=SCORE_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"pass": False, "error": f"score timeout after {SCORE_TIMEOUT}s",
                "extracted": True}
    except Exception as e:
        return {"pass": False, "error": f"docker error: {e}", "extracted": True}

    stdout = result.stdout or ""
    if "__PASS__" in stdout:
        return {"pass": True, "error": None, "extracted": True}
    elif "__FAIL_ASSERT__" in stdout:
        stderr = (result.stderr or "")[-500:]
        return {"pass": False, "error": f"assertion failure\n{stderr}",
                "extracted": True}
    elif "__FAIL_ERROR__" in stdout:
        stderr = (result.stderr or "")[-500:]
        return {"pass": False, "error": f"runtime error\n{stderr}",
                "extracted": True}
    else:
        stderr = (result.stderr or "")[-500:]
        return {"pass": False, "error": f"unexpected output\nstdout:{stdout[-300:]}\nstderr:{stderr}",
                "extracted": True}


# ---------------------------------------------------------------------------
# Result recording
# ---------------------------------------------------------------------------

def _build_result(run_id, problem, job_id, start_time, start_iso, status, data=None, score=None, error=None):
    data = data or {}
    now = time.time()
    elapsed = round(now - start_time, 1)
    started_at_str = data.get("started_at")
    queue_wait = None
    if started_at_str:
        try:
            queue_wait = round(datetime.fromisoformat(started_at_str).timestamp() - start_time, 1)
        except Exception:
            pass
    return {
        "run_id": run_id,
        "task_id": problem["task_id"],
        "entry_point": problem["entry_point"],
        "job_id": job_id,
        "start_time": start_iso,
        "end_time": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "queue_wait_seconds": queue_wait,
        "status": status,
        "sandbox_loop_count": data.get("sandbox_loop_count", 0),
        "files_generated": len(data.get("result", {}) or {}) if isinstance(data.get("result"), dict) else 0,
        "score_pass": score["pass"] if score else None,
        "score_error": (score or {}).get("error"),
        "extracted": (score or {}).get("extracted"),
        "error": error,
    }


def _append_result(result: dict):
    with open(RESULTS_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


# ---------------------------------------------------------------------------
# Concurrent execution
# ---------------------------------------------------------------------------

def _process_one(problem, run_id, per_q_timeout):
    """Submit one problem, poll until settled, score, and return the result row.

    Slot-based execution: each worker thread calls this in a loop, so at most
    `batch_size` jobs are ever in flight at once. Queue wait is therefore near
    zero, and the per-job timeout measures real pipeline work, not queue time.

    The per-job deadline is computed from `started_at` (set by the worker when
    it picks the job up), not from submission time. A short QUEUE_GRACE window
    covers submission acceptance only.
    """
    tid = problem["task_id"]
    submit_time = time.time()
    submit_iso = datetime.now(timezone.utc).isoformat()

    # Submit (with a short grace for the POST to return).
    try:
        job_id = submit_task(wrap_prompt(problem["prompt"]))
    except Exception as e:
        result = _build_result(run_id, problem, None, submit_time, submit_iso,
                               "submit_error", error=str(e))
        _append_result(result)
        print(f"  [{tid:<20}] SUBMIT_ERROR in {result['elapsed_seconds']}s: {e}")
        return result

    print(f"  [{tid:<20}] submitted {job_id}")

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
            cancel_task(job_id)
            result = _build_result(run_id, problem, job_id, submit_time, submit_iso,
                                   "queue_timeout", last_data,
                                   error=f"Worker did not start within {QUEUE_GRACE}s")
            _append_result(result)
            print(f"  [{tid:<20}] QUEUE_TIMEOUT (no worker start in {QUEUE_GRACE}s)")
            return result

        if deadline is not None and time.time() > deadline:
            cancel_task(job_id)
            result = _build_result(run_id, problem, job_id, submit_time, submit_iso,
                                   "timeout", last_data,
                                   error=f"Timed out after {per_q_timeout}s of worker time")
            _append_result(result)
            print(f"  [{tid:<20}] TIMEOUT after {result['elapsed_seconds']}s "
                  f"(worker budget {per_q_timeout}s exceeded)")
            return result

        try:
            data = poll_task(job_id)
            last_data = data
        except Exception as e:
            print(f"  [{tid:<20}] poll_error: {e}")
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
                print(f"  [{tid:<20}] worker started; deadline set "
                      f"(+{per_q_timeout}s from {data['started_at']})")
            except Exception:
                # If we can't parse started_at, fall back to submission time so
                # we don't lose the timeout entirely.
                deadline = submit_time + per_q_timeout

        if node != last_node:
            last_node = node
            print(f"  [{tid:<20}] {time.time() - submit_time:>7.1f}s -> {node}  "
                  f"(sbox={data.get('sandbox_loop_count', 0)})")

        if status in ("completed", "failed", "cancelled", "exhausted"):
            score = None
            if status == "completed":
                print(f"  [{tid:<20}] scoring against canonical tests...")
                try:
                    score = score_problem(job_id, problem)
                except Exception as e:
                    score = {"pass": False, "error": f"scorer error: {e}",
                             "extracted": False}

            result = _build_result(run_id, problem, job_id, submit_time, submit_iso,
                                   status, data, score=score)
            _append_result(result)
            verdict = "PASS" if (score or {}).get("pass") else "FAIL"
            score_str = f" score={verdict}" if score else ""
            print(f"  [{tid:<20}] {verdict} in {result['elapsed_seconds']}s  |  "
                  f"status={status} sbox={result['sandbox_loop_count']}{score_str}")
            return result

        time.sleep(POLL_INTERVAL)


def run_problems_concurrent(problems, run_id, batch_size=DEFAULT_BATCH_SIZE,
                             per_q_timeout=PER_Q_TIMEOUT):
    """Run problems slot-based: at most `batch_size` jobs in flight at once.

    Each worker thread loops: submit -> poll -> score -> pick up the next
    problem. This bounds queue wait to near zero (a job is only submitted when
    a worker slot is free), so the per-job timeout measures real pipeline
    work, not queue time. There is no whole-run kill switch -- each job has
    its own deadline computed from `started_at`.
    """
    run_start = time.time()
    print(f"\nHumanEval+ benchmark run: {run_id}")
    print(f"  Problems:    {len(problems)}")
    print(f"  API:         {BASE_URL}")
    print(f"  Concurrency: {batch_size} (slot-based)")
    print(f"  Per-job cap: {per_q_timeout}s (from worker start)")
    print(f"  Scoring:     docker {SCORE_IMAGE}, {SCORE_TIMEOUT}s timeout")

    # Each worker pops the next problem off a shared iterator.
    problem_iter = iter(problems)
    counter_lock = threading.Lock()
    results = []
    results_lock = threading.Lock()
    done_count = [0]

    def worker_loop():
        while True:
            with counter_lock:
                try:
                    problem = next(problem_iter)
                except StopIteration:
                    return
            result = _process_one(problem, run_id, per_q_timeout)
            with results_lock:
                results.append(result)
                done_count[0] += 1
                idx = done_count[0]
            print(f"  ---- [{idx}/{len(problems)}] done ----")

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(worker_loop) for _ in range(batch_size)]
        for f in futures:
            f.result()

    # Summary.
    results_map = {r["task_id"]: r for r in results}
    total = len(results)
    passed = sum(1 for r in results if r.get("score_pass"))
    completed = sum(1 for r in results if r["status"] == "completed")
    failed_pipeline = sum(1 for r in results
                          if r["status"] in ("failed", "exhausted", "cancelled"))
    timeouts = sum(1 for r in results if r["status"] == "timeout")
    submit_errs = sum(1 for r in results if r["status"] == "submit_error")
    queue_errs = sum(1 for r in results if r["status"] == "queue_timeout")

    print(f"\n{'=' * 60}")
    print(f"  HumanEval+ pass@1:  {passed}/{total} = {passed/total*100:.1f}%"
          if total else "\n  No results")
    print(f"  Pipeline completed: {completed}/{total}")
    print(f"  Pipeline failed:    {failed_pipeline}")
    print(f"  Timeouts:           {timeouts}")
    print(f"  Queue timeouts:     {queue_errs}")
    print(f"  Submit errors:      {submit_errs}")
    print(f"  Wall-clock:         {time.time() - run_start:.0f}s")
    if failed_pipeline:
        print(f"\n  Pipeline failures (non-completion):")
        for tid, r in sorted(results_map.items()):
            if r["status"] in ("failed", "exhausted", "cancelled"):
                print(f"    {tid}: {r['status']} - {(r.get('error') or '')[:80]}")
    score_fails = [(tid, r) for tid, r in results_map.items()
                   if r["status"] == "completed" and not r.get("score_pass")]
    if score_fails:
        print(f"\n  Completed but failed canonical tests ({len(score_fails)}):")
        for tid, r in score_fails[:20]:
            print(f"    {tid}: {(r.get('score_error') or '')[:80]}")
    print(f"{'=' * 60}")
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary():
    """Print pass@1 from the most recent run in the results file."""
    if not RESULTS_FILE.exists():
        print("No results file found.")
        return
    rows = []
    with open(RESULTS_FILE) as f:
        for line in f:
            rows.append(json.loads(line))
    if not rows:
        print("No results.")
        return
    # Group by run_id, pick the latest.
    latest_run = max(r["run_id"] for r in rows)
    run_rows = [r for r in rows if r["run_id"] == latest_run]
    total = len(run_rows)
    passed = sum(1 for r in run_rows if r.get("score_pass"))
    print(f"Latest run: {latest_run}")
    print(f"  Total:    {total}")
    print(f"  pass@1:   {passed}/{total} = {passed/total*100:.1f}%" if total else "  pass@1: n/a")


def rescore_existing(run_id=None, out_file=None):
    """Re-score completed jobs from their on-disk src/main.py.

    Reads RESULTS_FILE, finds rows whose status is 'completed' (i.e. the
    pipeline produced a workspace), and re-runs score_problem against the
    canonical HumanEval+ tests using the fixed scorer.  Writes one JSON line
    per re-scored task to out_file (default: humaneval_rescore.jsonl) and
    prints a summary.

    This does NOT re-run the pipeline — it only re-evaluates whatever
    src/main.py the pipeline already produced for each job.
    """
    if not RESULTS_FILE.exists():
        print("No results file found.")
        return

    # Load all problems so we can map task_id -> problem dict.
    all_problems = load_problems()
    by_tid = {p["task_id"]: p for p in all_problems}

    rows = []
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if run_id:
        rows = [r for r in rows if r.get("run_id") == run_id]

    # Only re-score rows that actually produced a workspace.
    candidates = [r for r in rows if r.get("status") == "completed" and r.get("job_id")]
    missing = [r for r in candidates if r["task_id"] not in by_tid]
    if missing:
        print(f"  WARNING: {len(missing)} rows have task_ids not in HumanEval+ dataset; skipping them.")
    candidates = [r for r in candidates if r["task_id"] in by_tid]

    if not candidates:
        print("No completed jobs to re-score.")
        return

    out_path = Path(out_file) if out_file else Path(__file__).parent / "humaneval_rescore.jsonl"
    print(f"\nRe-scoring {len(candidates)} completed jobs against canonical HumanEval+ tests")
    print(f"  Workspace: {WORKSPACE_ROOT}")
    print(f"  Output:    {out_path}")
    print(f"  Scoring:   docker {SCORE_IMAGE}, {SCORE_TIMEOUT}s timeout")

    results = []
    passed = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {}
        for r in candidates:
            problem = by_tid[r["task_id"]]
            futures[executor.submit(score_problem, r["job_id"], problem)] = r
        done = 0
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                score = fut.result()
            except Exception as e:
                score = {"pass": False, "error": f"rescore exception: {e}", "extracted": False}
            out = dict(r)
            out["score_pass"] = score["pass"]
            out["score_error"] = score.get("error")
            out["extracted"] = score.get("extracted")
            out["rescored_at"] = datetime.now(timezone.utc).isoformat()
            results.append(out)
            if score["pass"]:
                passed += 1
            done += 1
            status = "PASS" if score["pass"] else "FAIL"
            err = (score.get("error") or "")[:80].replace("\n", " ")
            print(f"  [{done}/{len(candidates)}] {r['task_id']:<14} {status}  {err}")

    # Sort by task_id for stable output.
    def _tid_key(r):
        n = r["task_id"].rsplit("/", 1)[-1]
        try:
            return (0, int(n))
        except ValueError:
            return (1, n)
    results.sort(key=_tid_key)

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    total = len(results)
    print(f"\n  RE-SCORE SUMMARY")
    print(f"    Re-scored: {total}")
    print(f"    Passed:    {passed}")
    print(f"    pass@1:    {passed}/{total} = {passed/total*100:.1f}%" if total else "    pass@1: n/a")
    # Breakdown of failure reasons.
    from collections import Counter
    reasons = Counter()
    for r in results:
        if r["score_pass"]:
            continue
        err = r.get("score_error") or ""
        if err.startswith("assertion failure"):
            reasons["assertion failure"] += 1
        elif err.startswith("runtime error"):
            reasons["runtime error"] += 1
        elif err.startswith("score timeout"):
            reasons["score timeout"] += 1
        elif err.startswith("src/main.py not found"):
            reasons["no main.py"] += 1
        elif err.startswith("docker error"):
            reasons["docker error"] += 1
        else:
            reasons["other"] += 1
    if reasons:
        print(f"    Failure breakdown:")
        for k, v in reasons.most_common():
            print(f"      {k:<20} {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global BASE_URL
    parser = argparse.ArgumentParser(description="HumanEval+ benchmark runner")
    parser.add_argument("--url", default=BASE_URL, help="Backbone API URL")
    parser.add_argument("--label", default=None, help="Run label (default: timestamp)")
    parser.add_argument("--limit", type=int, default=None, help="Only first N problems")
    parser.add_argument("--ids", nargs="*", default=None, help="Specific task_ids")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Concurrent submissions (default 6)")
    parser.add_argument("--per-q-timeout", type=int, default=PER_Q_TIMEOUT,
                        help=f"Per-job timeout in seconds, measured from worker "
                             f"start (default {PER_Q_TIMEOUT})")
    parser.add_argument("--summary", action="store_true", help="Print pass@1 from last run")
    parser.add_argument("--rescore", action="store_true",
                        help="Re-score completed jobs from on-disk src/main.py and exit")
    parser.add_argument("--rescore-run", default=None,
                        help="Only re-score rows with this run_id (used with --rescore)")
    parser.add_argument("--rescore-out", default=None,
                        help="Output file for --rescore (default: humaneval_rescore.jsonl)")
    args = parser.parse_args()

    if args.summary:
        print_summary()
        return

    if args.rescore:
        rescore_existing(run_id=args.rescore_run, out_file=args.rescore_out)
        return

    if args.url:
        BASE_URL = args.url

    run_id = args.label or f"humaneval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    problems = load_problems(ids=args.ids, limit=args.limit)
    if not problems:
        print("No problems to run.")
        return

    run_problems_concurrent(problems, run_id, batch_size=args.batch_size,
                             per_q_timeout=args.per_q_timeout)


if __name__ == "__main__":
    main()