#!/usr/bin/env python3
"""
BigCodeBench Instruct-Hard benchmark runner for the Coding Module.

Submits each BigCodeBench-Hard problem (148 tasks) through the full Mupin
pipeline as a normal coding job, then scores the generated src/main.py
against the canonical BigCodeBench unittest test suite.

Usage:
    python benchmarks/bigcodebench_runner.py                       # full 148-task run
    python benchmarks/bigcodebench_runner.py --limit 10             # first 10 problems (smoke test)
    python benchmarks/bigcodebench_runner.py --ids BigCodeBench/100 BigCodeBench/101
    python benchmarks/bigcodebench_runner.py --batch-size 6        # submit/poll concurrency (default 6)
    python benchmarks/bigcodebench_runner.py --scorer-concurrency 3  # scoring container concurrency (default 3)
    python benchmarks/bigcodebench_runner.py --per-q-timeout 3600  # per-job cap (default 3600s)
    python benchmarks/bigcodebench_runner.py --summary             # print pass@1 from last run
    python benchmarks/bigcodebench_runner.py --rescore             # re-score existing completed jobs in place
    python benchmarks/bigcodebench_runner.py --keep-workspaces    # retain task workspaces after scoring (debug)

Results are appended to benchmarks/bigcodebench_results.jsonl - one JSON line per
problem.  pass@1 = passing / total.

Execution model: slot-based. At most `--batch-size` jobs are submitted and
polled in parallel; each worker thread loops submit -> poll -> score -> next
problem. The heavy BigCodeBench scoring containers are gated independently by
`--scorer-concurrency` so the host is not overloaded by concurrent runs of the
~25 GB `bigcodebench/bigcodebench-evaluate:latest` image. Queue wait is
therefore near zero, and the per-job timeout (--per-q-timeout) measures real
pipeline work, measured from the worker's `started_at` timestamp -- NOT from
submission time. There is no whole-run kill switch.

To prevent the Proxmox guest VM from slowly accumulating page cache and freezing,
the runner also:
  - drops Linux page caches after each scoring task via a tiny privileged helper
    container (`docker run --rm --privileged --pid=host alpine sh -c "echo 3 > /proc/sys/vm/drop_caches"`)
  - deletes each task workspace after its result is recorded (use --keep-workspaces to retain)
  - runs `docker container prune -f` every 20 tasks to clear leftover metadata

The runner is designed to execute INSIDE the mupin_coding_worker container,
where the `bigcodebench` package and the backbone URL
(http://mupin-api-backbone:8000) are available. The benchmarks/ dir is mounted
rw into the worker so results can be written back to the host.

Requires: pip install bigcodebench requests  (already in the worker image)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

# When run inside the worker container, the backbone is at mupin-api-backbone:8000.
# When run on the host (debugging), fall back to localhost:8001 (backbone's exposed port).
BASE_URL = os.environ.get(
    "MUPIN_BACKBONE_URL",
    os.environ.get("BACKBONE_URL", "http://mupin-api-backbone:8000"),
)
RESULTS_FILE = Path(__file__).parent / "bigcodebench_results.jsonl"
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent / ".workspaces"

POLL_INTERVAL = 5          # seconds between status polls
PER_Q_TIMEOUT = 3600      # 1h per question, measured from worker start (started_at).
                           # BigCodeBench-Hard tasks are complex; 1h gives headroom.
QUEUE_GRACE = 1800         # seconds to wait for the worker to pick up a submitted
                           # job (i.e. for `started_at` to appear). Safety net for a
                           # dead/sick backbone, NOT a queue-wait cap. Raised from 600
                           # after bcb_r9 lost 30/148 tasks to queue_timeout under load.
DEFAULT_BATCH_SIZE = 6      # matches WORKER_MAX_JOBS

# Scoring container: the official BigCodeBench evaluate image has all 148 tasks'
# library deps (numpy, pandas, matplotlib, scipy, sklearn, opencv, librosa,
# tensorflow, tesseract, etc.).
SCORE_IMAGE = "bigcodebench/bigcodebench-evaluate:latest"
SCORE_TIMEOUT = 240          # seconds for the canonical test run (official default)

# The official BigCodeBench scoring image is ~25 GB on disk. Re-reading its
# heavy layers (tensorflow, librosa, opencv, etc.) from many containers at once
# can saturate the host disk bus, so scoring concurrency is gated independently
# of the pipeline submit/poll concurrency.
DEFAULT_SCORER_CONCURRENCY = 3
_SCORER_SEM: Optional[threading.Semaphore] = None

# Workspace cleanup. By default completed/failed task workspaces are deleted
# after scoring so the .workspaces/ dir does not grow without bound across
# many benchmark runs. Use --keep-workspaces to retain them for debugging.
_KEEP_WORKSPACES: bool = False

# Run docker container prune every N completed tasks to clear any leftover
# metadata from --rm scoring containers that the daemon may retain briefly.
_DOCKER_PRUNE_INTERVAL = 20


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_problems(ids=None, limit=None, subset="hard"):
    """Load BigCodeBench problems from the bigcodebench package.

    Returns a list of dicts with: task_id, entry_point, instruct_prompt,
    complete_prompt, canonical_solution, test, libs.
    """
    from bigcodebench.data import get_bigcodebench

    ds = get_bigcodebench(subset=subset)
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

def wrap_prompt(instruct_prompt: str) -> str:
    """Wrap a BigCodeBench instruct prompt for the Mupin pipeline.

    The pipeline's test_designer expects natural-language requirements.
    BigCodeBench's instruct_prompt is already natural language, but we add a
    light wrapper signalling 'implement this as a module defining task_func'
    so the coder produces an importable function rather than a script.
    """
    return (
        "Implement the following as a Python module. The module must define a "
        "function named `task_func` at the top level so it can be imported. "
        "Do not call the function at module level.\n\n"
        f"{instruct_prompt}"
    )


def _extract_contract(problem: dict) -> str:
    """Extract a verbatim starter contract from a BigCodeBench problem.

    Pulls (a) all top-level `import` / `from ... import` statements and (b) the
    `entry_point` function definition (signature + body as written) out of the
    problem's `complete_prompt`, which carries the canonical signature and
    module-level imports that canonical tests rely on.

    Returns "" if the entry_point cannot be found via AST or the source does not
    parse — the pipeline then falls back to test-derived skeleton inference.
    Partial extraction is avoided: a half-contract is worse than none because
    it would silently mislead skeleton_maker.
    """
    import ast

    source = problem.get("complete_prompt") or ""
    entry_point = problem.get("entry_point") or ""
    if not source or not entry_point:
        return ""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""

    lines = source.splitlines(keepends=True)

    # Top-level imports (preserve source text, in original order).
    import_lines = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            start = node.lineno - 1
            end = node.end_lineno if getattr(node, "end_lineno", None) else start + 1
            import_lines.append("".join(lines[start:end]))

    # The entry_point function definition (verbatim source slice).
    func_text = ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            start = node.lineno - 1
            end = node.end_lineno if getattr(node, "end_lineno", None) else len(lines)
            func_text = "".join(lines[start:end])
            break

    if not func_text:
        return ""

    contract = "".join(import_lines)
    if import_lines and not import_lines[-1].endswith("\n"):
        contract += "\n"
    contract += func_text
    return contract


# ---------------------------------------------------------------------------
# Backbone API helpers
# ---------------------------------------------------------------------------

_global_deps_cache_tag: str | None = None


def set_deps_cache_tag(tag: str | None) -> None:
    global _global_deps_cache_tag
    _global_deps_cache_tag = tag


def submit_task(prompt: str, contract_code: str = "") -> str:
    payload = {"job_type": "coding", "payload": {"prompt": prompt, "profile_name": "python"}}
    if contract_code:
        payload["payload"]["contract_code"] = contract_code
    if _global_deps_cache_tag:
        payload["payload"]["deps_cache_tag"] = _global_deps_cache_tag
    resp = requests.post(
        f"{BASE_URL}/jobs",
        json=payload,
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
# Scoring against canonical BigCodeBench tests
# ---------------------------------------------------------------------------

def _drop_caches() -> None:
    """Drop Linux page cache after a scoring container exits.

    Each fresh `docker run --rm` scoring container re-reads the ~25 GB
    BigCodeBench evaluate image layers into the guest's page cache. On the
    emulated virtio-scsi disk the kernel reclaims these pages sluggishly, so
    over 148 tasks the cache grows monotonically until the guest VM wedges.

    The runner executes inside the mupin_coding_worker container, where
    /proc/sys/vm/drop_caches is read-only. The worker has the host Docker
    socket mounted, so we drop the *host* (guest VM) page cache by running a
    tiny privileged container in the host PID namespace.
    """
    try:
        subprocess.run(
            [
                "docker", "run", "--rm", "--privileged", "--pid=host",
                "alpine", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches",
            ],
            timeout=30,
            capture_output=True,
        )
    except Exception:
        pass


def _docker_prune() -> None:
    """Remove exited container metadata that the daemon may retain.

    Called periodically during a long benchmark run so `docker run --rm`
    bookkeeping does not accumulate.
    """
    try:
        subprocess.run(
            ["docker", "container", "prune", "-f", "--filter", "status=exited"],
            timeout=30,
            capture_output=True,
        )
    except Exception:
        pass


def _cleanup_workspace(job_id: str) -> None:
    """Delete a task workspace after scoring so .workspaces/ does not grow.

    Workspaces are kept by default during a run for inspection; the runner now
    takes responsibility for removing them once results are recorded. The
    `--keep-workspaces` flag disables this.
    """
    if _KEEP_WORKSPACES:
        return
    if not job_id:
        return
    ws = WORKSPACE_ROOT / job_id
    try:
        shutil.rmtree(ws, ignore_errors=True)
    except Exception:
        pass


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
    """Score a completed job against the canonical BigCodeBench tests.

    Returns {"pass": bool, "error": str | None, "extracted": bool}.
    """
    entry_point = problem["entry_point"]
    test_src = problem["test"]

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
    # entry_point at module level, then run the canonical unittest.TestCase.
    # We use unittest.main() with a buffer to capture results, and print a
    # sentinel so we can parse the outcome regardless of exit code.
    #
    # Python requires `from __future__ import ...` to be the first statement
    # in a file. If the generated code starts with __future__ imports, we
    # split them out and prepend the scoring imports *after* them.
    future_lines = []
    code_lines = generated.splitlines(keepends=True)
    i = 0
    while i < len(code_lines):
        line = code_lines[i].strip()
        if not line or line.startswith("#"):
            future_lines.append(code_lines[i])
            i += 1
            continue
        if line.startswith("from __future__"):
            future_lines.append(code_lines[i])
            i += 1
            continue
        break
    rest_of_generated = "".join(code_lines[i:])
    future_prefix = "".join(future_lines)

    scoring_script = (
        f"{future_prefix}"
        "import sys, unittest, traceback\n"
        "sys.setrecursionlimit(2000)\n"
        f"{rest_of_generated}\n"
        f"\n\n{test_src}\n\n"
        f"if __name__ == '__main__':\n"
        f"    try:\n"
        f"        loader = unittest.TestLoader()\n"
        f"        suite = loader.loadTestsFromTestCase(TestCases)\n"
        f"        runner = unittest.TextTestRunner(verbosity=0, stream=sys.stderr)\n"
        f"        result = runner.run(suite)\n"
        f"        if result.wasSuccessful():\n"
        f"            print('__PASS__')\n"
        f"        else:\n"
        f"            print('__FAIL_ASSERT__')\n"
        f"    except Exception:\n"
        f"        print('__FAIL_ERROR__')\n"
        f"        traceback.print_exc()\n"
    )

    # Run in a docker container with all BigCodeBench library deps.
    # The official image's default entrypoint is `python3 -m bigcodebench.evaluate`
    # (the CLI evaluator). We override it to a plain `python` so we can pipe
    # our scoring script to stdin.
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "-i", "--entrypoint", "python",
             SCORE_IMAGE],
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
            queue_wait = round(datetime.fromisoformat(started_at_str.replace("Z", "+00:00")).timestamp() - start_time, 1)
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
        job_id = submit_task(wrap_prompt(problem["instruct_prompt"]),
                             _extract_contract(problem))
    except Exception as e:
        result = _build_result(run_id, problem, None, submit_time, submit_iso,
                               "submit_error", error=str(e))
        _append_result(result)
        print(f"  [{tid:<22}] SUBMIT_ERROR in {result['elapsed_seconds']}s: {e}")
        _cleanup_workspace(result.get("job_id"))
        return result

    print(f"  [{tid:<22}] submitted {job_id}")

    # Poll until settled or per-job timeout (clock starts at worker start).
    deadline = None              # set once started_at appears
    submit_deadline = submit_time + QUEUE_GRACE
    last_data = {}
    last_node = None

    while True:
        # If the worker hasn't started yet, only the submission grace applies.
        if deadline is None and time.time() > submit_deadline:
            cancel_task(job_id)
            result = _build_result(run_id, problem, job_id, submit_time, submit_iso,
                                   "queue_timeout", last_data,
                                   error=f"Worker did not start within {QUEUE_GRACE}s "
                                         f"(backbone health issue)")
            _append_result(result)
            print(f"  [{tid:<22}] QUEUE_TIMEOUT (no worker start in {QUEUE_GRACE}s)")
            _cleanup_workspace(job_id)
            return result

        if deadline is not None and time.time() > deadline:
            cancel_task(job_id)
            result = _build_result(run_id, problem, job_id, submit_time, submit_iso,
                                   "timeout", last_data,
                                   error=f"Timed out after {per_q_timeout}s of worker time")
            _append_result(result)
            print(f"  [{tid:<22}] TIMEOUT after {result['elapsed_seconds']}s "
                  f"(worker budget {per_q_timeout}s exceeded)")
            _cleanup_workspace(job_id)
            return result

        try:
            data = poll_task(job_id)
            last_data = data
        except Exception as e:
            print(f"  [{tid:<22}] poll_error: {e}")
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
                print(f"  [{tid:<22}] worker started; deadline set "
                      f"(+{per_q_timeout}s from {data['started_at']})")
            except Exception:
                deadline = submit_time + per_q_timeout

        if node != last_node:
            last_node = node
            print(f"  [{tid:<22}] {time.time() - submit_time:>7.1f}s -> {node}  "
                  f"(sbox={data.get('sandbox_loop_count', 0)})")

        if status in ("completed", "failed", "cancelled", "exhausted", "infra_exhausted"):
            score = None
            if status == "completed":
                print(f"  [{tid:<22}] scoring against canonical tests...")
                try:
                    global _SCORER_SEM
                    assert _SCORER_SEM is not None, "scorer semaphore not initialized"
                    with _SCORER_SEM:
                        score = score_problem(job_id, problem)
                    _drop_caches()
                except Exception as e:
                    score = {"pass": False, "error": f"scorer error: {e}",
                             "extracted": False}

            result = _build_result(run_id, problem, job_id, submit_time, submit_iso,
                                   status, data, score=score)
            _append_result(result)
            verdict = "PASS" if (score or {}).get("pass") else "FAIL"
            score_str = f" score={verdict}" if score else ""
            print(f"  [{tid:<22}] {verdict} in {result['elapsed_seconds']}s  |  "
                  f"status={status} sbox={result['sandbox_loop_count']}{score_str}")
            _cleanup_workspace(job_id)
            return result

        time.sleep(POLL_INTERVAL)


def run_problems_concurrent(problems, run_id, batch_size=DEFAULT_BATCH_SIZE,
                             per_q_timeout=PER_Q_TIMEOUT,
                             scorer_concurrency=DEFAULT_SCORER_CONCURRENCY):
    """Run problems slot-based: at most `batch_size` jobs in flight at once.

    Each worker thread loops: submit -> poll -> score -> pick up the next
    problem. This bounds queue wait to near zero. There is no whole-run kill
    switch -- each job has its own deadline computed from `started_at`.

    The heavy BigCodeBench scoring containers are gated by
    `scorer_concurrency` so at most that many can be running at once,
    regardless of `batch_size`.
    """
    global _SCORER_SEM
    _SCORER_SEM = threading.Semaphore(scorer_concurrency)

    run_start = time.time()
    print(f"\nBigCodeBench Instruct-Hard run: {run_id}")
    print(f"  Problems:    {len(problems)}")
    print(f"  API:         {BASE_URL}")
    print(f"  Concurrency: {batch_size} (slot-based)")
    print(f"  Scorer cap:  {scorer_concurrency} (docker {SCORE_IMAGE}, {SCORE_TIMEOUT}s timeout)")
    print(f"  Per-job cap: {per_q_timeout}s (from worker start)")
    print(f"  Cleanup:     drop_caches after scoring, workspace prune enabled"
          f"{' (--keep-workspaces)' if _KEEP_WORKSPACES else ''}")
    print(f"  Docker prune: every {_DOCKER_PRUNE_INTERVAL} tasks")

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
            if idx % _DOCKER_PRUNE_INTERVAL == 0:
                _docker_prune()
            print(f"  ---- [{idx}/{len(problems)}] done ----")

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(worker_loop) for _ in range(batch_size)]
        for f in futures:
            f.result()

    # Summary.
    total = len(results)
    passed = sum(1 for r in results if r.get("score_pass"))
    completed = sum(1 for r in results if r["status"] == "completed")
    failed_pipeline = sum(1 for r in results
                          if r["status"] in ("failed", "exhausted", "cancelled", "infra_exhausted"))
    timeouts = sum(1 for r in results if r["status"] == "timeout")
    submit_errs = sum(1 for r in results if r["status"] == "submit_error")
    queue_errs = sum(1 for r in results if r["status"] == "queue_timeout")

    print(f"\n{'=' * 60}")
    print(f"  BigCodeBench-Hard pass@1:  {passed}/{total} = {passed/total*100:.1f}%"
          if total else "\n  No results")
    print(f"  Pipeline completed: {completed}/{total}")
    print(f"  Pipeline failed:     {failed_pipeline}")
    print(f"  Timeouts:            {timeouts}")
    print(f"  Queue timeouts:     {queue_errs}")
    print(f"  Submit errors:      {submit_errs}")
    print(f"  Wall-clock:         {time.time() - run_start:.0f}s")
    if failed_pipeline:
        print(f"\n  Pipeline failures (non-completion):")
        for r in results:
            if r["status"] in ("failed", "exhausted", "cancelled", "infra_exhausted"):
                print(f"    {r['task_id']}: {r['status']} - {(r.get('error') or '')[:80]}")
    score_fails = [r for r in results
                   if r["status"] == "completed" and not r.get("score_pass")]
    if score_fails:
        print(f"\n  Completed but failed canonical tests ({len(score_fails)}):")
        for r in score_fails[:20]:
            print(f"    {r['task_id']}: {(r.get('score_error') or '')[:80]}")
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
    latest_run = max(r["run_id"] for r in rows)
    run_rows = [r for r in rows if r["run_id"] == latest_run]
    total = len(run_rows)
    passed = sum(1 for r in run_rows if r.get("score_pass"))
    print(f"Latest run: {latest_run}")
    print(f"  Total:    {total}")
    print(f"  pass@1:   {passed}/{total} = {passed/total*100:.1f}%" if total else "  pass@1: n/a")


def _scored_score_problem(job_id: str, problem: dict) -> dict:
    """Wrapper around score_problem that respects the scorer semaphore."""
    global _SCORER_SEM
    assert _SCORER_SEM is not None, "scorer semaphore not initialized"
    with _SCORER_SEM:
        score = score_problem(job_id, problem)
    _drop_caches()
    return score


def rescore_existing(run_id=None, out_file=None,
                     scorer_concurrency=DEFAULT_SCORER_CONCURRENCY):
    """Re-score completed jobs from their on-disk src/main.py.

    Reads RESULTS_FILE, finds rows whose status is 'completed', and re-runs
    score_problem against the canonical BigCodeBench tests. Writes one JSON
    line per re-scored task to out_file (default: bigcodebench_rescore.jsonl).
    Scoring container concurrency is gated by `scorer_concurrency`.
    """
    if not RESULTS_FILE.exists():
        print("No results file found.")
        return

    global _SCORER_SEM
    _SCORER_SEM = threading.Semaphore(scorer_concurrency)

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

    candidates = [r for r in rows if r.get("status") == "completed" and r.get("job_id")]
    missing = [r for r in candidates if r["task_id"] not in by_tid]
    if missing:
        print(f"  WARNING: {len(missing)} rows have task_ids not in BigCodeBench dataset; skipping them.")
    candidates = [r for r in candidates if r["task_id"] in by_tid]

    if not candidates:
        print("No completed jobs to re-score.")
        return

    out_path = Path(out_file) if out_file else Path(__file__).parent / "bigcodebench_rescore.jsonl"
    print(f"\nRe-scoring {len(candidates)} completed jobs against canonical BigCodeBench tests")
    print(f"  Workspace: {WORKSPACE_ROOT}")
    print(f"  Output:    {out_path}")
    print(f"  Scorer cap: {scorer_concurrency} (docker {SCORE_IMAGE}, {SCORE_TIMEOUT}s timeout)")

    results = []
    passed = 0
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {}
        for r in candidates:
            problem = by_tid[r["task_id"]]
            futures[executor.submit(_scored_score_problem, r["job_id"], problem)] = r
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
            print(f"  [{done}/{len(candidates)}] {r['task_id']:<22} {status}  {err}")

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
    parser = argparse.ArgumentParser(description="BigCodeBench Instruct-Hard benchmark runner")
    parser.add_argument("--url", default=None, help="Backbone API URL")
    parser.add_argument("--label", default=None, help="Run label (default: timestamp)")
    parser.add_argument("--limit", type=int, default=None, help="Only first N problems")
    parser.add_argument("--ids", nargs="*", default=None, help="Specific task_ids")
    parser.add_argument("--subset", default="hard", help="Dataset subset (default: hard)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Concurrent job submission/polling (default 6)")
    parser.add_argument("--scorer-concurrency", type=int, default=None,
                        help="Max concurrent BigCodeBench scoring containers "
                             "(default 3, or BCB_SCORER_CONCURRENCY env var). "
                             "Lower on hosts with slow disk or limited RAM.")
    parser.add_argument("--per-q-timeout", type=int, default=PER_Q_TIMEOUT,
                        help=f"Per-job timeout in seconds, measured from worker "
                             f"start (default {PER_Q_TIMEOUT})")
    parser.add_argument("--summary", action="store_true", help="Print pass@1 from last run")
    parser.add_argument("--rescore", action="store_true",
                        help="Re-score completed jobs from on-disk src/main.py and exit")
    parser.add_argument("--rescore-run", default=None,
                        help="Only re-score rows with this run_id (used with --rescore)")
    parser.add_argument("--rescore-out", default=None,
                        help="Output file for --rescore (default: bigcodebench_rescore.jsonl)")
    parser.add_argument("--deps-cache-tag", default=None,
                        help="Persist installed deps under a shared tag for project-long work")
    parser.add_argument("--keep-workspaces", action="store_true",
                        help="Do not delete task workspaces after scoring (for debugging)")
    parser.add_argument("--clear-deps-cache", action="store_true",
                        help="Wipe the shared .deps_cache directory and exit")
    args = parser.parse_args()

    global _KEEP_WORKSPACES
    _KEEP_WORKSPACES = args.keep_workspaces

    if args.clear_deps_cache:
        cache_root = WORKSPACE_ROOT.parent / ".deps_cache"
        if cache_root.exists():
            shutil.rmtree(cache_root)
            print(f"Cleared dependency cache: {cache_root}")
        else:
            print(f"No dependency cache to clear: {cache_root}")
        return

    if args.summary:
        print_summary()
        return

    scorer_conc = args.scorer_concurrency
    if scorer_conc is None:
        scorer_conc = int(os.environ.get("BCB_SCORER_CONCURRENCY", DEFAULT_SCORER_CONCURRENCY))

    if args.rescore:
        rescore_existing(run_id=args.rescore_run, out_file=args.rescore_out,
                         scorer_concurrency=scorer_conc)
        return

    if args.url:
        global_url = args.url
        BASE_URL = global_url

    if args.deps_cache_tag:
        set_deps_cache_tag(args.deps_cache_tag)

    run_id = args.label or f"bcb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    problems = load_problems(ids=args.ids, limit=args.limit, subset=args.subset)
    if not problems:
        print("No problems to run.")
        return

    run_problems_concurrent(problems, run_id, batch_size=args.batch_size,
                             per_q_timeout=args.per_q_timeout,
                             scorer_concurrency=scorer_conc)


if __name__ == "__main__":
    main()