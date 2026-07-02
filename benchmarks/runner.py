#!/usr/bin/env python3
"""
Benchmark runner for the Coding Module.

Usage:
    python benchmarks/runner.py                        # run all questions
    python benchmarks/runner.py --ids fibonacci stack  # run specific questions
    python benchmarks/runner.py --summary              # print summary of past runs

Results are appended to benchmarks/results.jsonl — one JSON line per run.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = os.environ.get("CODING_MODULE_URL", "http://localhost:8000")
QUESTIONS_FILE = Path(__file__).parent / "questions.json"
RESULTS_FILE = Path(__file__).parent / "results.jsonl"
METRICS_FILE = Path(__file__).parent / "metrics.jsonl"

POLL_INTERVAL = 5   # seconds between status polls
TIMEOUT = 1200      # seconds before we give up on a single task (20 min)
                    # Expert/hard questions may override this via questions.json


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
    resp = requests.post(f"{BASE_URL}/task", json={"prompt": prompt}, timeout=10)
    resp.raise_for_status()
    return resp.json()["task_id"]


def poll_task(task_id: str):
    resp = requests.get(f"{BASE_URL}/task/{task_id}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def cancel_task(task_id: str):
    """Ask the API to cancel a task so a timeout doesn't leave it running."""
    try:
        requests.post(f"{BASE_URL}/task/{task_id}/cancel", timeout=10)
    except Exception as e:
        print(f"  CANCEL ERROR: {e}")


def run_question(question: dict, run_id: str) -> dict:
    qid = question["id"]
    print(f"\n{'─' * 60}")
    print(f"  [{qid}]  difficulty={question['difficulty']}")
    print(f"  {question['prompt'][:100]}...")
    print(f"{'─' * 60}")

    start_time = time.time()
    start_iso = datetime.now(timezone.utc).isoformat()

    def _diagnostic_fields(data: dict):
        return {
            "node_history": data.get("node_history", []),
            "llm_usage": data.get("llm_usage", []),
            "docker_runs": data.get("docker_runs", []),
            "classifier_history": data.get("classifier_history", []),
            "latest_verification_error": data.get("error"),
        }

    def _metrics_rows(result: dict, diag: dict) -> list[dict]:
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

    try:
        task_id = submit_task(question["prompt"])
    except Exception as e:
        print(f"  SUBMIT ERROR: {e}")
        return {
            "run_id": run_id,
            "question_id": qid,
            "difficulty": question["difficulty"],
            "task_id": None,
            "start_time": start_iso,
            "end_time": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.time() - start_time, 1),
            "status": "submit_error",
            "error": str(e),
            "sandbox_loop_count": 0,
            "compliance_loop_count": 0,
            "files_generated": 0,
            "thoughts_count": 0,
            "node_history": [],
            "llm_usage": [],
            "docker_runs": [],
            "classifier_history": [],
            "latest_verification_error": None,
        }

    print(f"  task_id={task_id}")
    last_node = None
    last_data = {}
    # Allow per-question timeout overrides, falling back to the global default.
    per_q_timeout = question.get("timeout_seconds", TIMEOUT)
    deadline = start_time + per_q_timeout

    while time.time() < deadline:
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

        if status in ("completed", "failed", "cancelled", "exhausted"):
            end_iso = datetime.now(timezone.utc).isoformat()
            files = data.get("result") or {}
            diagnostics = _diagnostic_fields(data)
            result = {
                "run_id": run_id,
                "question_id": qid,
                "difficulty": question["difficulty"],
                "task_id": task_id,
                "start_time": start_iso,
                "end_time": end_iso,
                "elapsed_seconds": elapsed,
                "status": status,
                "error": data.get("error"),
                "sandbox_loop_count": data.get("sandbox_loop_count", 0),
                "compliance_loop_count": data.get("compliance_loop_count", 0),
                "files_generated": len(files),
                "thoughts_count": len(data.get("thoughts", [])),
                "compliance_status": data.get("compliance_status"),
                **diagnostics,
            }
            append_metrics(_metrics_rows(result, diagnostics))
            verdict = "PASS" if status == "completed" and files else "FAIL"
            print(
                f"\n  {verdict}  in {elapsed}s  |  "
                f"sandbox_loops={result['sandbox_loop_count']}  "
                f"compliance_loops={result['compliance_loop_count']}  "
                f"files={result['files_generated']}"
            )
            return result

        time.sleep(POLL_INTERVAL)

    # Timed out — cancel the task so it doesn't run on and contend with the next
    # one, and record the last-observed progress (not zeros).
    cancel_task(task_id)
    end_iso = datetime.now(timezone.utc).isoformat()
    elapsed = round(time.time() - start_time, 1)
    diagnostics = _diagnostic_fields(last_data)
    result = {
        "run_id": run_id,
        "question_id": qid,
        "difficulty": question["difficulty"],
        "task_id": task_id,
        "start_time": start_iso,
        "end_time": end_iso,
        "elapsed_seconds": elapsed,
        "status": "timeout",
        "error": f"Timed out after {per_q_timeout}s",
        "sandbox_loop_count": last_data.get("sandbox_loop_count", 0),
        "compliance_loop_count": last_data.get("compliance_loop_count", 0),
        "files_generated": len(last_data.get("result") or {}),
        "thoughts_count": len(last_data.get("thoughts", [])),
        "compliance_status": last_data.get("compliance_status"),
        **diagnostics,
    }
    append_metrics(_metrics_rows(result, diagnostics))
    print(f"\n  TIMEOUT after {elapsed}s — cancelled "
          f"(per-question limit={per_q_timeout}s, "
          f"last node={last_data.get('current_node')} "
          f"sandbox_loops={last_data.get('sandbox_loop_count', 0)})")
    return result


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
        f"  {'ID':<20} {'DIFF':<8} {'STATUS':<10} {'TIME':>8}  "
        f"{'SBOX':>4}  {'COMP':>4}  {'FILES':>5}"
    )
    print(f"  {'─'*20} {'─'*8} {'─'*10} {'─'*8}  {'─'*4}  {'─'*4}  {'─'*5}")
    total_elapsed = 0
    passed = 0
    for r in results:
        verdict = "PASS" if r["status"] == "completed" and r["files_generated"] > 0 else "FAIL"
        if verdict == "PASS":
            passed += 1
        total_elapsed += r["elapsed_seconds"]
        print(
            f"  {r['question_id']:<20} {r['difficulty']:<8} {verdict:<10} "
            f"{r['elapsed_seconds']:>7.1f}s  "
            f"{r['sandbox_loop_count']:>4}  "
            f"{r['compliance_loop_count']:>4}  "
            f"{r['files_generated']:>5}"
        )
    print(f"{'─' * 70}")
    print(f"  {passed}/{len(results)} passed   total time: {total_elapsed:.1f}s   avg: {total_elapsed/len(results):.1f}s")
    print(f"{'═' * 70}\n")


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
    global BASE_URL, TIMEOUT
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
    args = parser.parse_args()

    if args.url:
        BASE_URL = args.url

    if args.timeout:
        TIMEOUT = args.timeout

    if args.summary:
        print_historical_summary()
        return

    questions = load_questions(args.ids)
    if not questions:
        print("No questions matched.")
        sys.exit(1)

    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"\nBenchmark run: {run_id}")
    print(f"Questions:     {len(questions)}")
    print(f"API:           {BASE_URL}")
    if args.label:
        print(f"Label:         {args.label}")

    results = []
    for q in questions:
        r = run_question(q, run_id)
        r["label"] = args.label
        append_result(r)
        results.append(r)

    print_summary(results)
    if args.diagnostics:
        print_diagnostics(results)


if __name__ == "__main__":
    main()
