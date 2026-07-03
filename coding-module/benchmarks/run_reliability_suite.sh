#!/usr/bin/env bash
# Run the full 12-question benchmark suite 10 times sequentially.
# Total: 120 task executions. Intended for overnight reliability + token-usage
# characterization with the current writer-node model (kimi-k2.7-code:cloud).

set -uo pipefail

OUT="benchmarks/run_reliability_10x.out"
: > "$OUT"

for round in {1..10}; do
  run_id_suffix=$(date +%Y%m%d_%H%M%S)
  {
    echo ""
    echo "============================================="
    echo "  ROUND $round / 10  ($run_id_suffix)"
    echo "============================================="
    echo "[$(date -Iseconds)] Starting round $round"
    python3 benchmarks/runner.py --label "reliability-10x-r${round}" --diagnostics
    echo "[$(date -Iseconds)] Round $round finished"
  } >> "$OUT" 2>&1
done

python3 - <<'PY' >> "$OUT"
import json, statistics
from pathlib import Path
results_file = Path("benchmarks/results.jsonl")
if not results_file.exists():
    print("No results found.")
    raise SystemExit
rows = [json.loads(line) for line in results_file.open() if line.strip()]
# keep only the last 120 rows (this run set)
rows = rows[-120:]
passed = sum(1 for r in rows if r["status"] == "completed" and r["files_generated"] > 0)
total = len(rows)
times = [r["elapsed_seconds"] for r in rows]
print(f"\nRELIABILITY SUMMARY (last {total} tasks)")
print(f"  Passed: {passed}/{total}  ({100*passed/total:.1f}%)")
print(f"  Total time: {sum(times):.1f}s  ({sum(times)/3600:.2f}h)")
print(f"  Min/Avg/Median/Max: {min(times):.1f}s / {statistics.mean(times):.1f}s / {statistics.median(times):.1f}s / {max(times):.1f}s")
PY
