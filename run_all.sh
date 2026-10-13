#!/usr/bin/env bash
# Reproduce every number, table and figure of the paper and of the online supplement.
#   ./run_all.sh           full run (about 25 minutes on two cores)
#   ./run_all.sh --quick   smoke test with reduced grids (about 2 minutes; writes *_quick files
#                          and does NOT overwrite the full results or the paper's tables and figures)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
QUICK=""
if [[ "${1:-}" == "--quick" ]]; then QUICK="--quick"; fi

echo "== 1/7 download and preprocess the index data (pinned commit, SHA-256 checked)"
$PY scripts/download_data.py

echo "== 2/7 unit tests"
$PY -m pytest -q tests

echo "== 3/7 numerical check of every result (results/verification*.json)"
if [[ -n "$QUICK" ]]; then $PY scripts/verify_theory.py --quick; else $PY scripts/verify_theory.py; fi

echo "== 4/7 experiment E1: worst-case regret of reduction rules"
$PY scripts/run_e1_worst_case.py $QUICK

echo "== 5/7 experiment E2: time-decayed weights"
$PY scripts/run_e2_online_weights.py $QUICK

echo "== 6/7 experiment E3: decisions from reduced scenario sets"
$PY scripts/run_e3_decisions.py $QUICK

if [[ -n "$QUICK" ]]; then
  echo "Quick run finished (results/*_quick.csv). Run without --quick to rebuild the paper."
  exit 0
fi

echo "== 7/7 tables, figures and PDFs"
$PY scripts/make_tables.py
$PY scripts/make_figures.py
if command -v pdflatex > /dev/null; then
  ../paper/build.sh
else
  echo "pdflatex not found: tables and figures are in ../paper; compile ../paper/main.tex yourself."
fi
