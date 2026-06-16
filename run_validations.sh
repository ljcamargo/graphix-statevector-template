#!/usr/bin/env bash
# Run static analysis and linting checks (mypy, ruff) plus the full test suite.
# Captures all output to a timestamped log file.
set -euo pipefail

LOGFILE="validation_output_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee -a "$LOGFILE"
echo " graphix-statevector-template validations" | tee -a "$LOGFILE"
echo "==========================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

echo "=== Step 1: Update lockfile and install dependencies ===" | tee -a "$LOGFILE"
uv lock 2>&1 | tee -a "$LOGFILE"
uv sync --extra cuquantum --dev 2>&1 | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

echo "=== Step 2: Run mypy type checker ===" | tee -a "$LOGFILE"
uv run mypy . 2>&1 | tee -a "$LOGFILE" || true
echo "" | tee -a "$LOGFILE"

echo "=== Step 3: Run ruff linter ===" | tee -a "$LOGFILE"
uv run ruff check . 2>&1 | tee -a "$LOGFILE" || true
echo "" | tee -a "$LOGFILE"

echo "=== Step 4: Run ruff formatter check ===" | tee -a "$LOGFILE"
uv run ruff format --check . 2>&1 | tee -a "$LOGFILE" || true
echo "" | tee -a "$LOGFILE"

echo "=== Step 5: Run all tests ===" | tee -a "$LOGFILE"
uv run pytest tests/ -v 2>&1 | tee -a "$LOGFILE" || true
echo "" | tee -a "$LOGFILE"

echo "==========================================" | tee -a "$LOGFILE"
echo " All validations completed." | tee -a "$LOGFILE"
echo " Log saved to: $LOGFILE" | tee -a "$LOGFILE"
echo "==========================================" | tee -a "$LOGFILE"
