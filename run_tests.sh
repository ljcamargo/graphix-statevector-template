#!/usr/bin/env bash
# Test script for both backends: template (CPU) and cuQuantum (GPU).
# Captures all output to a log file for review.
set -euo pipefail

LOGFILE="test_output_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee -a "$LOGFILE"
echo " graphix-statevector-template test suite" | tee -a "$LOGFILE"
echo "==========================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

# ---- Config ----
REPO_URL="${1:-<your-repo-url>}"  # pass repo URL as first arg, or edit below
BRANCH="gpu-backend"

echo "=== Step 1: Clone repository (branch: $BRANCH) ===" | tee -a "$LOGFILE"
if [ -d "graphix-statevec-template" ]; then
    echo "Directory exists, pulling..." | tee -a "$LOGFILE"
    cd graphix-statevec-template
    git checkout "$BRANCH"
    git pull
else
    git clone "$REPO_URL" graphix-statevec-template
    cd graphix-statevec-template
    git checkout "$BRANCH"
fi
echo "" | tee -a "$LOGFILE"

echo "=== Step 2: Update lockfile and install with cuQuantum extras ===" | tee -a "$LOGFILE"
uv lock 2>&1 | tee -a "$LOGFILE"
uv sync --extra cuquantum --dev 2>&1 | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

echo "=== Step 3: Sanity check - cuQuantum import and basic init ===" | tee -a "$LOGFILE"
uv run python3 -c "
from graphix_statevec_cuquantum import Statevec, StatevectorBackend
from graphix.states import BasicStates
import numpy as np

# Basic init test
sv = Statevec(data=BasicStates.ZERO, nqubit=2)
print(f'nqubit: {sv.nqubit}')
print(f'flatten: {sv.flatten()}')

# Entangle + evolve
sv.entangle((0, 1))
sv.evolve_single(np.array([[0,1],[1,0]], dtype=np.complex128), 0)
print(f'after ops: {sv.flatten()}')

# Expectation
exp = sv.expectation_single(np.array([[1,0],[0,-1]], dtype=np.complex128), 1)
print(f'expectation: {exp}')

# Add nodes
sv.add_nodes(1, BasicStates.PLUS)
print(f'after add_nodes, nqubit: {sv.nqubit}')
print(f'flatten: {sv.flatten()}')

# Swap
sv.swap((0, 2))
print(f'after swap: {sv.flatten()}')

# Remove qubit
sv.remove_qubit(1)
print(f'after remove_qubit, nqubit: {sv.nqubit}')
print(f'flatten: {sv.flatten()}')

# Backend integration
from graphix.transpiler import Circuit
qc = Circuit(2)
qc.cz(0, 1)
qc.h(0)
pattern = qc.transpile().pattern
backend = StatevectorBackend()
result = pattern.simulate_pattern(backend=backend)
print(f'pattern simulation result: {result.flatten()}')
print('Sanity check PASSED')
" 2>&1 | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

echo "=== Step 4: Run template (CPU) backend unit tests ===" | tee -a "$LOGFILE"
# Template tests are marked with @pytest.mark.skip - we run them by deselecting the skip marker
uv run pytest tests/test_statevec.py -k "TestStatevec" -v --override-ini="filterwarnings=ignore" 2>&1 | tee -a "$LOGFILE" || true
echo "" | tee -a "$LOGFILE"

echo "=== Step 5: Run cuQuantum (GPU) backend unit tests ===" | tee -a "$LOGFILE"
uv run pytest tests/test_statevec_cuquantum.py -k "TestStatevec" -v --override-ini="filterwarnings=ignore" 2>&1 | tee -a "$LOGFILE" || true
echo "" | tee -a "$LOGFILE"

echo "=== Step 6: Run cuQuantum legacy comparison tests ===" | tee -a "$LOGFILE"
uv run pytest tests/test_statevec_cuquantum.py -k "TestStatevecLegacy" -v --override-ini="filterwarnings=ignore" 2>&1 | tee -a "$LOGFILE" || true
echo "" | tee -a "$LOGFILE"

echo "=== Step 7: Run cuQuantum pattern simulator test ===" | tee -a "$LOGFILE"
uv run pytest tests/test_statevec_cuquantum.py -k "test_pattern_simulator" -v --override-ini="filterwarnings=ignore" 2>&1 | tee -a "$LOGFILE" || true
echo "" | tee -a "$LOGFILE"

echo "==========================================" | tee -a "$LOGFILE"
echo " All tests completed." | tee -a "$LOGFILE"
echo " Log saved to: $LOGFILE" | tee -a "$LOGFILE"
echo "==========================================" | tee -a "$LOGFILE"
