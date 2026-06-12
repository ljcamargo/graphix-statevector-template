#!/usr/bin/env bash
# Run full test suite for both backends: template (CPU) and cuQuantum (GPU).
# Captures all output to a timestamped log file.
set -euo pipefail

LOGFILE="test_output_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================" | tee -a "$LOGFILE"
echo " graphix-statevector-template test suite" | tee -a "$LOGFILE"
echo "==========================================" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

echo "=== Step 1: Update lockfile and install with cuQuantum extras ===" | tee -a "$LOGFILE"
uv lock 2>&1 | tee -a "$LOGFILE"
uv sync --extra cuquantum --dev 2>&1 | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

echo "=== Step 2: Verify cuQuantum, cupy installed & GPU info ===" | tee -a "$LOGFILE"
uv run python3 -c "
import cupy; print(f'cupy {cupy.__version__} OK')
import cuquantum; print(f'cuquantum {cuquantum.__version__} OK')
" 2>&1 | tee -a "$LOGFILE"
echo "GPU info:" | tee -a "$LOGFILE"
nvidia-smi 2>&1 | tee -a "$LOGFILE" || echo "nvidia-smi not available" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

echo "=== Step 3: Sanity check - cuQuantum import and basic init ===" | tee -a "$LOGFILE"
uv run python3 -c "
from graphix_statevec_cuquantum import Statevec, StatevectorBackend
from graphix.states import BasicStates
import numpy as np

sv = Statevec(data=BasicStates.ZERO, nqubit=2)
print(f'nqubit: {sv.nqubit}')
print(f'flatten: {sv.flatten()}')

sv.entangle((0, 1))
sv.evolve_single(np.array([[0,1],[1,0]], dtype=np.complex128), 0)
print(f'after ops: {sv.flatten()}')

exp = sv.expectation_single(np.array([[1,0],[0,-1]], dtype=np.complex128), 1)
print(f'expectation: {exp}')

sv.add_nodes(1, BasicStates.PLUS)
print(f'after add_nodes, nqubit: {sv.nqubit}')

sv.swap((0, 2))
print(f'after swap: {sv.flatten()}')

sv.remove_qubit(1)
print(f'after remove_qubit, nqubit: {sv.nqubit}')

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

echo "=== Step 4: Run CPU comparison tests ===" | tee -a "$LOGFILE"
uv run pytest tests/test_statevec_cuquantum.py -k "TestStatevecCPU" -v 2>&1 | tee -a "$LOGFILE" || true
echo "" | tee -a "$LOGFILE"

echo "=== Step 5: Run cuQuantum GPU backend unit tests ===" | tee -a "$LOGFILE"
uv run pytest tests/test_statevec_cuquantum.py -k "TestStatevecCuQuantum" -v 2>&1 | tee -a "$LOGFILE" || true
echo "" | tee -a "$LOGFILE"

echo "=== Step 6: Run cuQuantum GPU pattern simulator test ===" | tee -a "$LOGFILE"
uv run pytest tests/test_statevec_cuquantum.py -k "test_pattern_simulator" -v 2>&1 | tee -a "$LOGFILE" || true
echo "" | tee -a "$LOGFILE"

echo "=== Step 7: Run cuQuantum GPU backend benchmarks ===" | tee -a "$LOGFILE"
uv run pytest benchmarks/bench_statevec_cuquantum.py --benchmark-only 2>&1 | tee -a "$LOGFILE" || true
echo "" | tee -a "$LOGFILE"

echo "=== Step 8: Run CPU backend benchmarks for comparison ===" | tee -a "$LOGFILE"
uv run pytest benchmarks/bench_statevec_cpu.py --benchmark-only 2>&1 | tee -a "$LOGFILE" || true
echo "" | tee -a "$LOGFILE"

echo "=== Step 9: Run MQT CPU benchmarks (if implemented) ===" | tee -a "$LOGFILE"
uv run pytest benchmarks/bench_mqtbench_cpu.py --benchmark-only -v -s --benchmark-verbose --tb=short 2>&1 | tee -a "$LOGFILE" || true
echo "" | tee -a "$LOGFILE"

echo "=== Step 10: Run MQT CuQuantum GPU benchmarks ===" | tee -a "$LOGFILE"
uv run pytest benchmarks/bench_mqtbench_cuquantum.py --benchmark-only -v -s --benchmark-verbose --tb=short 2>&1 | tee -a "$LOGFILE" || true
echo "" | tee -a "$LOGFILE"

echo "==========================================" | tee -a "$LOGFILE"
echo " All tests completed." | tee -a "$LOGFILE"
echo " Log saved to: $LOGFILE" | tee -a "$LOGFILE"
echo "==========================================" | tee -a "$LOGFILE"
