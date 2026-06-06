# graphix-statevec-cuquantum

GPU-accelerated statevector backend for Graphix pattern simulation, built on NVIDIA cuQuantum (cuStateVec) and CuPy.

## Requirements

- NVIDIA GPU with CUDA 12.x
- Python ≥ 3.13
- `cuquantum-python-cu12` (≥ 24.03) — pre-built wheels, no CUDA toolkit needed at build time
- `cupy-cuda12x` (≥ 13.0)

## Installation

```bash
uv lock
uv sync --extra cuquantum --dev
```

## Usage

```python
from graphix_statevec_cuquantum import Statevec, StatevectorBackend
from graphix.transpiler import Circuit

qc = Circuit(2)
qc.cz(0, 1)
qc.h(0)
pattern = qc.transpile().pattern

backend = StatevectorBackend()
sv = pattern.simulate_pattern(backend=backend)
print(sv.flatten())
```

## Design

### State representation

The quantum state is stored as a **flat** CuPy array of shape `(2**max_space,)` with dtype `complex128`. Only the first `2**nqubit` entries are meaningful; the tail is padding to avoid GPU reallocation as qubits come and go during MBQC pattern simulation.

### Qubit ordering

Graphix uses **MSB convention** (qubit 0 = most significant bit = axis 0 of the tensor). cuQuantum uses **LSB convention** (qubit 0 = least significant bit). All calls to cuQuantum APIs convert indices via `_msb_to_lsb()`.

### cuQuantum API used

| Operation | cuQuantum function | Notes |
|---|---|---|
| Single/multi-qubit gate | `custatevec.apply_matrix` | Workspace size queried via `apply_matrix_get_workspace_size` |
| Expectation value | `custatevec.compute_expectation` | Result written to host-side buffer, returned as `complex` |
| Qubit swap | `cp.swapaxes` (CuPy) | cuQuantum `swap_index_bits` available but CuPy tensor axis swap is simpler and equally GPU-accelerated |
| Entangle (CZ) | `custatevec.apply_matrix` with 4×4 CZ matrix | |

### Buffer growth

When `tensor()` needs more space than `max_space`, the buffer is reallocated to exactly the needed size. No doubling strategy is used, since MBQC pattern qubit counts fluctuate and doubling can overshoot GPU memory.

## Project structure

```
src/graphix_statevec_cuquantum/
    __init__.py                        # Exports: Statevec, StatevectorBackend
    graphix_statevec_cuquantum.py       # Main implementation
    README.md                           # This file

tests/
    test_statevec_cuquantum.py          # Unit tests + legacy comparison + pattern sim

benchmarks/
    bench_statevec_cuquantum.py         # Performance benchmarks

run_tests.sh                            # Full test suite script
```

## CI notes

Tests and benchmarks are decorated with `@pytest.mark.skipif(not _gpu_available(), reason="GPU not available")`. On CI runners without GPU they will be skipped. Formatting, linting, and type-checking still run normally.

## Benchmark results (Tesla T4)

Approximate per-operation latency (lower is better):

| Operation | 4 qubits | 8 qubits | 12 qubits | 16 qubits |
|---|---|---|---|---|
| `evolve_single` (H gate) | 45 µs | 46 µs | 45 µs | 46 µs |
| `entangle` (CZ) | 52 µs | 52 µs | 52 µs | 52 µs |
| `expectation_single` (Z) | 85 µs | 80 µs | 81 µs | 106 µs |
| `add_nodes` | 354 µs | 423 µs | 569 µs | 1.6 ms |
| `remove_qubit` | 389 µs | 477 µs | 614 µs | 1.6 ms |

Gate application times are near-constant across qubit counts due to cuQuantum's optimized GPU kernels. Structural operations (`add_nodes`, `remove_qubit`) scale with state size as they involve GPU memory transfers.

## Caveats

- **cuQuantum API version**: This implementation targets cuQuantum Python 26.x (`cuquantum.bindings.custatevec`). Earlier versions (24.x) with `cuquantum.custatevec` and `apply_gate` are **not** compatible.
- **`compute_expectation`**: cuQuantum's own API writes the scalar result to a host-side (CPU) buffer. The computation is fully GPU-accelerated; only the final 16-byte complex value is transferred to CPU.
- **Buffer reallocation**: Each `tensor()` growth triggers a new GPU allocation. For patterns with predictable `max_space`, initializing `Statevec` with the correct `max_space` avoids this.
- **No noise model**: The backend does not implement `apply_noise`.
- **Naming**: `Statevec` and `StatevectorBackend` follow the template's convention. The import path (`graphix_statevec_cuquantum`) disambiguates from the NumPy CPU backend (`graphix.sim.statevec`) and the template (`graphix_statevec_template`). Subject to change based on Graphix maintainer guidelines.
