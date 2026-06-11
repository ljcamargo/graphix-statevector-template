# This module benchmarks the GPU statevector simulator against the MQTBench suite
# with proper GPU synchronization for accurate timing.

from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pytest
from graphix_mqtbench import Benchmark, BenchmarkName, OptimizationPass
from graphix.sim.statevec import StatevectorBackend

if TYPE_CHECKING:
    from pytest_benchmark import BenchmarkFixture


class BenchTest:
    """GPU-accelerated MQT benchmarks with proper synchronization."""

    # Test different qubit counts to see scaling
    QUBIT_COUNTS = (8, 10, 12, 14, 16)

    @pytest.mark.benchmark(group="mqtbench_gpu_full_adder")
    @pytest.mark.parametrize("nqubits", QUBIT_COUNTS)
    def bench_full_adder_gpu(self, benchmark: BenchmarkFixture, nqubits: int) -> None:
        """Benchmark FULL_ADDER pattern on GPU."""
        pattern = Benchmark(BenchmarkName.FULL_ADDER, nqubits).to_pattern().minimize_space()
        backend = StatevectorBackend.with_capacity(max_qubits=nqubits)
        rng = np.random.default_rng(42)

        def run() -> None:
            pattern.simulate_pattern(backend=backend, rng=rng)
            cp.cuda.Device().synchronize()  # Ensure GPU work is complete

        benchmark(run)

    @pytest.mark.benchmark(group="mqtbench_gpu_qft")
    @pytest.mark.parametrize("nqubits", QUBIT_COUNTS)
    def bench_qft_gpu(self, benchmark: BenchmarkFixture, nqubits: int) -> None:
        """Benchmark QFT pattern on GPU."""
        pattern = Benchmark(BenchmarkName.QFT, nqubits).to_pattern().minimize_space()
        backend = StatevectorBackend.with_capacity(max_qubits=nqubits)
        rng = np.random.default_rng(42)

        def run() -> None:
            pattern.simulate_pattern(backend=backend, rng=rng)
            cp.cuda.Device().synchronize()

        benchmark(run)

    @pytest.mark.benchmark(group="mqtbench_gpu_random_circuit")
    @pytest.mark.parametrize("nqubits", QUBIT_COUNTS)
    def bench_random_circuit_gpu(self, benchmark: BenchmarkFixture, nqubits: int) -> None:
        """Benchmark RANDOMCIRCUIT pattern on GPU."""
        pattern = Benchmark(BenchmarkName.RANDOMCIRCUIT, nqubits).to_pattern().minimize_space()
        backend = StatevectorBackend.with_capacity(max_qubits=nqubits)
        rng = np.random.default_rng(42)

        def run() -> None:
            pattern.simulate_pattern(backend=backend, rng=rng)
            cp.cuda.Device().synchronize()

        benchmark(run)

    # Optional: Add CPU comparison benchmarks
    @pytest.mark.benchmark(group="mqtbench_cpu_qft")
    @pytest.mark.parametrize("nqubits", (8, 10, 12))  # CPU can't go as high
    def bench_qft_cpu(self, benchmark: BenchmarkFixture, nqubits: int) -> None:
        """Benchmark QFT pattern on CPU for comparison."""
        from graphix.sim.statevec import StatevectorBackend as CPUBackend

        pattern = Benchmark(BenchmarkName.QFT, nqubits).to_pattern().minimize_space()
        backend = CPUBackend()
        rng = np.random.default_rng(42)

        def run() -> None:
            pattern.simulate_pattern(backend=backend, rng=rng)

        benchmark(run)
