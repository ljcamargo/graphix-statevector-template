from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pytest
from graphix_mqtbench import Benchmark, BenchmarkName, OptimizationPass

from graphix_statevec_cuquantum import StatevectorBackend

if TYPE_CHECKING:
    from pytest_benchmark import BenchmarkFixture


class BenchTest:
    """GPU-accelerated MQT benchmarks with proper synchronization."""

    # Start with smaller qubit counts to verify it works
    QUBIT_COUNTS = (4, 6, 8, 10, 12)

    @pytest.mark.benchmark(group="mqtbench_gpu_full_adder")
    @pytest.mark.parametrize("nqubits", QUBIT_COUNTS)
    def bench_full_adder_gpu(self, benchmark: BenchmarkFixture, nqubits: int) -> None:
        """Benchmark FULL_ADDER pattern on GPU."""
        pattern = Benchmark(BenchmarkName.FULL_ADDER, nqubits).to_pattern().minimize_space()
        backend = StatevectorBackend()
        rng = np.random.default_rng(42)

        def run() -> None:
            pattern.simulate_pattern(backend=backend, rng=rng)
            cp.cuda.Device().synchronize()

        benchmark(run)

    @pytest.mark.benchmark(group="mqtbench_gpu_qft")
    @pytest.mark.parametrize("nqubits", QUBIT_COUNTS)
    def bench_qft_gpu(self, benchmark: BenchmarkFixture, nqubits: int) -> None:
        """Benchmark QFT pattern on GPU."""
        pattern = Benchmark(BenchmarkName.QFT, nqubits).to_pattern().minimize_space()
        backend = StatevectorBackend()
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
        backend = StatevectorBackend()
        rng = np.random.default_rng(42)

        def run() -> None:
            pattern.simulate_pattern(backend=backend, rng=rng)
            cp.cuda.Device().synchronize()

        benchmark(run)
