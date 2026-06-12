from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from graphix_mqtbench import Benchmark, BenchmarkName, OptimizationPass
from graphix.sim.statevec import StatevectorBackend as CPUBackend

if TYPE_CHECKING:
    from pytest_benchmark import BenchmarkFixture


class BenchTest:
    """CPU MQT benchmarks for comparison with GPU."""

    QUBIT_COUNTS = (2, 3)

    @pytest.mark.benchmark(group="mqtbench_cpu_full_adder", max_time=60, min_rounds=1, warmup=False)
    @pytest.mark.parametrize("nqubits", QUBIT_COUNTS)
    def bench_full_adder_cpu(self, benchmark: BenchmarkFixture, nqubits: int) -> None:
        """Benchmark FULL_ADDER pattern on CPU."""
        print(f"\n[CPU] Running FULL_ADDER with {nqubits} qubits...")
        pattern = Benchmark(BenchmarkName.FULL_ADDER, nqubits).to_pattern().minimize_space()
        backend = CPUBackend()
        rng = np.random.default_rng(42)

        def run() -> None:
            pattern.simulate_pattern(backend=backend, rng=rng)

        benchmark(run)

    @pytest.mark.benchmark(group="mqtbench_cpu_qft", max_time=60, min_rounds=1, warmup=False)
    @pytest.mark.parametrize("nqubits", QUBIT_COUNTS)
    def bench_qft_cpu(self, benchmark: BenchmarkFixture, nqubits: int) -> None:
        """Benchmark QFT pattern on CPU."""
        print(f"\n[CPU] Running QFT with {nqubits} qubits...")
        pattern = Benchmark(BenchmarkName.QFT, nqubits).to_pattern().minimize_space()
        backend = CPUBackend()
        rng = np.random.default_rng(42)

        def run() -> None:
            pattern.simulate_pattern(backend=backend, rng=rng)

        benchmark(run)

    @pytest.mark.benchmark(group="mqtbench_cpu_random_circuit", max_time=60, min_rounds=1, warmup=False)
    @pytest.mark.parametrize("nqubits", QUBIT_COUNTS)
    def bench_random_circuit_cpu(self, benchmark: BenchmarkFixture, nqubits: int) -> None:
        """Benchmark RANDOMCIRCUIT pattern on CPU."""
        print(f"\n[CPU] Running RANDOM_CIRCUIT with {nqubits} qubits...")
        pattern = Benchmark(BenchmarkName.RANDOMCIRCUIT, nqubits).to_pattern().minimize_space()
        backend = CPUBackend()
        rng = np.random.default_rng(42)

        def run() -> None:
            pattern.simulate_pattern(backend=backend, rng=rng)

        benchmark(run)
