"""Benchmark comparing CPU and GPU statevector backends for MBQC patterns."""

from __future__ import annotations

import timeit
from typing import Any

import cupy as _cp
import matplotlib.pyplot as plt
import numpy as np
from graphix.sim.statevec import StatevectorBackend as CPUBackend
from graphix_mqtbench import Benchmark, BenchmarkName

from graphix_statevec_cuquantum import StatevectorBackend as GPUBackend

cp: Any = _cp


def benchmark_backend(pattern: Any, backend: Any, rng: Any) -> Any:
    """Run pattern simulation with proper GPU synchronization.

    Parameters
    ----------
    pattern : Pattern
        Measurement-based quantum computation pattern.
    backend : Backend
        Simulation backend (CPU or GPU).
    rng : Generator
        Random number generator for measurements.

    Returns
    -------
    Any
        Simulation result from the backend.
    """

    def run() -> Any:
        # For GPU backend, synchronize before and after
        if "GPU" in str(type(backend)):
            cp.cuda.Device().synchronize()
            result = pattern.simulate_pattern(backend=backend, rng=rng)
            cp.cuda.Device().synchronize()
        else:
            result = pattern.simulate_pattern(backend=backend, rng=rng)
        return result

    return run


def run_benchmarks() -> None:
    """Benchmark QFT patterns at various qubit counts for CPU and GPU backends."""
    rng = np.random.default_rng(42)
    qubit_counts = [8, 10, 12, 14, 16, 18, 20]

    cpu_times = []
    gpu_times = []

    for nqubits in qubit_counts:
        print(f"Benchmarking {nqubits} qubits...")

        # Create pattern
        benchmark = Benchmark(BenchmarkName.QFT, nqubits)
        pattern = benchmark.to_pattern().minimize_space()

        # CPU benchmark
        cpu_run = benchmark_backend(pattern, CPUBackend(), rng)
        cpu_timer = timeit.Timer(cpu_run)
        cpu_time = min(cpu_timer.repeat(number=1, repeat=5))
        cpu_times.append(cpu_time)

        # GPU benchmark (with pre-allocated capacity)
        gpu_backend = GPUBackend.with_capacity(max_qubits=nqubits)
        gpu_run = benchmark_backend(pattern, gpu_backend, rng)
        gpu_timer = timeit.Timer(gpu_run)
        gpu_time = min(gpu_timer.repeat(number=1, repeat=5))
        gpu_times.append(gpu_time)

        print(f"  CPU: {cpu_time:.5f}s, GPU: {gpu_time:.5f}s, Speedup: {cpu_time / gpu_time:.2f}x")

    # Plot results

    plt.figure()
    plt.plot(qubit_counts, cpu_times, "o-", label="CPU")
    plt.plot(qubit_counts, gpu_times, "s-", label="GPU")
    plt.xlabel("Number of Qubits")
    plt.ylabel("Simulation Time (seconds)")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("benchmark_results.png", dpi=150)
    print("\nPlot saved as benchmark_results.png")


if __name__ == "__main__":
    run_benchmarks()
