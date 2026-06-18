from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from graphix.clifford import Clifford
from graphix.sim.statevec import Statevec
from graphix.states import BasicStates

if TYPE_CHECKING:
    from pytest_benchmark import BenchmarkFixture


nqubits = (4, 16)


class BenchTest:
    group = "bench_statevec"

    @pytest.mark.benchmark(group=group, max_time=1)
    @pytest.mark.parametrize("nqubit", nqubits)
    def bench_expectation_single(self, benchmark: BenchmarkFixture, nqubit: int) -> None:
        sv = Statevec(nqubit=nqubit, data=BasicStates.ZERO)
        op = Clifford.H.matrix
        q = nqubit - 1  # Apply gate on last qubit

        def run() -> complex:
            return sv.expectation_single(op, q)

        assert benchmark(run) == pytest.approx(1 / np.sqrt(2))

    @pytest.mark.benchmark(group=group, max_time=1)
    @pytest.mark.parametrize("nqubit", nqubits)
    def bench_evolve_single(self, benchmark: BenchmarkFixture, nqubit: int) -> None:
        sv = Statevec(nqubit=nqubit, data=BasicStates.ZERO)
        op = Clifford.H.matrix
        q = nqubit - 1  # Apply gate on last qubit

        # Since `sv` is modified in-place, the output of `benchmark(run)` will depend on
        # the number of benchmark iterations which is automatically calculated by
        # pytest-benchmark.  To test the output we would need to initialize a fresh
        # `Statevec` before each benchmark iteration.  Using `benchmark.pedantic` with a
        # setup function does not allow to have more than 1 benchmark iterations (as of
        # version 5.2.3), which leads to unreliable results for small instances.  A
        # possible solution is to instantiate a fresh `Statevec` inside the `run`
        # function, but here we choose not do it to have a more faithful benchmark of
        # `evolve_single`.

        def run() -> None:
            sv.evolve_single(op, q)

        benchmark(run)

    @pytest.mark.benchmark(group=group, max_time=1)
    @pytest.mark.parametrize("nqubit", nqubits)
    def bench_add_nodes(self, benchmark: BenchmarkFixture, nqubit: int) -> None:
        def run() -> Statevec:
            # Here we have to initialize a fresh statevector before each iteration,
            # otherwise the statevector becomes of size 2**(number_of_iterations).
            sv = Statevec(nqubit=nqubit, data=BasicStates.ZERO)
            sv.add_nodes(nqubit=1, data=BasicStates.PLUS)
            return sv

        sv = benchmark(run)
        sv_ref = Statevec(nqubit=nqubit + 1, data=[BasicStates.ZERO] * nqubit + [BasicStates.PLUS])
        assert np.allclose(sv.flatten(), sv_ref.flatten())

    @pytest.mark.benchmark(group=group, max_time=1)
    @pytest.mark.parametrize("nqubit", nqubits)
    def bench_remove_qubit(self, benchmark: BenchmarkFixture, nqubit: int) -> None:
        def run() -> Statevec:
            # Here we have to initialize a fresh statevector before each iteration,
            # otherwise the statevector does not have a constant number of qubits.
            sv = Statevec(nqubit=nqubit, data=BasicStates.PLUS)
            sv.remove_qubit(nqubit - 1)  # We remove last qubit
            return sv

        sv = benchmark(run)
        sv_ref = Statevec(nqubit=nqubit - 1, data=BasicStates.PLUS)
        assert np.allclose(sv.flatten(), sv_ref.flatten())

    @pytest.mark.benchmark(group=group, max_time=1)
    @pytest.mark.parametrize("nqubit", nqubits)
    def bench_entangle(self, benchmark: BenchmarkFixture, nqubit: int) -> None:
        sv = Statevec(nqubit=nqubit, data=BasicStates.PLUS)

        def run() -> None:
            sv.entangle((0, nqubit - 1))

        benchmark(run)
