from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from graphix.clifford import Clifford
from graphix.states import BasicStates
from graphix.sim.statevec import Statevec

if TYPE_CHECKING:
    from pytest_benchmark import BenchmarkFixture


nqubits = (4, 16)


class BenchTest:  # ← REMOVED the @pytest.mark.skip line
    group = "bench_statevec"

    @pytest.mark.benchmark(group=group, max_time=1)
    @pytest.mark.parametrize("nqubit", nqubits)
    def bench_expectation_single(self, benchmark: BenchmarkFixture, nqubit: int) -> None:
        sv = Statevec(nqubit=nqubit, data=BasicStates.ZERO)
        op = Clifford.H.matrix
        q = nqubit - 1

        def run() -> complex:
            return sv.expectation_single(op, q)

        assert benchmark(run) == pytest.approx(1 / np.sqrt(2))

    @pytest.mark.benchmark(group=group, max_time=1)
    @pytest.mark.parametrize("nqubit", nqubits)
    def bench_evolve_single(self, benchmark: BenchmarkFixture, nqubit: int) -> None:
        sv = Statevec(nqubit=nqubit, data=BasicStates.ZERO)
        op = Clifford.H.matrix
        q = nqubit - 1

        def run() -> None:
            sv.evolve_single(op, q)

        benchmark(run)

    @pytest.mark.benchmark(group=group, max_time=1)
    @pytest.mark.parametrize("nqubit", nqubits)
    def bench_add_nodes(self, benchmark: BenchmarkFixture, nqubit: int) -> None:
        def run() -> Statevec:
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
            sv = Statevec(nqubit=nqubit, data=BasicStates.PLUS)
            sv.remove_qubit(nqubit - 1)
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
