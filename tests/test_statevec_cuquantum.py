"""Tests for the cuQuantum-accelerated GPU statevector backend."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
from graphix.clifford import Clifford
from graphix.random_objects import rand_circuit
from graphix.sim.statevec import Statevec as StatevecCPU
from graphix.sim.statevec import StatevectorBackend as StatevectorBackendCPU
from graphix.states import BasicStates
from numpy.random import Generator

from graphix_statevec_cuquantum import Statevec, StatevectorBackend


def _gpu_available() -> bool:
    """Check if a CUDA GPU is available for testing."""
    try:
        import cupy as cp  # noqa: PLC0415

        cp.zeros(1)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return False
    else:
        return True


if TYPE_CHECKING:
    from graphix.states import State
    from numpy.random import PCG64


def generate_rnd_data(rng: Generator, nqubits: int) -> npt.NDArray[np.complex128]:
    length = 1 << nqubits
    data = rng.random(length) + 1j * rng.random(length)
    data /= np.sqrt(np.sum(np.abs(data) ** 2))
    return data


@pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
class TestStatevecCuQuantum:
    """Self-contained unit tests for the cuQuantum statevector."""

    N_JUMPS = 3

    @pytest.mark.parametrize(
        ("state", "data_ref"),
        [
            (BasicStates.PLUS, np.array([1, 1] / np.sqrt(2))),
            (BasicStates.MINUS, np.array([1, -1] / np.sqrt(2))),
            (BasicStates.ZERO, np.array([1, 0])),
            (BasicStates.ONE, np.array([0, 1])),
            (BasicStates.PLUS_I, np.array([1, 1j] / np.sqrt(2))),
            (BasicStates.MINUS_I, np.array([1, -1j] / np.sqrt(2))),
        ],
    )
    def test_init_basic_states(self, state: State, data_ref: npt.NDArray[np.complex128]) -> None:
        sv = Statevec(data=state)
        assert np.allclose(sv.flatten(), data_ref)

    @pytest.mark.parametrize("nqubit", range(5))
    def test_init_random_state(self, fx_rng: Generator, nqubit: int) -> None:
        data = generate_rnd_data(fx_rng, nqubit)
        sv = Statevec(data)
        assert np.allclose(sv.flatten(), data)

    @pytest.mark.parametrize(
        ("sv", "edge", "data_ref"),
        [
            (Statevec(data=BasicStates.ZERO, nqubit=2), (0, 1), np.array([1, 0, 0, 0])),
            (Statevec(data=[BasicStates.PLUS, BasicStates.PLUS]), (0, 1), np.array([1, 1, 1, -1]) / 2),
            (Statevec(data=[BasicStates.ONE, BasicStates.MINUS]), (0, 1), np.array([0, 0, 1, 1]) / np.sqrt(2)),
            (
                Statevec(data=np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)),
                (0, 2),
                np.array([1, 0, 0, 0, 0, 0, 0, -1]) / np.sqrt(2),
            ),
        ],
    )
    def test_entangle(self, sv: Statevec, edge: tuple[int, int], data_ref: npt.NDArray[np.complex128]) -> None:
        sv.entangle(edge)
        assert np.allclose(sv.flatten(), data_ref)

    @pytest.mark.parametrize(
        ("sv", "q", "op", "data_ref"),
        [
            (Statevec(data=BasicStates.ZERO, nqubit=2), 0, Clifford.X.matrix, np.array([0, 0, 1, 0])),
            (
                Statevec(data=[BasicStates.PLUS, BasicStates.PLUS]),
                1,
                Clifford.H.matrix,
                np.array([1, 0, 1, 0]) / np.sqrt(2),
            ),
            (
                Statevec(data=[BasicStates.PLUS, BasicStates.MINUS]),
                0,
                np.array([[1, 0], [0, np.exp(0.25j * np.pi)]]),
                np.array([1, -1, np.exp(0.25j * np.pi), -np.exp(0.25j * np.pi)]) / 2,
            ),
            (
                Statevec(data=np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)),
                1,
                Clifford.Z.matrix,
                np.array([1, 0, 0, 0, 0, 0, 0, -1]) / np.sqrt(2),
            ),
        ],
    )
    def test_evolve_single(
        self, sv: Statevec, q: int, op: npt.NDArray[np.complex128], data_ref: npt.NDArray[np.complex128]
    ) -> None:
        sv.evolve_single(op, q)
        assert np.allclose(sv.flatten(), data_ref)

    @pytest.mark.parametrize(
        ("sv", "q", "op", "exp_ref"),
        [
            (Statevec(data=BasicStates.ZERO, nqubit=2), 0, Clifford.X.matrix, 0),
            (Statevec(data=[BasicStates.PLUS, BasicStates.PLUS]), 1, Clifford.H.matrix, 1 / np.sqrt(2)),
            (
                Statevec(data=[BasicStates.PLUS, BasicStates.MINUS]),
                0,
                np.array([[1, 0], [0, np.exp(0.25j * np.pi)]]),
                (1 + np.exp(0.25j * np.pi)) / 2,
            ),
            (
                Statevec(data=np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2)),
                1,
                Clifford.Z.matrix,
                0,
            ),
        ],
    )
    def test_expectation_single(
        self, sv: Statevec, q: int, op: npt.NDArray[np.complex128], exp_ref: np.complex128
    ) -> None:
        assert np.isclose(sv.expectation_single(op, q), exp_ref)

    def test_add_nodes(self, fx_rng: Generator) -> None:
        max_qubits = 5
        sv_test = Statevec(nqubit=0)
        psi_ref = np.array([1.0 + 0.0j])

        for _ in range(max_qubits):
            data = generate_rnd_data(fx_rng, nqubits=1)
            psi_ref = np.kron(psi_ref, data)
            sv_test.add_nodes(1, data)
            assert np.allclose(sv_test.flatten(), psi_ref)

    @pytest.mark.parametrize(
        ("sv", "q", "sv_ref"),
        [
            (Statevec(data=BasicStates.ZERO, nqubit=2), 0, Statevec(data=BasicStates.ZERO, nqubit=1)),
            (Statevec(data=[BasicStates.PLUS, BasicStates.PLUS]), 1, Statevec(data=BasicStates.PLUS, nqubit=1)),
            (Statevec(data=[BasicStates.PLUS, BasicStates.MINUS]), 0, Statevec(data=BasicStates.MINUS, nqubit=1)),
            (Statevec(data=[BasicStates.ZERO, BasicStates.ONE]), 0, Statevec(data=BasicStates.ONE, nqubit=1)),
            (
                Statevec(data=[BasicStates.PLUS_I, BasicStates.ONE, BasicStates.PLUS]),
                1,
                Statevec(data=[BasicStates.PLUS_I, BasicStates.PLUS], nqubit=2),
            ),
        ],
    )
    def test_remove_qubit(self, sv: Statevec, q: int, sv_ref: Statevec) -> None:
        sv.remove_qubit(q)
        assert np.allclose(sv.flatten(), sv_ref.flatten())


@pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
class TestStatevecCPU:
    """Compare cuQuantum backend results against the reference NumPy statevector."""

    N_JUMPS = 3

    @pytest.mark.parametrize("jumps", range(1, N_JUMPS))
    def test_entangle(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        sv_test = Statevec(generate_rnd_data(rng, nqubits))
        sv_ref = StatevecCPU(data=sv_test.flatten())
        edge: tuple[int, int] = tuple(rng.choice(range(nqubits), size=2, replace=False))
        for sv in [sv_test, sv_ref]:
            sv.entangle(edge)

        assert sv_ref.isclose(StatevecCPU(data=sv_test.flatten()))

    @pytest.mark.parametrize("jumps", range(1, N_JUMPS))
    def test_swap(self, fx_bg: PCG64, jumps: int) -> None:
        rng = Generator(fx_bg.jumped(jumps))
        nqubits = 5
        sv_test = Statevec(generate_rnd_data(rng, nqubits))
        sv_ref = StatevecCPU(data=sv_test.flatten())
        edge: tuple[int, int] = tuple(rng.choice(range(nqubits), size=2, replace=False))
        for sv in [sv_test, sv_ref]:
            sv.swap(edge)

        assert sv_ref.isclose(StatevecCPU(data=sv_test.flatten()))

    def test_evolve_single(self, fx_rng: Generator) -> None:
        nqubits = 5
        for clifford in Clifford:
            sv_test = Statevec(generate_rnd_data(fx_rng, nqubits))
            sv_ref = StatevecCPU(data=sv_test.flatten())
            qubit = int(fx_rng.integers(0, nqubits))
            for sv in [sv_test, sv_ref]:
                sv.evolve_single(clifford.matrix, qubit)
            assert sv_ref.isclose(StatevecCPU(data=sv_test.flatten()))

    def test_expectation_single(self, fx_rng: Generator) -> None:
        nqubits = 5
        for clifford in Clifford:
            sv_test = Statevec(generate_rnd_data(fx_rng, nqubits))
            sv_ref = StatevecCPU(data=sv_test.flatten())
            qubit = int(fx_rng.integers(0, nqubits))

            val_test = sv_test.expectation_single(clifford.matrix, qubit)
            val_ref = sv_ref.expectation_single(clifford.matrix, qubit)

            assert math.isclose(val_test.real, val_ref.real, abs_tol=1e-12)
            assert math.isclose(val_test.imag, val_ref.imag, abs_tol=1e-12)

    def test_add_nodes(self, fx_rng: Generator) -> None:
        max_qubits = 5
        sv_test = Statevec(nqubit=0)
        sv_ref = StatevecCPU(nqubit=0)

        for _ in range(max_qubits):
            data = generate_rnd_data(fx_rng, nqubits=1)
            sv_test.add_nodes(1, data)
            sv_ref.add_nodes(1, data)

            assert sv_ref.isclose(StatevecCPU(data=sv_test.flatten()))

    @pytest.mark.parametrize(
        "projector",
        [np.array([[1, 0], [0, 0]], dtype=np.complex128), np.array([[0, 0], [0, 1]], dtype=np.complex128)],
    )
    def test_remove_nodes(self, fx_rng: Generator, projector: npt.NDArray[np.complex128]) -> None:
        nqubits = 5
        sv_test = Statevec(generate_rnd_data(fx_rng, nqubits))
        sv_ref = StatevecCPU(data=sv_test.flatten())
        q = 0
        for _ in range(nqubits - 1):
            sv_test.evolve_single(projector, q)
            sv_test.remove_qubit(q)
            sv_ref.evolve_single(projector, q)
            sv_ref.remove_qubit(q)

            assert sv_ref.isclose(StatevecCPU(data=sv_test.flatten()))


@pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
@pytest.mark.parametrize("jumps", range(1, 6))
def test_pattern_simulator(fx_bg: PCG64, jumps: int) -> None:
    """End-to-end pattern simulation with the cuQuantum backend."""
    rng = Generator(fx_bg.jumped(jumps))

    nqubits = 5

    pattern = rand_circuit(nqubits, depth=5, rng=rng).transpile().pattern
    pattern.remove_pauli_measurements()

    sv_test = pattern.simulate_pattern(backend=StatevectorBackend(), rng=rng)
    sv_ref = pattern.simulate_pattern(backend=StatevectorBackendCPU(), rng=rng)

    assert sv_ref.isclose(StatevecCPU(data=sv_test.flatten()))
