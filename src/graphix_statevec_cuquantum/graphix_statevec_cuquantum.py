"""MBQC state vector backend simulator using cuQuantum Python for GPU acceleration.

Requires ``cupy`` and ``cuquantum-python`` with a CUDA-capable GPU.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, SupportsComplex, SupportsFloat, override

import cupy as cp
import numpy as np
from cuquantum import custatevec
from graphix.parameter import Expression
from graphix.sim.base_backend import DenseState, DenseStateBackend, Matrix
from graphix.states import BasicStates, State

if TYPE_CHECKING:
    from collections.abc import Sequence

    from graphix.sim.data import Data

# ---------------------------------------------------------------------------
# CZ tensor for entangle
# ---------------------------------------------------------------------------
_CZ = cp.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
    dtype=cp.complex128,
)

# Global cuStateVec handle (created once, reused across Statevec instances)
_HANDLE: int | None = None


def _handle() -> int:
    global _HANDLE  # noqa: PLW0603
    if _HANDLE is None:
        _HANDLE = custatevec.create()
    return _HANDLE


def _cu_dtype(dt: cp.dtype) -> int:
    if dt == cp.complex128:
        return custatevec.cuDataType.CUDA_C_128F
    if dt == cp.uint64:
        return custatevec.cuDataType.CUDA_U_64
    raise TypeError(f"Unsupported dtype: {dt}")


def _msb_to_lsb(targets: list[int], nq: int) -> list[int]:
    """Graphix MSB → cuQuantum LSB."""
    return [nq - 1 - t for t in targets]


class Statevec(DenseState):
    """GPU-accelerated statevector using cuQuantum.

    The state is stored as a flat ``cupy.ndarray`` of shape ``(2**max_space,)``
    with dtype ``complex128``.  Only the first ``2**nqubit`` entries are
    meaningful; the tail is padding to avoid reallocation as qubits come
    and go during pattern simulation.

    Parameters
    ----------
    data : Data
        Input data.  Defaults to ``BasicStates.PLUS``.
    nqubit : int, optional
        Number of qubits.  Inferred from *data* if ``None``.
    max_space : int, optional
        Allocated qubit capacity.  Defaults to *nqubit*.

    Attributes
    ----------
    psi : cupy.ndarray
        Flat GPU array of shape ``(2**max_space,)``.
    _nqubit : int
        Currently active qubits.
    max_space : int
        Allocated qubit capacity.
    """

    psi: cp.ndarray
    _nqubit: int
    max_space: int

    def __init__(
        self,
        data: Data = BasicStates.PLUS,
        nqubit: int | None = None,
        max_space: int | None = None,
    ) -> None:
        if nqubit is not None and nqubit < 0:
            raise ValueError("nqubit must be a non-negative integer.")

        # Copy constructor
        if isinstance(data, Statevec):
            if nqubit is not None and nqubit != data._nqubit:
                raise ValueError(f"Inconsistent nqubit={nqubit} vs. Statevec nqubit={data._nqubit}")
            self._nqubit = data._nqubit
            self.max_space = data.max_space
            self.psi = data.psi.copy()
            return

        # Parse input
        if isinstance(data, State):
            if nqubit is None:
                nqubit = 1
            items: list = [data] * nqubit
        elif isinstance(data, Iterable):
            items = list(data)
        else:
            raise TypeError(f"Incorrect type for data: {type(data)}")

        if len(items) == 0:
            if nqubit is not None and nqubit != 0:
                raise ValueError("nqubit is not null but input state is empty.")
            nqubit = 0
            cpu = np.array([1.0 + 0.0j], dtype=np.complex128)
        elif isinstance(items[0], State):
            if nqubit is None:
                nqubit = len(items)
            elif nqubit != len(items):
                raise ValueError("Mismatch between nqubit and length of input state.")
            vecs = [s.to_statevector() for s in items]
            tmp = functools.reduce(lambda a, b: np.kron(a, b).astype(np.complex128), vecs)
            cpu = tmp.ravel()
        elif isinstance(items[0], (Expression, SupportsComplex, SupportsFloat)):
            if nqubit is None:
                nvals = len(items)
                if nvals & (nvals - 1):
                    raise ValueError("Length is not a power of two")
                nqubit = nvals.bit_length() - 1
            elif nqubit != (len(items).bit_length() - 1):
                raise ValueError("Mismatch between nqubit and length of input state")
            cpu = np.array(items, dtype=np.complex128)
            if cpu.dtype != "O" and not np.allclose(np.sqrt(np.sum(np.abs(cpu) ** 2)), 1):
                raise ValueError("Input state is not normalized")
        else:
            raise TypeError(f"First element of data has type {type(items[0])} whereas a State or number is expected")

        if max_space is None:
            max_space = nqubit
        elif max_space < nqubit:
            raise ValueError("max_space must be >= nqubit")

        self._nqubit = nqubit
        self.max_space = max_space
        self.psi = cp.zeros(1 << max_space, dtype=cp.complex128)
        if nqubit > 0:
            self.psi[: 1 << nqubit] = cp.asarray(cpu, dtype=cp.complex128)
        elif nqubit == 0:
            self.psi[0] = cp.asarray(cpu, dtype=cp.complex128)

    # -- properties ------------------------------------------------------ #

    @property
    @override
    def nqubit(self) -> int:
        return self._nqubit

    # -- flatten --------------------------------------------------------- #

    @override
    def flatten(self) -> Matrix:
        return cp.asnumpy(self.psi[: 1 << self._nqubit])

    # -- add_nodes ------------------------------------------------------- #

    @override
    def add_nodes(self, n: int, data: Data) -> None:
        sv = Statevec(nqubit=n, data=data)
        self.tensor(sv)

    # -- entangle -------------------------------------------------------- #

    @override
    def entangle(self, edge: tuple[int, int]) -> None:
        self._apply_gate(_CZ, list(edge))

    # -- evolve ---------------------------------------------------------- #

    @override
    def evolve(self, op: Matrix, qargs: Sequence[int]) -> None:
        self._apply_gate(cp.asarray(op, dtype=cp.complex128), list(qargs))

    @override
    def evolve_single(self, op: Matrix, i: int) -> None:
        self._apply_gate(cp.asarray(op, dtype=cp.complex128), [i])

    # -- expectation_single ---------------------------------------------- #

    @override
    def expectation_single(self, op: Matrix, loc: int) -> complex:
        g = cp.asarray(op, dtype=cp.complex128)
        return complex(self._expectation(g, [loc]))

    # -- remove_qubit ---------------------------------------------------- #

    @override
    def remove_qubit(self, qarg: int) -> None:
        """Remove a separable qubit, keeping the branch with non-zero norm."""
        n = self._nqubit
        t = self.psi[: 1 << n].reshape((2,) * n)

        idx: list[slice | int] = [slice(None)] * n
        idx[qarg] = 0
        br = t[tuple(idx)].ravel()
        nrm2 = float(cp.sum(cp.abs(br) ** 2))
        if math.isclose(nrm2, 0, abs_tol=1e-15):
            idx[qarg] = 1
            br = t[tuple(idx)].ravel()
            nrm2 = float(cp.sum(cp.abs(br) ** 2))
        br /= math.sqrt(nrm2)

        self.psi[: 1 << (n - 1)] = br
        self._nqubit -= 1

    # -- swap ------------------------------------------------------------ #

    @override
    def swap(self, qubits: tuple[int, int]) -> None:
        i, j = qubits
        if i == j or self._nqubit == 0:
            return
        n = self._nqubit
        bit_i = n - 1 - i
        bit_j = n - 1 - j

        idxs = cp.arange(1 << n, dtype=cp.uint64)
        bi = (idxs >> bit_i) & 1
        bj = (idxs >> bit_j) & 1
        mask = cp.where(bi != bj, (1 << bit_i) | (1 << bit_j), 0)
        gather = idxs ^ mask

        h = _handle()
        active = self.psi[: 1 << n]
        custatevec.swap_index_bits(
            h,
            active.data.ptr,
            _cu_dtype(active.dtype),
            gather.data.ptr,
            _cu_dtype(gather.dtype),
            1 << n,
        )

    # -- tensor ---------------------------------------------------------- #

    def tensor(self, other: Statevec) -> None:
        """In-place tensor product ``self ⊗ other``."""
        ns = self._nqubit
        no = other._nqubit
        total = ns + no

        if total > self.max_space:
            new_max = max(total, self.max_space * 2) if self.max_space > 0 else total
            buf = cp.zeros(1 << new_max, dtype=cp.complex128)
            buf[: 1 << ns] = self.psi[: 1 << ns]
            self.psi = buf
            self.max_space = new_max

        a = self.psi[: 1 << ns]
        b = other.psi[: 1 << no]
        self.psi[: 1 << total] = cp.kron(a, b)
        self._nqubit = total

    # -- cuQuantum helpers ----------------------------------------------- #

    def _apply_gate(self, gate: cp.ndarray, targets: list[int]) -> None:
        if self._nqubit == 0:
            return
        n = self._nqubit
        active = self.psi[: 1 << n]
        t = _msb_to_lsb(targets, n)
        h = _handle()
        custatevec.apply_gate(
            h,
            active.data.ptr,
            _cu_dtype(active.dtype),
            gate.data.ptr,
            _cu_dtype(gate.dtype),
            t,
            len(t),
            0,
            0,
            0,  # no controls
            0,  # adjoint = False
        )

    def _expectation(self, gate: cp.ndarray, targets: list[int]) -> float:
        if self._nqubit == 0:
            return 1.0
        n = self._nqubit
        active = self.psi[: 1 << n]
        t = _msb_to_lsb(targets, n)
        h = _handle()
        r = custatevec.compute_expectation(
            h,
            active.data.ptr,
            _cu_dtype(active.dtype),
            gate.data.ptr,
            _cu_dtype(gate.dtype),
            t,
            len(t),
            0,
            0,
            0,
        )
        return float(r)

    # -- helpers --------------------------------------------------------- #

    def normalize(self) -> None:
        """Normalise the state in-place."""
        a = self.psi[: 1 << self._nqubit]
        a /= math.sqrt(float(cp.sum(cp.abs(a) ** 2)))

    def dims(self) -> tuple[int, ...]:
        """Return the tensor shape ``(2,) * nqubit``."""
        return (2,) * self._nqubit

    def isclose(self, other: Statevec, *, rtol: float = 1e-09, atol: float = 0.0) -> bool:
        """Check equality up to global phase via fidelity."""
        return math.isclose(self.fidelity(other), 1, rel_tol=rtol, abs_tol=atol)

    def fidelity(self, other: Statevec) -> float:
        r"""Fidelity :math:`|\langle\psi_1|\psi_2\rangle|^2`."""
        a = self.psi[: 1 << self._nqubit]
        b = other.psi[: 1 << other._nqubit]
        ip = float(cp.dot(a.conj(), b))
        return ip.real**2 + ip.imag**2

    def copy(self) -> Statevec:
        """Return a deep copy."""
        return copy.deepcopy(self)


@dataclass(frozen=True)
class StatevectorBackend(DenseStateBackend[Statevec]):
    """MBQC backend with cuQuantum-accelerated statevector simulation."""

    state: Statevec = dataclasses.field(init=False, default_factory=lambda: Statevec(nqubit=0))
