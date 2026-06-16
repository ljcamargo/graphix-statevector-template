"""MBQC state vector backend simulator using cuQuantum Python for GPU acceleration.

Requires ``cupy`` and ``cuquantum-python`` with a CUDA-capable GPU.
"""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, Tuple, cast, override  # noqa: UP035

import cupy as _cp
import numpy as np
from cuquantum import cudaDataType  # type: ignore[attr-defined]
from cuquantum.bindings import custatevec
from graphix.sim.base_backend import DenseState, DenseStateBackend, Matrix
from graphix.sim.statevec import Statevec as BaseStatevec
from graphix.states import BasicStates

# cupy has no type stubs — treat as Any to avoid per-line attr-defined errors
cp: Any = _cp

if TYPE_CHECKING:
    from collections.abc import Sequence

    from graphix.sim.data import Data

# ---------------------------------------------------------------------------
# cuQuantum constants
# ---------------------------------------------------------------------------
_SV_DTYPE = cudaDataType.CUDA_C_64F
_LAYOUT = custatevec.MatrixLayout.ROW
_COMPUTE = custatevec.ComputeType.COMPUTE_DEFAULT

# Global cuStateVec handle (created once, reused across Statevec instances)
_HANDLE: int = custatevec.create()

# Common quantum gates
_CZ = cp.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
    dtype=cp.complex128,
)
_PLUS_STATE = cp.array([1.0, 1.0], dtype=cp.complex128) / cp.sqrt(2.0)


def _handle() -> int:
    """Return the global cuStateVec handle, creating it on first access.

    Returns
    -------
    int
        Opaque handle to the cuStateVec library context.
    """
    return _HANDLE


def _msb_to_lsb(targets: list[int], nq: int) -> Tuple[int, ...]:  # noqa: UP006
    """Convert qubit indices from Graphix MSB convention to cuQuantum LSB convention.

    Parameters
    ----------
    targets : list of int
        Target qubit indices in MSB order.
    nq : int
        Total number of qubits.

    Returns
    -------
    tuple of int
        Indices in LSB order.
    """
    return tuple(nq - 1 - t for t in targets)


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
    """

    psi: cp.ndarray
    _nqubit: int
    max_space: int

    def __init__(
        self,
        data: Data | cp.ndarray = BasicStates.PLUS,
        nqubit: int | None = None,
        max_space: int | None = None,
    ) -> None:
        # Handle GPU array input
        if isinstance(data, cp.ndarray):
            if nqubit is None:
                # Infer nqubit from array length
                length = len(data)
                if length & (length - 1):
                    raise ValueError("Array length must be a power of 2")
                nqubit = length.bit_length() - 1

            # Validate normalization
            norm = cp.sqrt(cp.sum(cp.abs(data) ** 2))
            if not cp.isclose(norm, 1.0):
                raise ValueError("Input state is not normalized")

            # Determine max_space
            actual_max_space: int
            if max_space is None:
                actual_max_space = nqubit
            else:
                if max_space < nqubit:
                    raise ValueError("max_space must be >= nqubit")
                actual_max_space = max_space

            self._nqubit = nqubit
            self.max_space = actual_max_space
            self.psi = cp.zeros(1 << actual_max_space, dtype=cp.complex128)
            self.psi[: 1 << nqubit] = data.astype(cp.complex128)
            return

        base = BaseStatevec(data, nqubit)

        # Determine the actual max_space value
        if max_space is None:
            actual_max_space = base.nqubit
        else:
            if max_space < base.nqubit:
                raise ValueError("max_space must be >= nqubit")
            actual_max_space = max_space

        # Initializing GPU state with padding
        self._nqubit = base.nqubit
        self.max_space = actual_max_space
        self.psi = cp.zeros(1 << actual_max_space, dtype=cp.complex128)

        # Copying the validated state to GPU
        size = 1 << self._nqubit
        if self._nqubit > 0:
            self.psi[:size] = cp.asarray(base.psi.flatten()[:size], dtype=cp.complex128)
        elif self._nqubit == 0:
            self.psi[0] = cp.asarray(base.psi.item(), dtype=cp.complex128)

    # -- properties ------------------------------------------------------ #

    @property
    def _active_psi(self) -> Any:
        """Return the active portion of the state vector (first 2^n elements)."""
        return self.psi[: 1 << self._nqubit]

    @_active_psi.setter
    def _active_psi(self, value: cp.ndarray) -> None:
        """Set the active portion of the state vector."""
        self.psi[: 1 << self._nqubit] = value

    @property
    @override
    def nqubit(self) -> int:
        """Return the number of qubits currently in the state.

        Returns
        -------
        int
            Number of qubits.
        """
        return self._nqubit

    # -- capacity management --------------------------------------------- #

    def _ensure_capacity(self, required_qubits: int) -> None:
        """Ensure the state vector has capacity for at least `required_qubits` qubits.

        If current capacity is insufficient, reallocate with larger padding.

        Parameters
        ----------
        required_qubits : int
            Minimum number of qubits needed.
        """
        required_size = 1 << required_qubits
        current_capacity = 1 << self.max_space

        if required_size <= current_capacity:
            return

        # Grow capacity: at least double or add 1 qubit, whichever is larger
        new_max = max(self.max_space + 1, required_qubits)
        new_psi = cp.zeros(1 << new_max, dtype=cp.complex128)
        new_psi[: 1 << self._nqubit] = self.psi[: 1 << self._nqubit]
        self.psi = new_psi
        self.max_space = new_max

    # -- public methods -------------------------------------------------- #
    # -- flatten --------------------------------------------------------- #

    @override
    def flatten(self) -> Matrix:
        """Return the flattened state vector as a CPU numpy array.

        Copies the active portion of the GPU state back to the host.

        Returns
        -------
        Matrix
            Numpy array of shape ``(2**nqubit,)`` with dtype ``complex128``.
        """
        return cast("Matrix", cp.asnumpy(self.psi[: 1 << self._nqubit]))

    # -- add_nodes ------------------------------------------------------- #

    @override
    def add_nodes(self, nqubit: int, data: Data) -> None:
        r"""Add qubits and initialise them in a given state.

        A fast path is used when adding a single qubit in the
        :math:`|+\rangle` state.

        Parameters
        ----------
        nqubit : int
            Number of qubits to add.
        data : Data
            State in which to initialise the new qubits.
        """
        if nqubit == 1 and data is BasicStates.PLUS:
            old_size = 1 << self._nqubit
            new_size = old_size * 2
            new_nqubit = self._nqubit + 1

            # Ensure we have enough capacity
            self._ensure_capacity(new_nqubit)

            # Use cp.kron for correct tensor product with |+>
            new_state = cp.kron(self.psi[:old_size], _PLUS_STATE)
            self.psi[:new_size] = new_state
            self._nqubit = new_nqubit
        else:
            # General case: use the standard tensor product
            sv = Statevec(nqubit=nqubit, data=data)
            self.tensor(sv)

    # -- entangle -------------------------------------------------------- #

    @override
    def entangle(self, edge: tuple[int, int]) -> None:
        """Apply a CZ gate between two qubits.

        Parameters
        ----------
        edge : tuple of int
            (control, target) qubit indices.
        """
        self._apply_matrix(_CZ, list(edge))

    # -- evolve ---------------------------------------------------------- #

    @override
    def evolve(self, op: Matrix, qargs: Sequence[int]) -> None:
        r"""Apply a multi-qubit gate operation.

        Parameters
        ----------
        op : Matrix
            2\ :sup:`n` x 2\ :sup:`n` operator matrix.
        qargs : Sequence of int
            Target qubit indices.
        """
        self._apply_matrix(cp.asarray(op, dtype=cp.complex128), list(qargs))

    @override
    def evolve_single(self, op: Matrix, i: int) -> None:
        """Apply a single-qubit gate operation.

        Parameters
        ----------
        op : Matrix
            2 x 2 operator matrix.
        i : int
            Target qubit index.
        """
        self._apply_matrix(cp.asarray(op, dtype=cp.complex128), [i])

    # -- expectation_single ---------------------------------------------- #

    @override
    def expectation_single(self, op: Matrix, loc: int) -> complex:
        """Compute the expectation value of a single-qubit operator.

        Parameters
        ----------
        op : Matrix
            2 x 2 operator matrix.
        loc : int
            Target qubit index.

        Returns
        -------
        complex
            Expectation value ``<psi|op|psi>``.
        """
        gate = cp.asarray(op, dtype=cp.complex128)
        return self._expectation(gate, [loc])

    # -- remove_qubit ---------------------------------------------------- #

    @override
    def remove_qubit(self, qarg: int) -> None:
        """Remove a separable qubit from the state.

        The qubit is traced out by keeping only the branch with non-zero
        norm and renormalising.

        Parameters
        ----------
        qarg : int
            Index of the qubit to remove.

        Raises
        ------
        ValueError
            If both branches have zero norm (qubit is not separable).
        """
        n = self._nqubit
        t = self._active_psi.reshape((2,) * n)

        idx: list[slice | int] = [slice(None)] * n
        br: cp.ndarray | None = None
        nrm2: float | None = None
        for val in (0, 1):
            idx[qarg] = val
            branch = t[tuple(idx)].ravel()
            branch_nrm2 = float(cp.sum(cp.abs(branch) ** 2))
            if not math.isclose(branch_nrm2, 0, abs_tol=1e-15):
                br = branch
                nrm2 = branch_nrm2
                break
        else:
            raise ValueError(f"Both branches for qubit {qarg} have zero norm — qubit may not be separable.")

        assert br is not None
        assert nrm2 is not None

        br /= math.sqrt(nrm2)

        self._nqubit -= 1
        self._active_psi = br

    # -- swap ------------------------------------------------------------ #

    @override
    def swap(self, qubits: tuple[int, int]) -> None:
        """Swap two qubits in the state.

        Parameters
        ----------
        qubits : tuple of int
            (index1, index2) qubit indices to swap.
        """
        i, j = qubits
        if i == j or self._nqubit == 0:
            return

        t = self._active_psi.reshape((2,) * self._nqubit)
        self._active_psi = cp.swapaxes(t, i, j).ravel()

    # -- tensor ---------------------------------------------------------- #

    def tensor(self, other: Statevec) -> None:
        """In-place tensor product with another state.

        The resulting state has ``nqubit_self + nqubit_other`` qubits.

        Parameters
        ----------
        other : Statevec
            State to tensor with ``self``.
        """
        n_self = self._nqubit
        n_other = other._nqubit
        n_total = n_self + n_other

        self._ensure_capacity(n_total)

        a = self.psi[: 1 << n_self]
        b = other.psi[: 1 << n_other]
        self.psi[: 1 << n_total] = cp.kron(a, b)
        self._nqubit = n_total

    # -- cuQuantum helpers ----------------------------------------------- #

    def _apply_matrix(self, gate: cp.ndarray, targets: list[int]) -> None:
        """Apply a matrix gate using the cuStateVec library.

        Parameters
        ----------
        gate : cp.ndarray
            Gate matrix on GPU.
        targets : list of int
            Target qubit indices.
        """
        n = self._nqubit
        if n == 0:
            return
        active = self._active_psi
        t = _msb_to_lsb(targets, n)
        h = _handle()

        n_targets = len(t)
        n_controls = 0

        ws_size: int = custatevec.apply_matrix_get_workspace_size(
            handle=h,
            sv_data_type=_SV_DTYPE,
            n_index_bits=n,
            matrix=gate.data.ptr,
            matrix_data_type=_SV_DTYPE,
            layout=_LAYOUT,
            adjoint=False,
            n_targets=n_targets,
            n_controls=n_controls,
            compute_type=_COMPUTE,
        )

        ws_ptr: int = cp.zeros(ws_size, dtype=cp.uint8).data.ptr if ws_size > 0 else 0

        custatevec.apply_matrix(
            handle=h,
            sv=active.data.ptr,
            sv_data_type=_SV_DTYPE,
            n_index_bits=n,
            matrix=gate.data.ptr,
            matrix_data_type=_SV_DTYPE,
            layout=_LAYOUT,
            adjoint=False,
            targets=t,
            n_targets=n_targets,
            controls=0,
            control_bit_values=0,
            n_controls=n_controls,
            compute_type=_COMPUTE,
            extra_workspace=ws_ptr,
            extra_workspace_size_in_bytes=ws_size,
        )

    def _expectation(self, gate: cp.ndarray, targets: list[int]) -> complex:
        r"""Compute the expectation value :math:`\langle\psi|\texttt{gate}|\psi\rangle`.

        Uses ``custatevec.compute_expectation`` with a host-side result buffer.

        Parameters
        ----------
        gate : cp.ndarray
            Operator matrix on GPU.
        targets : list of int
            Target qubit indices.

        Returns
        -------
        complex
            Expectation value.
        """
        n = self._nqubit
        if n == 0:
            return 1.0 + 0.0j
        active = self._active_psi
        t = _msb_to_lsb(targets, n)
        h = _handle()

        # Host-side buffer for the result (cpu numpy array)
        result = np.zeros(1, dtype=np.complex128)
        result_ptr = result.ctypes.data

        n_basis_bits = len(t)

        ws_size: int = custatevec.compute_expectation_get_workspace_size(
            handle=h,
            sv_data_type=_SV_DTYPE,
            n_index_bits=n,
            matrix=gate.data.ptr,
            matrix_data_type=_SV_DTYPE,
            layout=_LAYOUT,
            n_basis_bits=n_basis_bits,
            compute_type=_COMPUTE,
        )

        ws_ptr: int = cp.zeros(ws_size, dtype=cp.uint8).data.ptr if ws_size > 0 else 0

        custatevec.compute_expectation(
            handle=h,
            sv=active.data.ptr,
            sv_data_type=_SV_DTYPE,
            n_index_bits=n,
            expectation_value=result_ptr,
            expectation_data_type=_SV_DTYPE,
            matrix=gate.data.ptr,
            matrix_data_type=_SV_DTYPE,
            layout=_LAYOUT,
            basis_bits=t,
            n_basis_bits=n_basis_bits,
            compute_type=_COMPUTE,
            extra_workspace=ws_ptr,
            extra_workspace_size_in_bytes=ws_size,
        )

        return complex(result[0])

    # -- helpers --------------------------------------------------------- #

    def normalize(self) -> None:
        """Normalise the state vector in-place.

        Divides the active portion of the state by its L2-norm so that
        the state is renormalised to unit length.
        """
        a = self.psi[: 1 << self._nqubit]
        a /= math.sqrt(float(cp.sum(cp.abs(a) ** 2)))

    def dims(self) -> tuple[int, ...]:
        """Return the tensor shape of the state.

        Returns
        -------
        tuple of int
            Shape ``(2, 2, ..., 2)`` with ``nqubit`` elements.
        """
        return (2,) * self._nqubit

    def isclose(self, other: Statevec, *, rtol: float = 1e-09, atol: float = 0.0) -> bool:
        r"""Check approximate equality up to global phase.

        Equality is determined by checking whether the fidelity
        :math:`|\langle\psi_1|\psi_2\rangle|^2` is close to 1 within the
        given tolerances.

        Parameters
        ----------
        other : Statevec
            State to compare with.
        rtol : float
            Relative tolerance (passed to :func:`math.isclose`).
        atol : float
            Absolute tolerance (passed to :func:`math.isclose`).

        Returns
        -------
        bool
            ``True`` if the states are equivalent up to global phase.
        """
        return math.isclose(self.fidelity(other), 1, rel_tol=rtol, abs_tol=atol)

    def fidelity(self, other: Statevec) -> float:
        r"""Compute the fidelity with another state.

        .. math::
            F = |\langle\psi_1|\psi_2\rangle|^2

        Parameters
        ----------
        other : Statevec
            State to compare with.

        Returns
        -------
        float
            Fidelity value in ``[0, 1]``.
        """
        a = self.psi[: 1 << self._nqubit]
        b = other.psi[: 1 << other._nqubit]
        ip = float(cp.dot(a.conj(), b))
        return ip.real**2 + ip.imag**2

    def copy(self) -> Statevec:
        """Return a deep copy of the state.

        Returns
        -------
        Statevec
            Independent copy with the same state and capacity.
        """
        return copy.deepcopy(self)


@dataclass(frozen=True)
class StatevectorBackend(DenseStateBackend[Statevec]):
    """MBQC backend with cuQuantum-accelerated statevector simulation."""

    state: Statevec = dataclasses.field(init=False, default_factory=lambda: Statevec(nqubit=0))

    @classmethod
    def with_capacity(cls, max_qubits: int, state: Statevec | None = None, **kwargs: Any) -> Self:
        """Create a backend with preallocated statevector capacity.

        Pre-allocating capacity avoids repeated GPU reallocations as qubits
        are added during pattern simulation.

        Parameters
        ----------
        max_qubits : int
            Maximum number of qubits to allocate capacity for.
        state : Statevec, optional
            Initial state to use.  If ``None``, starts with a zero-qubit state.
        **kwargs : Any
            Additional arguments passed to the backend constructor
            (e.g. ``branch_selector``, ``symbolic``).

        Returns
        -------
        Self
            A new backend instance with the specified capacity.
        """
        if state is None:
            state_init = Statevec(nqubit=0, max_space=max_qubits)
        else:
            gpu_state = state.psi[: 1 << state._nqubit].copy()
            state_init = Statevec(data=gpu_state, nqubit=state._nqubit, max_space=max_qubits)

        backend = cls(**kwargs)
        object.__setattr__(backend, "state", state_init)  # noqa: PLC2801
        return backend
