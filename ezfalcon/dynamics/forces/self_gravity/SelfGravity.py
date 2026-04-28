"""Self-gravity solvers as :class:`ConservativeForceField` objects.

These wrap the C-extension gravity functions in the Force-object API used at
the integrator boundary. Construction params (eps, theta, kernel) are frozen
at instantiation time so the solver instance is reusable across substeps.

Until the Phase-4 C-wrapper split (gravity / gravity_force / gravity_pot) lands,
all three entry points call the existing single-pass ``gravity()`` and slice
the output. The Python-side API is forward-compatible: when the wrappers gain
dedicated force-only / pot-only paths, only the body of these methods changes.
"""

from typing import Tuple, Union
import numpy as np

from ...state import State
from ..ConservativeForceField import ConservativeForceField
from .falcON import _falcON_gravity
from .direct_summation import _direct_summation_C, _direct_summation_py


class FalcONGravity(ConservativeForceField):
    """Self-gravity via the falcON fast-multipole tree.

    Parameters
    ----------
    eps : float or (N,) array
        Gravitational softening length(s) [kpc].
    theta : float, optional
        Tree opening angle (default 0.6). Smaller = more accurate but slower.
    kernel : int, optional
        Softening kernel: 0=Plummer, 1=default (~r^-7), 2,3=faster decay.
    """

    def __init__(
        self,
        eps: Union[float, np.ndarray],
        theta: float = 0.6,
        kernel: int = 1,
    ):
        self.eps = eps
        self.theta = theta
        self.kernel = kernel

    def force_and_potential(
        self, state: State
    ) -> Tuple[np.ndarray, np.ndarray]:
        return _falcON_gravity(
            state.pos, state.mass, self.eps, self.theta, self.kernel,
            return_potential=True,
        )

    def force(self, state: State) -> np.ndarray:
        return _falcON_gravity(
            state.pos, state.mass, self.eps, self.theta, self.kernel,
            return_potential=False,
        )

    def potential(self, state: State) -> np.ndarray:
        return self.force_and_potential(state)[1]


class DirectSummationGravity(ConservativeForceField):
    """Self-gravity via direct O(N^2) pair summation.

    Parameters
    ----------
    eps : float or (N,) array
        Gravitational softening length(s) [kpc]. If an array, pairwise
        softening uses the arithmetic mean ``(eps_i + eps_j) / 2``.
    use_C : bool, optional
        Use the C extension (default ``True``). Set to ``False`` for the
        pure-Python reference implementation.
    """

    def __init__(self, eps: Union[float, np.ndarray], use_C: bool = True):
        self.eps = eps
        self.use_C = use_C

    def force_and_potential(
        self, state: State
    ) -> Tuple[np.ndarray, np.ndarray]:
        impl = _direct_summation_C if self.use_C else _direct_summation_py
        return impl(state.pos, state.mass, self.eps, return_potential=True)

    def force(self, state: State) -> np.ndarray:
        impl = _direct_summation_C if self.use_C else _direct_summation_py
        return impl(state.pos, state.mass, self.eps, return_potential=False)

    def potential(self, state: State) -> np.ndarray:
        return self.force_and_potential(state)[1]
