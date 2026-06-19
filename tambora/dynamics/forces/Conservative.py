from abc import ABC, abstractmethod
import numpy as np


class Conservative(ABC):
    """Capability mixin for forces that are the gradient of a scalar potential.

    Mixed into a :class:`~.Force` subclass, it marks the force as conservative:
    it provides a potential (used for energy diagnostics) and is therefore
    velocity-independent (which lets symplectic integrators evaluate it once at
    the drifted position).

    Concrete classes implement the author-facing ``potential`` (whatever
    reduced signature fits the force) together with the :meth:`_eval_potential`
    adapter onto the uniform ``(pos, vel, mass, t)`` call. Solvers that produce
    acceleration and potential in a single pass (falcON, direct summation)
    should override :meth:`_eval_acc_and_potential` to avoid a second sweep.
    """

    @abstractmethod
    def _eval_potential(self, pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, t) -> np.ndarray:
        """Potential per unit mass for each particle, shape ``(N,)``."""
        ...

    def _eval_acc_and_potential(self, pos, vel, mass, t):
        """Default: evaluate acceleration and potential separately.

        Override for solvers that produce both arrays in one pass.
        """
        return self._eval_acc(pos, vel, mass, t), self._eval_potential(pos, vel, mass, t)
