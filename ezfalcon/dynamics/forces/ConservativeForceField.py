from abc import abstractmethod
from typing import Tuple
import numpy as np

from ..state import State
from .BaseForce import BaseForce


class ConservativeForceField(BaseForce):
    """Base class for forces that are the gradient of a potential.

    Concrete subclasses must implement both :meth:`force` and
    :meth:`potential`. Solvers that compute both arrays in a single shared
    pass (falcON, direct summation) should additionally override
    :meth:`force_and_potential` to skip the second sweep.
    """

    @abstractmethod
    def force(self, state: State) -> np.ndarray:
        ...

    @abstractmethod
    def potential(self, state: State) -> np.ndarray:
        """Specific gravitational potential at each particle, shape ``(N,)``."""
        ...

    def force_and_potential(
        self, state: State
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Default: call ``force`` and ``potential`` separately.

        Override for solvers that produce both arrays in one pass.
        """
        return self.force(state), self.potential(state)
