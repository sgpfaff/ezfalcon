from ..BaseForce import BaseForce
from ...state import State


class ChandrasekarDF(BaseForce):
    """Chandrasekar dynamical friction.

    Velocity-dependent (i.e. not a :class:`ConservativeForceField`); the
    integrator detects this by elimination and predicts the velocity to
    the half-step before each call to :meth:`force`.
    """

    def force(self, state: State):
        raise NotImplementedError(
            "Chandrasekar dynamical friction not implemented yet."
        )
