from ..ConservativeForceField import ConservativeForceField
from ...state import State


class ExternalAgamaPotential(ConservativeForceField):
    """Wrap an agama potential as an :class:`ExternalForce`. (Stub.)"""

    def __init__(self, potential):
        # TODO: build _force_fn / _potential_fn from the agama potential,
        # mirroring the galpy bridge.
        raise NotImplementedError(
            "ExternalAgamaPotential is not implemented yet."
        )

    def force(self, state: State):
        return self._force_fn(state.pos, state.t)

    def potential(self, state: State):
        return self._potential_fn(state.pos, state.t)
