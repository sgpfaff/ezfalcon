from ..ConservativeForceField import ConservativeForceField
from ...state import State
from ....util._galpy_bridge import _galpy_pot_to_acc_fn, _galpy_pot_to_pot_fn


class ExternalGalpyPotential(ConservativeForceField):
    """Wrap a galpy ``Potential`` as an :class:`ExternalForce`.

    Conservative (has a potential) and not C-backed. Combine with other
    galpy potentials via ``+``::

        sim.add_external_force(ExternalGalpyPotential(nfw)
                               + ExternalGalpyPotential(disk))
    """

    def __init__(self, potential):
        self._force_fn = _galpy_pot_to_acc_fn(potential)
        self._potential_fn = _galpy_pot_to_pot_fn(potential)

    def force(self, state: State):
        return self._force_fn(state.pos, state.t)

    def potential(self, state: State):
        return self._potential_fn(state.pos, state.t)
