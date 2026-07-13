from .ExternalConservativeForce import ExternalConservativeForce
import galpy
from ....tools.util._galpy_bridge import (
                _galpy_pot_to_acc_fn, _galpy_pot_to_pot_fn,
                _check_physical, _check_supported_pot,
                _ensure_pot, _iter_components,
            )
import numpy as np
import pickle

class ExternalGalpyPotential(ExternalConservativeForce):
    """Wrap a galpy ``Potential`` as an :class:`ExternalForce`.

    Conservative (has a potential) and not C-backed. Combine with other
    galpy potentials via ``+``::

        sim.add_external_force(ExternalGalpyPotential(nfw)
                               + ExternalGalpyPotential(disk))
    """

    def __init__(self, potential):
        potential = _ensure_pot(potential)
        for p in _iter_components(potential):
            if not isinstance(p, galpy.potential.Potential):
                raise TypeError("External potential must be a galpy Potential object.")
            _check_physical(p)
        _check_supported_pot(potential)
        self._pot = potential                    # kept for dedup identity
        self._acc_fn = _galpy_pot_to_acc_fn(potential)
        self._potential_fn = _galpy_pot_to_pot_fn(potential)

    def _dedup_key(self):
        # pickle serialises the full parameter set, so equal bytes <=> 
        # same type and all parameters. Falls back to object identity 
        # for the rare unpicklable potential.
        try:
            return (type(self), pickle.dumps(self._pot))
        except Exception:
            return (type(self), id(self._pot))

    def acc(self, pos: np.ndarray, t) -> np.ndarray:
        return self._acc_fn(pos, t)

    def potential(self,  pos: np.ndarray, t) -> np.ndarray:
        return self._potential_fn(pos, t)
    