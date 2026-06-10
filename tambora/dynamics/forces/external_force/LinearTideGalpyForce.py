from .MassIndependentConservForce import MassIndependentConservForce
import numpy as np
from tambora.tools.util._galpy_bridge import (
                _galpy_pot_to_acc_fn, _galpy_pot_to_pot_fn,
                _check_physical, _check_supported_pot,
                _ensure_pot, _iter_components, UNVECTORIZED_WRAPPERS, 
                get_physical, _unwrap_pot, _get_ro_vo, _needs_scalar_loop,

             )

from tambora.tools.util.units import KMS_TO_KPCGYR
from galpy.util.coords import rect_to_cyl
from galpy import potential

class LinearTideGalpyForce(MassIndependentConservForce):
    def __init__(self, pot, center=None):
        pot = _ensure_pot(pot)
        for p in _iter_components(pot):
            if not isinstance(p, potential.Potential):
                raise TypeError("External potential must be a galpy Potential object.")
            _check_physical(p)
        _check_supported_pot(pot)
        self._pot = pot
        self._center = None if center is None else np.asarray(center, float)
        self._ro, vo = _get_ro_vo(pot)
        self._vo_int = vo * KMS_TO_KPCGYR

    def _tensor_disp(self, pos):
        pos = np.atleast_2d(np.asarray(pos, float))
        c = np.median(pos, axis=0) if self._center is None else self._center
        Rc, phic, zc = rect_to_cyl(*np.atleast_2d(c).T)
        T = np.asarray(self._pot.ttensor(R=Rc/self._ro, phi=phic, z=zc/self._ro,
                                         t=0.0, use_physical=False)).reshape(3, 3)
        return T, pos - c

    def acc(self, pos, t):
        T, d = self._tensor_disp(pos)
        return d @ T.T * self._vo_int**2 / self._ro**2

    def potential(self, pos, t):          # per unit mass; -grad = acc
        T, d = self._tensor_disp(pos)
        return -0.5 * np.einsum('ni,ij,nj->n', d, T, d) * self._vo_int**2 / self._ro**2
