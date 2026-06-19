from .ExternalConservativeForce import ExternalConservativeForce
from .ExternalGalpyPotential import ExternalGalpyPotential
import galpy
from ....tools.util._galpy_bridge import (
                _galpy_pot_to_acc_fn, _galpy_pot_to_pot_fn,
                _check_physical, _check_supported_pot,
                _ensure_pot, _iter_components,
            )
import numpy as np
from galpy.potential import MultipoleExpansionPotential

class WidrowKaiserFDM(ExternalConservativeForce):
    def __init_(self, targetPotential, update_freq):
        self._target_pot = _ensure_pot(targetPotential)
        for p in _iter_components(targetPotential):
            if not isinstance(p, galpy.potential.Potential):
                raise TypeError("External potential must be a galpy Potential object.")
            _check_physical(p)
        _check_supported_pot(targetPotential)

        self._a = None
        self._last_force = None # cached last FDM potential
        self._last_update_t = None
        self.update_freq = update_freq
        self._targetForce = ExternalGalpyPotential(targetPotential)

    def _update_force(self, t):
        dens_args = _dens_solver(t)
        pot = MultipoleExpansionPotential(*dens_args)
        self._last_force = ExternalGalpyPotential(pot) - self._targetForce

    def acc(self, pos, t):
        # if (t % self.update_freq) == 0:
        if t != self._last_update_t:
            self._update_force(t)
        return self._last_force.acc(pos, t)
    
    def potential(self, pos, t):
        if t != self._last_update_t:
            self._update_force(t)
        return self._last_force.potential(pos, t)
        
mysim = Sim()
targetPotential = someGalpyPotential()
targetPotential.turn_physical_on()

mysim.add_external_pot(targetPotential)

dt = 0.01 # Gyr
fdmForce = WidrowKaiserFDM(targetPotential)
mysim.add_external_force(fdmForce)

mysim.add_particles(...)
mysim.run(t_end=2., dt=dt, dt_out=0.1, eps=0.01)
