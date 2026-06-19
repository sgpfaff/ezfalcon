from tambora.dynamics.forces.external_force.ExternalConservativeForce import ExternalConservativeForce
from tambora.dynamics.forces.external_force.ExternalGalpyPotential import ExternalGalpyPotential
import galpy
from tambora.tools.util._galpy_bridge import (
                _galpy_pot_to_acc_fn, _galpy_pot_to_pot_fn,
                _check_physical, _check_supported_pot,
                _ensure_pot, _iter_components,
            )
from tambora.tools.util.units import KMS_TO_KPCGYR
import numpy as np
from galpy.potential import MultipoleExpansionPotential
from galpy.df import isotropicHernquistdf

from src.potential import HaloModel
from src.helper_functions import create_r_array, Solution
from src.wavefunction import EigenmodeBasis, WaveFunction
from src.solver import build_wavefunction
from src.galpy_interface import build_fdm_multipole_potential, build_fdm_multipole_snapshot
from src.fast_eval import build_force_tables, forces_cyl
from galpy.util.coords import rect_to_cyl, cyl_to_rect_vec
from astropy.constants import G, hbar, c
import astropy.units as u
import time

class WidrowKaiserFDM(ExternalConservativeForce):
    def __init__(self, targetPotential, mboson, L, update_freq, Rmax=300.,
                 N=1500, ro=8., vo=220., build_Nr=300):
        '''
        FDM halo solved with Widrow-Kaiser method.

        Parameters
        ----------
        targetPotential : galpy potential
            target potential for Widrow-Kaiser
        mboson : float
            FMD axion mass
        L : int
            Highest multiple l galpy expands to.
        Rmax_eval : float
            Maximum of radial grid used to evaluate the density.
        N_eval : int
            Number of radial grid points used to evaluate the density.
        Rmax_solve : float
            Maximum of radial grid used to solve the wave function.
        N_solve : int
            Number of radial grid points used to solve the wave function.
        ro : float
            galpy internal position scaling.
        vo : float
            galpy internal velocity scaling.
        build_Nr : int
            Number of radial grid points used to build the multipole potential
            (galpy's radial Poisson grid). Independent of ``N`` (the wavefunction
            solve grid). The build cost scales with this; ~300 matches the default
            600 to <1e-4 in force while building ~2x faster. Reduce further to
            speed up on-the-fly rebuilds at the cost of radial resolution.

        Returns
        -------
        None
        '''
        self._target_pot = _ensure_pot(targetPotential)
        for p in _iter_components(targetPotential):
            if not isinstance(p, galpy.potential.Potential):
                raise TypeError("External potential must be a galpy Potential object.")
            _check_physical(p)
        _check_supported_pot(targetPotential)
        self.ro, self.vo = ro, vo
        self.vo_int = vo * KMS_TO_KPCGYR  # kpc/Gyr
        self.eps = (hbar / (mboson * vo*(u.km/u.s) * ro*(u.kpc))).decompose()
        self.L = L
        self.build_Nr = build_Nr
        self._a = None
        self._last_force = None # cached last FDM potential
        self._last_update_t = None
        self.update_freq = update_freq
        self._targetForce = ExternalGalpyPotential(targetPotential)

        self._r_array = None

        self.halo = HaloModel(pot=self._target_pot, df_class = isotropicHernquistdf, ro=ro, vo=vo)
        self.Rmax = Rmax
        self.Emax = self.halo.potential(Rmax/ro)

        self._r_array = create_r_array(Rmax, N)
        self.wf = build_wavefunction(self.halo, self._r_array, mboson, ro, vo, self.Emax)

    def _potential_solver(self, t):
        t_int = t * self.vo_int / self.ro
        return build_fdm_multipole_snapshot(self.wf, self.ro, self.vo, t0=t_int,
                                            L=self.L, Nr=self.build_Nr)
        #print(x, "| _tdep =", x._tdep, "(constant in time)", "| L =", x._L)

    def _update_force(self, t):
        pot = self._potential_solver(t)
        pot.turn_physical_on(self.ro, self.vo)
        self._fdm_pot = pot
        self._last_force = ExternalGalpyPotential(pot) #- self._targetForce
        # Extract radial multipole tables for the vectorized (numba) force
        # evaluator, which bypasses galpy's per-point Python loop in acc().
        t_int = t * self.vo_int / self.ro
        self._force_tables = build_force_tables(pot, t_int)

    def _grid_time(self, t):
        """Snap t down onto the update grid of spacing ``update_freq``.

        The FDM field is rebuilt only at these grid times; between them the
        frozen snapshot (built at the cell's start) is held. Returns ``t``
        unchanged if ``update_freq`` is unset/<=0 (rebuild every distinct t).
        """
        if not self.update_freq or self.update_freq <= 0:
            return t
        return np.floor(t / self.update_freq) * self.update_freq

    def _maybe_update(self, t):
        """Rebuild the snapshot only when t enters a new update-grid cell.

        This is the throttle that makes the FDM force usable in a sim: the
        expensive multipole build happens at most once per ``update_freq``
        (frozen at the cell's start time), not every timestep, so the force is
        piecewise-constant in time between grid nodes.
        """
        node = self._grid_time(t)
        if self._last_update_t is None or node != self._last_update_t:
            self._update_force(node)
            self._last_update_t = node

    def acc(self, pos, t):
        self._maybe_update(t)
        # Vectorized force eval over all particles at once (numba), instead of
        # galpy's per-point evaluateRforces/zforces/phitorques. The snapshot is
        # constant in time, so the tables (built above) need no t.
        pos = np.asarray(pos, dtype=float)
        R, phi, z = rect_to_cyl(*pos.T)
        Rforce, zforce, phitorque = forces_cyl(
            self._force_tables, R / self.ro, z / self.ro, phi
        )
        scale = self.vo_int ** 2 / self.ro  # galpy internal force -> kpc/Gyr^2
        aR = Rforce * scale
        az = zforce * scale
        aphi = phitorque * self.vo_int ** 2 / R
        ax, ay, az = cyl_to_rect_vec(aR, aphi, az, phi)
        return np.array([ax, ay, az]).T
    
    def potential(self, pos, t):
        self._maybe_update(t)
        return self._last_force.potential(pos, t)
    

# fdmpot = ...
# for i, t in enumerate(ts_update):
#     int_ts = np.arange(ts_update[i-1], t, dt)
#     o.integrate(pot, int_ts)
#     fdmpot = rebuild()