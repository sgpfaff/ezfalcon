'''galpy bridge functions.'''

from .coords import cyl2cart
from galpy.util.coords import rect_to_cyl, cyl_to_rect, cyl_to_rect_vec
from galpy.util.conversion import get_physical
from galpy import potential
from .units import KMS_TO_KPCMYR, GYR_TO_MYR
import numpy as np
import astropy.units as u
import warnings

# galpy physical units --> ezfalcon internal units conversion factors
FROM_GALPY_TO_INTERNAL = {
    'pos': 1.0,                 # kpc --> kpc
    'vel': KMS_TO_KPCMYR,       # km/s --> kpc/Myr 
    'mass': 1.0,                # Msun --> Msun
    'time' : GYR_TO_MYR,        # Gyr --> Myr
    'pot' : KMS_TO_KPCMYR**2,   # (km/s)² --> (kpc/Myr)²
    'acc' : KMS_TO_KPCMYR,      # km/s/Myr --> kpc/Myr²
}

SUPPORTED_POTENTIALS = (
    potential.BurkertPotential,
    potential.DehnenCoreSphericalPotential,
    potential.DehnenSphericalPotential,
    potential.EinastoPotential,
    potential.HernquistPotential,
    potential.interpSphericalPotential,
    potential.IsochronePotential,
    potential.JaffePotential,
    potential.KeplerPotential,
    potential.KingPotential,
    potential.NFWPotential,
    potential.PlummerPotential,
    potential.PowerSphericalPotential,
    potential.PowerSphericalPotentialwCutoff,
    potential.PseudoIsothermalPotential,
    potential.TwoPowerSphericalPotential,
    potential.KuzminDiskPotential,
)

RMIN = 1e-15 * u.kpc

def _check_supported_pot(pot):
    if not isinstance(pot, SUPPORTED_POTENTIALS):
        raise TypeError(
            f"{type(pot).__name__} is not supported by ezfalconv2. "
            f"Supported potentials: {', '.join(p.__name__ for p in SUPPORTED_POTENTIALS)}"
        )
    if not isinstance(pot, potential.SphericalPotential.SphericalPotential):
        warnings.warn(f"More tests needed for axisymmetric potentials.")

def _check_physical_pot(pot):
    if not pot._roSet and not pot._voSet:
        warnings.warn("The provided galpy potential has physical outputs turned off. Using galpy get_physical to determine ro and vo.")
        pot.turn_physical_on(ro=8.0, vo=220.0)

def _galpy_pot_to_pot_fn(pot):
    def pot_fn(pos, t):
        R, phi, z = rect_to_cyl(*np.array(pos).T*u.kpc)
        return pot(R, z, phi=phi, quantity=True, t=t*u.Myr).to(u.kpc**2/u.Myr**2).value
    return pot_fn

def _galpy_pot_to_acc_fn(pot):
    '''
    Convert a galpy potential to a function that 
    returns accelerations in ezfalcon internal units.
    
    Parameters
    ----------
    pot : galpy.potential.Potential
        A galpy potential object. You 
        must turn physical on before passing it in, e.g. with `pot.turn_physical_on()`.
    
    Returns
    -------
    acc_fn : function
        A function that takes positions (x, y, z) and returns accelerations (ax, ay, az)
        in ezfalcon internal units.

    '''
    def acc_fn(pos, t):
        R, phi, z = rect_to_cyl(*np.array(pos).T*u.kpc)
        aR = pot.Rforce(R, z, phi=phi, quantity=True, t=t*u.Myr)
        aphitorque = pot.phitorque(R, z, phi=phi, quantity=True, t=t*u.Myr)
        az = pot.zforce(R, z, phi=phi, quantity=True, t=t*u.Myr)
        aphi = aphitorque / R
        ax, ay, az = cyl_to_rect_vec(aR, aphi, az, phi)
        return np.array([ax.to(u.kpc/u.Myr**2).value, 
                ay.to(u.kpc/u.Myr**2).value, 
                az.to(u.kpc/u.Myr**2).value]).T
    return acc_fn
        
