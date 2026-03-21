'''
Accelerations from self-gravity with pyfalcon.
'''
import pyfalcon
from ...util import G_INTERNAL

def self_gravity(pos, mass, eps, theta=0.6):
    """
    Compute accelerations and potentials from self-gravity using pyfalcon.

    Parameters
    ----------
    pos : (N, 3) array
        Positions of particles.
        Unit: kpc
    mass : (N,) array
        Masses of particles.
        Unit: Msun
    eps : float
        Gravitational softening length.
        Unit: kpc
    theta : float, optional
        Tree opening angle (default 0.6). Smaller = more accurate but slower.

    Returns
    -------
    acc : (N, 3) array
        Accelerations.
        Unit: kpc/Myr² (internal units)
    pot : (N,) array
        Gravitational potential.
        Unit: kpc²/Myr² (internal units)
    """
    return pyfalcon.gravity(pos, mass * G_INTERNAL, eps, theta=theta)
