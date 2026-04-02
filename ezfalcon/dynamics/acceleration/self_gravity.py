'''
Accelerations from self-gravity with pyfalcon.
'''
import pyfalcon
from ...util import G_INTERNAL
import numpy as np

def self_gravity(pos, mass, method='falcON', **kwargs):
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
    method : str, optional
        Method to use for computing self-gravity. Options are:
        - 'falcON' (default): Use the fast multipole method implemented in falcON.
        - 'direct': Use direct summation.
    **kwargs
        Additional keyword arguments to pass to the gravity method. 

        For 'falcON', these include:
        - eps: Gravitational softening length (kpc)
        - theta: Tree opening angle (default 0.6). Smaller = more accurate but slower.

        For 'direct', these include:
        - eps: Gravitational softening length (kpc)

    Returns
    -------
    acc : (N, 3) array
        Accelerations.
        Unit: kpc / Myr^2 (internal units)
    pot : (N,) array
        Gravitational potential.
        Unit: kpc^2 / Myr^2 (internal units)
    """
    if method == 'falcON':
        if 'eps' not in kwargs and 'theta' not in kwargs:
            raise ValueError("Must provide 'eps' and 'theta' keyword arguments for falcON method.")
        elif 'eps' not in kwargs:
            raise ValueError("Must provide 'eps' keyword argument for falcON method.")
        elif 'theta' not in kwargs:
            raise ValueError("Must provide 'theta' keyword argument for falcON method.")
        else:
            eps = kwargs.get('eps', 0.05)
            theta = kwargs.get('theta', 0.6)
        return _falcON_gravity(pos, mass, eps, theta)
    if method == 'direct':
        if 'eps' not in kwargs:
            raise ValueError("Must provide 'eps' keyword argument for direct summation method.")
        else:
            eps = kwargs.get('eps', 0.05)
        return _direct_summation(pos, mass, eps)
    else:
        raise ValueError(f"Unknown method '{method}' for self-gravity.")

def _falcON_gravity(pos, mass, eps, theta):
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
        Unit: kpc / Myr^2 (internal units)
    pot : (N,) array
        Gravitational potential.
        Unit: kpc^2 / Myr^2 (internal units)
    """
    return pyfalcon.gravity(pos, mass * G_INTERNAL, eps, theta=theta)

def _direct_summation(pos, mass, eps):
    '''
    Compute accelerations and potentials from self-gravity using direct summation.
    
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

    Returns
    -------
    acc : (N, 3) array
        Accelerations.
        Unit: kpc / Myr^2 (internal units)
    pot : (N,) array
        Gravitational potential.
        Unit: kpc^2 / Myr^2 (internal units)
    '''
    N = len(mass)
    acc = np.zeros_like(pos)
    pot = np.zeros(N)
    for i in range(N):
        mask = np.arange(N) != i
        dx = pos[mask] - pos[i]  # (N-1, 3)
        r2 = np.sum(dx**2, axis=1) + eps**2  # (N-1,)
        inv_r3 = 1.0 / r2**1.5  # (N-1,)
        acc[i] = G_INTERNAL * np.sum(mass[mask, None] * dx * inv_r3[:, None], axis=0)
        pot[i] = -G_INTERNAL * np.sum(mass[mask] / np.sqrt(r2))
    return acc, pot

