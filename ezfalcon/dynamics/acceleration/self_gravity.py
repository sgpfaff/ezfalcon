'''
Accelerations from self-gravity with pyfalcon.
'''
# import pyfalcon
from ._falcon import gravity as _falcon_grav
from ._direct_summation import gravity as direct_summation
from ...util import G_INTERNAL
import numpy as np

SELF_GRAVITY_KEYS = {'eps', 'theta'}
SELF_GRAVITY_METHODS = ['direct', 'direct_C','falcON']

def self_gravity(pos, mass, method='falcON', return_potential=True, **kwargs):
    """
    Compute accelerations and potentials from self-gravity.

    Parameters
    ----------
    pos : (N, 3) array
        Positions of particles (x, y, z).
        Unit: kpc
    mass : (N,) array
        Masses of particles.
        Unit: Msun
    method : str, optional
        Method to use for computing self-gravity. Options are:
        - 'falcON' (default): Use the fast multipole method implemented in falcON.
        - 'direct': Use direct summation.
    return_potential : bool, optional
        Whether to return the self-gravitational potential. Default is True.
    **kwargs
        Additional keyword arguments to pass to the gravity method. 

        For 'falcON', these include:
        - eps: Gravitational softening length (kpc)
        - theta: Tree opening angle. Smaller = more accurate but slower.
        - kernel (int, optional): Softening kernel: 0=Plummer, 1=default (~r^-7), 2,3=faster decay.

        For 'direct', these include:
        - eps: Gravitational softening length (kpc)

        For 'direct_C', these include:
        - eps: Gravitational softening length (kpc)

    Returns
    -------
    acc : (N, 3) array
        Accelerations.
        Unit: kpc / Myr^2 (internal units)
    pot : (N,) array, optional
        Specific gravitational potential.
        Only returned if return_potential is True.
        Unit: kpc^2 / Myr^2 (internal units)
    """
    if method not in SELF_GRAVITY_METHODS:
        raise ValueError(f"Unknown method '{method}' for self-gravity. Supported methods: {SELF_GRAVITY_METHODS}")
   
    if method == 'falcON':
        if 'eps' not in kwargs and 'theta' not in kwargs:
            raise ValueError("Must provide 'eps' and 'theta' keyword arguments for falcON method.")
        elif 'eps' not in kwargs:
            raise ValueError("Must provide 'eps' keyword argument for falcON method.")
        else:
            eps = kwargs.get('eps', 0.05)
            theta = kwargs.get('theta', 0.6)
            kernel = kwargs.get('kernel', 1)
        if set(kwargs.keys()) - {'eps', 'theta', 'kernel'}:
            raise ValueError(f"{set(kwargs.keys()) - {'eps', 'theta', 'kernel'}} is (are) invalid kwarg(s) for 'falcON' self-gravity method. Only kwargs for self-gravity methods are allowed.")
        return _falcON_gravity(pos, mass, eps, theta, kernel, return_potential)
    if method == 'direct_C':
        if 'eps' not in kwargs:
            raise ValueError("Must provide 'eps' keyword argument for direct_C summation method.")
        else:
            eps = kwargs.get('eps', 0.05)
        if set(kwargs.keys()) - {'eps'}:
            raise ValueError(f"{set(kwargs.keys()) - {'eps'}} is (are) invalid kwarg(s) for 'direct_C' self-gravity method. Only kwargs for self-gravity methods are allowed.")
        return _direct_summation_C(pos, mass, eps, return_potential)
    if method == 'direct':
        if 'eps' not in kwargs:
            raise ValueError("Must provide 'eps' keyword argument for direct summation method.")
        else:
            eps = kwargs.get('eps', 0.05)
        if set(kwargs.keys()) - {'eps'}:
            raise ValueError(f"{set(kwargs.keys()) - {'eps'}} is (are) invalid kwarg(s) for 'direct' self-gravity method. Only kwargs for self-gravity methods are allowed.")
        return _direct_summation(pos, mass, eps, return_potential)
    else:
        raise ValueError(f"Unknown method '{method}' for self-gravity.")

def _falcON_gravity(pos, mass, eps, theta, kernel, return_potential):
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
    kernel : int, optional
        Softening kernel to use.
    return_potential : bool
        Whether to return the self-gravitational potential.

    Returns
    -------
    acc : (N, 3) array
        Accelerations.
        Unit: kpc / Myr^2 (internal units)
    pot : (N,) array, optional
        Specific gravitational potential.
        Only returned if return_potential is True.
        Unit: kpc^2 / Myr^2 (internal units)
    """
    if return_potential:
        return _falcon_grav(pos, mass * G_INTERNAL, eps, theta=theta, kernel=kernel)
    else:
        return _falcon_grav(pos, mass * G_INTERNAL, eps, theta=theta, kernel=kernel)[0]

def _direct_summation_C(pos, mass, eps, return_potential):
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
    return_potential : bool
        Whether to return the self-gravitational potential. Default is True.

    Returns
    -------
    acc : (N, 3) array
        Accelerations.
        Unit: kpc / Myr^2 (internal units)
    pot : (N,) array, optional
        Specific gravitational potential.
        Only returned if return_potential is True.
        Unit: kpc^2 / Myr^2 (internal units)
    '''
    if return_potential:
        return direct_summation(pos, mass * G_INTERNAL, eps)#acc, pot
    else:
        return direct_summation(pos, mass * G_INTERNAL, eps)[0]

def _direct_summation(pos, mass, eps, return_potential):
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
    return_potential : bool
        Whether to return the self-gravitational potential. Default is True.

    Returns
    -------
    acc : (N, 3) array
        Accelerations.
        Unit: kpc / Myr^2 (internal units)
    pot : (N,) array, optional
        Specific gravitational potential.
        Only returned if return_potential is True.
        Unit: kpc^2 / Myr^2 (internal units)
    '''
    N = len(mass)
    acc = np.zeros_like(pos)
    if return_potential:
        pot = np.zeros(N)
    for i in range(N):
        mask = np.arange(N) != i
        dx = pos[mask] - pos[i]  # (N-1, 3)
        r2 = np.sum(dx**2, axis=1) + eps**2  # (N-1,)
        inv_r3 = 1.0 / r2**1.5  # (N-1,)
        acc[i] = G_INTERNAL * np.sum(mass[mask, None] * dx * inv_r3[:, None], axis=0)
        if return_potential:
            pot[i] = -G_INTERNAL * np.sum(mass[mask] / np.sqrt(r2))
    if return_potential:
        return acc, pot
    else:
        return acc