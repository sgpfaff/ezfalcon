from .leapfrog import _leapfrog_step #, leapfrog_drift, leapfrog_kick
from ..acceleration import self_gravity
import numpy as np
from tqdm import tqdm
from functools import partial

def _integrate(pos, vel, mass, 
              include_self_gravity,
              self_gravity_method,
              extra_acc,
              t_end, dt, dt_out, 
              return_self_potential=True,
              return_self_gravity=True,
              return_ext_acc=True,
              **kwargs):
    '''
    Integrate particle trajectories under self-gravity 
    using leapfrog with falcON.

    Parameters
    ----------
    pos : (N, 3) array
        Initial positions of particles. 
        Units: `kpc`
    vel : (N, 3) array
        Initial velocities of particles.
        Units: `kpc / Gyr`
    mass : (N,) array
        Masses of particles.
        Units: `Msun`
    include_self_gravity : bool
        Whether to include self-gravity in the integration.
    self_gravity_method : str
        Method to use for computing self-gravity. Included options are:
        - 'falcON': Use the fast multipole method implemented in falcON.
        - 'direct': Use direct summation.
    extra_acc : list of callables
        Additional accelerations to be added to the self-gravity accelerations.
    t_end : float
        End time of integration.
        Units: `Gyr`
    dt : float
        Timestep for integration.
        Units: `Gyr`
    dt_out : float
        Output interval.
        Units: `Gyr`
    return_self_potential : bool, optional
        Whether to return the self-gravitational potential at each output snapshot. Default is True.
    return_self_gravity : bool, optional
        Whether to return the self-gravitational acceleration at each output snapshot. Default is True.
    **kwargs
        Additional keyword arguments to pass to the self-gravity method.

    Returns
    -------
    positions : (nsnaps, N, 3) array
        Positions at each output snapshot.
        Units: `kpc`
    velocities : (nsnaps, N, 3) array
        Velocities at each output snapshot.
        Units: `kpc / Gyr`
    ts : (nsnaps,) array
        Times of each output snapshot.
        Units: `Gyr`
    self_gravities : (nsnaps, N, 3) array or None
        Self-gravitational accelerations at each output snapshot. 
        Returns None if return_self_gravity is False.
        Units: `kpc / Gyr^2`
    self_potentials : (nsnaps, N) array or None
        Self-gravitational potentials at each output snapshot.
        Returns None if return_self_potential is False.
        Units: `kpc^2 / Myr^2`
    '''
    
    def acc_fn(pos, mass, method, **kwargs):
        '''
        Returns None for self_acc (self_pot) if 
        return_self_gravity (return_self_potential) is False.
        '''
        acc = np.zeros_like(vel)
        ext_acc = np.zeros_like(vel)
        if include_self_gravity:
            if return_self_potential:
                self_acc, self_pot = self_gravity(pos, mass, method=method, return_potential=True, **kwargs)
            else:
                self_pot = None
                self_acc = self_gravity(pos, mass, method=method, return_potential=False, **kwargs)
            acc += self_acc
        else:
            self_acc = np.zeros_like(vel)
            if return_self_potential:
                self_pot = np.zeros(pos.shape[0])
            else:
                self_pot = None
        for fn in extra_acc:
            ext_acc += fn(pos, t=0)
        acc += ext_acc
        return acc, self_acc, self_pot
    
    ratio = t_end / dt
    n_steps = round(ratio) if abs(ratio - round(ratio)) < 1e-9 else int(ratio)
    steps_per_output = round(dt_out / dt)
    nsnaps = n_steps // steps_per_output + 1  # +1 for initial snapshot

    ts_out = np.arange(nsnaps, dtype=np.float64) * dt_out
    ts_integrate = np.arange(n_steps + 1, dtype=np.float64) * dt

    positions = np.zeros((nsnaps, mass.shape[0], 3), dtype=np.float64)
    velocities = np.zeros((nsnaps, mass.shape[0], 3), dtype=np.float64)
    positions[0] = pos.copy()
    velocities[0] = vel.copy()

    self_potentials = None
    self_gravities = None
    acc, self_acc, self_pot = acc_fn(pos, mass, method=self_gravity_method, **kwargs)

    if return_self_potential:
        self_potentials = np.zeros((nsnaps, mass.shape[0]), dtype=np.float64)
        self_potentials[0] = self_pot.copy()
    if return_self_gravity:
        self_gravities = np.zeros((nsnaps, mass.shape[0], 3), dtype=np.float64)
        self_gravities[0] = self_acc.copy()

    i_out = 1
    for step, t in enumerate(tqdm(ts_integrate[1:]), start=1):
        (pos, vel, acc, 
        self_acc, self_pot) = _leapfrog_step(pos, vel, acc, partial(acc_fn, mass=mass, 
                                                                    method=self_gravity_method, **kwargs), 
                                            dt=dt)
        if step % steps_per_output == 0 and i_out < nsnaps:
            positions[i_out] = pos.copy()
            velocities[i_out] = vel.copy()
            if return_self_gravity:
                self_gravities[i_out] = self_acc.copy()
            if return_self_potential:
                self_potentials[i_out] = self_pot.copy()
            i_out += 1
    return positions, velocities, ts_out, self_gravities, self_potentials