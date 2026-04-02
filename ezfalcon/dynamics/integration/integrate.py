from .leapfrog import leapfrog_step #, leapfrog_drift, leapfrog_kick
from ..acceleration import self_gravity
import numpy as np
from tqdm import tqdm
from functools import partial

def _integrate(pos, vel, mass, 
              include_self_gravity,
              extra_acc,
              t_end, dt, dt_out, 
              method,
              **kwargs):
    '''
    Integrate particle trajectories under self-gravity 
    using leapfrog with falcON.

    Parameters
    ----------
    pos : (N, 3) array
        Initial positions of particles. 
        Units: kpc
    vel : (N, 3) array
        Initial velocities of particles.
        Units: kpc/Myr
    mass : (N,) array
        Masses of particles.
        Units: Msun
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
        Units: Myr
    dt : float
        Timestep for integration.
        Units: Myr
    dt_out : float
        Output interval.
        Units: Myr

    Returns
    -------
    positions : (nsnaps, N, 3) array
        Positions at each output snapshot.
        Units: kpc
    velocities : (nsnaps, N, 3) array
        Velocities at each output snapshot.
        Units: kpc / Myr
    self_accelerations : (nsnaps, N, 3) array
        Self-gravity accelerations at each output snapshot.
        Units: kpc / Myr^2
    self_potentials : (nsnaps, N) array
        Self-gravity potentials at each output snapshot.
        Units: kpc^2 / Myr^2
    ts : (nsnaps,) array
        Times of each output snapshot.
        Units: Myr
    '''
    
    def acc_fn(pos, mass, method, **kwargs):
        acc = np.zeros_like(vel)
        self_acc = np.zeros_like(vel)
        self_pot = np.zeros(mass.shape[0])
        ext_acc = np.zeros_like(vel)
        if include_self_gravity:
            self_acc, self_pot = self_gravity(pos, mass, method=method, **kwargs)
            acc += self_acc
        for fn in extra_acc:
            ext_acc += fn(pos, t=0)
        acc += ext_acc
        return acc, self_acc, self_pot
    
    ts_out = np.arange(0, t_end + dt_out, dt_out)
    nsnaps = len(ts_out)
    ts_integrate = np.arange(0, t_end + dt, dt)

    # Determine which integration steps correspond to output snapshots
    steps_per_output = round(dt_out / dt)

    positions = np.zeros((nsnaps, mass.shape[0], 3), dtype=np.float64)
    velocities = np.zeros((nsnaps, mass.shape[0], 3), dtype=np.float64)
    positions[0] = pos.copy()
    velocities[0] = vel.copy()
    
    _, self_acc, self_pot = acc_fn(pos, mass, method=method, **kwargs)

    self_potentials = np.zeros((nsnaps, mass.shape[0]), dtype=np.float64)
    self_accelerations = np.zeros((nsnaps, mass.shape[0], 3), dtype=np.float64)
    self_accelerations[0] = self_acc.copy()
    self_potentials[0] = self_pot.copy()

    i_out = 1
    for step, t in enumerate(tqdm(ts_integrate[1:]), start=1):
        pos, vel = leapfrog_step(pos, vel, 
                                 partial(acc_fn, mass=mass, method=method, **kwargs), dt)
        if step % steps_per_output == 0 and i_out < nsnaps:
            positions[i_out] = pos.copy()
            velocities[i_out] = vel.copy()
            i_out += 1

    return (positions, velocities, ts_out)