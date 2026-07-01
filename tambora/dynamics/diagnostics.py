"""
Pure diagnostic computations shared by the live hooks (``StepState``) and the
post-run accessors (``Sim``).

Keeping the physics here — as plain array-in functions — is what lets the live
and post-run paths agree: both call the same code, one on live integration
state, the other on stored snapshots.
"""

import numpy as np


def bound_mask(pos, vel, mass, eps, method='falcON', theta=0.6, max_iter=50):
    """Boolean mask of self-bound particles via iterative unbinding.

    A particle is bound if its energy in the COM frame is negative,

        0.5 |v - v_com|^2 + phi_self < 0,

    where ``phi_self`` is the *specific* self-gravity potential of the currently
    bound set. Since both ``v_com`` and ``phi_self`` depend on which particles
    are bound, this is iterated to convergence: recompute the COM velocity and
    self-potential from the bound set, drop particles with E >= 0, repeat.

    Parameters
    ----------
    pos : (N, 3) array
        Positions [kpc].
    vel : (N, 3) array
        Velocities [kpc / Gyr] (internal units).
    mass : (N,) array
        Masses [Msun].
    eps : float
        Softening length [kpc].
    method : str, optional
        Self-gravity solver: ``'falcON'`` (default), ``'direct'``, ``'direct_C'``.
    theta : float, optional
        falcON opening angle (ignored by the direct methods).
    max_iter : int, optional
        Maximum unbinding iterations.

    Returns
    -------
    bound : (N,) bool array
        True for bound particles.

    Notes
    -----
    The self-potential is recomputed from the *bound subset* each iteration.
    When called on a single component of a multi-component system this measures
    binding to that component alone. It must NOT be shortcut to a cached
    full-system self-potential, which would fold in other components' particles
    and give the wrong binding energy.
    """
    from .forces.self_gravity import self_gravity

    mass = np.asarray(mass)
    bound = np.ones(len(mass), dtype=bool)
    kw = dict(eps=eps, theta=theta) if method == 'falcON' else dict(eps=eps)
    for _ in range(max_iter):
        mb = mass[bound]
        v_com = (mb[:, None] * vel[bound]).sum(0) / mb.sum()
        _, phi = self_gravity(pos[bound], mb, method=method, **kw)   # specific potential
        keep = 0.5 * np.sum((vel[bound] - v_com) ** 2, axis=-1) + phi < 0
        if keep.all():
            break                              # converged: nothing new unbound
        idx = np.flatnonzero(bound)
        bound[idx[~keep]] = False              # monotonic: once unbound, stays unbound
        if not bound.any():
            break
    return bound


def reconstruct_mask(initial_mask, events, t):
    """Reconstruct a bound mask at time *t* from an initial mask + transitions.

    Parameters
    ----------
    initial_mask : (N,) bool array
        Bound mask at the first fire.
    events : sequence of (idx, time, direction, *payload)
        Transition events; ``direction`` is +1 (became bound) or -1 (unbound).
        Any trailing payload (e.g. captured position) is ignored here.
    t : float
        Query time. The state is reconstructed by replaying every event with
        ``time <= t``.

    Returns
    -------
    mask : (N,) bool array
        Bound mask at time *t* (piecewise-constant between fires).
    """
    m = initial_mask.copy()
    for e in events:
        idx, te, direction = e[0], e[1], e[2]
        if te <= t:
            m[idx] = (direction == +1)
    return m
