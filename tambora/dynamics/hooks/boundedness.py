"""
Boundedness diagnostics as integration hooks.

``BoundednessHook`` computes the self-bound mask of a component once per fire
(via the cached ``StepState.bound_mask``) and, from it, always maintains a
memory-light transition log. Boolean quantities (bound fraction, counts, full
mask history) are *derived* from that log on demand. Coordinate-dependent
reductions (COM, dispersion) can't be derived from booleans, so they are opted
into with ``track=(...)`` and stored per fire.

Anything needing its own cadence — e.g. periodically storing pos/vel of the
bound set — is a *separate* hook (see ``BoundKinematics``); the cached
``bound_mask`` means such a hook shares the computation for free on overlapping
steps.
"""

import numpy as np

from .base import Hook
from .cadence import EveryOutput
from ..diagnostics import reconstruct_mask


# -- coordinate-dependent reductions (must be tracked; not derivable from deltas) --

def _bound_com(x, mass, mask):
    m = mass[mask]
    return (m[:, None] * x[mask]).sum(0) / m.sum()


def _bound_dispersion(vel, mass, mask):
    m = mass[mask]
    v_com = (m[:, None] * vel[mask]).sum(0) / m.sum()
    return np.sqrt((m * np.sum((vel[mask] - v_com) ** 2, axis=-1)).sum() / m.sum())


_REDUCTIONS = {
    'com':        lambda c, mask: _bound_com(c.pos(), c.mass(), mask),
    'com_vel':    lambda c, mask: _bound_com(c.vel(), c.mass(), mask),
    'dispersion': lambda c, mask: _bound_dispersion(c.vel(), c.mass(), mask),
}


class BoundednessHook(Hook):
    """Track boundedness of a component over time.

    Always maintains a transition log (initial mask + flip events), from which
    boolean diagnostics are derived. Optionally tracks coordinate reductions and
    captures per-transition payloads.

    Parameters
    ----------
    component : str
        Component to test for boundedness.
    eps : float
        Softening length [kpc].
    track : sequence of str, optional
        Coordinate reductions to store each fire. Available: ``'com'``,
        ``'com_vel'``, ``'dispersion'``. Each becomes a list attribute of the
        same name (aligned with ``self.t``).
    capture_transitions : sequence of str, optional
        Per-flip payload to attach to each event. Subset of ``('pos', 'vel')``.
    method, theta, max_iter
        Passed to the iterative unbinding solver (must match across hooks to
        share the cached ``bound_mask``).

    Attributes
    ----------
    t : list of float
        Times at which the hook fired.
    initial_mask : (n,) bool array
        Bound mask at the first fire.
    events : list of tuple
        ``(idx, time, direction, *payload)`` per flip; ``direction`` is +1
        (became bound) or -1 (became unbound). Indices are component-local.
    """

    default_cadence = EveryOutput()

    def __init__(self, component, eps, track=(), capture_transitions=(),
                 method='falcON', theta=0.6, max_iter=50):
        for name in track:
            if name not in _REDUCTIONS:
                raise ValueError(
                    f"Unknown track quantity {name!r}. Available: {sorted(_REDUCTIONS)}")
        for name in capture_transitions:
            if name not in ('pos', 'vel'):
                raise ValueError(
                    f"Unknown capture {name!r}. Available: ('pos', 'vel')")

        self.component = component
        self.eps = eps
        self.track = tuple(track)
        self.capture_transitions = tuple(capture_transitions)
        self.method = method
        self.theta = theta
        self.max_iter = max_iter

        self.t = []
        self.initial_mask = None
        self.events = []
        self._prev = None
        for name in self.track:                 # tracked reductions -> list attributes
            setattr(self, name, [])

    def __call__(self, state):
        c = state.component(self.component)
        mask = state.bound_mask(self.component, eps=self.eps, method=self.method,
                                theta=self.theta, max_iter=self.max_iter)
        self.t.append(state.t)

        # transition log (always on -- essentially free once we have the mask)
        if self._prev is None:
            self.initial_mask = mask.copy()
        else:
            for i in np.flatnonzero(mask != self._prev):
                direction = 1 if mask[i] else -1
                payload = []
                if 'pos' in self.capture_transitions:
                    payload.append(c.pos()[i].copy())
                if 'vel' in self.capture_transitions:
                    payload.append(c.vel()[i].copy())
                self.events.append((int(i), state.t, direction, *payload))
        self._prev = mask.copy()

        # coordinate reductions (stored; not delta-derivable)
        for name in self.track:
            getattr(self, name).append(_REDUCTIONS[name](c, mask))

        state.report(**{f"n_bound({self.component})": int(mask.sum())})

    # --- derived quantities (computed from the transition log, not stored) ---

    def mask_at(self, t):
        """Bound mask at time *t*, reconstructed from the transition log."""
        return reconstruct_mask(self.initial_mask, self.events, t)

    def history(self, times=None):
        """Dense ``(len(times), n)`` bound-mask history (default: fire times)."""
        times = self.t if times is None else times
        return np.array([self.mask_at(t) for t in times])

    def n_bound(self, times=None):
        """Number of bound particles at each time (default: fire times)."""
        return self.history(times).sum(axis=1)

    def n_unbound(self, times=None):
        return self.initial_mask.size - self.n_bound(times)

    def fraction(self, times=None):
        """Bound fraction at each time (default: fire times)."""
        return self.n_bound(times) / self.initial_mask.size

    def transition_times(self, direction=None):
        """Times of transitions, optionally filtered by direction (+1 / -1)."""
        return np.array([e[1] for e in self.events
                         if direction is None or e[2] == direction])

    def transitions(self, direction=None):
        """Full transition events, optionally filtered by direction (+1 / -1)."""
        return [e for e in self.events if direction is None or e[2] == direction]


class BoundKinematics(Hook):
    """Store pos/vel of the bound particles at this hook's cadence.

    The reference "separate capture hook": an O(N)-per-fire diagnostic that
    typically wants a coarser cadence than the boolean tracking above. It shares
    the cached ``bound_mask`` with ``BoundednessHook`` on any step where both
    fire (pass matching ``component``/``eps``/``method``/``theta``).

    Attributes
    ----------
    t : list of float
        Fire times.
    pos, vel : list of arrays
        Positions/velocities of the bound particles at each fire (ragged: the
        bound count varies over time). Internal units (kpc, kpc/Gyr).
    """

    default_cadence = EveryOutput()

    def __init__(self, component, eps, method='falcON', theta=0.6, max_iter=50):
        self.component = component
        self.eps = eps
        self.method = method
        self.theta = theta
        self.max_iter = max_iter
        self.t = []
        self.pos = []
        self.vel = []

    def __call__(self, state):
        c = state.component(self.component)
        mask = state.bound_mask(self.component, eps=self.eps, method=self.method,
                                theta=self.theta, max_iter=self.max_iter)
        self.t.append(state.t)
        self.pos.append(c.pos()[mask].copy())
        self.vel.append(c.vel()[mask].copy())
