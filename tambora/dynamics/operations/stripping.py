"""Stripping tracker: edge-detect a component's boundedness across steps."""

import numpy as np

from .accumulator import Accumulator
from ...tools.util.units import KMS_TO_KPCGYR


class StrippingTracker(Accumulator):
    """Track stripping events for a component (or the whole system) at its own cadence.

    Runs every ``cadence`` steps (``1`` = finest, per step). Recapture is detected by
    comparing the confirmed bound state across checks. ``confirm=k`` requires ``k``
    consecutive checks in the new state before a transition counts, suppressing
    tidal-boundary chatter; the recorded time/phase-space is that of the *crossing*
    (when the flip began), not the confirmation instant.

    finalize() returns flat, per-particle fields (suffixed with the component name):
    ``n_strip``, ``n_recapture``, ``tstrip_first``, ``tstrip_last`` [Gyr], and
    ``strip_pos`` [kpc] / ``strip_vel`` [km/s] at first stripping.

    Parameters
    ----------
    sim : Sim
        Used only to resolve the component's slice at construction time.
    component : str or None
        Component name, or ``None`` for the whole system.
    cadence : int
        Steps between checks (default 1).
    confirm : int
        Consecutive checks in the new state before a transition counts (default 1).
    """

    FIELDS = ("n_strip", "n_recapture", "tstrip_first", "tstrip_last", "strip_pos", "strip_vel")

    def __init__(self, sim, component=None, cadence=1, confirm=1):
        self._sl = slice(None) if component is None else sim._slices[component]
        self.component = component
        self.dep = "bound" if component is None else f"bound_{component}"
        self.name = "stripping" if component is None else f"stripping_{component}"
        self.cadence = cadence
        self.k = confirm
        self.requires = (self.dep,)

    def init(self, N):
        n = len(range(*self._sl.indices(N)))
        self._initialized = False
        self._state = None                            # confirmed bound state (set on first update)
        self._streak = np.zeros(n, dtype=int)         # consecutive checks disagreeing with _state
        self._cand_t = np.full(n, np.nan)             # time the pending flip began (the crossing)
        self._cand_x = np.full((n, 3), np.nan)
        self._cand_v = np.full((n, 3), np.nan)
        self.n_strip = np.zeros(n, dtype=int)
        self.n_recapture = np.zeros(n, dtype=int)
        self.tstrip_first = np.full(n, np.nan)
        self.tstrip_last = np.full(n, np.nan)
        self.strip_pos = np.full((n, 3), np.nan)
        self.strip_vel = np.full((n, 3), np.nan)

    def update(self, ctx):
        bound = ctx.get(self.dep)                      # (n,) over the component; memoized this step
        if not self._initialized:                      # establish baseline; no transition yet
            self._state = bound.copy()
            self._initialized = True
            return
        pos, vel = ctx.pos[self._sl], ctx.vel[self._sl]
        disagree = bound != self._state
        starting = disagree & (self._streak == 0)      # crossing just began -> stamp candidate
        self._cand_t[starting] = ctx.t
        self._cand_x[starting] = pos[starting]
        self._cand_v[starting] = vel[starting]
        self._streak[disagree] += 1
        self._streak[~disagree] = 0                    # back in confirmed state -> reset
        confirmed = disagree & (self._streak >= self.k)
        to_unbound = confirmed & self._state           # confirmed strip
        to_bound = confirmed & ~self._state            # confirmed recapture
        self.n_strip += to_unbound
        self.n_recapture += to_bound
        first = to_unbound & np.isnan(self.tstrip_first)
        self.tstrip_first[first] = self._cand_t[first]
        self.strip_pos[first] = self._cand_x[first]
        self.strip_vel[first] = self._cand_v[first]
        self.tstrip_last[to_unbound] = self._cand_t[to_unbound]
        self._state[confirmed] = bound[confirmed]
        self._streak[confirmed] = 0

    def finalize(self):
        suffix = "" if self.component is None else f"_{self.component}"
        out = {
            "n_strip": self.n_strip,
            "n_recapture": self.n_recapture,
            "tstrip_first": self.tstrip_first,
            "tstrip_last": self.tstrip_last,
            "strip_pos": self.strip_pos,                # kpc (internal == output for length)
            "strip_vel": self.strip_vel / KMS_TO_KPCGYR,  # kpc/Gyr -> km/s
        }
        return {f"{k}{suffix}": v for k, v in out.items()}
