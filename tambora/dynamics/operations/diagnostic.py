"""Diagnostics: per-step values derived from state.

A :class:`Diagnostic` is recorded during the run (if it has a ``name``) and is
always recomputable afterward. ``compute`` is pure; the cached-vs-recompute policy
lives in ``at``. The recompute path reaches snapshot state through a narrow
:class:`StateView` rather than the whole ``Sim``.
"""

from .Operation import Operation
from .step_context import StepContext


class Diagnostic(Operation):
    def compute(self, ctx):
        """Produce this diagnostic's value from the step context. Pure."""
        raise NotImplementedError

    def at(self, t, store, state, use_cached=True):
        """Cached-or-recompute access at time *t*.

        ``store`` is the run's record dict (``name -> (nsnaps, ...) array``);
        ``state`` is a :class:`StateView` onto the snapshot data + providers.
        """
        if use_cached:
            values = store.get(self.name)
            if values is None:
                raise ValueError(
                    f"{self.name!r} was not recorded this run; "
                    "pass use_cached=False to recompute."
                )
            return values[state.ti(t)]
        if t is ...:
            raise NotImplementedError(
                "Recompute over all snapshots (t=...) is not yet supported; "
                "pass an index/time, or use_cached=True."
            )
        return self.compute(state.ctx_at(t))


class StateView:
    """Narrow, read-only handle the recompute path uses instead of the whole Sim."""

    def __init__(self, sim):
        self._sim = sim

    def ti(self, t):
        return self._sim._ti(t)

    def ctx_at(self, t):
        i = self._sim._ti(t, vectorized=False)
        return StepContext(
            self._sim._positions[i], self._sim._velocities[i], self._sim._mass,
            float(self._sim._times[i]), None, None, self._sim._providers(),
        )
