"""Accumulators: stateful operations folded over the trajectory."""

from .Operation import Operation


class Accumulator(Operation):
    """A stateful operation evaluated every ``cadence`` steps.

    Unlike a :class:`~.diagnostic.Diagnostic` (a pure function of one snapshot, so
    recomputable), an accumulator holds state across steps via :meth:`update` and
    surfaces its result through :meth:`finalize`, which returns a ``{field: array}``
    dict of per-particle summaries. It is *not* recomputable from a single snapshot.

    Resolution is set by ``cadence`` (steps between updates), independent of the
    output cadence ``dt_out`` -- use ``cadence=1`` for per-step resolution.
    """

    def update(self, ctx):
        """Fold this step into the accumulator's state."""
        raise NotImplementedError
