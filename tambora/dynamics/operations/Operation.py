from abc import ABC


class Operation(ABC):
    """Base class for operations evaluated during integration.

    Subclasses declare:

    * ``name``     -- cache key for recorded output (``None`` => not recorded).
    * ``cadence``  -- run every ``cadence`` integration steps.
    * ``requires`` -- derived quantities the operation consumes (advisory; can be
      validated against the provider set at run time).

    An operation only knows how to *produce* its value(s) from state. Storage and
    the cached-vs-recompute policy live elsewhere (see :class:`~.diagnostic.Diagnostic`),
    so ``compute`` stays pure -- it is what *fills* the cache, never what reads it.
    """

    name = None
    cadence = 1
    requires = ()
    display = None       # optional Display attached for progress-bar readout

    def init(self, N):
        """Optional hook: allocate per-run state before integration begins."""

    def finalize(self):
        """Optional hook: return a summary after the run (used by accumulators)."""
        return None
