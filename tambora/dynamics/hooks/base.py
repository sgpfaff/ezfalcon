from abc import ABC, abstractmethod
import numpy as np
from ..integration import StepState
from .cadence import EveryOutput


class Hook(ABC):
    """
    Abstract base class for integration hooks that run between integration steps.

    A hook is a callable that receives a :class:`StepState` (a live, read-only
    view of the simulation at the current step) and observes it, typically
    accumulating results on ``self`` for inspection after the run.

    Attributes
    ----------
    default_cadence : Cadence
        Cadence used if none is supplied to ``Sim.add_hook``.
    """

    default_cadence = EveryOutput()

    @abstractmethod
    def __call__(self, state: StepState):
        """
        Apply the hook to the given StepState.

        Parameters
        ----------
        state : StepState
            The current state of the simulation at this step.
        """
        ...

    def _dedup_key(self):
        """
        Identity used by ``Sim.add_hook`` to reject duplicate registrations.

        Return a hashable/comparable key built from the hook's *meaningful
        configuration*. These are the flags that make two instances mean different
        things (e.g. ``component``, ``eps``, an output ``path``, an index set).
        Two hooks with equal, non-``None`` keys are treated as duplicates.
        Never include mutable/result state (accumulated logs, sampled times).

        The default is ``None``: opt out of dedup entirely (any number of
        instances allowed). 
        """
        return None


class EnergyMonitor(Hook):
    """
    Track the fractional change in total system energy over the run.

    Results accumulate on the instance: ``t`` holds the sampled times and ``dE``
    the corresponding ``|(E - E0) / E0|`` values, where ``E0`` is the energy the
    first time the hook fires.
    """

    default_cadence = EveryOutput()

    def __init__(self):
        self.E0 = None
        self.t = []
        self.dE = []

    def _dedup_key(self):
        # Config-less whole-system diagnostic: at most one per simulation.
        return (type(self),)

    def __call__(self, state: StepState):
        E = state.system_energy()
        if self.E0 is None:
            self.E0 = E
        dE = np.abs((E - self.E0) / self.E0)
        self.t.append(state.t)
        self.dE.append(dE)
        state.report(**{"|dE/E0|": f"{float(dE):.2e}"})
