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
    mutates : bool
        Whether the hook writes to the simulation state. Mutating hooks are not
        yet supported; this flag exists so the runner can recognise them and so
        ``Sim.add_hook`` can reject them cleanly.
    """

    default_cadence = EveryOutput()
    mutates = False

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

    def __call__(self, state: StepState):
        E = state.system_energy()
        if self.E0 is None:
            self.E0 = E
        dE = np.abs((E - self.E0) / self.E0)
        self.t.append(state.t)
        self.dE.append(dE)
        state.report(**{"|dE/E0|": f"{float(dE):.2e}"})
