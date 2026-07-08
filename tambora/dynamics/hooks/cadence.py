from abc import ABC, abstractmethod


class Cadence(ABC):
    """
    Abstract base class for integration cadence hooks that determine when to output data.
    """

    @abstractmethod
    def due(self, step: int, steps_per_output: int) -> bool:
        """
        Determine whether to output data at the given step.

        Parameters
        ----------
        step : int
            The current integration step.
        steps_per_output : int
            The number of steps between outputs.

        Returns
        -------
        bool
            True if data should be output at this step, False otherwise.
        """
        ...

class EveryStep(Cadence):
    def due(self, step, steps_per_output): return True

class EveryNSteps(Cadence):
    def __init__(self, n): self.n = n
    def due(self, step, steps_per_output): return step % self.n == 0

class EveryOutput(Cadence):
    # default; aligned with snapshots
    def due(self, step, steps_per_output): return step % steps_per_output == 0

class EveryNOutputs(Cadence):
    def __init__(self, n): self.n = n
    def due(self, step, steps_per_output): return step % (self.n * steps_per_output) == 0


