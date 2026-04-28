from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class State:
    """Immutable snapshot of an N-body system at a single instant.

    Used at the Force/integrator boundary so a Force implementation can never
    accidentally mutate the integrator's working arrays. Also the natural
    handoff type from IC generators (samplers return a State).

    The arrays are stored as read-only numpy views; attempting to write to
    `state.pos`, `state.vel`, or `state.mass` raises ``ValueError``. Make a
    copy if you need a writable buffer.
    """

    pos: np.ndarray   # (N, 3) kpc
    vel: np.ndarray   # (N, 3) kpc / Gyr (internal)
    mass: np.ndarray  # (N,)   Msun
    t: float = 0.0    # Gyr

    def __post_init__(self):
        for name in ('pos', 'vel', 'mass'):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                arr = np.asarray(arr, dtype=np.float64)
                object.__setattr__(self, name, arr)
            arr.flags.writeable = False

    @property
    def n(self) -> int:
        return self.mass.shape[0]
