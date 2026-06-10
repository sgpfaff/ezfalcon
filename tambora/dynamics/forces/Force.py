from abc import ABC, abstractmethod
import numpy as np


class Force(ABC):
    """Root of all forces acting on N-body particles.

    The canonical interface used at the integrator boundary is
    :meth:`_eval_acc`, which always takes the full ``(pos, vel, mass, t)``
    state. Concrete subclasses implement the author-facing :meth:`acc`, whose
    signature may be narrower (e.g. ``acc(pos, t)`` for an external potential);
    the matching ``_eval_*`` adapter maps the uniform call onto it.

    Forces split along two orthogonal axes:

    * **Coupling** — :class:`ExternalForce` (single-particle, evaluated per
      particle against a fixed external field) vs :class:`InteractionForce`
      (collective N-body, requires a solver). This determines how the force is
      routed and whether it needs the whole-ensemble arrays.
    * **Conservative** — the :class:`~.Conservative` mixin adds a scalar
      potential (and hence velocity-independence) for energy diagnostics.

    Forces compose with ``+``::

        combined = ExternalGalpyPotential(nfw) + ExternalGalpyPotential(disk)

    The result is a :class:`CompositeForce` that itself satisfies the
    :class:`Force` interface.
    """

    @abstractmethod
    def acc(self, pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, t) -> np.ndarray:
        """Acceleration on each particle, shape ``(N, 3)``, internal units."""
        ...

    def _eval_acc(self, pos: np.ndarray, vel: np.ndarray, mass: np.ndarray, t) -> np.ndarray:
        return self.acc(pos, vel, mass, t)

    def __add__(self, other: "Force") -> "Force":
        from .CompositeForce import CompositeForce
        return CompositeForce([self, other])


class ExternalForce(Force):
    """Single-particle force: ``a_i = f(x_i, v_i, m_i, t)``.

    The field is imposed from outside the simulated system and each particle is
    evaluated independently (embarrassingly parallel; no solver). Note that an
    external force *may* depend on a particle's own mass (e.g. dynamical
    friction); what makes it external is that it does not couple particles to
    one another.
    """


class InteractionForce(Force):
    """Collective N-body force: ``a_i = f({x_j}, {v_j}, {m_j}, t)``.

    The field is sourced by the simulation particles themselves, so evaluating
    it couples every particle to every other and requires a solver (tree,
    direct summation). Self-gravity is the canonical example. These are routed
    through the solver/cache path, never through ``add_external_force``.
    """


class NullExternalForce(ExternalForce):
    """No-op external force. Used as an empty external slot."""

    def __init__(self):
        self._zeros_array_3d = None
        self._zeros_array_1d = None

    def acc(self, pos, vel, mass, t):
        if self._zeros_array_3d is None:
            self._zeros_array_3d = np.zeros_like(pos)
        return self._zeros_array_3d

    def potential(self, pos, vel, mass, t):
        if self._zeros_array_1d is None:
            self._zeros_array_1d = np.zeros(pos.shape[0])
        return self._zeros_array_1d

    def acc_and_potential(self, pos, vel, mass, t):
        return self.acc(pos, vel, mass, t), self.potential(pos, vel, mass, t)
