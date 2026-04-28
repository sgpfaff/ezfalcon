from typing import List, Tuple
import numpy as np

from ..state import State
from .BaseForce import BaseForce
from .ConservativeForceField import ConservativeForceField


def _flatten(forces: List[BaseForce]) -> List[BaseForce]:
    """Flatten nested composites so ``(a + b) + c`` and ``a + (b + c)`` agree."""
    out: List[BaseForce] = []
    for f in forces:
        if isinstance(f, _CompositeMixin):
            out.extend(f.members)
        else:
            out.append(f)
    return out


class _CompositeMixin:
    """Shared state for both composite variants. Not used directly."""

    members: List[BaseForce]

    def __init__(self, members: List[BaseForce]):
        self.members = members

    def force(self, state: State) -> np.ndarray:
        acc = self.members[0].force(state)
        for f in self.members[1:]:
            acc = acc + f.force(state)
        return acc

    def _c_handle(self):
        # Return a single C handle iff every member has one. Building the
        # iterating-shim handle is part of the C-fast-path work; for now
        # just signal "not all C-backed" whenever any member returns None.
        if any(f._c_handle() is None for f in self.members):
            return None
        # Placeholder: construct a composite C handle here when the C
        # integrator path lands. Until then, the auto resolver treats this
        # as "not all C-backed" because there is no shim.
        return None


class _CompositePlain(_CompositeMixin, BaseForce):
    """Composite whose member set includes at least one non-conservative force."""


class _CompositeConservative(_CompositeMixin, ConservativeForceField):
    """Composite of only-conservative members. Adds potential / one-pass path."""

    def potential(self, state: State) -> np.ndarray:
        pot = self.members[0].potential(state)
        for f in self.members[1:]:
            pot = pot + f.potential(state)
        return pot

    def force_and_potential(
        self, state: State
    ) -> Tuple[np.ndarray, np.ndarray]:
        acc, pot = self.members[0].force_and_potential(state)
        for f in self.members[1:]:
            a_i, p_i = f.force_and_potential(state)
            acc = acc + a_i
            pot = pot + p_i
        return acc, pot


def CompositeForce(members: List[BaseForce]) -> BaseForce:
    """Combine several forces into one. Returned by :meth:`BaseForce.__add__`.

    The result is a :class:`ConservativeForceField` iff every member is
    conservative — otherwise a plain :class:`BaseForce`. This means
    ``(NFW + DynFric).potential(state)`` is a clear ``AttributeError``
    rather than a silent half-answer.
    """
    flat = _flatten(members)
    if all(isinstance(f, ConservativeForceField) for f in flat):
        return _CompositeConservative(flat)
    return _CompositePlain(flat)
