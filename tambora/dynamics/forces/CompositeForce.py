from typing import List, Tuple
import numpy as np
from .Force import Force
from .Conservative import Conservative


def _flatten(forces: List[Force]) -> List[Force]:
    """Flatten nested composites so ``(a + b) + c`` and ``a + (b + c)`` agree."""
    out: List[Force] = []
    for f in forces:
        if isinstance(f, _CompositeMixin):
            out.extend(f.members)
        else:
            out.append(f)
    return out


class _CompositeMixin:
    """Shared state for both composite variants. Not used directly."""

    members: List[Force]

    def __init__(self, members: List[Force]):
        self.members = members
    def _eval_acc(self, pos, vel, mass, t):
        a = self.members[0]._eval_acc(pos, vel, mass, t)
        for f in self.members[1:]:
            a = a + f._eval_acc(pos, vel, mass, t)
        return a

    # def _c_handle(self):
    #     # Return a single C handle iff every member has one. Building the
    #     # iterating-shim handle is part of the C-fast-path work; for now
    #     # just signal "not all C-backed" whenever any member returns None.
    #     if any(f._c_handle() is None for f in self.members):
    #         return None
    #     # Placeholder: construct a composite C handle here when the C
    #     # integrator path lands. Until then, the auto resolver treats this
    #     # as "not all C-backed" because there is no shim.
    #     return None


class _CompositePlain(_CompositeMixin, Force):
    """Composite whose member set includes at least one non-conservative force."""
    def acc(self, pos, vel, mass, t):
        if not self.members:
            return np.zeros_like(pos)
        a = self.members[0]._eval_acc(pos, vel, mass, t)
        for f in self.members[1:]:
            a = a + f._eval_acc(pos, vel, mass, t)
        return a

class _CompositeConservative(_CompositeMixin, Conservative, Force):
    """Composite of only-conservative members. Adds potential / one-pass path."""
    def acc(self, pos, mass, t):
        if not self.members:
            return np.zeros_like(pos)
        a = self.members[0]._eval_acc(pos, None, mass, t)
        for f in self.members[1:]:
            a = a + f._eval_acc(pos, None, mass, t)
        return a

    def potential(self, pos, mass, t):
        if not self.members:
            return np.zeros(pos.shape[0])
        p = self.members[0]._eval_potential(pos, None, mass, t)
        for f in self.members[1:]:
            p = p + f._eval_potential(pos, None, mass, t)
        return p

    def _eval_potential(self, pos, vel, mass, t):
        if not self.members:
            return np.zeros(pos.shape[0])
        p = self.members[0]._eval_potential(pos, vel, mass, t)
        for f in self.members[1:]:
            p = p + f._eval_potential(pos, vel, mass, t)
        return p

    def acc_and_potential(self, pos, mass, t):
        if not self.members:
            return np.zeros_like(pos), np.zeros(pos.shape[0])
        a, p = self.members[0]._eval_acc_and_potential(pos, None, mass, t)
        for f in self.members[1:]:
            a_i, p_i = f._eval_acc_and_potential(pos, None, mass, t)
            a, p = a + a_i, p + p_i
        return a, p


def CompositeForce(members: List[Force]) -> Force:
    """Combine several forces into one. Returned by :meth:`Force.__add__`.

    The result is a :class:`_CompositeConservative` iff every member is
    :class:`~.Conservative` — otherwise a plain :class:`_CompositePlain`. This
    means ``(NFW + DynFric).potential(state)`` is a clear ``AttributeError``
    rather than a silent half-answer.
    """
    flat = _flatten(members)
    if all(isinstance(f, Conservative) for f in flat):
        return _CompositeConservative(flat)
    return _CompositePlain(flat)