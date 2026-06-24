"""Live progress-bar readouts for operations.

Attach a :class:`Display` to any operation (``op.display = ...``) to surface a scalar
in the integration progress bar. The reducer maps the current step context to a scalar
and may keep state across steps (e.g. the initial energy for a drift ratio). Values are
pulled through the same ``StepContext`` the recorder uses, so nothing is computed twice.
"""

import numpy as np

from .Operation import Operation


class Display:
    """A scalar progress-bar readout.

    Parameters
    ----------
    label : str
        Key shown in the tqdm postfix.
    reduce : callable(ctx, op, state) -> float
        Maps the step context (and owning operation) to a scalar. ``state`` is a
        per-display dict the reducer may use to remember values across steps.
    fmt : str
        ``str.format`` spec applied to the scalar.
    every : int or None
        Refresh cadence in steps. ``None`` means "at the output cadence" (resolved
        by the runner), which keeps derived quantities on their fast path.
    """

    def __init__(self, label, reduce, fmt="{:.2e}", every=None):
        self.label = label
        self.reduce = reduce
        self.fmt = fmt
        self.every = every
        self._state = {}

    def text(self, ctx, op):
        return self.label, self.fmt.format(self.reduce(ctx, op, self._state))


class DisplayOnly(Operation):
    """An operation that only carries a :class:`Display` -- records nothing."""

    def __init__(self, display):
        self.display = display


def fraction_bound(label="f_bound", fmt="{:.0%}"):
    """Fraction of the operation's particles currently bound (uses ``op.name``)."""
    return Display(label, lambda ctx, op, st: float(np.mean(ctx.get(op.name))), fmt)

def number_unbound(label="n_unbound", fmt="{}"):
    """Fraction of the operation's particles currently bound (uses ``op.name``)."""
    return Display(label, lambda ctx, op, st: int(np.sum(~ctx.get(op.name))), fmt)

def energy_drift(label="|dE/E0|", fmt="{:.2e}"):
    """Relative drift of KE + self-potential energy from its first sampled value."""
    def reduce(ctx, op, st):
        E = float(np.sum(0.5 * ctx.mass * np.sum(ctx.vel ** 2, axis=1)
                         + ctx.mass * ctx.get("self_pot")))
        st.setdefault("E0", E)
        return abs((E - st["E0"]) / st["E0"]) if st["E0"] else 0.0
    return Display(label, reduce, fmt)
