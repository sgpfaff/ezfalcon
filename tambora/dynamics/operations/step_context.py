"""Per-step view of integration state, with a memo for derived quantities.

Operations read raw state off a :class:`StepContext` and pull derived quantities
via :meth:`StepContext.get`. Whatever ``get`` computes is memoized for the life of
the step, so a quantity needed by several operations (and by recording) is computed
exactly once. ``StepResult`` is merely the integrator's fast-path cache (``self_acc``,
``self_pot``, ...); anything else is derived on demand from the providers table.
"""

import numpy as np


class StepContext:
    def __init__(self, pos, vel, mass, t, step, step_result, providers):
        self.pos, self.vel, self.mass = pos, vel, mass
        self.t, self.step = t, step
        self._sr = step_result          # fast-path fields the integrator already produced
        self._providers = providers     # name -> callable(ctx[, *args]) for everything else
        self._memo = {}

    def get(self, name):
        if name in self._memo:
            return self._memo[name]
        val = getattr(self._sr, name, None)         # fast path: integrator already has it
        if val is None:
            provider = self._providers.get(name)    # derive (may recurse into get())
            if provider is None:
                raise KeyError(f"no provider for derived quantity {name!r}")
            val = provider(self)
        self._memo[name] = val
        return val

    def self_pot(self, subset=None):
        """Self-potential at a subset's particles, *sourced by that subset only*.

        ``subset=None`` returns the full-system potential (``get("self_pot")``). A
        slice/array runs the solver on just those particles -- which is what
        'bound to this component' needs; you cannot slice it out of the full
        potential. The provider, not the context, owns the self-gravity solver.
        Memoized per step on a hashable rendering of *subset*.
        """
        if subset is None:
            return self.get("self_pot")
        hashable = subset if isinstance(subset, slice) else np.asarray(subset).tobytes()
        key = ("self_pot_of", hashable)
        if key not in self._memo:
            self._memo[key] = self._providers["self_pot_of"](self, subset)
        return self._memo[key]


def base_providers(self_gravity_force):
    """Providers shared by the run and by post-hoc recompute.

    During a run, ``StepContext.get`` short-circuits ``self_acc``/``self_pot`` via the
    ``StepResult`` fast path; these are used on recompute (no ``StepResult``) and for
    subset solves. ``self_grav`` returns both arrays in one pass so the memo can feed
    ``self_acc`` and ``self_pot`` without sweeping twice.
    """
    sg = self_gravity_force

    def self_grav(ctx):
        return sg.acc_and_potential(ctx.pos, ctx.mass)

    return {
        "self_grav":   self_grav,
        "self_acc":    lambda ctx: ctx.get("self_grav")[0],
        "self_pot":    lambda ctx: ctx.get("self_grav")[1],
        # parametric (called directly by StepContext.self_pot, not via get): potential
        # at a subset, sourced by that subset.
        "self_pot_of": lambda ctx, subset: sg.acc_and_potential(ctx.pos[subset],
                                                                ctx.mass[subset])[1],
    }
