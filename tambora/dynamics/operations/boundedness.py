"""Particle boundedness via iterative self-unbinding."""

import numpy as np

from .diagnostic import Diagnostic


class Boundedness(Diagnostic):
    """Whole-system boundedness: a particle is bound iff its kinetic energy in the
    bound-set COM frame plus its self-potential is negative.

    Parameters
    ----------
    max_iter : int
        Cap on unbinding iterations (default 10). The COM is recomputed over the
        currently-bound set each iteration; convergence is usually fast.
    """

    name = "bound"
    requires = ("self_pot",)

    def __init__(self, max_iter=10):
        self.max_iter = max_iter

    def compute(self, ctx):
        phi = ctx.get("self_pot")                 # (N,) per unit mass; free during a run
        vel, mass = ctx.vel, ctx.mass
        bound = np.ones(len(mass), dtype=bool)    # monotonic: members only leave the set
        for _ in range(self.max_iter):
            if not bound.any():
                break
            vb, mb = vel[bound], mass[bound]
            vcom = np.average(vb, weights=mb, axis=0)
            ke_b = 0.5 * np.sum((vb - vcom) ** 2, axis=1)
            still = (ke_b + phi[bound]) < 0
            if still.all():
                break
            new = bound.copy()
            new[bound] = still
            bound = new
        return bound


class ComponentBoundedness(Diagnostic):
    """Boundedness of one component via monotonic iterative unbinding.

    Parameters
    ----------
    sim : Sim
        Used only to resolve the component's slice at construction time.
    component : str
        Component name (must already be added to ``sim``).
    source : {'self', 'all'}
        Whose potential enters the energy: ``'self'`` = sourced by this component
        only (a subset solve); ``'all'`` = the full-system potential.
    max_iter : int
        Cap on unbinding iterations (default 10).
    recompute_every : int
        How often to re-solve the *bound-set* self-potential during the iteration.
        ``0`` (default) freezes the potential at the whole-component value; ``1`` =
        strict (re-solve every iteration); ``k`` = every ``k`` iterations. Ignored
        when ``source='all'``.
    """

    def __init__(self, sim, component, source="self", max_iter=10, recompute_every=0):
        self.sl = sim._slices[component]
        self.name = f"bound_{component}"
        self.source = source
        self.max_iter = max_iter
        self.recompute_every = recompute_every
        self.requires = ("self_pot",) if source == "all" else ()

    def compute(self, ctx):
        sl = self.sl
        gidx = np.arange(sl.start, sl.stop)              # global indices of this component
        vel, mass = ctx.vel[sl], ctx.mass[sl]
        phi_full = ctx.get("self_pot")[sl] if self.source == "all" else ctx.self_pot(gidx)
        bound = np.ones(len(gidx), dtype=bool)           # shrinks only (monotonic)
        for it in range(self.max_iter):
            if not bound.any():
                break
            if self.source == "self" and self.recompute_every and it % self.recompute_every == 0:
                phi_b = ctx.self_pot(gidx[bound])        # re-solve over the current bound set
            else:
                phi_b = phi_full[bound]                  # frozen (sliced to current members)
            vb, mb = vel[bound], mass[bound]
            vcom = np.average(vb, weights=mb, axis=0)
            ke_b = 0.5 * np.sum((vb - vcom) ** 2, axis=1)
            still = (ke_b + phi_b) < 0
            if still.all():
                break
            new = bound.copy()
            new[bound] = still                           # map back to the component frame
            bound = new
        return bound
