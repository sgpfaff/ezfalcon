from dataclasses import dataclass
import numpy as np

@dataclass
class StepContext:
    """Per-run constants shared by every StepState. Built once in _runner."""
    slices: dict # component name -> slice (from Sim._slices)
    self_gravity_force: object
    conserv_ext_force: object
    base_ext_force: object
    progress: object = None # _Progress wrapping the run's progress bar, or None


class _Progress:
    """
    Reporting channel from hooks to the run's progress bar.

    Hooks call ``state.report(**fields)``; fields accumulate here and are pushed
    to the (duck-typed) progress bar's ``set_postfix``. Kept tqdm-agnostic so
    the runner owns the bar and StepState/hooks stay decoupled from it.
    """
    def __init__(self, pbar=None):
        self._pbar = pbar
        self._fields = {}

    def report(self, **fields):
        self._fields.update(fields)
        if self._pbar is not None:
            self._pbar.set_postfix(self._fields, refresh=False)


_FULL = slice(None)

class StepState:
    """
    Live view of the simulation at the current step, passed to hooks.

    Mirrors the API of Sim (pos(), vel(), KE(), energy(), self_gravity(), ...)
    with three main differences:
        * only accesses current time (accessors do not take `t`)
        * returns INTERNAL units for speed/consistency
        * borrows arrays + force objects by reference (no copies); expensive
        potentials are computed lazily and cached for the life of the step.
    """
    def __init__(self, result, ctx, sl=_FULL, cache=None):
        self._result = result       # StepResult: live pos/vel/mass/t + precomputed accs/pot
        self._ctx = ctx             # StepContext: slices + force objects
        self._sl = sl               # whole system, or one compontent
        self._cache = {} if cache is None else cache    # shared across component sub-views

    # -- reuse one instance across steps: refresh data, drop cached results --
    def _update(self, result):
        self._result = result
        self._cache.clear()

    # --- step metadata ---
    @property
    def step(self): return self._result.step

    @property
    def t(self): return self._result.t

    # -- reporting: push display fields to the run's progress bar --
    def report(self, **fields):
        '''
        Report display fields (e.g. ``|dE/E0|``) to the run's progress bar.

        Fields accumulate across hooks and persist between updates. No-op if the
        run has no progress bar (e.g. in tests).
        '''
        if self._ctx.progress is not None:
            self._ctx.progress.report(**fields)

    # -- component access: state.sat.pos() or state.component('sat') --
    def component(self, name):
        return StepState(self._result, self._ctx, self._ctx.slices[name], cache=self._cache)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        slices = self._ctx.slices
        if name in slices:
            return self.component(name)
        raise AttributeError(f"StepState has no attribute or component named {name!r}")
    
    # -- kinematics --
    def pos(self):  return self._result.pos[self._sl]
    def vel(self):  return self._result.vel[self._sl]
    def mass(self): return self._result.mass[self._sl]
    def x(self):    return self.pos()[..., 0]
    def y(self):    return self.pos()[..., 1]
    def z(self):    return self.pos()[..., 2]
    def vx(self):   return self.vel()[..., 0]
    def vy(self):   return self.vel()[..., 1]
    def vz(self):   return self.vel()[..., 2]
    def r(self):    return np.linalg.norm(self.pos(), axis=-1)
    def KE(self):   return 0.5 * self.mass() * np.sum(self.vel()**2, axis=-1)

    # -- accelerations -- uses integrator has already computed
    def self_gravity(self):
        a = self._result.self_acc
        return np.zeros_like(self.pos()) if a is None else a[self._sl]
    
    def external_acc(self):
        c, b = self._result.conserv_ext_acc, self._result.base_ext_acc
        out = 0.0
        if c is not None: out = out + c[self._sl]
        if b is not None: out = out + b[self._sl]
        return out
    
    # -- potentials --
    def self_potential(self):
        phi = self._result.self_pot
        if phi is None:
            phi = self._cache.setdefault(
                "self_pot",
                self._ctx.self_gravity_force.acc_and_potential(
                    self._result.pos, self._result.mass)[1])
        return self.mass() * phi[self._sl]
    
    def external_pot(self):
        # lazy compute on full array and slice after
        phi = self._cache.get("ext_pot")
        if phi is None:
            phi = self._ctx.conserv_ext_force.potential(
                self._result.pos, self._result.mass, self._result.t)
            self._cache["ext_pot"] = phi
        return self.mass() * phi[self._sl]
    
    def PE(self):       return self.self_potential() + self.external_pot()
    def energy(self):   return self.KE() + self.PE()
    def system_energy(self):
        return (np.sum(self.KE())
                + 0.5 * np.sum(self.self_potential())
                + np.sum(self.external_pot()))

    # -- boundedness (iterative unbinding), cached per step --
    def bound_mask(self, component=None, *, eps, method='falcON', theta=0.6, max_iter=50):
        '''
        Boolean mask of self-bound particles, via iterative unbinding.

        Cached for the life of the step and keyed by the particle subset and
        parameters, so multiple hooks asking on the same step (at any cadence)
        share a single computation.

        Parameters
        ----------
        component : str, optional
            Component to test. None (default) uses this view's particles (the
            whole system for a top-level state).
        eps : float
            Softening length [kpc]. Required.
        method, theta, max_iter
            Passed through to the iterative unbinding solver.
        '''
        from ..diagnostics import bound_mask as _bound_mask
        c = self if component is None else self.component(component)
        sl = c._sl
        # key by the resolved slice (not the component name) so a whole-system
        # call and a component call can never collide in the shared cache.
        key = ("bound_mask", (sl.start, sl.stop, sl.step), eps, method, theta, max_iter)
        if key not in self._cache:
            self._cache[key] = _bound_mask(c.pos(), c.vel(), c.mass(),
                                           eps=eps, method=method, theta=theta,
                                           max_iter=max_iter)
        return self._cache[key]
