# Operations framework — design sketch

Status: **proposal to react to**, not wired in. Code below is illustrative; signatures
match the current `Sim` / `_runner` / `StepResult` but nothing here has been run.

## The shape, in one breath

- **Storage is central** (`Sim` owns one `_records` dict) → single source of truth, serializable.
- **Policy lives on the operation** (compute / when-to-recompute / cadence) → the
  `use_cached`-vs-recompute logic exists once per quantity, not copy-pasted across accessors.
- **Dependencies resolve by pull** through a per-step `StepContext.get(name)` memo → no manual
  ordering of data dependencies, and "not in `StepResult`" stops being a special case.
- **Side-effect order is explicit** = registration order in a list.

Three operation categories, plus a function tier for the common case:

| category        | method            | runs        | recorded?        | recompute from a snapshot? |
|-----------------|-------------------|-------------|------------------|----------------------------|
| `Diagnostic`    | `compute(ctx)`    | per output  | yes (if `name`)  | yes                        |
| `StateTransform`| `apply(ctx)`      | per step    | no               | n/a (mutates state)        |
| `Accumulator`   | `update(ctx)`     | per step    | via `finalize()` | no — needs history         |

Stateless diagnostics can be a plain function (`@sim.diagnostic`); you only reach for a class
when you need state (`Accumulator`) or want to override the caching policy.

---

## Core types

### `StepContext` — the read-only view + per-step memo

```python
class StepContext:
    """State at one step, plus a memo for derived quantities.

    Operations read raw state off the attributes and pull derived quantities via
    `get(name)`. Whatever `get` computes is memoized for the life of the step, so a
    quantity needed by several ops (and by recording) is computed exactly once.
    """
    def __init__(self, pos, vel, mass, t, step, step_result, providers):
        self.pos, self.vel, self.mass = pos, vel, mass
        self.t, self.step = t, step
        self._sr = step_result          # fast-path fields the integrator already produced
        self._providers = providers     # name -> callable(ctx[, *args]) for everything else
        self._memo = {}

    def get(self, name, *args):
        key = (name, args) if args else name
        if key in self._memo:
            return self._memo[key]
        val = None if args else getattr(self._sr, name, None)   # fast path (no-arg only)
        if val is None:
            provider = self._providers.get(name)                # derive (may recurse into get())
            if provider is None:
                raise KeyError(f"no provider for derived quantity {name!r}")
            val = provider(self, *args)
        self._memo[key] = val
        return val

    def self_pot(self, subset=None):
        """Sugar over the `self_pot` / `self_pot_of` providers. `subset=None` is the
        full-system potential; a slice/mask is the potential *sourced by that subset
        only* — which you cannot slice out of the full potential. The provider, not the
        context, owns the self-gravity solver."""
        if subset is None:
            return self.get("self_pot")
        hashable = subset if isinstance(subset, slice) else subset.tobytes()
        return self.get("self_pot_of", hashable, subset)
```

`StepContext` is now agnostic — no `sg`, no self-gravity knowledge. The runner installs the
self-gravity-specific bit as just another provider:

```python
def base_providers(sg):
    return {
        "self_grav":   lambda ctx: sg.acc_and_potential(ctx.pos, ctx.mass),
        "self_acc":    lambda ctx: ctx.get("self_grav")[0],
        "self_pot":    lambda ctx: ctx.get("self_grav")[1],
        # parametric: potential at a subset, sourced by that subset. `_h` is the hashable
        # memo key; `subset` is the real indexer.
        "self_pot_of": lambda ctx, _h, subset: sg.acc_and_potential(ctx.pos[subset],
                                                                    ctx.mass[subset])[1],
    }
```

`StepResult` is just the integrator's fast-path cache (`self_acc`, `self_pot`, …). Anything else
is derived on demand; the only thing genuinely out of reach is *within-step ephemera* (intermediate
sub-step velocities), which would need a hook inside the integrator, not an end-of-step observer.

### Operations

```python
class Operation:
    name: str | None = None     # set => output is recorded and queryable
    cadence: int = 1            # fire every `cadence` steps
    requires: tuple = ()        # derived quantities it consumes (docs/validation)
    def init(self, N): ...      # optional: allocate state before the run
    def finalize(self): return None

class Diagnostic(Operation):
    """A value derived from state. Recorded if `name` is set; always recomputable."""
    def compute(self, ctx): ...

    # default caching policy — override on a subclass to customize
    def at(self, t, store, state, use_cached=True):
        if use_cached:
            rec = store.get(self.name)
            if rec is None:
                raise ValueError(f"{self.name!r} was not recorded this run; "
                                 "pass use_cached=False to recompute.")
            values, _times = rec
            return values[state.ti(t)]
        return self.compute(state.ctx_at(t))         # recompute via the narrow view

class StateTransform(Operation):
    """Mutates the carried (pos, vel, mass). Returns the new triple."""
    def apply(self, ctx): ...                         # -> (pos, vel, mass)

class Accumulator(Operation):
    """Folds over the trajectory; result surfaced by finalize()."""
    def update(self, ctx): ...                        # -> None
    def finalize(self): ...
```

### `StateView` — the narrow recompute handle

The recompute path needs `pos(t)`, `mass`, and the forces, which live on `Sim`. Hand the op *this*,
not the whole `Sim`:

```python
class StateView:
    def __init__(self, sim): self._sim = sim
    def ti(self, t): return self._sim._ti(t)
    def ctx_at(self, t):
        i = self._sim._ti(t, vectorized=False)
        pos, vel = self._sim._positions[i], self._sim._velocities[i]
        # On recompute there's no StepResult; providers fall back to the forces.
        return StepContext(pos, vel, self._sim._mass, self._sim._times[i],
                           step=None, step_result=None, providers=self._sim._providers())
```

Base providers (shared by run and recompute) wrap the forces — see `base_providers` under
`StepContext` above. During a run, `getattr(step_result, "self_pot")` short-circuits them; on
recompute they call the force once and the memo feeds both `self_acc` and `self_pot`.

---

## The runner loop

```python
# providers = base_providers(sg) | {d.name: d.compute for d in diagnostics}
# records   = {d.name: (values_buf, times_buf) for recorded diagnostics}  (each its own cadence)
# write_idx = {d.name: 0}

for op in operations:
    op.init(N)

for step, t in enumerate(ts_integrate[1:], start=1):
    sr = integrator.step(pos, vel, mass, current_t, dt, sg, cons_ext, base_ext)
    pos, vel, current_t = sr.pos, sr.vel, sr.t
    ctx = StepContext(pos, vel, mass, current_t, step, sr, providers)

    # 1) state transforms, in registration order; refresh ctx after each mutation
    for tr in transforms:
        if step % tr.cadence == 0:
            pos, vel, mass = tr.apply(ctx)
            ctx = StepContext(pos, vel, mass, current_t, step, sr, providers)

    # 2) accumulators (stateful)
    for acc in accumulators:
        if step % acc.cadence == 0:
            acc.update(ctx)                       # may call ctx.get("bound"), etc.

    # 3) snapshot + recorded diagnostics, each on its own cadence
    if step % steps_per_output == 0:
        positions[i_out], velocities[i_out] = pos.copy(), vel.copy()
        i_out += 1
    for d in recorded_diagnostics:
        if step % d.cadence == 0:
            k = write_idx[d.name]
            values, times = records[d.name]
            values[k] = ctx.get(d.name)           # pull => compute once, memoized this step
            times[k]  = current_t
            write_idx[d.name] = k + 1

for op in operations:
    result = op.finalize()                        # accumulators hand back their array here
```

Two ordering rules made concrete: transforms run in **list order** (side effects you control), and a
diagnostic that needs another's output just calls `ctx.get(...)` (data deps, impossible to mis-order).

---

## Sim wiring

```python
class Sim:
    def add_operation(self, op):
        self._operations.append(op)               # ordered

    def diagnostic(self, name, every=1, requires=()):   # decorator for the function tier
        def deco(fn):
            d = _FnDiagnostic(fn, name=name, cadence=every, requires=requires)
            self._operations.append(d)
            return fn
        return deco

    def _providers(self):
        prov = base_providers(self._self_gravity_force)
        prov.update({d.name: d.compute for d in self._operations if isinstance(d, Diagnostic)})
        return prov

    # one generic accessor replaces the ~8 hand-written use_cached branches
    def record(self, name, t=..., use_cached=True):
        op = self._op_by_name[name]
        return op.at(t, self._records, StateView(self), use_cached)

    # --- convenience wrappers (iii): construct + register the common ops --------------------
    def track_boundedness(self, component=None, source="self", **kw):
        """Record per-particle boundedness. `component=None` => whole system."""
        op = Boundedness(**kw) if component is None else ComponentBoundedness(self, component, source, **kw)
        self.add_operation(op)
        return op            # return the handle so the user can tweak cadence etc.

    def track_stripping(self, component=None, cadence=1, record_events=False, source="self"):
        """Track stripping time / phase-space / counts at `cadence` steps (1 = finest).

        Registers the boundedness it depends on if not already present, so a single call
        is enough: sim.track_stripping("sat", cadence=1).
        """
        if (f"bound_{component}" if component else "bound") not in self._op_by_name:
            self.track_boundedness(component, source)
        self.add_operation(StrippingTracker(self, component, cadence, record_events))
```

The wrappers are thin — they just construct and register — so power users can still
`add_operation(...)` a hand-built op. Keeping them as one-liners avoids `Sim` re-accreting the
per-quantity logic we just pulled out of it.

### Auto-generated accessors (iii)

Rather than hand-write a `bound()` method, generate the accessor surface from the registry. Two
routing points, both extending the `__getattr__` you already have:

```python
# On Sim: whole-system diagnostics and accumulator results become attributes/methods.
def __getattr__(self, name):
    if name.startswith("_"): raise AttributeError(name)
    slices = self.__dict__.get("_slices", {})
    if name in slices:
        return Component(self, slices[name], name)
    ops = self.__dict__.get("_op_by_name", {})
    if name in ops:                                   # e.g. sim.bound(t)  /  sim.energy(t)
        return lambda t=..., use_cached=True: self.record(name, t, use_cached)
    summaries = self.__dict__.get("_summaries", {})
    if name in summaries:                             # e.g. sim.stripping -> StrippingResult
        return summaries[name]
    raise AttributeError(f"{type(self).__name__!r} has no attribute or component {name!r}")
```

```python
# On Component: per-component diagnostics and accumulator fields, both scoped by suffix.
def __getattr__(self, name):
    if name.startswith("_"): raise AttributeError(name)
    full = f"{name}_{self._name}"
    if full in self._sim._op_by_name:                 # diagnostic: sim.sat.bound(t) -> "bound_sat"
        return lambda t=..., use_cached=True: self._sim.record(full, t, use_cached)
    if full in self._sim._summaries:                  # accumulator field: sim.sat.n_strip -> array
        return self._sim._summaries[full]
    raise AttributeError(name)

# Registration forwarders so you declare tracking from the proxy (point iii):
def track_boundedness(self, **kw): return self._sim.track_boundedness(self._name, **kw)
def track_stripping(self, **kw):   return self._sim.track_stripping(self._name, **kw)
```

So the surface comes out as:

```python
sim.sat.track_boundedness()                  # declare from the proxy, component is implicit
sim.sat.track_stripping(cadence=1, confirm=3)
sim.run(...)

sim.bound(t=3.0)                 # whole-system Boundedness (named "bound"): a method
sim.sat.bound(t=3.0)             # ComponentBoundedness "bound_sat": a method
sim.sat.bound(t=3.0, use_cached=False)
sim.sat.n_strip                  # accumulator field: a plain (N_sat,) array attribute
sim.sat.tstrip_last              # (N_sat,) Gyr
sim.sat.strip_pos                # (N_sat, 3) kpc
```

Note the deliberate split: **diagnostics are methods** (they take `t`/`use_cached`), **accumulator
fields are bare attributes** (one value per particle for the whole run, no time argument) — which is
exactly the `sim.sat.tstrip_last` / `sim.sat.n_strip` form you asked for. The same `__getattr__`
mechanism lets `self_acc`/`self_pot`/`energy` (now diagnostics) keep their current method names while
their bodies collapse to one generic path. No per-quantity method maintenance.

`self.bound(t)` etc. become thin wrappers over `record("bound", t)`, or you read the raw buffer via
`sim._records["bound"]`.

---

## Immediate usage — particle boundedness

A stateless diagnostic that depends on `self_pot` (pulled, computed once per step):

```python
import numpy as np
from tambora.dynamics.operations import Diagnostic

class Boundedness(Diagnostic):
    """Iterative self-unbinding: bound iff KE in the bound-set COM frame + self-Φ < 0."""
    name = "bound"
    requires = ("self_pot",)

    def compute(self, ctx):
        phi = ctx.get("self_pot")                 # (N,) per unit mass — free during a run
        mass, vel = ctx.mass, ctx.vel
        bound = np.ones(len(mass), dtype=bool)
        for _ in range(10):                       # converges fast; cap iterations
            vcom = np.average(vel[bound], weights=mass[bound], axis=0)
            ke = 0.5 * np.sum((vel - vcom) ** 2, axis=1)
            new = (ke + phi) < 0
            if np.array_equal(new, bound):
                break
            bound = new
        return bound
```

Use it:

```python
sim.add_operation(Boundedness())
sim.run(t_end=5, dt=1e-3, dt_out=1e-2, eps=0.05)

sim._records["bound"]            # (n_out, N) bool history  — recorded during the run
sim.record("bound", t=3.0)       # bound mask at t≈3 Gyr (from cache)
sim.record("bound", t=3.0, use_cached=False)   # recompute at that snapshot instead
sim.sat.pos(t=3.0)[sim.record("bound", t=3.0)]  # positions of still-bound satellite stars
```

That's the whole story for your current need: define `compute`, register, run, read.

With the convenience layer (iii), the full intended workflow — boundedness recorded at snapshots
**plus** stripping tracked at per-step resolution with phase space and counts — is three calls:

```python
sim.sat.track_boundedness()                          # dense bound_sat mask at dt_out
sim.sat.track_stripping(cadence=1, confirm=3)        # precise debounced events, every step
sim.run(t_end=5, dt=1e-3, dt_out=1e-2, eps=0.05)

sim.sat.bound(t=3.0)          # bound mask at t≈3 (from cache)
sim.sat.tstrip_last           # (N_sat,) last stripping time per star [Gyr]
sim.sat.n_strip, sim.sat.n_recapture, sim.sat.strip_pos   # direct attributes
```

### Per-component boundedness (multiple components, mutual effects intact)

Two things to separate cleanly:

- **Mutual effects during integration** are automatic — every component feels every other through
  the shared force evaluation. Nothing in the diagnostic changes that.
- **Whose potential/COM defines "bound"** is a *diagnostic* choice. For "bound to the satellite" you
  want the potential sourced by the satellite *only*, in the satellite's own COM frame — hence
  `ctx.self_pot(slice)`, which does a subset solve (you can't slice it out of the full potential).

```python
class ComponentBoundedness(Diagnostic):
    """Boundedness of one component via monotonic iterative unbinding.

    source          : 'self' = potential sourced by this component only;
                      'all'  = the full-system potential.
    max_iter        : cap on unbinding iterations (default 10).
    recompute_every : how often to re-solve the *bound-set* self-potential during the
                      iteration. 0 (default) freezes phi at the whole-component value;
                      1 = strict (re-solve every iteration); k = every k iterations.
                      Ignored when source='all'.
    """
    def __init__(self, sim, component, source="self", max_iter=10, recompute_every=0):
        self.sl = sim._slices[component]
        self.name = f"bound_{component}"
        self.source, self.max_iter, self.recompute_every = source, max_iter, recompute_every

    def compute(self, ctx):
        sl = self.sl
        gidx = np.arange(sl.start, sl.stop)              # global indices of this component
        vel, mass = ctx.vel[sl], ctx.mass[sl]
        phi_full = ctx.get("self_pot")[sl] if self.source == "all" else ctx.self_pot(gidx)
        bound = np.ones(len(gidx), dtype=bool)           # shrinks only (monotonic)
        for it in range(self.max_iter):
            members = gidx[bound]
            if self.source == "self" and self.recompute_every and it % self.recompute_every == 0:
                phi_b = ctx.self_pot(members)            # re-solve over the current bound set
            else:
                phi_b = phi_full[bound]                  # frozen (sliced to current members)
            vb, mb = vel[bound], mass[bound]
            vcom = np.average(vb, weights=mb, axis=0)
            ke_b = 0.5 * np.sum((vb - vcom) ** 2, axis=1)
            still = (ke_b + phi_b) < 0
            if still.all():
                break
            new = bound.copy(); new[bound] = still       # map back to the component frame
            bound = new
        return bound                                     # (N_component,) over the slice
```

```python
sim.add_operation(ComponentBoundedness(sim, "sat"))      # bound_sat    -> (n_out, N_sat)
sim.add_operation(ComponentBoundedness(sim, "stream"))   # bound_stream, independent
sim.run(...)
sim._records["bound_sat"]
```

Each component records its own `(n_out, N_component)` mask; "multiple separately" is just multiple
registrations. Notes:

- **Monotonic is within a single snapshot only — it does *not* prevent recapture.** Each call starts
  from *all* component particles and iterates down, so a particle stripped at an earlier step is
  re-tested from scratch here and can come back bound; the `StrippingTracker` sees that as recapture
  across steps. Monotonicity only means that *inside one convergence loop* a removed member isn't
  re-added — which keeps the potential solve a plain self-solve on the shrinking bound set (so even
  `recompute_every` needs no falcON change). The only thing that would need a sources≠sinks solve is
  within-snapshot re-binding, which nobody wants and we deliberately skip.
- **Cost.** `recompute_every=0` is one subset solve per component per record; `recompute_every=1`
  adds up to `max_iter` solves. All at output cadence, so still cheap relative to the run. The
  full-system `self_pot` is computed once and shared via the memo.

---

## Advanced patterns this unlocks

### 1. Stripping tracker — a stateful `Accumulator` with its own resolution

An edge-detecting tracker over a component's boundedness. It runs at **its own cadence** (independent
of `dt_out`) — `cadence=1` is per-step. Recapture is detected by comparing the confirmed state across
checks; **debounce** (`confirm=k`, point v) suppresses tidal-boundary chatter by requiring `k`
consecutive checks in the new state before committing a transition. It exposes a flat set of **named
per-particle outputs** (point i) — `n_strip`, `n_recapture`, `tstrip_first`, `tstrip_last`,
`strip_pos`, `strip_vel` — each surfaced directly on the component proxy.

```python
class StrippingTracker(Accumulator):
    """Edge-detect a component's boundedness across steps, with hysteresis.

    component : name, or None for the whole system.
    cadence   : steps between checks (1 = finest; default 1).
    confirm   : consecutive checks in the new state before a transition counts (1 = raw flips).
    """
    FIELDS = ("n_strip", "n_recapture", "tstrip_first", "tstrip_last", "strip_pos", "strip_vel")

    def __init__(self, sim, component=None, cadence=1, confirm=1):
        self._sl = slice(None) if component is None else sim._slices[component]
        self.component, self.dep = component, ("bound" if component is None else f"bound_{component}")
        self.name = "stripping" if component is None else f"stripping_{component}"
        self.cadence, self.k, self.requires = cadence, confirm, (self.dep,)

    def init(self, N):
        n = len(range(*self._sl.indices(N)))
        self._state  = np.ones(n, dtype=bool)         # confirmed bound state (start all bound)
        self._streak = np.zeros(n, dtype=int)         # consecutive checks disagreeing with _state
        self._cand_t = np.full(n, np.nan)             # time the pending flip began (the *crossing*)
        self._cand_x = np.full((n, 3), np.nan)
        self._cand_v = np.full((n, 3), np.nan)
        self.n_strip = np.zeros(n, int); self.n_recapture = np.zeros(n, int)
        self.tstrip_first = np.full(n, np.nan); self.tstrip_last = np.full(n, np.nan)
        self.strip_pos = np.full((n, 3), np.nan); self.strip_vel = np.full((n, 3), np.nan)

    def update(self, ctx):
        bound = ctx.get(self.dep)                      # fresh-from-all-particles, so recapture shows
        pos, vel = ctx.pos[self._sl], ctx.vel[self._sl]
        disagree = bound != self._state
        starting = disagree & (self._streak == 0)      # crossing just began -> stamp candidate
        self._cand_t[starting], self._cand_x[starting], self._cand_v[starting] = ctx.t, pos[starting], vel[starting]
        self._streak[disagree] += 1
        self._streak[~disagree] = 0                    # back in confirmed state -> reset
        confirmed = disagree & (self._streak >= self.k)
        to_unbound = confirmed &  self._state          # confirmed strip (use crossing time/phase)
        to_bound   = confirmed & ~self._state          # confirmed recapture
        self.n_strip += to_unbound; self.n_recapture += to_bound
        first = to_unbound & np.isnan(self.tstrip_first)
        self.tstrip_first[first] = self._cand_t[first]
        self.strip_pos[first], self.strip_vel[first] = self._cand_x[first], self._cand_v[first]
        self.tstrip_last[to_unbound] = self._cand_t[to_unbound]
        self._state[confirmed] = bound[confirmed]; self._streak[confirmed] = 0

    def finalize(self):                                # flat named outputs -> proxy attributes
        return {f: getattr(self, f) for f in self.FIELDS}
```

The accumulator pulls `bound_sat` via `ctx.get`, so it reuses the exact `ComponentBoundedness`
computation the recorder uses — at every step — while the dense `bound_sat` mask is only *recorded*
at `dt_out`. **Dense mask at snapshot cadence, precise events at fine cadence**, both off one
boundedness definition. The recapture check works precisely because `compute` re-derives boundedness
from *all* component particles each step (it never seeds from the previous bound set).

`finalize()` returns a `{field: array}` dict; the runner stores each into `_summaries[f"{field}_sat"]`
so they read back as `sim.sat.n_strip`, `sim.sat.tstrip_last`, etc. (point i, accessors below). Times
are in `Gyr`; `strip_pos`/`strip_vel` are converted to output units (`kpc`, `km/s`) at finalize.

If `dt_out` resolution is enough you don't need this — first-strip time is a one-line reduction over
the recorded `bound_sat` array. Reach for the tracker for sub-`dt_out` precision, phase-space at
stripping, recapture counts, or debounced transition counts.

### 2. Center-of-mass snapping — a per-step `StateTransform`

```python
class SnapCOM(StateTransform):
    cadence = 1
    def __init__(self, target=(0, 0, 0)): self.target = np.asarray(target)
    def apply(self, ctx):
        com = np.average(ctx.pos, weights=ctx.mass, axis=0)
        return ctx.pos - com + self.target, ctx.vel, ctx.mass
```

This is the one in the hot path — fine in Python for now, and the canonical thing to port to C later.

### 3. Ad-hoc diagnostic with no class — the function tier

```python
@sim.diagnostic(name="r_half", every=10)          # half-mass radius, every 10 steps
def half_mass_radius(ctx):
    r = np.linalg.norm(ctx.pos - np.average(ctx.pos, weights=ctx.mass, axis=0), axis=1)
    order = np.argsort(r)
    cum = np.cumsum(ctx.mass[order])
    return r[order][np.searchsorted(cum, 0.5 * cum[-1])]
```

`sim._records["r_half"]` then holds `(values, times)` on its own 10-step cadence.

### 4. Dependency chaining for free

A diagnostic that consumes another diagnostic — e.g. bound-only velocity dispersion — just pulls it:

```python
@sim.diagnostic(name="sigma_bound", requires=("bound",))
def sigma_bound(ctx):
    b = ctx.get("bound")                           # Boundedness.compute, memoized this step
    v = ctx.vel[b]
    return np.std(v - v.mean(axis=0))
```

**Ordering is irrelevant here** — and that's deliberate. `ctx.get("bound")` resolves through the
*provider* table (`"bound"` -> `Boundedness.compute`), not through the operation list, so it computes
`bound` on the spot regardless of whether `sigma_bound` was registered before or after `Boundedness`.
`bound` is computed once per step and reused by the recorder, by `StrippingTime`, and by this.

The one real precondition is that a provider for `"bound"` exists — i.e. you registered a
`Boundedness`. That's what `requires=("bound",)` declares; validate it against the provider set at
`run()` start so a missing/misspelled dependency fails fast instead of mid-run (open decision #2).

### 5. Triggered side effects — checkpoint on energy drift

`compute` returning `None` (or an `Accumulator`) makes a fine event hook:

```python
class CheckpointOnDrift(Accumulator):
    cadence = 100
    def __init__(self, tol=1e-3, path="chk.npz"):
        self.tol, self.path, self.E0 = tol, path, None
    def update(self, ctx):
        E = np.sum(0.5 * ctx.mass * np.sum(ctx.vel**2, axis=1) + ctx.mass * ctx.get("self_pot"))
        if self.E0 is None: self.E0 = E
        elif abs((E - self.E0) / self.E0) > self.tol:
            np.savez(self.path, pos=ctx.pos, vel=ctx.vel, mass=ctx.mass, t=ctx.t)
    def finalize(self): return None
```

### 6. Recompute with a *different* method than the run used

Because policy lives on the op, "recompute differently" is "construct the op you want and ask it":

```python
Boundedness().at(t=3.0, store={}, state=StateView(sim), use_cached=False)
```

### 7. Live progress-bar readouts (iv)

Any operation can surface a scalar in the `tqdm` bar by attaching a `display` spec. The spec carries a
label, a **reducer** (per-particle array -> scalar, may hold state for things like energy drift), a
format, and its own refresh cadence (so the bar isn't recomputed every step):

```python
class Display:
    def __init__(self, label, reduce, fmt="{:.2e}", every=None):
        self.label, self.reduce, self.fmt, self.every, self._st = label, reduce, fmt, every, {}
    def text(self, ctx, op):
        return self.label, self.fmt.format(self.reduce(ctx, op, self._st))

# presets
def fraction_bound(label="f_bound", fmt="{:.0%}"):
    return Display(label, lambda ctx, op, st: float(np.mean(ctx.get(op.name))), fmt)

def energy_drift(label="|dE/E0|"):
    def r(ctx, op, st):
        E = float(np.sum(0.5 * ctx.mass * np.sum(ctx.vel**2, 1) + ctx.mass * ctx.get("self_pot")))
        st.setdefault("E0", E)
        return abs((E - st["E0"]) / st["E0"]) if st["E0"] else 0.0
    return Display(label, r)
```

The runner collects every op with a `display`, and at each op's display cadence updates the bar
(reusing `ctx.get`, so the value is shared with recording — no extra solves):

```python
shown = {}
for op in operations:
    if getattr(op, "display", None) and step % (op.display.every or steps_per_output) == 0:
        k, v = op.display.text(ctx, op); shown[k] = v
pbar.set_postfix(shown)        # -> "... |dE/E0|=3.1e-04, f_bound=87%"
```

Wiring it up — energy conservation on **by default**, fraction-bound opt-in:

```python
sim.track_energy(display=True)                          # default: energy_drift() in the bar
sim.sat.track_boundedness(display=fraction_bound("sat bound"))
```

`track_energy(display=True)` just attaches `energy_drift()` to the energy diagnostic; pass
`display=False` to silence it. Because the reducer can hold state (`E0`), "conservation" (a ratio to
the initial value) works even though the bar only ever sees one step at a time.

---

## (iv) The existing self-gravity / energy caching *is* this framework

This is the message-one cleanup. Today `Sim` has `run(cache_self_gravity_acc=, cache_self_gravity_pot=)`,
the `_cached_self_acc/_cached_self_pot` arrays, and `self_gravity()/self_potential()/energy()/PE()/
system_energy()/dE()` each carrying the same `use_cached`-vs-recompute branch. All of that maps onto
the framework with no new concepts:

- **`self_acc`, `self_pot` become recorded diagnostics.** They're special only in that, *during a run*,
  `ctx.get("self_pot")` hits the `StepResult` fast path — recording is a tap, not a recompute. The
  `compute`/provider (the self-gravity force) is used only on the `use_cached=False` path. So the
  `cache_self_gravity_*` run flags just become "is `self_pot` in the recorded set."
- **`KE` is a derived quantity, never cached** — it's `0.5 m v²`, cheaper to recompute than to store.
- **`PE`, `energy`, `system_energy` are derived diagnostics that pull.** `PE = self_pot (recorded or
  recomputed) + external_pot`; `energy = KE + PE`. The `use_cached` decision now lives in exactly one
  place — `self_pot`'s `at()` — instead of being re-implemented in five accessors.

```python
@sim.diagnostic(name="energy", requires=("self_pot",))
def energy(ctx):
    ke  = 0.5 * ctx.mass * np.sum(ctx.vel ** 2, axis=1)
    pe  = ctx.mass * (ctx.get("self_pot") + ctx.get("ext_pot"))
    return ke + pe
```

One API consequence (flagged earlier): the current accessors let you recompute with a *different*
`method=`/`**kwargs` than the run used. Under this model "recompute differently" becomes "construct
the diagnostic you want and call it." Cleaner, but it's a deliberate change to those signatures —
decide it rather than letting it drift.

## What stays Python vs C

- Everything at **output cadence** (all recorded diagnostics, recording) stays Python essentially for
  free — it fires every `steps_per_output` steps, not every step.
- **Per-step** ops (`SnapCOM`, `StrippingTime`) sit in the hot path. The per-step *cost* is tiny
  (a masked write / a COM subtract), and they're the natural first things to reimplement C-side.
- Design rule: **zero registered ops ⇒ the loop never calls back into Python**, so the eventual
  pure-C fast path pays nothing. A registered Python op is an explicit escape hatch that re-enters
  Python at its cadence — correct and convenient, and as cheap as that cadence allows.

## Open decisions (your call)

1. **Per-op cadence vs snapshot-locked.** Sketch gives each recorded op its own buffer + counter
   (true "every couple of steps"). Simpler alternative: force diagnostic cadence to be a multiple of
   `dt_out` and share the snapshot index. The former is more flexible; the latter is less bookkeeping.
2. **`requires` — documentation or enforcement?** Either just advisory, or validated against
   available providers at `run()` time so a typo fails fast instead of mid-run.
3. **Accessor surface.** Sketch auto-generates `sim.bound(t)` / `sim.sat.bound(t)` via `__getattr__`
   (iii). Alternative is the explicit `sim.record(name, t)` only — less magic, more typing. The
   `__getattr__` route can collide with real attribute names, so decide a precedence rule
   (components > diagnostics > summaries, as sketched).
4. **`finalize()` results.** Decided: accumulators return a flat `{field: array}` dict, stored in
   `sim._summaries[f"{field}_{component}"]` and read as bare proxy attributes (`sim.sat.n_strip`),
   kept distinct from the per-snapshot `_records`. Open sub-point: do the phase-space fields
   (`strip_pos`/`strip_vel`) get unit-converted at finalize (sketch) or stay internal with a
   `return_internal` escape like the snapshot accessors?
5. **Stripping debounce.** Decided: `confirm=k` hysteresis (default `k=1` = raw flips). Sub-point:
   should the recorded *time* be the crossing (sketch) or the confirmation instant?
6. **Display refresh cadence.** Per-op `display.every` (sketch) vs a single global bar-refresh cadence
   on `run()`. Also: should `track_energy(display=True)` be auto-registered when self-gravity is on,
   or always require the explicit call?
