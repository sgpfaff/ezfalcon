# Test plan — boundedness diagnostics & integration hooks

Coverage plan for the code added in `381bd14` (boundedness hooks) plus the
`criterion='jacobi'` extension. Grouped by source module; the grouping mirrors
the proposed `tests/` subpackage layout (see the bottom of this file).

Legend: `[x]` implemented in `test_boundedness.py` · `[ ]` to add.

---

## 1. `dynamics/diagnostics.py`

### 1a. `bound_mask` — energy criterion (pure physics)
The most bug-prone layer; test it directly on hand-built inputs.

- [ ] Isolated cold Plummer/blob in its COM frame → all bound; inject a few
      fast particles → exactly those return unbound.
- [ ] Frame-independence: adding a bulk velocity to every particle leaves the
      mask unchanged (uses `v - v_com`).
- [ ] Solver equivalence: `method='falcON'`, `'direct'`, `'direct_C'` give the
      same mask for small N.
- [ ] Monotonic convergence: the bound set only shrinks across iterations;
      `max_iter=1` yields a superset of the converged mask.
- [ ] Edge — single-particle component: direct self-potential is 0, so the
      particle is marked **unbound**. Pin this behavior (surprising but defined).
- [ ] Edge — all particles unbound: returns all-`False`, no divide-by-zero on
      the empty COM.

### 1b. `bound_mask` — Jacobi criterion (tidal)
- [x] Tidal-tensor sign convention vs analytic Kepler host
      (`diag(-GM/R³, -GM/R³, +2GM/R³)`).
- [x] `r_t = (Gm/λ₁)^(1/3)` matches the King form `R(m/2M)^(1/3)`.
- [x] Strips escaped tracers that the energy criterion keeps (jacobi ⊂ energy).
- [x] Reduces to the energy criterion when the tide → 0 (distant host).
- [x] Robust to an offset COM from a developed stream (regression: energy-first
      seeding must recover the core instead of collapsing to 0).
- [x] `criterion='jacobi'` without `tidal_force` → `ValueError`.
- [x] Unknown `criterion` → `ValueError`.
- [x] Purely compressive tensor (no positive eigenvalue) → `ValueError`.
- [ ] Jacobi result is always a subset of the energy result (property test over
      several random configs).
- [ ] Tensor is evaluated at the **core**, not the global median: a one-sided
      stream must not shift `λ₁`/`r_t` (compare against tensor at the core COM).

### 1c. `reconstruct_mask` (pure)
- [ ] Replays `+1`/`-1` events correctly; `time <= t` boundary is inclusive.
- [ ] Does not mutate `initial_mask` (assert input unchanged after the call).
- [ ] Ignores trailing payload entries (`idx, t, dir, pos, vel`).
- [ ] Documents order-dependence: an out-of-chronological-order log with a
      double flip on one index reconstructs wrong (guards the assumption).

---

## 2. `dynamics/hooks/`

### 2a. `BoundednessHook`
- [ ] Transition detection: flip one particle between two fires → one event with
      the right component-local `idx` and `direction`.
- [ ] Derived quantities mutually consistent and match a brute-force mask:
      `n_bound`, `n_unbound`, `fraction`, `history`, `mask_at`.
- [ ] `track=('com','com_vel','dispersion')` lists align with `self.t`.
- [ ] `capture_transitions=('pos','vel')` attaches correct per-flip payload.
- [ ] Validation: unknown `track` / `capture_transitions` name → `ValueError`.
- [x] `criterion='jacobi'` without `tidal_force` → `ValueError`.
- [ ] `transition_times(direction=±1)` / `transitions(direction=±1)` filter.
- [ ] First-fire `initial_mask` set, no spurious event on the first fire.

### 2b. `BoundKinematics`
- [ ] Stores `pos`/`vel` of the bound set each fire; ragged lengths track the
      shrinking bound count; arrays are copies (not views into live state).
- [x] `criterion='jacobi'` without `tidal_force` → `ValueError`.

### 2c. `base.py` (`Hook`, `EnergyMonitor`)
- [ ] `EnergyMonitor`: `E0` set on first fire; `dE` is `|(E−E0)/E0|`; near-zero
      for an energy-conserving run.
- [ ] `Hook` is abstract — instantiating a subclass without `__call__` raises.

### 2d. `cadence.py`
- [ ] `due()` truth tables for `EveryStep`, `EveryNSteps`, `EveryOutput`,
      `EveryNOutputs`.
- [ ] All cadences fire at `step=0` (the t0 fire), since `0 % n == 0`.

---

## 3. `dynamics/integration/step_state.py`

- [ ] Accessor parity with `Sim` on a fixed `StepResult`: `pos/vel/mass/KE/
      energy/self_potential/external_pot` return expected internal-unit values.
- [ ] Component views: `state.component('sat')` and `state.sat` slice correctly;
      unknown name → `AttributeError`.
- [ ] `bound_mask` caching: two identical calls compute once (patch/​spy the
      underlying `bound_mask`, assert one call).
- [ ] Cache keys don't collide: whole-system vs component, differing `eps` /
      `method` / `criterion` / `tidal_force` each get a distinct entry.
- [ ] `report()` is a no-op when there is no progress bar.

---

## 4. `dynamics/integration/integrate.py` (`_runner`, `_fire`)

- [ ] Hooks fire once on the initial state (t0) and then per cadence.
- [ ] A no-op cadence never fires; `_fire` returns `False` (no mutators today).
- [ ] `state` is refreshed only when at least one hook is due.
- [ ] Mutator-ordering hook: a `mutates=True` stub is sorted before observers
      (even though `add_hook` rejects it — `_fire`'s ordering is separate logic).

---

## 5. `simulation/simulation.py`

- [ ] `add_hook` after `run()` → `RuntimeError`.
- [ ] `add_hook` with a `mutates=True` hook → `NotImplementedError`.
- [ ] `add_hook` cadence resolution: uses `default_cadence`, else `EveryOutput`.
- [ ] `boundedness`: `eps=None` → `ValueError`; unknown component → `ValueError`.
- [ ] `boundedness` int vs float `t` indexing selects the same snapshot.
- [ ] `boundedness(criterion='jacobi', tidal_force=...)` passes through.

---

## 6. Cross-cutting integration tests

- [ ] **Keystone consistency**: run a `Sim` with a `BoundednessHook` at
      `EveryOutput`, then assert `sim.boundedness(c, t=snap) == hook.mask_at(snap)`
      for every snapshot — the live and post-run paths must agree.
- [ ] **Transition correctness** (`EveryStep`): a recorded event's `time` and
      captured `pos` equal the step state at that step — no off-by-one between
      "detected at step k" and the stored payload.
- [ ] **Transition convergence**: refining the cadence
      (`EveryOutput → EveryNSteps → EveryStep`) makes each transition time
      converge monotonically toward a reference.
- [ ] **End-to-end physics**: a cluster on a tidal orbit — bound fraction trends
      downward and stabilizes; `jacobi` bound fraction ≤ `energy` throughout.

---

## Proposed `tests/` subpackage layout

```
tambora/tests/
├── conftest.py                    # stays at root; fixtures propagate to all subdirs
├── __init__.py
├── dynamics/
│   ├── __init__.py
│   ├── test_diagnostics.py        # §1  (bound_mask energy+jacobi, reconstruct_mask)
│   ├── forces/
│   │   ├── __init__.py
│   │   ├── test_direct_summation.py
│   │   ├── test_falcON.py
│   │   ├── test_self_gravity_fn.py
│   │   ├── test_CompositeForce.py
│   │   └── test_tidal_tensor.py   # tidal_tensor() accessor + sign convention
│   ├── hooks/
│   │   ├── __init__.py
│   │   ├── test_boundedness.py    # §2a, §2b (move existing here)
│   │   ├── test_base.py           # §2c
│   │   └── test_cadence.py        # §2d
│   └── integration/
│       ├── __init__.py
│       ├── test_step_state.py     # §3
│       ├── test_runner.py         # §4  (existing)
│       └── test_leapfrog.py       # (existing)
├── simulation/
│   ├── __init__.py
│   └── test_simulation.py         # §5  (existing) + boundedness accessor
├── tools/
│   ├── __init__.py
│   ├── test_galpy_bridge.py
│   ├── test_galpy_tools.py
│   ├── test_imf_tools.py
│   └── test_units.py
├── test_component.py              # (existing; place under simulation/ if apt)
└── test_integration_boundedness.py  # §6 cross-cutting
```

**Reorg checklist**
- Add `__init__.py` to every new subpackage (package-style imports require it).
- Keep `conftest.py` at `tests/` root — its autouse RNG-seed fixture applies to
  all nested tests.
- Update the two hard-coded paths in `tox.ini`
  (`test_galpy_bridge.py`, `test_galpy_tools.py`) to their new locations.
- `testpaths = ["tambora"]` already recurses — no pyproject change needed.
