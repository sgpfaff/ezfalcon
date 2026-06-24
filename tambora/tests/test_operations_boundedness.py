'''
End-to-end tests for the operations framework via the boundedness diagnostic:
recording during a run, the cached vs recompute access policy, the Sim/Component
accessors, and per-component independence.
'''

import numpy as np
import pytest

from tambora.simulation import Sim


def _blob(n, scale, vsigma, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=scale, size=(n, 3)), rng.normal(scale=vsigma, size=(n, 3))


def _run_single(track_component=False, n=60, n_escapers=5, **run_kw):
    """A cold, deeply-bound blob plus a few fast escapers, with boundedness tracked."""
    pos, vel = _blob(n, scale=0.3, vsigma=0.3, seed=0)
    vel[:n_escapers] += 400.0                       # km/s -> clearly unbound
    mass = np.full(n, 2e7)
    sim = Sim()
    sim.add_particles('sat', pos=pos, vel=vel, mass=mass)
    sim.track_boundedness()                         # whole-system "bound"
    if track_component:
        sim.sat.track_boundedness()                 # "bound_sat"
    kw = dict(t_end=0.02, dt=1e-3, dt_out=5e-3, method='direct', eps=0.1)
    kw.update(run_kw)
    sim.run(**kw)
    return sim, n, n_escapers


# --- recording ------------------------------------------------------------------------------ #

def test_records_shape_and_dtype():
    sim, n, _ = _run_single()
    rec = sim._records['bound']
    nsnaps = len(sim._times)
    assert rec.shape == (nsnaps, n)
    assert rec.dtype == np.bool_


def test_initial_snapshot_is_recorded():
    # index 0 (t0) is filled before the loop, not left as empty-buffer garbage:
    # it both reflects the clean initial split and matches a fresh recompute.
    sim, n, n_esc = _run_single()
    assert sim._records['bound'][0].sum() == n - n_esc
    assert np.array_equal(sim.bound(t=0), sim.bound(t=0, use_cached=False))


def test_escapers_are_unbound_and_blob_initially_bound():
    sim, n, n_esc = _run_single()
    # the cold blob is fully self-bound at t0; the boosted particles are not
    assert sim.bound(t=0)[n_esc:].all()
    assert (~sim.bound(t=0)[:n_esc]).all()
    # and the escapers stay unbound at every recorded snapshot
    assert (~sim._records['bound'][:, :n_esc]).all()


# --- access policy --------------------------------------------------------------------------- #

def test_recompute_matches_cached():
    sim, _, _ = _run_single()
    for t in (0, 1, -1):
        assert np.array_equal(sim.bound(t=t), sim.bound(t=t, use_cached=False))


def test_use_cached_without_record_raises():
    # A registered-but-unrecorded name can't be read from cache.
    sim, _, _ = _run_single()
    sim._records.pop('bound')                        # simulate "not recorded"
    with pytest.raises(ValueError):
        sim.bound(t=-1, use_cached=True)


def test_unregistered_diagnostic_raises():
    sim, _, _ = _run_single()
    with pytest.raises(KeyError):
        sim.record('does_not_exist', t=-1)


# --- accessors ------------------------------------------------------------------------------- #

def test_sim_and_component_accessors_present():
    sim, _, _ = _run_single(track_component=True)
    assert callable(sim.bound)                       # Sim auto-accessor
    assert callable(sim.sat.bound)                   # Component auto-accessor


def test_single_component_matches_whole_system():
    # For one component, monotonic component boundedness == whole-system boundedness.
    sim, _, _ = _run_single(track_component=True)
    assert np.array_equal(sim.bound(t=-1), sim.sat.bound(t=-1))


# --- multiple components --------------------------------------------------------------------- #

def test_multi_component_independent_records():
    pos_a, vel_a = _blob(40, scale=0.3, vsigma=2.0, seed=1)
    pos_b, vel_b = _blob(25, scale=0.3, vsigma=2.0, seed=2)
    pos_b += 50.0                                     # well-separated second blob
    sim = Sim()
    sim.add_particles('a', pos=pos_a, vel=vel_a, mass=np.full(40, 5e6))
    sim.add_particles('b', pos=pos_b, vel=vel_b, mass=np.full(25, 5e6))
    sim.a.track_boundedness()
    sim.b.track_boundedness()
    sim.run(t_end=0.01, dt=1e-3, dt_out=5e-3, method='direct', eps=0.1)

    nsnaps = len(sim._times)
    assert sim._records['bound_a'].shape == (nsnaps, 40)
    assert sim._records['bound_b'].shape == (nsnaps, 25)
    # 'a' uses its own self-gravity (source='self'), unaffected by the distant 'b'.
    assert sim.a.bound(t=-1).all()


# --- registration guards --------------------------------------------------------------------- #

def test_duplicate_operation_name_raises():
    pos, vel = _blob(10, 0.3, 2.0, 0)
    sim = Sim()
    sim.add_particles('sat', pos=pos, vel=vel, mass=np.full(10, 5e6))
    sim.track_boundedness()                          # registers "bound"
    with pytest.raises(ValueError):
        sim.track_boundedness()                      # duplicate "bound"


def test_add_operation_after_run_raises():
    sim, _, _ = _run_single()
    with pytest.raises(RuntimeError):
        sim.track_boundedness(component=None)
