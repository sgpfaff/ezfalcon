'''
Storage-unification tests: self-gravity acceleration/potential now flow through the
same record store as user diagnostics, are reachable via Sim.record(), and remain
identical to the dedicated cached accessors.
'''

import numpy as np
import pytest

from tambora.simulation import Sim


def _sg_sim(**run_kw):
    rng = np.random.default_rng(0)
    pos = rng.normal(scale=0.3, size=(30, 3))
    vel = rng.normal(scale=0.3, size=(30, 3))
    sim = Sim()
    sim.add_particles('sat', pos=pos, vel=vel, mass=np.full(30, 2e7))
    kw = dict(t_end=0.02, dt=1e-3, dt_out=5e-3, method='direct', eps=0.1, show_energy=False)
    kw.update(run_kw)
    sim.run(**kw)
    return sim


def test_self_gravity_in_unified_records():
    sim = _sg_sim()
    nsnaps = len(sim._times)
    assert sim._records["self_acc"].shape == (nsnaps, 30, 3)
    assert sim._records["self_pot"].shape == (nsnaps, 30)
    # same array the dedicated cache attribute points at -- one store of truth
    assert sim._records["self_acc"] is sim._cached_self_acc
    assert sim._records["self_pot"] is sim._cached_self_pot


def test_record_reads_self_gravity():
    sim = _sg_sim()
    np.testing.assert_array_equal(sim.record("self_pot", t=-1), sim._cached_self_pot[-1])
    np.testing.assert_array_equal(sim.record("self_acc", t=2), sim._cached_self_acc[2])


def test_record_self_gravity_recompute_unsupported():
    sim = _sg_sim()
    with pytest.raises(ValueError):
        sim.record("self_pot", t=-1, use_cached=False)


def test_record_unknown_name_raises():
    sim = _sg_sim()
    with pytest.raises(KeyError):
        sim.record("not_a_thing", t=-1)


def test_self_gravity_record_matches_dedicated_accessor():
    # The dedicated accessor returns mass-weighted energy; the raw record is the
    # per-unit-mass potential. They must agree up to the mass factor.
    sim = _sg_sim()
    raw = sim.record("self_pot", t=-1)
    via_accessor = sim.self_potential(t=-1, return_internal=True)   # mass * pot
    np.testing.assert_allclose(via_accessor, sim.mass * raw)


def test_records_coexist_with_user_diagnostics():
    rng = np.random.default_rng(1)
    sim = Sim()
    sim.add_particles('sat', pos=rng.normal(scale=0.3, size=(20, 3)),
                      vel=rng.normal(scale=0.3, size=(20, 3)), mass=np.full(20, 2e7))
    sim.sat.track_boundedness()
    sim.run(t_end=0.01, dt=1e-3, dt_out=5e-3, method='direct', eps=0.1, show_energy=False)
    # self-gravity and the user diagnostic share the same store
    assert {"self_acc", "self_pot", "bound_sat"} <= set(sim._records)


def test_caching_flag_off_omits_from_records():
    sim = _sg_sim(cache_self_gravity_pot=False)
    assert "self_pot" not in sim._records
    assert "self_acc" in sim._records          # acc still cached by default
    with pytest.raises(KeyError):
        sim.record("self_pot", t=-1)
