'''
Tests for the StrippingTracker accumulator: edge detection (strip / recapture),
first/last stripping time, phase-space capture, and confirm-based debounce -- driven
deterministically through a synthetic context -- plus an end-to-end Sim integration.
'''

import numpy as np
import pytest

from tambora.simulation import Sim
from tambora.dynamics.operations import StrippingTracker
from tambora.tools.util.units import KMS_TO_KPCGYR


class _FakeCtx:
    """Minimal StepContext stand-in: get(dep) returns a preset bound mask."""
    def __init__(self, bound, pos, vel, t):
        self._bound = np.asarray(bound, dtype=bool)
        self.pos, self.vel, self.t = np.asarray(pos, float), np.asarray(vel, float), t

    def get(self, name):
        return self._bound


def _drive(tracker, n, frames):
    """Init a whole-system tracker over n particles and feed (bound, t) frames."""
    tracker.init(n)
    for bound, t in frames:
        pos = np.tile(np.array([float(t), 0.0, 0.0]), (n, 1))   # x encodes the time
        vel = np.tile(np.array([0.0, float(t), 0.0]), (n, 1))   # vy encodes the time
        tracker.update(_FakeCtx(bound, pos, vel, t))


# --- edge detection ------------------------------------------------------------------------- #

def test_strip_recapture_counts_and_times():
    # p0 always bound; p1 strips at t=2; p2 strips at t=2, recaptured t=3, strips again t=4.
    trk = StrippingTracker(sim=None, component=None, cadence=1, confirm=1)
    _drive(trk, 3, [
        ([True,  True,  True],  1),   # baseline
        ([True,  False, False], 2),
        ([True,  False, True],  3),
        ([True,  False, False], 4),
    ])
    assert list(trk.n_strip) == [0, 1, 2]
    assert list(trk.n_recapture) == [0, 0, 1]
    np.testing.assert_array_equal(trk.tstrip_first, [np.nan, 2, 2])
    np.testing.assert_array_equal(trk.tstrip_last, [np.nan, 2, 4])


def test_phase_space_captured_at_first_strip():
    trk = StrippingTracker(sim=None, component=None, cadence=1, confirm=1)
    _drive(trk, 2, [
        ([True, True],  1),
        ([True, False], 2),           # p1 strips at t=2
        ([True, False], 3),
    ])
    # x encodes t, vy encodes t (in km/s after the finalize conversion)
    assert trk.strip_pos[1, 0] == 2.0
    np.testing.assert_allclose(trk.strip_vel[1, 1], 2.0)        # internal kpc/Gyr
    out = trk.finalize()
    np.testing.assert_allclose(out["strip_vel"][1, 1], 2.0 / KMS_TO_KPCGYR)  # -> km/s


# --- debounce ------------------------------------------------------------------------------- #

def test_confirm_suppresses_transient_flip():
    trk = StrippingTracker(sim=None, component=None, cadence=1, confirm=2)
    _drive(trk, 1, [
        ([True],  1),                 # baseline
        ([False], 2),                 # 1 check unbound...
        ([True],  3),                 # ...flips back before confirm=2 -> not counted
    ])
    assert trk.n_strip[0] == 0
    assert np.isnan(trk.tstrip_first[0])


def test_confirm_records_crossing_time_not_confirmation():
    trk = StrippingTracker(sim=None, component=None, cadence=1, confirm=2)
    _drive(trk, 1, [
        ([True],  1),                 # baseline
        ([False], 2),                 # crossing begins at t=2
        ([False], 3),                 # confirmed at t=3
    ])
    assert trk.n_strip[0] == 1
    assert trk.tstrip_first[0] == 2.0   # crossing time, not the confirmation instant


def test_particle_unbound_from_baseline_is_not_a_strip():
    trk = StrippingTracker(sim=None, component=None, cadence=1, confirm=1)
    _drive(trk, 1, [
        ([False], 1),                 # baseline already unbound
        ([False], 2),
    ])
    assert trk.n_strip[0] == 0


# --- finalize naming ------------------------------------------------------------------------ #

def test_finalize_field_names_unsuffixed_for_whole_system():
    trk = StrippingTracker(sim=None, component=None)
    trk.init(2)
    assert set(trk.finalize()) == {
        "n_strip", "n_recapture", "tstrip_first", "tstrip_last", "strip_pos", "strip_vel"}


# --- integration ---------------------------------------------------------------------------- #

def _blob(n, scale, vsigma, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=scale, size=(n, 3)), rng.normal(scale=vsigma, size=(n, 3))


def test_track_stripping_end_to_end():
    pos, vel = _blob(60, 0.3, 0.3, 0)
    vel[:5] += 400.0
    sim = Sim()
    sim.add_particles('sat', pos=pos, vel=vel, mass=np.full(60, 2e7))
    sim.sat.track_stripping(cadence=1, confirm=1)
    sim.run(t_end=0.02, dt=1e-3, dt_out=5e-3, method='direct', eps=0.1)

    # summaries populated with suffixed per-particle fields, sized to the component
    for field in ("n_strip", "n_recapture", "tstrip_first", "tstrip_last"):
        assert sim._summaries[f"{field}_sat"].shape == (60,)
    assert sim._summaries["strip_pos_sat"].shape == (60, 3)

    # accessors on the component proxy
    np.testing.assert_array_equal(sim.sat.n_strip, sim._summaries["n_strip_sat"])
    assert sim.sat.tstrip_last.shape == (60,)
    # counts are non-negative ints; recaptures cannot exceed strips per particle
    assert (sim.sat.n_strip >= 0).all()
    assert (sim.sat.n_recapture <= sim.sat.n_strip).all()


def test_track_stripping_autoregisters_boundedness():
    pos, vel = _blob(20, 0.3, 0.3, 1)
    sim = Sim()
    sim.add_particles('sat', pos=pos, vel=vel, mass=np.full(20, 2e7))
    sim.sat.track_stripping()
    # the dependency was registered automatically
    assert "bound_sat" in sim._op_by_name
    assert "stripping_sat" in sim._op_by_name
