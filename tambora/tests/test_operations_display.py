'''
Tests for the progress-bar display layer: the Display reducers (fraction_bound and
the stateful energy_drift), and that displays actually fire through the runner at the
expected cadence without disturbing results.
'''

import numpy as np
import pytest

from tambora.simulation import Sim
from tambora.dynamics.operations import Display, DisplayOnly, fraction_bound, energy_drift


class _FakeCtx:
    def __init__(self, bound=None, mass=None, vel=None, t=0.0, self_pot=None):
        self._bound = None if bound is None else np.asarray(bound, bool)
        self.mass = None if mass is None else np.asarray(mass, float)
        self.vel = None if vel is None else np.asarray(vel, float)
        self.t = t
        self._self_pot = self_pot

    def get(self, name):
        if name == "self_pot":
            return self._self_pot
        return self._bound


class _NamedOp:
    name = "bound"


# --- reducers ------------------------------------------------------------------------------- #

def test_fraction_bound_reducer():
    d = fraction_bound()
    label, text = d.text(_FakeCtx(bound=[True, True, False, False]), _NamedOp())
    assert label == "f_bound"
    assert text == "50%"


def test_display_format_is_applied():
    d = Display("x", lambda ctx, op, st: 0.5, fmt="{:.3f}")
    assert d.text(_FakeCtx(), None) == ("x", "0.500")


def test_energy_drift_is_zero_at_first_sample_then_tracks():
    d = energy_drift()
    mass = np.ones(2)
    # first sample establishes E0
    _, first = d.text(_FakeCtx(mass=mass, vel=np.array([[1., 0, 0], [1., 0, 0]]),
                               self_pot=np.zeros(2)), None)
    assert float(first) == 0.0
    # same state -> still zero drift
    _, same = d.text(_FakeCtx(mass=mass, vel=np.array([[1., 0, 0], [1., 0, 0]]),
                              self_pot=np.zeros(2)), None)
    assert float(same) == 0.0
    # double the speed -> KE x4 -> |dE/E0| = 3
    _, drift = d.text(_FakeCtx(mass=mass, vel=np.array([[2., 0, 0], [2., 0, 0]]),
                               self_pot=np.zeros(2)), None)
    assert float(drift) == pytest.approx(3.0)


def test_display_only_carries_display_and_records_nothing():
    d = energy_drift()
    op = DisplayOnly(d)
    assert op.display is d
    assert op.name is None          # not registered as a recorded diagnostic
    assert op.finalize() is None


# --- runner integration --------------------------------------------------------------------- #

def _mini_sim():
    rng = np.random.default_rng(0)
    pos = rng.normal(scale=0.3, size=(20, 3))
    vel = rng.normal(scale=0.3, size=(20, 3))
    sim = Sim()
    sim.add_particles('sat', pos=pos, vel=vel, mass=np.full(20, 2e7))
    return sim


def test_display_fires_during_run():
    fired = []
    probe = Display("probe", lambda ctx, op, st: fired.append(ctx.t) or 1.0)
    sim = _mini_sim()
    sim.show(probe)
    sim.run(t_end=0.02, dt=1e-3, dt_out=5e-3, method='direct', eps=0.1, show_energy=False)
    assert len(fired) >= 1                       # fired at the output cadence


def test_display_every_controls_cadence():
    coarse, fine = [], []
    sim = _mini_sim()
    sim.show(Display("coarse", lambda ctx, op, st: coarse.append(ctx.t) or 1.0))          # every=None
    sim.show(Display("fine", lambda ctx, op, st: fine.append(ctx.t) or 1.0, every=1))     # every step
    sim.run(t_end=0.02, dt=1e-3, dt_out=5e-3, method='direct', eps=0.1, show_energy=False)
    assert len(fine) > len(coarse)


def test_show_energy_default_does_not_disturb_results():
    sim_on = _mini_sim()
    sim_on.run(t_end=0.01, dt=1e-3, dt_out=5e-3, method='direct', eps=0.1)               # default on
    sim_off = _mini_sim()
    sim_off.run(t_end=0.01, dt=1e-3, dt_out=5e-3, method='direct', eps=0.1, show_energy=False)
    np.testing.assert_array_equal(sim_on.pos(t=-1, return_internal=True),
                                  sim_off.pos(t=-1, return_internal=True))


def test_track_boundedness_display_true_attaches_fraction_bound():
    sim = _mini_sim()
    op = sim.sat.track_boundedness(display=True)
    assert op.display is not None
    assert op.display.label == "f_bound(sat)"
