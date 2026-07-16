from tambora.dynamics.forces import Force, ExternalConservativeForce
from tambora.dynamics.forces.CompositeForce import _CompositePlain, _CompositeConservative
import numpy as np
import pytest

class ExampleExternalForce(Force):
    def __init__(self, multiplier):
        self.multiplier = multiplier
    def acc(self, pos, vel, mass, t):
        return 2 * pos * vel * self.multiplier + np.sum(mass) + t

class ExampleConservativeForce(ExternalConservativeForce):
    def __init__(self, multiplier):
        self.multiplier = multiplier
    def acc(self, pos, t):
        return 2 * pos * self.multiplier + t
    def potential(self, pos, t):
        return np.sum(pos**2) * self.multiplier + t

# --- Composite of ExternalForces ------------------------------------------------------ #

def test_sum_of_base_forces_makes_CompositePlain():
    baseForce1 = ExampleExternalForce(multiplier=1)
    baseForce2 = ExampleExternalForce(multiplier=2)
    composite = baseForce1 + baseForce2
    assert isinstance(composite, _CompositePlain)

def test_base_CompositeForce_requires_pos_vel_mass_t():
    baseForce1 = ExampleExternalForce(multiplier=1)
    baseForce2 = ExampleExternalForce(multiplier=2)
    composite = baseForce1 + baseForce2
    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([0.5, 0.5, 0.5])
    mass = np.array([1.0, 1.0, 1.0])
    t = 0.0
    # Should not raise an error
    composite.acc(pos, vel, mass, t)
    pytest.raises(TypeError, composite.acc, pos, mass, t)  # missing vel
    pytest.raises(TypeError, composite.acc, pos, vel, t)  # missing mass
    pytest.raises(TypeError, composite.acc, pos, vel, mass)  # missing t

def test_sum_of_base_forces_equals_composite():
    baseForce1 = ExampleExternalForce(multiplier=1)
    baseForce2 = ExampleExternalForce(multiplier=2)
    composite = baseForce1 + baseForce2
    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([0.5, 0.5, 0.5])
    mass = np.array([1.0, 1.0, 1.0])
    t = 0.0
    expected_acc = baseForce1.acc(pos, vel, mass, t) + baseForce2.acc(pos, vel, mass, t)
    assert np.allclose(composite.acc(pos, vel, mass, t), expected_acc)

def test_composite_of_base_forces_equals_analytic():
    baseForce1 = ExampleExternalForce(multiplier=1)
    baseForce2 = ExampleExternalForce(multiplier=2)
    composite = baseForce1 + baseForce2
    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([0.5, 0.5, 0.5])
    mass = np.array([1.0, 1.0, 1.0])
    t = 0.0
    expected_acc = 2 * pos * vel * (baseForce1.multiplier + baseForce2.multiplier) + 2*np.sum(mass) + 2*t
    assert np.allclose(composite.acc(pos, vel, mass, t), expected_acc)

# --- Composite of ConservativeForces ------------------------------------------------------ #

def test_sum_of_conservative_forces_makes_CompositeConservative():
    conservativeForce1 = ExampleConservativeForce(multiplier=1)
    conservativeForce2 = ExampleConservativeForce(multiplier=2)
    composite = conservativeForce1 + conservativeForce2
    assert isinstance(composite, _CompositeConservative)

def test_sum_of_conservative_forces_equals_composite():
    conservativeForce1 = ExampleConservativeForce(multiplier=1)
    conservativeForce2 = ExampleConservativeForce(multiplier=2)
    composite = conservativeForce1 + conservativeForce2
    pos = np.array([1.0, 2.0, 3.0])
    mass = np.array([1.0, 1.0, 1.0])
    t = 0.0
    expected_acc = conservativeForce1.acc(pos, t) + conservativeForce2.acc(pos, t)
    assert np.allclose(composite.acc(pos, mass, t), expected_acc)

def test_composite_of_conservative_forces_equals_analytic():
    conservativeForce1 = ExampleConservativeForce(multiplier=1)
    conservativeForce2 = ExampleConservativeForce(multiplier=2)
    composite = conservativeForce1 + conservativeForce2
    pos = np.array([1.0, 2.0, 3.0])
    mass = np.array([1.0, 1.0, 1.0])
    t = 0.0
    expected_acc = 2 * pos * (conservativeForce1.multiplier + conservativeForce2.multiplier) + 2*t
    assert np.allclose(composite.acc(pos, mass, t), expected_acc)

def test_composite_potential_equals_sum_of_potentials():
    conservativeForce1 = ExampleConservativeForce(multiplier=1)
    conservativeForce2 = ExampleConservativeForce(multiplier=2)
    composite = conservativeForce1 + conservativeForce2
    pos = np.array([1.0, 2.0, 3.0])
    mass = np.array([1.0, 1.0, 1.0])
    t = 0.0
    expected_potential = conservativeForce1.potential(pos, t) + conservativeForce2.potential(pos, t)
    assert np.allclose(composite.potential(pos, mass, t), expected_potential)

# --- Composite of ExternalForces and ConservativeForces ---------------------------------------------- #

def test_sum_of_mixed_forces_makes_CompositePlain():
    baseForce1 = ExampleExternalForce(multiplier=1)
    conservativeForce1 = ExampleConservativeForce(multiplier=1)
    composite = baseForce1 + conservativeForce1
    assert isinstance(composite, _CompositePlain)

def test_sum_of_mixed_forces_equals_composite():
    baseForce1 = ExampleExternalForce(multiplier=1)
    conservativeForce1 = ExampleConservativeForce(multiplier=1)
    composite = baseForce1 + conservativeForce1
    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([0.5, 0.5, 0.5])
    mass = np.array([1.0, 1.0, 1.0])
    t = 0.0
    expected_acc = baseForce1.acc(pos, vel, mass, t) + conservativeForce1.acc(pos, t)
    assert np.allclose(composite.acc(pos, vel, mass, t), expected_acc)

def test_composite_of_mixed_forces_equals_analytic():
    baseForce1 = ExampleExternalForce(multiplier=1)
    conservativeForce1 = ExampleConservativeForce(multiplier=1)
    composite = baseForce1 + conservativeForce1
    pos = np.array([1.0, 2.0, 3.0])
    vel = np.array([0.5, 0.5, 0.5])
    mass = np.array([1.0, 1.0, 1.0])
    t = 0.0
    expected_acc = 2 * pos * (baseForce1.multiplier * vel + conservativeForce1.multiplier) + np.sum(mass) + 2*t
    assert np.allclose(composite.acc(pos, vel, mass, t), expected_acc)


# --- the _eval_* / one-pass paths ------------------------------------------------------ #
#
# The tests above drive the public acc()/potential(). These cover the internal
# entry points the integrator and StepState use, which take the uniform
# (pos, vel, mass, t) signature regardless of what the concrete force wants:
#
#   _eval_acc            - dispatches to acc(), dropping vel for conservative forces
#   _eval_potential      - same, for potentials
#   acc_and_potential    - the one-pass path, so a solver sweeps once not twice

_POS = np.array([[1., 2., 3.], [4., 5., 6.]])
_VEL = np.array([[0.5, 0.5, 0.5], [1., 1., 1.]])
_MASS = np.array([10., 20.])
_T = 0.5


def test_eval_acc_folds_over_members_of_a_plain_composite():
    f1, f2 = ExampleExternalForce(multiplier=1), ExampleExternalForce(multiplier=2)
    composite = f1 + f2
    np.testing.assert_allclose(
        composite._eval_acc(_POS, _VEL, _MASS, _T),
        f1.acc(_POS, _VEL, _MASS, _T) + f2.acc(_POS, _VEL, _MASS, _T))


def test_eval_acc_folds_over_members_of_a_conservative_composite():
    # Conservative forces take acc(pos, t) -- no vel. _eval_acc must drop it.
    f1, f2 = ExampleConservativeForce(multiplier=1), ExampleConservativeForce(multiplier=2)
    composite = f1 + f2
    np.testing.assert_allclose(
        composite._eval_acc(_POS, _VEL, _MASS, _T),
        f1.acc(_POS, _T) + f2.acc(_POS, _T))


def test_eval_acc_agrees_with_the_public_acc():
    # The two entry points must not drift apart.
    f1, f2 = ExampleConservativeForce(multiplier=1), ExampleConservativeForce(multiplier=3)
    composite = f1 + f2
    np.testing.assert_allclose(composite._eval_acc(_POS, None, _MASS, _T),
                               composite.acc(_POS, _MASS, _T))


def test_eval_potential_folds_over_members():
    f1, f2 = ExampleConservativeForce(multiplier=1), ExampleConservativeForce(multiplier=2)
    composite = f1 + f2
    np.testing.assert_allclose(
        composite._eval_potential(_POS, _VEL, _MASS, _T),
        f1.potential(_POS, _T) + f2.potential(_POS, _T))


def test_eval_potential_agrees_with_the_public_potential():
    f1, f2 = ExampleConservativeForce(multiplier=1), ExampleConservativeForce(multiplier=3)
    composite = f1 + f2
    np.testing.assert_allclose(composite._eval_potential(_POS, None, _MASS, _T),
                               composite.potential(_POS, _MASS, _T))


def test_acc_and_potential_matches_computing_each_separately():
    # The one-pass path exists to avoid sweeping twice; it must agree with the
    # two-sweep answer or the optimisation is a bug.
    f1, f2 = ExampleConservativeForce(multiplier=1), ExampleConservativeForce(multiplier=2)
    composite = f1 + f2
    a, p = composite.acc_and_potential(_POS, _MASS, _T)
    np.testing.assert_allclose(a, composite.acc(_POS, _MASS, _T))
    np.testing.assert_allclose(p, composite.potential(_POS, _MASS, _T))


def test_acc_and_potential_folds_over_three_members():
    # More than two, so a fold that silently used only the first two would fail.
    fs = [ExampleConservativeForce(multiplier=m) for m in (1, 2, 5)]
    composite = fs[0] + fs[1] + fs[2]
    a, p = composite.acc_and_potential(_POS, _MASS, _T)
    np.testing.assert_allclose(a, sum(f.acc(_POS, _T) for f in fs))
    np.testing.assert_allclose(p, sum(f.potential(_POS, _T) for f in fs))


def test_a_plain_composite_has_no_potential_paths():
    # CompositeForce's documented promise: mixing in a non-conservative force
    # makes .potential() a clear AttributeError, not a silent half-answer.
    composite = ExampleConservativeForce(multiplier=1) + ExampleExternalForce(multiplier=2)
    assert isinstance(composite, _CompositePlain)
    for name in ('potential', 'acc_and_potential', '_eval_potential'):
        assert not hasattr(composite, name), f"_CompositePlain should not expose {name}"


# --- empty-composite guards ----------------------------------------------------------- #
#
# Every method opens with `if not self.members`. Sim seeds its force slots with
# empty composites (_CompositeConservative([]) / _CompositePlain([])), so these
# run on every force evaluation of every hookless, host-free simulation.

def test_an_empty_conservative_composite_is_a_zero_force():
    empty = _CompositeConservative([])
    np.testing.assert_array_equal(empty.acc(_POS, _MASS, _T), np.zeros_like(_POS))
    np.testing.assert_array_equal(empty.potential(_POS, _MASS, _T), np.zeros(len(_POS)))
    np.testing.assert_array_equal(empty._eval_potential(_POS, _VEL, _MASS, _T), np.zeros(len(_POS)))


def test_an_empty_conservative_composite_one_pass_is_zero():
    a, p = _CompositeConservative([]).acc_and_potential(_POS, _MASS, _T)
    np.testing.assert_array_equal(a, np.zeros_like(_POS))
    np.testing.assert_array_equal(p, np.zeros(len(_POS)))


def test_an_empty_plain_composite_is_a_zero_force():
    np.testing.assert_array_equal(
        _CompositePlain([]).acc(_POS, _VEL, _MASS, _T), np.zeros_like(_POS))


def test_empty_composite_zeros_have_the_shape_of_pos_not_a_scalar():
    # This is why Sim can seed empty composites and still report a proper
    # (N, 3) external_acc for a sim with no external force.
    assert _CompositeConservative([]).acc(_POS, _MASS, _T).shape == _POS.shape
    assert _CompositePlain([]).acc(_POS, _VEL, _MASS, _T).shape == _POS.shape
