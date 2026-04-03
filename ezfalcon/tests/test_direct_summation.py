import pytest
import numpy as np
from ezfalcon.dynamics.acceleration.self_gravity import _direct_summation
from ezfalcon.util.units import G_INTERNAL

np.random.seed(42)

def test_newtons_third_law():
    '''
    Test that the direct summation method correctly computes that
    the force on each particle is equal and opposite. This ensures that the
    direct summation method is correctly implementing Newton's third law.
    ''' 
    pos = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    mass = np.array([1.0, 10.0])
    acc = _direct_summation(pos, mass, eps=0.0, return_potential=False)
    assert np.allclose(mass[0] * acc[0], -mass[1] * acc[1], rtol=1e-15)

def test_two_body_acceleration():
    '''
    Test that the direct summation method correctly computes the acceleration
    for a simple two-body system. Compare to the analytical solution for two 
    point masses.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    acc = _direct_summation(pos, mass, eps=0.0, return_potential=False)
    acc_analytical_0 = G_INTERNAL * mass[1] * (pos[1] - pos[0]) / np.linalg.norm(pos[1] - pos[0])**3
    acc_analytical_1 = G_INTERNAL * mass[0] * (pos[0] - pos[1]) / np.linalg.norm(pos[0] - pos[1])**3
    np.testing.assert_allclose(acc[0], acc_analytical_0, rtol=1e-15)
    np.testing.assert_allclose(acc[1], acc_analytical_1, rtol=1e-15)

def test_two_body_potential():
    '''
    Test that the direct summation method correctly computes the potential
    for a simple two-body system. Compare to the analytical solution for two 
    point masses.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    _, pot = _direct_summation(pos, mass, eps=0.0, return_potential=True)
    pot_analytical_0 = -G_INTERNAL * mass[1] / np.linalg.norm(pos[1] - pos[0])
    pot_analytical_1 = -G_INTERNAL * mass[0] / np.linalg.norm(pos[0] - pos[1])
    np.testing.assert_allclose(pot[0], pot_analytical_0, rtol=1e-15)
    np.testing.assert_allclose(pot[1], pot_analytical_1, rtol=1e-15)

def test_acceleration_with_softening_length():
    '''
    Test that the direct summation method correctly implements gravitational softening.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    eps = 2.0
    acc, _ = _direct_summation(pos, mass, eps=eps, return_potential=True)
    acc_analytical_0 = G_INTERNAL * mass[1] * (pos[1] - pos[0]) / (np.linalg.norm(pos[1] - pos[0])**2 + eps**2)**1.5
    acc_analytical_1 = G_INTERNAL * mass[0] * (pos[0] - pos[1]) / (np.linalg.norm(pos[0] - pos[1])**2 + eps**2)**1.5
    np.testing.assert_allclose(acc[0], acc_analytical_0, rtol=1e-15)
    np.testing.assert_allclose(acc[1], acc_analytical_1, rtol=1e-15)

def test_potential_with_softening_length():
    '''
    Test that the direct summation method correctly implements gravitational softening for the potential.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    eps = 2.0
    _, pot = _direct_summation(pos, mass, eps=eps, return_potential=True)
    pot_analytical_0 = -G_INTERNAL * mass[1] / np.sqrt(np.linalg.norm(pos[1] - pos[0])**2 + eps**2)
    pot_analytical_1 = -G_INTERNAL * mass[0] / np.sqrt(np.linalg.norm(pos[0] - pos[1])**2 + eps**2)
    np.testing.assert_allclose(pot[0], pot_analytical_0, rtol=1e-15)
    np.testing.assert_allclose(pot[1], pot_analytical_1, rtol=1e-15)

def test_zero_acceleration_for_single_particle():
    '''
    Test that the direct summation method correctly returns zero acceleration for a single particle.
    '''
    pos = np.random.normal(size=(1,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(1,))
    acc, pot = _direct_summation(pos, mass, eps=0.0, return_potential=True)
    np.testing.assert_allclose(acc[0], np.zeros(3), rtol=1e-15)
    np.testing.assert_allclose(pot[0], 0.0, rtol=1e-15)

def test_acc_and_pot_shapes():
    '''
    Test that the direct summation method returns acceleration and potential arrays of the correct shape.
    '''
    pos = np.random.normal(size=(3,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(3,))
    acc, pot = _direct_summation(pos, mass, eps=0.0, return_potential=True)
    assert acc.shape == (3, 3)
    assert pot.shape == (3,)


# --- return_potential flag -----------------------------------------------------------------------------

def test_return_potential_true_returns_tuple():
    '''
    Test that the return_potential flag correctly controls whether the potential is returned.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    out = _direct_summation(pos, mass, eps=0.0, return_potential=True)
    assert isinstance(out, tuple)

def test_return_potential_false_returns_acc_only():
    '''
    Test that the return_potential flag correctly controls whether the potential is returned.
    '''
    pos = np.random.normal(size=(2,3))
    mass = 10**np.random.normal(loc = 10, scale=1, size=(2,))
    out = _direct_summation(pos, mass, eps=0.0, return_potential=False)
    assert isinstance(out, np.ndarray)