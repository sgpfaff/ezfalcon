#-----------------------#
#    pyfalcon tests     #
#-----------------------#

import pytest
from ezfalcon.dynamics.acceleration.self_gravity import _falcON_gravity, _direct_summation
from ezfalcon.util.units import G_INTERNAL
import numpy as np
np.random.seed(3)

def test_newtons_third_law():
    '''
    Test that the falcON method correctly computes that
    the force on each particle is equal and opposite. This ensures that the
    falcON method is correctly implementing Newton's third law.
    ''' 
    pos = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    mass = np.array([1.0, 10.0])
    acc, _ = _falcON_gravity(pos, mass, eps=0.0, theta=0.1)
    assert np.allclose(mass[0] * acc[0], -mass[1] * acc[1], rtol=1e-15)

def test_two_body_acceleration():
    '''
    Test that the falcON method correctly computes the acceleration
    for a simple two-body system. Compare to the analytical solution for two 
    point masses.
    '''
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    mass = np.array([1e5, 1e10], dtype=np.float32)
    acc, _ = _falcON_gravity(pos, mass, eps=0.0, theta=0.0)
    acc_analytical_0 = G_INTERNAL * mass[1] * (pos[1] - pos[0]) / np.linalg.norm(pos[1] - pos[0])**3
    acc_analytical_1 = G_INTERNAL * mass[0] * (pos[0] - pos[1]) / np.linalg.norm(pos[0] - pos[1])**3
    np.testing.assert_allclose(acc[0], acc_analytical_0, rtol=1e-15)
    np.testing.assert_allclose(acc[1], acc_analytical_1, rtol=1e-15)

def test_acceleration_against_direct():
    '''
    Test that the falcON method correctly computes the acceleration
    against direct summation.
    '''
    pos = np.array(np.random.normal(size=(10, 3)), dtype=np.float32)
    mass = np.array(np.random.normal(loc=1e9, scale=1e8, size=(10,)), dtype=np.float32)
    acc, _ = _falcON_gravity(pos, mass, eps=0.0, theta=0.0)
    acc_direct, _ = _direct_summation(pos, mass, eps=0.0)
    np.testing.assert_allclose(acc, acc_direct, rtol=1e-5)

def test_potential_against_direct():
    '''
    Test that the falcON method correctly computes the potential
    against direct summation.
    '''
    pos = np.array(np.random.normal(size=(10, 3)),  dtype=np.float64)
    mass = np.array(np.random.normal(loc=1e9, scale=1e8, size=(10,)), dtype=np.float64)
    _, pot = _falcON_gravity(pos, mass, eps=0.0, theta=0.0)
    _, pot_direct = _direct_summation(pos, mass, eps=0.0)
    np.testing.assert_allclose(pot, pot_direct, rtol=1e-5)

def test_two_body_potential():
    '''
    Test that the falcON method correctly computes the potential
    for a simple two-body system. Compare to the analytical solution for two 
    point masses.
    '''
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    mass = np.array([1e8, 1e10], dtype=np.float32)
    _, pot = _falcON_gravity(pos, mass, eps=0.0, theta=0.0)
    pot_analytical_0 = -G_INTERNAL * mass[1] / np.linalg.norm(pos[1] - pos[0])
    pot_analytical_1 = -G_INTERNAL * mass[0] / np.linalg.norm(pos[0] - pos[1])
    np.testing.assert_allclose(pot[0], pot_analytical_0, rtol=1e-15)
    np.testing.assert_allclose(pot[1], pot_analytical_1, rtol=1e-15)

def test_zero_acceleration_for_single_particle():
    '''
    Test that the falcON method correctly returns zero acceleration for a single particle.
    '''
    pos = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    mass = np.array([1e8], dtype=np.float32)
    acc, pot = _falcON_gravity(pos, mass, eps=0.0, theta=0.1)
    np.testing.assert_allclose(acc[0], np.zeros(3), rtol=1e-15)
    np.testing.assert_allclose(pot[0], 0.0, rtol=1e-15)