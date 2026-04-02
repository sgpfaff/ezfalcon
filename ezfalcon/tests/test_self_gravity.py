'''
Test the self_gravity function
'''

import pytest
from ezfalcon.dynamics import self_gravity
import numpy as np


SELF_GRAVITY_METHODS = ['direct', 'falcON']

def test_raises_error_for_unknown_method():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mass = np.array([1.0, 1.0])
    with pytest.raises(ValueError):
        self_gravity(pos, mass, method='unknown_method')

def test_passes_additional_kwargs_to_falcON():
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    mass = np.array([1.0, 1.0])
    # Just test that it doesn't raise an error when we pass additional kwargs
    self_gravity(pos, mass, method='falcON', eps=0.1, theta=0.5)

def test_raises_error_for_no_eps_with_direct():
    '''
    Test that the direct summation method raises an error if eps is not provided.
    '''
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mass = np.array([1e8, 1e10])
    with pytest.raises(ValueError, match="Must provide 'eps' keyword argument for direct summation method."):
        self_gravity(pos, mass, method='direct')

def test_raises_error_for_no_theta_with_falcon():
    '''
    Test that error is raised if theta is not provided for falcON.
    '''
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mass = np.array([1e8, 1e10])
    with pytest.raises(ValueError, match="Must provide 'theta' keyword argument for falcON method."):
        self_gravity(pos, mass, method='falcON', eps=0.1)

def test_raises_error_for_no_eps_with_falcon():
    '''
    Test that error is raised if eps is not provided for falcON.
    '''
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mass = np.array([1e8, 1e10])
    with pytest.raises(ValueError, match="Must provide 'eps' keyword argument for falcON method."):
        self_gravity(pos, mass, method='falcON', theta=0.5)

def test_raises_error_for_no_eps_and_theta_with_falcon():
    '''
    Test that error is raised if neither eps nor theta is provided for falcON.
    '''
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    mass = np.array([1e8, 1e10])
    with pytest.raises(ValueError, match="Must provide 'eps' and 'theta' keyword arguments for falcON method."):
        self_gravity(pos, mass, method='falcON')