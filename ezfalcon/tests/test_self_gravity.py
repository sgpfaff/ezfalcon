

#-----------------------#
#    pyfalcon tests     #
#-----------------------#

import pytest
from ezfalcon.dynamics.acceleration import self_gravity
import numpy as np

binaries = [
    # Two equal-mass particles in circular orbit
    (np.array([[0, 0, 0], [1, 0, 0]]),  np.array([1, 1]), 0.01, 0.6),
    # Two equal-mass particles in elliptical orbit
    (np.array([[0, 0, 0], [1, 1, 0]]), np.array([1, 1]), 0.01, 0.6),
    # Two unequal-mass particles
    (np.array([[0, 0, 0], [1, 0, 0]]), np.array([1, 10]), 0.01, 0.6),
]

@pytest.mark.parametrize("pos, mass, eps, theta", binaries)
def test_mutual_acceleration(pos, mass, eps, theta):
    '''
    Test that the self_gravity function correctly computes the 
    same gravitational acceleration between two particles.
    '''
    print(pos.shape)
    acc, pot = self_gravity(pos, mass, eps, theta)
    # For two particles, the acceleration on each should be equal and opposite
    assert np.allclose(acc[0], -acc[1], rtol=1e-5)
    
    
    
    # The potential should be negative and larger in magnitude for the more massive particle
    # assert pot[0] < 0 and pot[1] < 0
    # if mass[0] != mass[1]:
    #     assert abs(pot[0]) < abs(pot[1])  # More massive particle should have deeper potential well
    

