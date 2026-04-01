import pytest
import numpy as np
import astropy.units as u
from ezfalcon.simulation import Sim, Component

np.random.seed(42)

COMP1_NPTS = 50
COMP1_POS = np.random.rand(COMP1_NPTS, 3)
COMP1_VEL = np.random.rand(COMP1_NPTS, 3)
COMP1_MASS = np.random.rand(COMP1_NPTS)

COMP2_NPTS = 30
COMP2_POS = np.random.rand(COMP2_NPTS, 3)
COMP2_VEL = np.random.rand(COMP2_NPTS, 3)
COMP2_MASS = np.random.rand(COMP2_NPTS)

multicomp = Sim()
multicomp.add_particles('comp1',
                    pos=COMP1_POS, 
                    vel=COMP1_VEL, 
                    mass=COMP1_MASS)
multicomp.add_particles('comp2',
                    pos=COMP2_POS, 
                    vel=COMP2_VEL, 
                    mass=COMP2_MASS)

def test_component_attribute_access_pre_run():
    '''
    Test that component attribute access 
    returns the correct slices of the data.
    '''
    comp1 = multicomp.comp1
    assert isinstance(comp1, Component)
    assert np.all(comp1.pos(t=0) == multicomp._init_pos[multicomp._slices['comp1']])
    assert np.all(comp1.vel(t=0) == multicomp._init_vel[multicomp._slices['comp1']])
    assert np.all(comp1.mass == multicomp._mass[multicomp._slices['comp1']])

multicomp.run(t_end=1., dt=0.1, dt_out=0.1, eps=0.1)

def test_component_accessor_shapes_after_run():
    '''
    Test that the component accessors return arrays of the correct shape after run.
    '''
    comp1 = multicomp.comp1
    comp2 = multicomp.comp2
    assert comp1.pos().shape == (11, COMP1_NPTS, 3)
    assert comp1.vel().shape == (11, COMP1_NPTS, 3)
    assert comp1.mass.shape == (COMP1_NPTS,)
    assert comp2.pos().shape == (11, COMP2_NPTS, 3)
    assert comp2.vel().shape == (11, COMP2_NPTS, 3)
    assert comp2.mass.shape == (COMP2_NPTS,)