'''
Test the Sim class and its methods.
'''

import pytest 
from ezfalcon.simulation import Sim, Component
import numpy as np
from ezfalcon.util import G_INTERNAL
import astropy.units as u


### Setup ###

np.random.seed(42)

COMP1_NPTS = 50
COMP1_POS = np.random.rand(COMP1_NPTS, 3)
COMP1_VEL = np.random.rand(COMP1_NPTS, 3)
COMP1_MASS = np.random.rand(COMP1_NPTS)

COMP2_NPTS = 30
COMP2_POS = np.random.rand(COMP2_NPTS, 3)
COMP2_VEL = np.random.rand(COMP2_NPTS, 3)
COMP2_MASS = np.random.rand(COMP2_NPTS)


### .add_particles() tests ###

def test_add_particles_stores_correct_shape():
    sim = Sim()
    sim.add_particles('test', 
                      pos=COMP1_POS, 
                      vel=COMP1_VEL, 
                      mass=COMP1_MASS)
    assert sim._init_pos.shape == (COMP1_NPTS, 3)
    assert sim._init_vel.shape == (COMP1_NPTS, 3)
    assert sim._mass.shape == (COMP1_NPTS,)

def test_add_particles_multiple_components():
    '''
    Test that adding multiple components stores the
    correct shapes.
    '''
    sim = Sim()
    sim.add_particles('comp1',
                      pos=COMP1_POS, 
                      vel=COMP1_VEL, 
                      mass=COMP1_MASS)
    sim.add_particles('comp2',
                      pos=COMP2_POS, 
                      vel=COMP2_VEL, 
                      mass=COMP2_MASS)
    assert sim._init_pos.shape == (COMP1_NPTS + COMP2_NPTS, 3)
    assert sim._init_vel.shape == (COMP1_NPTS + COMP2_NPTS, 3)
    assert sim._mass.shape == (COMP1_NPTS + COMP2_NPTS,)

def test_add_component_with_same_name():
    '''
    Test that adding a component with the same 
    name raises an error.
    '''
    sim = Sim()
    sim.add_particles('comp1',
                      pos=COMP1_POS, 
                      vel=COMP1_VEL, 
                      mass=COMP1_MASS)
    with pytest.raises(ValueError, match="Component 'comp1' already exists."):
        sim.add_particles('comp1',
                        pos=COMP2_POS, 
                        vel=COMP2_VEL, 
                        mass=COMP2_MASS)
    
def test_add_component_with_non_string_name():
    '''
    Test that adding a component with a non-string
    name raises an error.
    '''
    sim = Sim()
    with pytest.raises(TypeError, match="name must be a string."):
        sim.add_particles(123,
                        pos=COMP1_POS, 
                        vel=COMP1_VEL, 
                        mass=COMP1_MASS)

def test_add_component_after_run():
    '''
    Test that adding a component after run() raises an error.
    '''
    sim = Sim()
    sim.add_particles('comp1',
                      pos=COMP1_POS, 
                      vel=COMP1_VEL, 
                      mass=COMP1_MASS)
    sim.run(t_end=1., dt=0.5, dt_out=0.5, eps=0.1)
    with pytest.raises(RuntimeError, match="Cannot add components after run()"):
        sim.add_particles('comp2',
                        pos=COMP2_POS, 
                        vel=COMP2_VEL, 
                        mass=COMP2_MASS)

def test_add_particles_invalid_pos_shapes():
    '''
    Test that adding particles with invalid 
    position shapes raises an error.
    '''
    sim = Sim()
    with pytest.raises(ValueError, match="pos must be shape \(N, 3\), received \(50, 2\)"):
        sim.add_particles('comp1',
                          pos=np.random.rand(COMP1_NPTS, 2),
                          vel=COMP1_VEL,
                          mass=COMP1_MASS)
    with pytest.raises(ValueError, match="pos must be shape \(N, 3\), received \(50, 4\)"):
        sim.add_particles('comp1',
                          pos=np.random.rand(COMP1_NPTS, 4),
                          vel=COMP1_VEL,
                          mass=COMP1_MASS)

def test_add_particles_invalid_vel_shapes():
    '''
    Test that adding particles with invalid
    velocity shapes raises an error.
    '''
    sim = Sim()
    with pytest.raises(ValueError, match="vel must be shape \(N, 3\), received \(50, 2\)"):
        sim.add_particles('comp1',
                          pos=COMP1_POS,
                          vel=np.random.rand(COMP1_NPTS, 2),
                          mass=COMP1_MASS)
    with pytest.raises(ValueError, match="vel must be shape \(N, 3\), received \(50, 4\)"):
        sim.add_particles('comp1',
                          pos=COMP1_POS,
                          vel=np.random.rand(COMP1_NPTS, 4),
                          mass=COMP1_MASS)

def test_add_particles_invalid_mass_shapes():
    '''
    Test that adding particles with invalid
    mass shapes raises an error.
    '''
    sim = Sim()
    with pytest.raises(ValueError, match="mass must be shape \(N,\), received \(50, 2\)"):
        sim.add_particles('comp1',
                          pos=COMP1_POS,
                          vel=COMP1_VEL,
                          mass=np.random.rand(COMP1_NPTS, 2))
    with pytest.raises(ValueError, match="mass must be shape \(N,\), received \(50, 4\)"):
        sim.add_particles('comp1',
                          pos=COMP1_POS,
                          vel=COMP1_VEL,
                          mass=np.random.rand(COMP1_NPTS, 4))

def test_add_particles_different_number_of_particles():
    '''
    Test that adding particles with different number
    of particles in pos, vel, and mass raises an error.
    '''
    sim = Sim()
    with pytest.raises(ValueError, match="pos, vel, mass must have same number of particles, received 50, 30, 50."):
        sim.add_particles('comp1',
                          pos=COMP1_POS,
                          vel=COMP2_VEL,
                          mass=COMP1_MASS)
        






### Multi-component slicing tests ###

singlecomp = Sim()
singlecomp.add_particles('comp1',
                    pos=COMP1_POS, 
                    vel=COMP1_VEL, 
                    mass=COMP1_MASS)

                
multicomp = Sim()
multicomp.add_particles('comp1',
                    pos=COMP1_POS, 
                    vel=COMP1_VEL, 
                    mass=COMP1_MASS)
multicomp.add_particles('comp2',
                    pos=COMP2_POS, 
                    vel=COMP2_VEL, 
                    mass=COMP2_MASS)

def test_component_slices_are_contiguous():
    '''
    Test that the component slices are contiguous and
    non-overlapping.
    '''
    comp1_slice = multicomp._slices['comp1']
    comp2_slice = multicomp._slices['comp2']
    assert comp1_slice.stop == comp2_slice.start
    assert comp1_slice == slice(0, COMP1_NPTS)
    assert comp2_slice == slice(COMP1_NPTS, COMP1_NPTS + COMP2_NPTS)

def test_component_attribute_access():
    '''
    Test that component attribute access 
    returns the correct slices of the data.
    '''
    comp1 = multicomp.comp1
    assert isinstance(comp1, Component)
    assert np.all(comp1.pos(t=0) == multicomp._init_pos[multicomp._slices['comp1']])
    assert np.all(comp1.vel(t=0) == multicomp._init_vel[multicomp._slices['comp1']])
    assert np.all(comp1.mass == multicomp._mass[multicomp._slices['comp1']])

def test_non_existent_component_access():
    '''
    Test that accessing a non-existent component raises an error.
    '''
    with pytest.raises(AttributeError, match="\'Sim\' has no attribute or component named 'comp3'"):
        _ = multicomp.comp3




### test accessors pre-run ###

def test_accessors_are_correct_initially():
    '''
    Test that the pos, vel, mass, x, y, z, vx, vy, vz
    match the input values.
    '''
    np.testing.assert_array_equal(singlecomp.pos(0), COMP1_POS)
    np.testing.assert_array_equal(singlecomp.vel(0), COMP1_VEL)
    np.testing.assert_array_equal(singlecomp.mass, COMP1_MASS)
    np.testing.assert_array_equal(singlecomp.x(0), COMP1_POS[:, 0])
    np.testing.assert_array_equal(singlecomp.y(0), COMP1_POS[:, 1])
    np.testing.assert_array_equal(singlecomp.z(0), COMP1_POS[:, 2])
    np.testing.assert_array_equal(singlecomp.vx(0), COMP1_VEL[:, 0])
    np.testing.assert_array_equal(singlecomp.vy(0), COMP1_VEL[:, 1])
    np.testing.assert_array_equal(singlecomp.vz(0), COMP1_VEL[:, 2])

def test_accessors_match_init_values():
    '''
    Test that the pos, vel, mass, x, y, z, vx, vy, vz
    match the ._init_ values.
    '''
    np.testing.assert_array_equal(multicomp.pos(0), multicomp._init_pos)
    np.testing.assert_array_equal(multicomp.vel(0), multicomp._init_vel)
    np.testing.assert_array_equal(multicomp.mass, multicomp._mass)
    np.testing.assert_array_equal(multicomp.x(0), multicomp._init_pos[:, 0])
    np.testing.assert_array_equal(multicomp.y(0), multicomp._init_pos[:, 1])
    np.testing.assert_array_equal(multicomp.z(0), multicomp._init_pos[:, 2])
    np.testing.assert_array_equal(multicomp.vx(0), multicomp._init_vel[:, 0])
    np.testing.assert_array_equal(multicomp.vy(0), multicomp._init_vel[:, 1])
    np.testing.assert_array_equal(multicomp.vz(0), multicomp._init_vel[:, 2])






##### POST-RUN, NO EXTERNAL POTENTIAL TESTS ######

multicomp.run(t_end=1., dt=0.1, dt_out=0.1, eps=0.1)
singlecomp.run(t_end=1., dt=0.1, dt_out=0.1, eps=0.1)

### ._ti time indexing tests ###

def test_ti_int_passthrough():
    '''
    Test that passing an integer to _ti returns the same integer.
    '''
    assert multicomp._ti(5) == 5

def test_ti_float_to_int():
    '''
    Test that passing a float returns that index of the nearest
    snapshot.
    '''
    assert multicomp._ti(0.2) == 2
    assert multicomp._ti(0.25) == 2
    assert multicomp._ti(0.26) == 3




### Run output tests ###

def test_single_component_run_output_shapes():
    '''
    Test that the output arrays have the correct shapes
    for single component sims.
    '''
    assert singlecomp._positions.shape == (11, COMP1_NPTS, 3)
    assert singlecomp._velocities.shape == (11, COMP1_NPTS, 3)
    assert singlecomp._self_acc.shape == (11, COMP1_NPTS, 3)
    assert singlecomp._self_pot.shape == (11, COMP1_NPTS)
    assert singlecomp._times.shape == (11,)

def test_multicomponent_run_output_shapes():
    '''
    Test that the output arrays have the correct shapes
    for multi-component sims.
    '''
    assert multicomp._positions.shape == (11, COMP1_NPTS + COMP2_NPTS, 3)
    assert multicomp._velocities.shape == (11, COMP1_NPTS + COMP2_NPTS, 3)
    assert multicomp._self_acc.shape == (11, COMP1_NPTS + COMP2_NPTS, 3)
    assert multicomp._self_pot.shape == (11, COMP1_NPTS + COMP2_NPTS)
    assert multicomp._times.shape == (11,)

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



### Test energy methods ###

BINARY_TEST_POS, BINARY_TEST_VELS, BINARY_TEST_MASS = (
    np.array([[0, -1, 0], [0, 1, 0]]),
    np.array([[1, 0, 0], [0, 0, 2]]),
    np.array([1, 1])
)
BINARY_TEST_SIM = Sim()
BINARY_TEST_SIM.add_particles('comp1', 
                            pos=BINARY_TEST_POS, 
                            vel=BINARY_TEST_VELS, 
                            mass=BINARY_TEST_MASS)
BINARY_TEST_SIM.run(t_end=1, dt=0.5, dt_out=0.5, eps=0.01)

BINARY_TEST_KE = 0.5 * BINARY_TEST_MASS * np.linalg.norm(BINARY_TEST_VELS, axis=-1)**2

BINARY_TEST_SELF_PE = np.array([-G_INTERNAL * BINARY_TEST_MASS[0] * BINARY_TEST_MASS[1] / (
    np.linalg.norm(BINARY_TEST_POS[1] - BINARY_TEST_POS[0])), 0])

def test_KE_spot_check():
    '''
    Test that .KE returns the correct kinetic energy
    for a simple case.
    '''
    assert np.all(np.isclose(BINARY_TEST_SIM.KE(t=0), BINARY_TEST_KE, rtol=1e-10))

def test_self_PE_spot_check():
    '''
    Test that .PE returns the correct potential energy
    for a simple case.
    '''
    assert np.all(np.isclose(BINARY_TEST_SIM.self_PE(t=0), BINARY_TEST_SELF_PE, rtol=1e-10))

def test_total_energy_spot_check():
    '''
    Test that .energy returns the correct total energy
    for a simple case.
    '''
    assert np.all(np.isclose(BINARY_TEST_SIM.energy(t=0), BINARY_TEST_KE + BINARY_TEST_SELF_PE, rtol=1e-10))

def test_system_energy_spot_check():
    '''Test that .system_energy returns the correct total energy'''
    assert np.isclose(BINARY_TEST_SIM.system_energy(t=0), np.sum(BINARY_TEST_KE + BINARY_TEST_SELF_PE), rtol=1e-10)




### Test self-gravity toggle ###

def test_self_gravity_off_gives_zero_acc():
    sim = Sim()
    sim.add_particles('a', np.random.normal(size=(30, 3)) * 0.5, np.zeros((30, 3)), np.ones(30) * 1e4)
    sim.turn_self_gravity_off()
    sim.run(t_end=2, dt=1, dt_out=2, eps=0.1)
    np.testing.assert_array_equal(sim.self_gravity_acc(-1), 0)
    assert np.all(np.isclose(sim.compute_self_gravity(0.1, 0.6, 0), 0, atol=1e-10))

def test_self_gravity_on_gives_nonzero_acc():
    sim = Sim()
    sim.add_particles('a', np.random.normal(size=(30, 3)) * 0.5, np.zeros((30, 3)), np.ones(30) * 1e4)
    sim.turn_self_gravity_on()
    sim.run(t_end=2, dt=1, dt_out=2, eps=0.1)
    acc = sim.self_gravity_acc(-1)
    assert not np.all(np.isclose(acc, 0, atol=1e-10))
    acc_computed = sim.compute_self_gravity(0.1, 0.6, 0)
    assert np.all(np.isclose(acc, acc_computed, rtol=1e-10))





### Test self-gravity accessors ###

SELF_GRAVITY_TEST = np.array(-G_INTERNAL * BINARY_TEST_MASS[0] * BINARY_TEST_MASS[1] * (BINARY_TEST_POS[1] - BINARY_TEST_POS[0])/ (
    np.linalg.norm(BINARY_TEST_POS[1] - BINARY_TEST_POS[0]))**3)

def test_self_gravity_acc_spot_check():
    assert np.all(np.isclose(np.abs(BINARY_TEST_SIM.self_gravity_acc(t=0)), SELF_GRAVITY_TEST, rtol=1e-10))
def test_self_ax_spot_check():
    assert np.all(np.isclose(BINARY_TEST_SIM.self_ax(t=0), SELF_GRAVITY_TEST[0], rtol=1e-10))
def test_self_ay_spot_check():
    assert np.all(np.isclose(BINARY_TEST_SIM.self_ay(t=0), SELF_GRAVITY_TEST[1], rtol=1e-10))
def test_self_az_spot_check():
    assert np.all(np.isclose(BINARY_TEST_SIM.self_az(t=0), SELF_GRAVITY_TEST[2], rtol=1e-10))

### test .add_external_pot() ###
def test_add_external_pot_rejection():
    sim = Sim()
    with pytest.raises(TypeError, match="External potential must be a galpy Potential object."):
        sim.add_external_pot("not a potential", lambda pos, t: pos)


### test external acceleration accessors ###

def test_external_acc_zero_without_pot():
    sim = Sim()
    sim.add_particles('a', np.random.normal(size=(30, 3)) * 0.5, np.zeros((30, 3)), np.ones(30) * 1e4)
    sim.run(t_end=2, dt=1, dt_out=2, eps=0.1)
    np.testing.assert_array_equal(sim.external_acc(-1), 0)


### Test external acceleration accessors ###

KEPLER_SIM = Sim()
from galpy.potential import KeplerPotential
kepler_pot = KeplerPotential(amp=1.0*u.Msun)
KEPLER_SIM.add_external_pot('kepler', kepler_pot)
KEPLER_SIM.add_particles('a', pos=np.array([[0.1, 0.1, 0.1]]), vel=np.array([[0, 0.1, 0]]), mass=np.array([0.01]))
KEPLER_ACC = -G_INTERNAL * 1.0 * 0.01 * np.array([[0.1, 0.1, 0.1]]) / (0.1**2 + 0.1**2 + 0.1**2)**(3/2)
KEPLER_POT = -G_INTERNAL * 1.0 * 0.01 / np.sqrt(0.1**2 + 0.1**2 + 0.1**2)

def test_external_acc_with_pot():
    acc = KEPLER_SIM.external_acc(0)
    assert np.all(np.isclose(acc, KEPLER_ACC, rtol=1e-10))

def test_external_ax_with_pot():
    ax = KEPLER_SIM.external_ax(0)
    assert np.all(np.isclose(ax, KEPLER_ACC[:, 0], rtol=1e-10))

def test_external_ay_with_pot():
    ay = KEPLER_SIM.external_ay(0)
    assert np.all(np.isclose(ay, KEPLER_ACC[:, 1], rtol=1e-10))

def test_external_az_with_pot():
    az = KEPLER_SIM.external_az(0)
    assert np.all(np.isclose(az, KEPLER_ACC[:, 2], rtol=1e-10))

def test_external_pot_with_pot():
    pot = KEPLER_SIM.compute_external_pot(0)
    assert np.all(np.isclose(pot, KEPLER_POT, rtol=1e-10))


KEPLER_SIM.add_particles('b', pos=np.array([[0.2, 0.2, 0.2]]), vel=np.array([[0, 0.1, 0]]), mass=np.array([0.02]))
KEPLER_SIM.run(t_end=2, dt=1, dt_out=2, eps=0.1)

def test_total_PE():
    expected_PE_a = -G_INTERNAL * 1.0 * 0.01 / np.linalg.norm([0.1, 0.1, 0.1]) + -G_INTERNAL * 0.02 * 0.01 / np.linalg.norm([0.1, 0.1, 0.1])
    expected_PE_b = -G_INTERNAL * 1.0 * 0.02 / np.linalg.norm([0.2, 0.2, 0.2]) + -G_INTERNAL * 0.02 * 0.01 / np.linalg.norm([0.1, 0.1, 0.1])
    expected_PE = np.array([expected_PE_a, expected_PE_b])
    assert np.all(np.isclose(KEPLER_SIM.PE(t=0), expected_PE, rtol=1e-10))

def test_dE():
    E0 = KEPLER_SIM.system_energy(t=0)
    E1 = KEPLER_SIM.system_energy(t=1)
    dE = KEPLER_SIM.dE()
    assert np.all(np.isclose(E1 - E0, dE, rtol=1e-10))

