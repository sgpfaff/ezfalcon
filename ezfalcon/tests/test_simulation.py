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

def test_ti_positive_int_out_of_bounds():
    '''
    Test that passing a positive integer out of bounds raises an error.
    '''
    with pytest.raises(IndexError, match="Time index 100 is out of bounds for simulation with 11 snapshots. Please provide an index within \[-11, 10\]."):
        multicomp._ti(100)

def test_ti_negative_int_out_of_bounds():
    '''
    Test that passing a negative integer out of bounds raises an error.
    '''
    with pytest.raises(IndexError, match="Time index -100 is out of bounds for simulation with 11 snapshots. Please provide an index within \[-11, 10\]."):
        multicomp._ti(-100)

def test_ti_float_out_of_bounds():
    '''
    Test that passing a float out of bounds raises an error.
    '''
    with pytest.raises(ValueError, match="t=100.0 Myr is out of bounds for simulation time range \[0.0, 1.0\] Myr."):
        multicomp._ti(100.0)

def test_ti_fails_with_list():
    with pytest.raises(TypeError, match="t must be an int index, a float time, or ellipsis."):
        multicomp._ti([0, 5], vectorized=False)

def test_ti_vectorized_false_fails_with_ellipse():
    '''
    Test that passing a list when vectorized=False raises an error.
    '''
    with pytest.raises(TypeError, match="This method is not vectorized, so t cannot be a list or ellipse. Please provide an integer index or a float time."):
        multicomp._ti(..., vectorized=False)

### Run output tests ###

def test_single_component_run_output_shapes():
    '''
    Test that the output arrays have the correct shapes
    for single component sims.
    '''
    assert singlecomp._positions.shape == (11, COMP1_NPTS, 3)
    assert singlecomp._velocities.shape == (11, COMP1_NPTS, 3)
    assert singlecomp._times.shape == (11,)

def test_multicomponent_run_output_shapes():
    '''
    Test that the output arrays have the correct shapes
    for multi-component sims.
    '''
    assert multicomp._positions.shape == (11, COMP1_NPTS + COMP2_NPTS, 3)
    assert multicomp._velocities.shape == (11, COMP1_NPTS + COMP2_NPTS, 3)
    assert multicomp._times.shape == (11,)


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
    assert np.all(np.isclose(BINARY_TEST_SIM.compute_self_potential(t=0, eps=0.01, theta=0.3), BINARY_TEST_SELF_PE, rtol=1e-10))

def test_total_energy_spot_check():
    '''
    Test that .energy returns the correct total energy
    for a simple case.
    '''
    assert np.all(np.isclose(BINARY_TEST_SIM.energy(t=0, eps=0.01, theta=0.3), BINARY_TEST_KE + BINARY_TEST_SELF_PE, rtol=1e-10))

def test_system_energy_spot_check():
    '''Test that .system_energy returns the correct total energy'''
    assert np.isclose(BINARY_TEST_SIM.system_energy(t=0, eps=0.01, theta=0.3), np.sum(BINARY_TEST_KE + BINARY_TEST_SELF_PE), rtol=1e-10)


### Test self-gravity toggle ###

def test_self_gravity_off_gives_zero_acc():
    sim = Sim()
    sim.add_particles('a', np.random.normal(size=(30, 3)) * 0.5, np.zeros((30, 3)), np.ones(30) * 1e4)
    sim.turn_self_gravity_off()
    sim.run(t_end=2, dt=1, dt_out=2, eps=0.1)
    acc = [sim.compute_self_gravity(t=i, eps=0.01, theta=0.3) for i in range(len(sim.times))]
    np.testing.assert_array_equal(acc, 0)

def test_self_gravity_on_gives_nonzero_acc():
    sim = Sim()
    sim.add_particles('a', np.random.normal(size=(30, 3)) * 0.5, np.zeros((30, 3)), np.ones(30) * 1e4)
    sim.turn_self_gravity_on()
    sim.run(t_end=2, dt=1, dt_out=2, eps=0.1)
    acc = [sim.compute_self_gravity(t=i, eps=0.01, theta=0.3) for i in range(len(sim.times))]
    assert not np.all(np.isclose(acc, 0, atol=1e-10))


### Test self-gravity accessors ###

SELF_GRAVITY_TEST = np.array(-G_INTERNAL * BINARY_TEST_MASS[0] * BINARY_TEST_MASS[1] * (BINARY_TEST_POS[1] - BINARY_TEST_POS[0])/ (
    np.linalg.norm(BINARY_TEST_POS[1] - BINARY_TEST_POS[0]))**3)

def test_compute_self_gravity_spot_check():
    assert np.all(np.isclose(np.abs(BINARY_TEST_SIM.compute_self_gravity(t=0, eps=0.01, theta=0.3)), SELF_GRAVITY_TEST, rtol=1e-10))
def test_self_ax_spot_check():
    assert np.all(np.isclose(BINARY_TEST_SIM.self_ax(t=0, eps=0.01, theta=0.3), SELF_GRAVITY_TEST[0], rtol=1e-10))
def test_self_ay_spot_check():
    assert np.all(np.isclose(BINARY_TEST_SIM.self_ay(t=0, eps=0.01, theta=0.3), SELF_GRAVITY_TEST[1], rtol=1e-10))
def test_self_az_spot_check():
    assert np.all(np.isclose(BINARY_TEST_SIM.self_az(t=0, eps=0.01, theta=0.3), SELF_GRAVITY_TEST[2], rtol=1e-10))


### test .add_external_pot() ###

def test_add_external_pot_rejection():
    sim = Sim()
    with pytest.raises(TypeError, match="External potential must be a galpy Potential object."):
        sim.add_external_pot(lambda pos, t: pos)


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
KEPLER_SIM.add_external_pot(kepler_pot)
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
    assert np.all(np.isclose(KEPLER_SIM.PE(t=0, eps=0.01, theta=0.3), expected_PE, rtol=1e-10))

def test_dE():
    E0 = KEPLER_SIM.system_energy(t=0, eps=0.01, theta=0.3)
    E1 = KEPLER_SIM.system_energy(t=1, eps=0.01, theta=0.3)
    dE = KEPLER_SIM.dE(eps=0.01, theta=0.3)
    assert np.all(np.isclose(E1 - E0, dE, rtol=1e-10))

### Test .run method ###
def test_negative_dt():
    with pytest.raises(ValueError, match="dt, dt_out, and t_end must be positive."):
        KEPLER_SIM.run(
                  t_end=1.0, 
                  dt=-0.1, 
                  dt_out=0.1,
                  eps=1.0,
                  theta=0.5)

def test_negative_dt_out():
    with pytest.raises(ValueError, match="dt, dt_out, and t_end must be positive."):
        KEPLER_SIM.run(
                  t_end=1.0, 
                  dt=0.1, 
                  dt_out=-0.1,
                  eps=1.0,
                  theta=0.5)

def test_negative_t_end():
    with pytest.raises(ValueError, match="dt, dt_out, and t_end must be positive."):
        KEPLER_SIM.run(
                  t_end=-1.0, 
                  dt=0.1, 
                  dt_out=0.1,
                  eps=1.0,
                  theta=0.5)

def test_dt_out_less_than_dt():
    with pytest.raises(ValueError, match="dt_out must be greater than or equal to dt."):
        KEPLER_SIM.run(
                  t_end=1.0, 
                  dt=0.1, 
                  dt_out=0.05,
                  eps=1.0,
                  theta=0.5)
        

### Energy tests with non-unit masses ###
#
# All tests above use mass=1, so m*Φ == Φ and missing mass factors are invisible.
# These tests use deliberately non-unit, non-equal masses so that if mass is
# missing from any PE/energy term, the analytical expected value will not match.

E_POS  = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])   # 2 kpc apart
E_VEL  = np.array([[0.5,  0.0, 0.0], [0.0, 0.0, 0.3]])
E_MASS = np.array([3e8, 5e8])           # non-unit, non-equal, large enough that G*M*M/r >> 1e-8
E_SEP  = np.linalg.norm(E_POS[1] - E_POS[0])   # 2.0 kpc
E_EPS  = 0.001
E_THETA = 0.0
E_KEPLER_MASS = 1e10   # Msun, for external Kepler potential


def _energy_sim(with_ext_pot=False):
    """Two massive particles, optionally in an external Kepler potential."""
    sim = Sim()
    sim.add_particles('pair', pos=E_POS.copy(), vel=E_VEL.copy(), mass=E_MASS.copy())
    if with_ext_pot:
        kep = KeplerPotential(amp=E_KEPLER_MASS * u.Msun)
        sim.add_external_pot(kep)
    sim.run(t_end=0.5, dt=0.25, dt_out=0.25, eps=E_EPS)
    return sim


# --- KE -----------------------------------------------------------------

def test_KE_nonunit_mass():
    """KE = ½ m |v|² must scale with particle mass."""
    sim = _energy_sim()
    ke = sim.KE(t=0)
    expected = 0.5 * E_MASS * np.sum(E_VEL ** 2, axis=-1)
    np.testing.assert_allclose(ke, expected, rtol=1e-10)
    # Confirm it *differs* from the unit-mass answer
    assert not np.allclose(ke, 0.5 * np.sum(E_VEL ** 2, axis=-1))


# --- Self-gravitational PE -----------------------------------------------

def test_self_potential_is_mass_weighted():
    """compute_self_potential must return m_i * Φ_i, not bare Φ_i."""
    sim = _energy_sim()
    pe = sim.compute_self_potential(t=0, eps=E_EPS, theta=E_THETA)
    # Two-body: PE_i = m_i * (-G * m_j / r_ij)
    expected_0 = -E_MASS[0] * G_INTERNAL * E_MASS[1] / E_SEP
    expected_1 = -E_MASS[1] * G_INTERNAL * E_MASS[0] / E_SEP
    np.testing.assert_allclose(pe[0], expected_0, rtol=1e-2)
    np.testing.assert_allclose(pe[1], expected_1, rtol=1e-2)

def test_self_potential_differs_from_bare_phi():
    """If mass were missing, pe[0] would equal Φ_0 = -G*m1/r, not m0*Φ_0."""
    sim = _energy_sim()
    pe = sim.compute_self_potential(t=0, eps=E_EPS, theta=E_THETA)
    bare_phi_0 = -G_INTERNAL * E_MASS[1] / E_SEP   # potential, not PE
    assert not np.isclose(pe[0], bare_phi_0, rtol=1e-2, atol=0)


# --- External potential (per unit mass) -----------------------------------

def test_external_pot_is_mass_weighted():
    """compute_external_pot should return m_i * Φ_ext,i."""
    sim = _energy_sim(with_ext_pot=True)
    ext = sim.compute_external_pot(t=0)
    r0 = np.linalg.norm(E_POS[0])
    r1 = np.linalg.norm(E_POS[1])
    expected_0 = -E_MASS[0] * G_INTERNAL * E_KEPLER_MASS / r0
    expected_1 = -E_MASS[1] * G_INTERNAL * E_KEPLER_MASS / r1
    np.testing.assert_allclose(ext[0], expected_0, rtol=1e-2)
    np.testing.assert_allclose(ext[1], expected_1, rtol=1e-2)


# --- Total PE (self + external, both mass-weighted) -----------------------

def test_PE_includes_mass_on_external():
    """PE = m*Φ_self + m*Φ_ext — mass must multiply the external term."""
    sim = _energy_sim(with_ext_pot=True)
    pe = sim.PE(t=0, eps=E_EPS, theta=E_THETA)

    r0 = np.linalg.norm(E_POS[0])
    r1 = np.linalg.norm(E_POS[1])

    self_pe = -G_INTERNAL * E_MASS[0] * E_MASS[1] / E_SEP   # same for both
    ext_pe_0 = -E_MASS[0] * G_INTERNAL * E_KEPLER_MASS / r0
    ext_pe_1 = -E_MASS[1] * G_INTERNAL * E_KEPLER_MASS / r1

    expected = np.array([self_pe + ext_pe_0, self_pe + ext_pe_1])
    np.testing.assert_allclose(pe, expected, rtol=1e-2)

def test_PE_external_without_mass_is_wrong():
    """Verify PE differs from Φ_self + Φ_ext (missing mass on external)."""
    sim = _energy_sim(with_ext_pot=True)
    pe = sim.PE(t=0, eps=E_EPS, theta=E_THETA)

    self_pe_0 = -E_MASS[0] * G_INTERNAL * E_MASS[1] / E_SEP
    bare_ext_0 = -G_INTERNAL * E_KEPLER_MASS / np.linalg.norm(E_POS[0])
    wrong_pe_0 = self_pe_0 + bare_ext_0   # missing mass on external
    assert not np.isclose(pe[0], wrong_pe_0, rtol=1e-2, atol=0)


# --- System energy --------------------------------------------------------

def test_system_energy_analytical():
    """E = Σ KE + ½ Σ(m·Φ_self) + Σ(m·Φ_ext), all computed analytically."""
    sim = _energy_sim(with_ext_pot=True)
    E = sim.system_energy(t=0, eps=E_EPS, theta=E_THETA)

    ke = np.sum(0.5 * E_MASS * np.sum(E_VEL ** 2, axis=-1))

    # ½ * Σ m_i Φ_i = ½*(m0*Φ0 + m1*Φ1) = -G*m0*m1/r  (double-counting factor)
    self_pe = -G_INTERNAL * E_MASS[0] * E_MASS[1] / E_SEP

    r0 = np.linalg.norm(E_POS[0])
    r1 = np.linalg.norm(E_POS[1])
    ext_pe = (-E_MASS[0] * G_INTERNAL * E_KEPLER_MASS / r0
              - E_MASS[1] * G_INTERNAL * E_KEPLER_MASS / r1)

    expected = ke + self_pe + ext_pe
    np.testing.assert_allclose(E, expected, rtol=1e-10)

def test_system_energy_mass_on_external_matters():
    """system_energy must differ from the wrongly unweighted external term."""
    sim = _energy_sim(with_ext_pot=True)
    E = sim.system_energy(t=0, eps=E_EPS, theta=E_THETA)

    ke = np.sum(0.5 * E_MASS * np.sum(E_VEL ** 2, axis=-1))
    self_pe = -G_INTERNAL * E_MASS[0] * E_MASS[1] / E_SEP
    r0 = np.linalg.norm(E_POS[0])
    r1 = np.linalg.norm(E_POS[1])
    wrong_ext = (-G_INTERNAL * E_KEPLER_MASS / r0
                 - G_INTERNAL * E_KEPLER_MASS / r1)  # missing mass
    wrong_E = ke + self_pe + wrong_ext
    assert not np.isclose(E, wrong_E, rtol=1e-2, atol=0)


### Test conservation of energy with single orbit in external galpy potential ###
