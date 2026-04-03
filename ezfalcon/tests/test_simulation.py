'''
Test the Sim class and its methods.
'''

import pytest 
from ezfalcon.simulation import Sim, Component
import numpy as np
from ezfalcon.util import G_INTERNAL
from ezfalcon.dynamics.acceleration.self_gravity import _direct_summation
import astropy.units as u


# --- Setup ----------------------------------------------------------------------------------- #

np.random.seed(42)

COMP1_NPTS = 50
COMP1_POS = np.random.rand(COMP1_NPTS, 3)
COMP1_VEL = np.random.rand(COMP1_NPTS, 3)
COMP1_MASS = np.random.rand(COMP1_NPTS)

COMP2_NPTS = 30
COMP2_POS = np.random.rand(COMP2_NPTS, 3)
COMP2_VEL = np.random.rand(COMP2_NPTS, 3)
COMP2_MASS = np.random.rand(COMP2_NPTS)


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

# --- .add_particles() ------------------------------------------------------------------------ #

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
    sim.run(t_end=1., dt=0.5, dt_out=0.5, method='direct', eps=0.0)
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

# --- Multi-component slicing ------------------------------------------------------------------------ #
#
# The slicing itself is tested in test_component.py,
# but here we test that the slices don't overlap and
# that the correct errors are raised when accessing non-existent components.


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


# --- accessors pre-run ------------------------------------------------------------------------ #

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


# --- POST-RUN, NO EXTERNAL POTENTIAL TESTS ------------------------------------------------------------------- #

multicomp.run(t_end=1., dt=0.1, dt_out=0.1, eps=0.1, theta=0.3)
singlecomp.run(t_end=1., dt=0.1, dt_out=0.1, eps=0.1, theta=0.3)

# --- ._ti time indexing ------------------------------------------------------------------------ #

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
    '''
    Test that passing a list raises an error.
    '''
    with pytest.raises(TypeError, match="t must be an int index, a float time, or ellipsis."):
        multicomp._ti([0, 5], vectorized=False)

def test_ti_vectorized_false_fails_with_ellipsis():
    '''
    Test that passing an ellipsis when vectorized=False raises an error.
    '''
    with pytest.raises(TypeError, match="This method is not vectorized, so t cannot be a list or ellipse. Please provide an integer index or a float time."):
        multicomp._ti(..., vectorized=False)

# --- output shapes ------------------------------------------------------------------------ #

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


# --- self-gravity toggle ------------------------------------------------------------------------ #

def test_self_gravity_off_gives_zero_acc():
    sim = Sim()
    sim.add_particles('a', np.random.normal(size=(30, 3)) * 0.5, np.zeros((30, 3)), np.ones(30) * 1e4)
    sim.turn_self_gravity_off()
    sim.run(t_end=2, dt=1, dt_out=2, method='direct', eps=0.0)
    acc = sim.self_gravity()
    np.testing.assert_array_equal(acc, 0)

def test_self_gravity_on_gives_nonzero_acc():
    sim = Sim()
    sim.add_particles('a', np.random.normal(size=(30, 3)) * 0.5, np.zeros((30, 3)), np.ones(30) * 1e4)
    sim.turn_self_gravity_on()
    sim.run(t_end=2, dt=1, dt_out=2, method='direct', eps=0.0)
    acc = sim.self_gravity()
    assert not np.all(np.isclose(acc, 0, atol=1e-10))

# --- .add_external_pot() ------------------------------------------------------------------------ #

def test_add_external_pot_rejection():
    sim = Sim()
    with pytest.raises(TypeError, match="External potential must be a galpy Potential object."):
        sim.add_external_pot(lambda pos, t: pos)

# --- self-gravity acceleration accessors ------------------------------------------------------------------------ #

def test_acc_matches_direct():
    '''
    Test that the acceleration accessors match the directly
    obtained values.
    '''
    pos = np.array(np.random.normal(size=(10, 3)))
    vel = np.array(np.random.normal(size=(10, 3)))
    mass = np.array(np.random.normal(loc=1e9, scale=1e8, size=(10,)))
    sim = Sim()
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    acc = sim.self_gravity(t=0, method='direct', eps=0.0)
    acc_direct= _direct_summation(pos, mass, eps=0.0, return_potential=False)
    np.testing.assert_allclose(acc, acc_direct, rtol=1e-10)

# --- external acceleration accessors -------------------------------------------------------- #

def test_external_acc_zero_without_pot():
    sim = Sim()
    sim.add_particles('a', np.random.normal(size=(30, 3)) * 0.5, np.zeros((30, 3)), np.ones(30) * 1e4)
    sim.run(t_end=2, dt=1, dt_out=2, method='direct', eps=0.0)
    np.testing.assert_array_equal(sim.external_acc(-1), 0)

KEPLER_SIM = Sim()
from galpy.potential import KeplerPotential
kepler_pot = KeplerPotential(amp=1e9*u.Msun)
KEPLER_SIM.add_external_pot(kepler_pot)
KEPLER_SIM.add_particles('a', pos=np.array([[0.1, 0.2, 0.3]]), vel=np.array([[0, 0.0, 0]]), mass=np.array([1e8]))
KEPLER_ACC = -G_INTERNAL * 1e9 * np.array([[0.1, 0.2, 0.3]]) / (0.1**2 + 0.2**2 + 0.3**2)**(3/2)
KEPLER_POT = -G_INTERNAL * 1e8 * 1e9 / np.sqrt(0.1**2 + 0.2**2 + 0.3**2)

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

# --- potential accessors ------------------------------------------------------------------------ #

def test_external_pot_against_direct():
    '''
    Aim: Verify compute_external_pot() returns m * phi_ext for a single particle
    in a Kepler potential, compared to the hand-computed value.

    If this fails: galpy bridge is returning wrong potential values, or
    compute_external_pot is not multiplying by mass.
    Relies on: galpy KeplerPotential being correct, _galpy_bridge conversion.
    '''
    pot = KEPLER_SIM.compute_external_pot(0)
    assert np.all(np.isclose(pot, KEPLER_POT, rtol=1e-10))

def test_self_potential_against_direct():
    '''
    Aim: Verify self_potential() equals mass * _direct_summation()
    potential. This tests that the Sim accessor correctly multiplies by mass
    on top of the raw solver output. Uses non-unit masses [1e8, 1e10].

    If this fails: self_potential is not multiplying by mass, or is
    calling self_gravity incorrectly (wrong method, wrong kwargs).
    Relies on: _direct_summation being correct (test_direct_summation.py).
    '''
    pos = np.array([[1.0, 0, 0], [0, 1.5, 0]])
    vel = np.zeros_like(pos)
    mass = np.array([1e8, 1e10])
    sim = Sim()
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    _, pot_direct = _direct_summation(pos, mass, eps=0.0, return_potential=True)
    pot_sim = sim.self_potential(t=0, method='direct', eps=0.0)
    np.testing.assert_allclose(pot_sim[0], mass[0] * pot_direct[0], rtol=1e-15)
    np.testing.assert_allclose(pot_sim[1], mass[1] * pot_direct[1], rtol=1e-15)

def test_PE_against_direct():
    '''
    Aim: Verify PE() = mass*phi_self + mass*phi_ext by computing both terms
    analytically and comparing to the Sim accessor. Uses non-unit masses
    and an external Kepler potential.

    If this fails: PE() is not correctly summing self + external, or one
    of the terms is missing its mass factor.
    Relies on: test_self_potential_against_direct, test_external_pot_against_direct.
    '''
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 1.5, 0.0]])
    vel = np.zeros_like(pos)
    mass = np.array([1e8, 1e10])
    sim = Sim()
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.add_external_pot(kepler_pot)
    _, self_pot_direct = _direct_summation(pos, mass, eps=0.0, return_potential=True)
    ext_pot_direct = ([-G_INTERNAL * mass[0] * 1e9 / np.linalg.norm(pos[0]), 
                       -G_INTERNAL * mass[1] * 1e9 / np.linalg.norm(pos[1])])
    pot_direct = mass * self_pot_direct + ext_pot_direct
    pot_sim = sim.PE(t=0, method='direct', eps=0.0)
    np.testing.assert_allclose(pot_sim[0], pot_direct[0], rtol=1e-10)
    np.testing.assert_allclose(pot_sim[1], pot_direct[1], rtol=1e-10)

def test_PE_is_sum_of_self_and_external():
    '''
    Aim: Verify PE() == self_potential() + compute_external_pot().
    This is a pure consistency check — it does not compare to analytical
    values, so it can pass even if both compute_ methods have the same bug.

    If this fails: PE() is doing something other than adding the two
    methods together (e.g. extra terms, wrong signs).
    Relies on: nothing external — only tests internal consistency of Sim.
    '''
    pos = np.array([[1.0, 0, 0], [0, 1.5, 0]])
    vel = np.zeros_like(pos)
    mass = np.array([1e8, 1e10])
    sim = Sim()
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.add_external_pot(kepler_pot)
    self_pot = sim.self_potential(t=0, method='direct', eps=0.0)
    ext_pot = sim.compute_external_pot(t=0)
    total_pot = sim.PE(t=0, method='direct', eps=0.0)
    np.testing.assert_allclose(total_pot, self_pot + ext_pot, rtol=1e-15)

# --- total energy accessors ------------------------------------------------------------------------ #

def test_energy_is_sum_of_KE_and_PE():
    '''
    Aim: Verify energy() == KE() + PE() at t=0 (before integration).
    Pure consistency check — if KE and PE are both wrong in the same
    way, this still passes. The individual KE/PE correctness is
    validated by the analytical spot-check tests above.

    If this fails: energy() has extra logic beyond summing KE + PE.
    Relies on: nothing external — only tests internal consistency of Sim.
    '''
    pos = np.array([[1.0, 0, 0], [0, 1.5, 0]])
    vel = np.zeros_like(pos)
    mass = np.array([1e8, 1e10])
    sim = Sim()
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.add_external_pot(kepler_pot)
    KE = sim.KE(t=0)
    PE = sim.PE(t=0, method='direct', eps=0.0)
    energy = sim.energy(t=0, method='direct', eps=0.0)
    np.testing.assert_allclose(energy, KE + PE, rtol=1e-15)

def test_energy_is_sum_of_KE_and_PE_after_run():
    '''
    Aim: Same as test_energy_is_sum_of_KE_and_PE, but checked at t=0.5
    after the simulation has evolved. Ensures that energy() remains
    consistent with KE() + PE() even after integration moves particles.

    If this fails: energy() uses stale/cached data instead of recomputing
    from the current snapshot positions and velocities.
    Relies on: run() completing without error, pos/vel accessors working
    at non-zero timesteps.
    '''
    pos = np.array([[1.0, 0, 0], [0, 1.5, 0]])
    vel = np.zeros_like(pos)
    mass = np.array([1e8, 1e10])
    sim = Sim()
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.add_external_pot(kepler_pot)
    sim.run(t_end=0.5, dt=0.25, dt_out=0.25, method='direct', eps=0.0)
    np.testing.assert_allclose(sim.energy(t=0.5, method='direct', eps=0.0), 
                               sim.KE(t=0.5) + sim.PE(t=0.5, method='direct', eps=0.0), rtol=1e-15)

def test_system_energy_is_sum_of_energies():
    '''
    Aim: Verify system_energy() == Σ KE + ½ Σ(m·Φ_self) + Σ(m·Φ_ext).
    The ½ on the self-PE avoids double-counting pairwise interactions.
    This checks the scalar reduction logic, NOT whether individual
    terms have correct mass factors (that's the analytical tests' job).

    NOTE: the expected value here is built from Sim's own accessors
    (self_potential, compute_external_pot), so this is a
    consistency test. A shared mass-factor bug would escape.

    If this fails: system_energy() is combining terms incorrectly
    (e.g. missing the ½ on self-PE, or not summing over particles).
    Relies on: KE(), self_potential(), compute_external_pot()
    all returning per-particle arrays of the right shape.
    '''
    pos = np.array([[1.0, 0, 0], [0, 1.5, 0]])
    vel = np.zeros_like(pos)
    mass = np.array([1e8, 1e10])
    sim = Sim()
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.add_external_pot(kepler_pot)
    KE = np.sum(sim.KE(t=0))
    PE = 0.5 * np.sum(sim.self_potential(t=0, method='direct', eps=0.0)) + np.sum(sim.compute_external_pot(t=0))
    system_energy = sim.system_energy(t=0, method='direct', eps=0.0)
    np.testing.assert_allclose(system_energy, KE + PE, rtol=1e-15)
   
def test_system_energy_is_sum_of_energies_after_run():
    pos = np.array([[1.0, 0, 0], [0, 1.5, 0]])
    vel = np.zeros_like(pos)
    mass = np.array([1e8, 1e10])
    sim = Sim()
    sim.add_particles('test', pos=pos, vel=vel, mass=mass)
    sim.add_external_pot(kepler_pot)
    sim.run(t_end=0.5, dt=0.25, dt_out=0.25, method='direct', eps=0.0)
    KE = np.sum(sim.KE(t=0.5))
    PE = 0.5 *np.sum(sim.self_potential(t=0.5, method='direct', eps=0.0)) + np.sum(sim.compute_external_pot(t=0.5))
    system_energy = sim.system_energy(t=0.5, method='direct', eps=0.0)
    np.testing.assert_allclose(system_energy, KE + PE, rtol=1e-15)


# --- Test .run method ------------------------------------------------------------------------ #

def test_negative_dt():
    with pytest.raises(ValueError, match="dt, dt_out, and t_end must be positive."):
        KEPLER_SIM.run(
                  t_end=1.0, 
                  dt=-0.1, 
                  dt_out=0.1,
                  method='direct',
                  eps=0.0)

def test_negative_dt_out():
    with pytest.raises(ValueError, match="dt, dt_out, and t_end must be positive."):
        KEPLER_SIM.run(
                  t_end=1.0, 
                  dt=0.1, 
                  dt_out=-0.1,
                  method='direct',
                  eps=0.0)

def test_negative_t_end():
    with pytest.raises(ValueError, match="dt, dt_out, and t_end must be positive."):
        KEPLER_SIM.run(
                  t_end=-1.0, 
                  dt=0.1, 
                  dt_out=0.1,
                  method='direct',
                  eps=0.0)

def test_dt_out_less_than_dt():
    with pytest.raises(ValueError, match="dt_out must be greater than or equal to dt."):
        KEPLER_SIM.run(
                  t_end=1.0, 
                  dt=0.1, 
                  dt_out=0.05,
                  method='direct',
                  eps=0.0)
def test_invalid_method():
    with pytest.raises(ValueError, match="Unknown method 'invalid_method' for self-gravity. Supported methods: \['direct', 'falcON'\]"):
        KEPLER_SIM.run(
                  t_end=1.0, 
                  dt=0.1, 
                  dt_out=0.1,
                  method='invalid_method',
                  eps=0.0) 

def test_invalid_kwargs():
    with pytest.raises(ValueError, match="{'invalid_kwarg'} is \(are\) invalid kwarg\(s\) for 'direct' self-gravity method. Only kwargs for self-gravity methods are allowed."):
        KEPLER_SIM.run(
                  t_end=1.0, 
                  dt=0.1, 
                  dt_out=0.1,
                  method='direct',
                  eps=0.0,
                  invalid_kwarg=42)
    with pytest.raises(ValueError, match="{'invalid_kwarg'} is \(are\) invalid kwarg\(s\) for 'falcON' self-gravity method. Only kwargs for self-gravity methods are allowed."):
        KEPLER_SIM.run(
                  t_end=1.0, 
                  dt=0.1, 
                  dt_out=0.1,
                  method='falcON',
                  eps=0.0,
                  theta=0.3,
                  invalid_kwarg=42)

# --- Energy tests with non-unit masses ------------------------------------------------------------------------ #
#
# All tests above use mass=1, so m*Φ == Φ and missing mass factors are invisible.
# These tests use deliberately non-unit, non-equal masses so that if mass is
# missing from any PE/energy term, the analytical expected value will not match.

E_POS  = np.array([[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]])   # 2 kpc apart
E_VEL  = np.array([[0.5,  0.0, 0.0], [0.0, 0.0, 0.3]])
E_MASS = np.array([3e8, 5e8])           # non-unit, non-equal, large enough that G*M*M/r >> 1e-8
E_SEP  = np.linalg.norm(E_POS[1] - E_POS[0])   # 2.0 kpc
E_EPS  = 0.0
E_KEPLER_MASS = 1e10   # Msun, for external Kepler potential


def _energy_sim(with_ext_pot=False):
    """Two massive particles, optionally in an external Kepler potential."""
    sim = Sim()
    sim.add_particles('pair', pos=E_POS.copy(), vel=E_VEL.copy(), mass=E_MASS.copy())
    if with_ext_pot:
        kep = KeplerPotential(amp=E_KEPLER_MASS * u.Msun)
        sim.add_external_pot(kep)
    sim.run(t_end=0.5, dt=0.25, dt_out=0.25, method='direct', eps=E_EPS)
    return sim

# --- KE -----------------------------------------------------------------

def test_KE_nonunit_mass():
    '''
    Aim: Verify KE = ½ m |v|² with non-unit masses [3e8, 5e8] Msun.
    Also asserts the result differs from the unit-mass answer, so a
    missing mass factor would be caught.

    If this fails: KE() is not multiplying by particle mass.
    Relies on: vel accessor returning correct initial velocities.
    '''
    sim = _energy_sim()
    ke = sim.KE(t=0)
    expected = 0.5 * E_MASS * np.sum(E_VEL ** 2, axis=-1)
    np.testing.assert_allclose(ke, expected, rtol=1e-10)
    # Confirm it *differs* from the unit-mass answer
    assert not np.allclose(ke, 0.5 * np.sum(E_VEL ** 2, axis=-1))

# --- Self-gravitational PE -----------------------------------------------

def test_self_potential_is_mass_weighted():
    '''
    Aim: Verify compute_self_potential returns m_i * Φ_i (not just Φ_i)
    by comparing to the analytical two-body PE: m_i * (-G * m_j / r_ij).
    Uses non-unit masses so m*Φ ≠ Φ.

    If this fails: compute_self_potential is returning bare Φ without
    the mass multiplication, or _direct_summation potential is wrong.
    Relies on: _direct_summation being correct (test_direct_summation.py).
    '''
    sim = _energy_sim()
    pe = sim.self_potential(t=0, method='direct',eps=E_EPS)
    # Two-body: PE_i = m_i * (-G * m_j / r_ij)
    expected_0 = -E_MASS[0] * G_INTERNAL * E_MASS[1] / E_SEP
    expected_1 = -E_MASS[1] * G_INTERNAL * E_MASS[0] / E_SEP
    np.testing.assert_allclose(pe[0], expected_0, rtol=1e-2)
    np.testing.assert_allclose(pe[1], expected_1, rtol=1e-2)

def test_self_potential_differs_from_bare_phi():
    '''
    Aim: Negative sanity check — verify that the returned PE does NOT
    equal the bare (unweighted) potential Φ_0 = -G*m1/r. If the
    mass factor were missing, they would match and this test would fail.

    If this fails: compute_self_potential is returning Φ without
    multiplying by particle mass — the exact bug this was written to catch.
    Relies on: test_self_potential_is_mass_weighted (if that passes and
    this fails, there is a contradiction).
    '''
    sim = _energy_sim()
    pe = sim.self_potential(t=0, method='direct',eps=E_EPS)
    bare_phi_0 = -G_INTERNAL * E_MASS[1] / E_SEP   # potential, not PE
    assert not np.isclose(pe[0], bare_phi_0, rtol=1e-2, atol=0)

# --- External potential (per unit mass) -----------------------------------

def test_external_pot_is_mass_weighted():
    '''
    Aim: Verify compute_external_pot returns m_i * Φ_ext,i (not bare Φ_ext)
    by comparing to the analytical Kepler potential m * (-G * M / r).
    Uses non-unit masses so m*Φ ≠ Φ.

    If this fails: compute_external_pot is not multiplying by mass, or
    the galpy bridge is returning wrong potential values.
    Relies on: galpy KeplerPotential, _galpy_bridge unit conversion.
    '''
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
    '''
    Aim: Verify PE() = m*Φ_self + m*Φ_ext by computing both terms
    analytically with non-unit masses. This is the main end-to-end check
    that the total PE has mass factors on BOTH the self and external terms.

    If this fails: PE() is missing mass on either the self-gravity or
    external potential term, or the summation of the two is wrong.
    Relies on: test_self_potential_is_mass_weighted,
    test_external_pot_is_mass_weighted (individual terms correct).
    '''
    sim = _energy_sim(with_ext_pot=True)
    pe = sim.PE(t=0, method='direct', eps=E_EPS)

    r0 = np.linalg.norm(E_POS[0])
    r1 = np.linalg.norm(E_POS[1])

    self_pe = -G_INTERNAL * E_MASS[0] * E_MASS[1] / E_SEP   # same for both
    ext_pe_0 = -E_MASS[0] * G_INTERNAL * E_KEPLER_MASS / r0
    ext_pe_1 = -E_MASS[1] * G_INTERNAL * E_KEPLER_MASS / r1

    expected = np.array([self_pe + ext_pe_0, self_pe + ext_pe_1])
    np.testing.assert_allclose(pe, expected, rtol=1e-2)

def test_PE_external_without_mass_is_wrong():
    '''
    Aim: Negative sanity check — verify PE does NOT match the value you
    would get if mass were missing from the external potential term.
    Computes the "wrong" answer (m*Φ_self + Φ_ext) and asserts PE differs.

    If this fails: PE() is not multiplying mass onto the external potential
    — the exact bug this was written to catch.
    Relies on: test_PE_includes_mass_on_external (if that passes and this
    fails, there is a contradiction).
    '''
    sim = _energy_sim(with_ext_pot=True)
    pe = sim.PE(t=0, method='direct', eps=E_EPS)

    self_pe_0 = -E_MASS[0] * G_INTERNAL * E_MASS[1] / E_SEP
    bare_ext_0 = -G_INTERNAL * E_KEPLER_MASS / np.linalg.norm(E_POS[0])
    wrong_pe_0 = self_pe_0 + bare_ext_0   # missing mass on external
    assert not np.isclose(pe[0], wrong_pe_0, rtol=1e-2, atol=0)


# --- System energy --------------------------------------------------------

def test_system_energy_analytical():
    '''
    Aim: Verify system_energy() matches a fully hand-computed value:
    E = Σ(½ m |v|²) + (-G m0 m1 / r) + Σ(-m_i G M_ext / r_i).
    The ½ on the self-PE double-counting cancels with the pairwise sum.
    All values are analytical — no Sim accessors in the expected value.

    If this fails: system_energy has a wrong coefficient (e.g. missing ½
    on self-PE), a missing mass factor on any term, or wrong signs.
    Relies on: _direct_summation being correct, galpy bridge being correct.
    This is the strongest energy test — independent of all other accessors.
    '''
    sim = _energy_sim(with_ext_pot=True)
    E = sim.system_energy(t=0, method='direct', eps=E_EPS, use_cached=False)

    ke = np.sum(0.5 * E_MASS * np.sum(E_VEL ** 2, axis=-1))

    self_pe = -G_INTERNAL * E_MASS[0] * E_MASS[1] / E_SEP

    r0 = np.linalg.norm(E_POS[0])
    r1 = np.linalg.norm(E_POS[1])
    ext_pe = (-E_MASS[0] * G_INTERNAL * E_KEPLER_MASS / r0
              - E_MASS[1] * G_INTERNAL * E_KEPLER_MASS / r1)

    expected = ke + self_pe + ext_pe
    np.testing.assert_allclose(E, expected, rtol=1e-10)

def test_system_energy_mass_on_external_matters():
    '''
    Aim: Negative sanity check — verify system_energy does NOT match
    the value you get if mass is missing from the external potential sum.
    Computes the "wrong" answer with bare phi_ext and asserts it differs.

    If this fails: system_energy is not multiplying mass onto the
    external potential — the exact bug this was written to catch.
    Relies on: test_system_energy_analytical (if that passes and this
    fails, there is a contradiction).
    '''
    sim = _energy_sim(with_ext_pot=True)
    E = sim.system_energy(t=0, method='direct', eps=E_EPS)

    ke = np.sum(0.5 * E_MASS * np.sum(E_VEL ** 2, axis=-1))
    self_pe = -G_INTERNAL * E_MASS[0] * E_MASS[1] / E_SEP
    r0 = np.linalg.norm(E_POS[0])
    r1 = np.linalg.norm(E_POS[1])
    wrong_ext = (-G_INTERNAL * E_KEPLER_MASS / r0
                 - G_INTERNAL * E_KEPLER_MASS / r1)  # missing mass
    wrong_E = ke + self_pe + wrong_ext
    assert not np.isclose(E, wrong_E, rtol=1e-2, atol=0)


# --- self-gravity caching ------------------------------------------------------------------------ #

def _caching_test_sim():
    sim = Sim()
    sim.add_particles('test', pos=np.random.normal(size=(10, 3)), vel=np.random.normal(size=(10, 3)), mass=np.random.normal(loc=1e9, scale=1e8, size=(10,)))
    return sim
def _run_caching_test_sim(sim, cache_self_gravity, cache_self_potential):
    sim.run(t_end=1, dt=0.1, dt_out=0.1, 
            method='direct', eps=0.0, 
            cache_self_gravity=cache_self_gravity, 
            cache_self_potential=cache_self_potential)
    
def test_no_self_gravity_pot_or_acc_caching_after_run():
    '''
    Test that self-gravity potential and acceleration are not cached during
    run if return_self_acceleration and return_potential are False. 
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=False, cache_self_potential=False)
    assert sim._cached_self_pot is None
    assert sim._cached_self_acc is None
    
def test_self_gravity_acc_caching_only_after_run():
    '''
    Test that self-gravity acceleration is cached and potential
    is not during run if cache_self_gravity is True and cache_self_potential is False.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=True, cache_self_potential=False)
    assert sim._cached_self_acc.shape == (len(sim._times), sim._positions.shape[1], 3)
    assert sim._cached_self_pot is None

def test_self_gravity_pot_caching_only_after_run():
    '''
    Test that self-gravity potential is cached and acceleration is not during run
    if cache_self_potential is True and cache_self_gravity is False.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=False, cache_self_potential=True)
    assert sim._cached_self_pot.shape == (len(sim._times), sim._positions.shape[1])
    assert sim._cached_self_acc is None

def test_self_gravity_acc_and_pot_caching_after_run():
    '''
    Test that self-gravity acceleration and potential are cached during run
    if cache_self_gravity and cache_self_potential are True.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=True, cache_self_potential=True)
    assert sim._cached_self_acc.shape == (len(sim._times), sim._positions.shape[1], 3)
    assert sim._cached_self_pot.shape == (len(sim._times), sim._positions.shape[1])

def test_self_gravity_acc_caching_matches_internal_array():
    '''
    Test that the self-gravity acceleration values when 
    using caching are the same as the internal arrays.
    
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=True, cache_self_potential=False)
    np.testing.assert_allclose(sim.self_gravity(), sim._cached_self_acc, rtol=1e-10)

def test_self_gravity_pot_caching_matches_internal_array():
    '''
    Test that the self-gravity potential values when 
    using caching are the same as the internal arrays.
    
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=False, cache_self_potential=True)
    np.testing.assert_allclose(sim.self_potential(), sim._mass * sim._cached_self_pot, rtol=1e-10)

def test_self_gravity_acc_cache_matches_direct_computation():
    '''
    Test that the self-gravity acceleration cached results match the internal array after the run.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=True, cache_self_potential=False)
    np.testing.assert_allclose(sim.self_gravity(t=0, method='direct', eps=0.0), sim.self_gravity(t=0, use_cached=True), rtol=1e-10)

def test_self_gravity_pot_cache_matches_direct_computation():
    '''
    Test that the self-gravity potential cached results match the internal array after the run.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=False, cache_self_potential=True)
    np.testing.assert_allclose(sim.self_potential(t=0, method='direct', eps=0.0), sim.self_potential(t=0, use_cached=True), rtol=1e-10)

def test_provides_method_but_use_cached_true_raises_error():
    '''
    Test that requesting a method-specific computation with use_cached=True
    raises an error, since the cache is not method-specific.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=True, cache_self_potential=True)
    with pytest.raises(ValueError, match="`method` should not be specified"):
        sim.self_gravity(t=0, method='direct', eps=0.0, use_cached=True)

def test_use_cache_true_before_run_raises_error():
    '''
    Test that trying to use the self-gravity cache before it has been populated by a run raises an error.
    '''
    sim = _caching_test_sim()
    with pytest.raises(ValueError, match="Cannot use cached results before run"):
        sim.self_gravity(t=0, method=None, eps=0.0, use_cached=True)
    with pytest.raises(ValueError, match="Cannot use cached results before run"):
        sim.self_gravity(t=0, method='direct', eps=0.0, use_cached=True)

def test_no_caching_without_method_raises_error():
    '''
    Test that setting use_cached=False without specifying a method raises an error.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=True, cache_self_potential=True)
    with pytest.raises(ValueError, match="No cached results available"):
        sim.self_gravity(t=0, use_cached=False)

def test_caching_defaults_to_false_before_run():
    '''
    Test that use_cached defaults to False before run(), so calling an
    accessor without a method raises the "must provide a method" error
    (not a cache-lookup error).
    '''
    sim = _caching_test_sim()
    # No explicit use_cached → decorator should resolve to False (pre-run).
    # With method=None that triggers the "must provide a method" error.
    with pytest.raises(ValueError, match="No cached results available"):
        sim.self_gravity(t=0)
    with pytest.raises(ValueError, match="No cached results available"):
        sim.self_potential(t=0)

def test_caching_fails_if_run_did_not_cache():
    '''
    Test that if you set cache_self_gravity=False but then try to use the cache after the run, it raises an error.
    '''
    sim = _caching_test_sim()
    _run_caching_test_sim(sim, cache_self_gravity=False, cache_self_potential=False)
    with pytest.raises(ValueError, match="Cached self-potential is not available"):
        sim.self_potential(t=0, use_cached=True)
    with pytest.raises(ValueError, match="Cached self-gravity is not available"):
        sim.self_gravity(t=0, use_cached=True)