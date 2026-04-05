import pytest
import numpy as np
import astropy.units as u
from ezfalcon.simulation import Sim, Component
from ezfalcon.util import G_INTERNAL
from ezfalcon.dynamics.acceleration.self_gravity import _direct_summation

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

# --- Basic attribute access ------------------------------------------------- #

def test_component_attribute_access_pre_run():
    '''
    Aim: Verify that accessing a named component via sim.comp1 returns a
    Component that slices into the correct portion of the Sim arrays.

    If this fails: __getattr__ is not creating Component with the right slice.
    Relies on: add_particles correctly building _slices.
    '''
    comp1 = multicomp.comp1
    assert isinstance(comp1, Component)
    assert np.all(comp1.pos(t=0) == multicomp._init_pos[multicomp._slices['comp1']])
    assert np.all(comp1.vel(t=0) == multicomp._init_vel[multicomp._slices['comp1']])
    assert np.all(comp1.mass == multicomp._mass[multicomp._slices['comp1']])

def test_component_returns_view_not_copy():
    '''
    Aim: Verify that component.mass is a view into (not a copy of)
    the parent Sim's mass array, so they share memory.

    If this fails: Component is copying data instead of slicing.
    Relies on: add_particles, __getattr__.
    '''
    comp1 = multicomp.comp1
    assert np.shares_memory(comp1.mass, multicomp._mass)

def test_both_components_partition_full_arrays():
    '''
    Aim: Verify comp1 and comp2 together cover all particles exactly once.
    Concatenating their masses should reproduce the full Sim mass array.

    If this fails: Slices are overlapping, have gaps, or are wrong size.
    Relies on: add_particles building contiguous non-overlapping slices.
    '''
    comp1 = multicomp.comp1
    comp2 = multicomp.comp2
    combined_mass = np.concatenate([comp1.mass, comp2.mass])
    np.testing.assert_array_equal(combined_mass, multicomp._mass)

def test_component_mass_is_read_only_property():
    '''
    Aim: Verify mass is a property, not a callable — calling it should
    raise TypeError, indexing should work.

    If this fails: mass was accidentally defined as a method instead of property.
    Relies on: Component.mass @property.
    '''
    comp1 = multicomp.comp1
    assert isinstance(comp1.mass, np.ndarray)
    with pytest.raises(TypeError):
        comp1.mass()

# --- Shapes after run ------------------------------------------------------- #

multicomp.run(t_end=1., dt=0.1, dt_out=0.1, eps=0.1, theta=0.3)

def test_component_accessor_shapes_after_run():
    '''
    Aim: Verify that all position/velocity accessors return arrays with
    the correct component-specific shapes after run.

    If this fails: Slicing is grabbing too many or too few particles.
    Relies on: run() populating _positions/_velocities correctly.
    '''
    comp1 = multicomp.comp1
    comp2 = multicomp.comp2
    assert comp1.pos().shape == (11, COMP1_NPTS, 3)
    assert comp1.vel().shape == (11, COMP1_NPTS, 3)
    assert comp1.mass.shape == (COMP1_NPTS,)
    assert comp2.pos().shape == (11, COMP2_NPTS, 3)
    assert comp2.vel().shape == (11, COMP2_NPTS, 3)
    assert comp2.mass.shape == (COMP2_NPTS,)

# --- Scalar accessors (x, y, z, vx, vy, vz) -------------------------------- #

def test_component_xyz_match_pos_columns():
    '''
    Aim: Verify x(), y(), z() return the correct columns of pos().

    If this fails: Column indices are swapped in the axis slicing.
    Relies on: pos() being correct.
    '''
    comp1 = multicomp.comp1
    np.testing.assert_array_equal(comp1.x(0), comp1.pos(0)[:, 0])
    np.testing.assert_array_equal(comp1.y(0), comp1.pos(0)[:, 1])
    np.testing.assert_array_equal(comp1.z(0), comp1.pos(0)[:, 2])

def test_component_vxvyvz_match_vel_columns():
    '''
    Aim: Verify vx(), vy(), vz() return the correct columns of vel().

    If this fails: Column indices are swapped in the axis slicing.
    Relies on: vel() being correct.
    '''
    comp1 = multicomp.comp1
    np.testing.assert_array_equal(comp1.vx(0), comp1.vel(0)[:, 0])
    np.testing.assert_array_equal(comp1.vy(0), comp1.vel(0)[:, 1])
    np.testing.assert_array_equal(comp1.vz(0), comp1.vel(0)[:, 2])

# --- Slicing correctness: component data != other component's data ---------- #

def test_comp1_data_differs_from_comp2():
    '''
    Aim: Verify that comp1 and comp2 return different data (they have
    different initial conditions). Guards against both pointing to
    the same slice.

    If this fails: Both components reference the same slice.
    Relies on: Different random ICs for COMP1 and COMP2.
    '''
    comp1 = multicomp.comp1
    comp2 = multicomp.comp2
    assert not np.array_equal(comp1.pos(0), comp2.pos(0)[:COMP1_NPTS])
    assert not np.array_equal(comp1.mass, comp2.mass[:COMP1_NPTS])

# --- Time indexing through Component ---------------------------------------- #

def test_component_int_time_index():
    '''
    Aim: Verify integer time indexing returns a single snapshot.

    If this fails: Time resolution in Component is broken.
    Relies on: Sim._ti() for index resolution.
    '''
    comp1 = multicomp.comp1
    assert comp1.pos(0).shape == (COMP1_NPTS, 3)
    assert comp1.pos(-1).shape == (COMP1_NPTS, 3)

def test_component_float_time_index():
    '''
    Aim: Verify float time indexing resolves to the nearest snapshot
    and returns data matching the integer-indexed snapshot.

    If this fails: Float-to-index resolution is wrong inside Component.
    Relies on: Sim._ti() for float resolution.
    '''
    comp1 = multicomp.comp1
    np.testing.assert_array_equal(comp1.pos(0.0), comp1.pos(0))
    np.testing.assert_array_equal(comp1.vel(0.0), comp1.vel(0))

def test_component_ellipsis_returns_all_snapshots():
    '''
    Aim: Verify that the default t=... returns all snapshots.

    If this fails: Ellipsis handling in Sim._ti() is broken.
    Relies on: Sim._ti() ellipsis support.
    '''
    comp1 = multicomp.comp1
    all_pos = comp1.pos()  # default t=...
    assert all_pos.shape == (11, COMP1_NPTS, 3)
    # First and last snapshots should match explicit indexing
    np.testing.assert_array_equal(all_pos[0], comp1.pos(0))
    np.testing.assert_array_equal(all_pos[-1], comp1.pos(-1))

# --- Component data matches parent Sim sliced data ------------------------- #

def test_component_pos_matches_sim_sliced():
    '''
    Aim: Verify that comp.pos(t) == sim.pos(t)[slice] at every snapshot.
    This is the fundamental contract of Component.

    If this fails: Component is slicing the wrong array or wrong axis.
    Relies on: Sim.pos() being correct.
    '''
    comp1 = multicomp.comp1
    sl = multicomp._slices['comp1']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(comp1.pos(t), multicomp.pos(t)[sl])
        np.testing.assert_array_equal(comp1.vel(t), multicomp.vel(t)[sl])

def test_component_vel_matches_sim_sliced():
    '''
    Aim: Same as above but for comp2, ensuring the second slice is correct.

    If this fails: Second component's offset is wrong.
    Relies on: Sim.vel() being correct.
    '''
    comp2 = multicomp.comp2
    sl = multicomp._slices['comp2']
    for t in [0, 5, -1]:
        np.testing.assert_array_equal(comp2.vel(t), multicomp.vel(t)[sl])

# --- KE -------------------------------------------------------------------- #

def test_component_KE_matches_manual():
    '''
    Aim: Verify comp.KE() = 0.5 * comp.mass * |comp.vel|^2.

    If this fails: KE formula is wrong or uses wrong mass/vel arrays.
    Relies on: mass and vel() being correctly sliced.
    '''
    comp1 = multicomp.comp1
    ke = comp1.KE(t=0)
    expected = 0.5 * comp1.mass * np.sum(comp1.vel(0)**2, axis=-1)
    np.testing.assert_allclose(ke, expected, rtol=1e-15)

def test_component_KE_differs_between_components():
    '''
    Aim: Verify comp1.KE and comp2.KE are different (different ICs).

    If this fails: Both components reference the same underlying data.
    Relies on: Different random ICs.
    '''
    assert not np.array_equal(multicomp.comp1.KE(0), multicomp.comp2.KE(0)[:COMP1_NPTS])

def test_component_KE_sums_to_sim_KE():
    '''
    Aim: Verify that the sum of per-component KEs equals the full Sim KE.

    If this fails: Component slicing is losing or duplicating particles.
    Relies on: Sim.KE() being correct.
    '''
    sim_ke = multicomp.KE(t=0)
    comp1_ke = multicomp.comp1.KE(t=0)
    comp2_ke = multicomp.comp2.KE(t=0)
    combined = np.concatenate([comp1_ke, comp2_ke])
    np.testing.assert_allclose(combined, sim_ke, rtol=1e-15)

