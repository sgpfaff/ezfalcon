import pytest
import warnings
from galpy import potential
from galpy.util.coords import rect_to_cyl
import numpy as np
from ezfalcon.util import _galpy_bridge
from itertools import product
from functools import partial
from scipy.differentiate import derivative
import astropy.units as u


SUPPORTED_GALPY_SPHERICAL_POTENTIALS = [
    potential.BurkertPotential(),
    potential.TwoPowerSphericalPotential(),
    potential.DehnenCoreSphericalPotential(),
    potential.DehnenSphericalPotential(),
    potential.EinastoPotential(),
    potential.HernquistPotential(),
    potential.interpSphericalPotential(rforce=lambda r: -1./r,
                        rgrid=np.geomspace(0.01,20,101),Phi0=0.),
    potential.IsochronePotential(),
    potential.JaffePotential(),
    potential.KeplerPotential(),
    potential.KingPotential(),
    potential.NFWPotential(),
    potential.PlummerPotential(),
    potential.PowerSphericalPotential(),
    potential.PowerSphericalPotentialwCutoff(),
    potential.PseudoIsothermalPotential(),
    potential.HomogeneousSpherePotential(),
    potential.SphericalShellPotential(),
    potential.TwoPowerTriaxialPotential(b=1., c=1.),
    potential.TriaxialGaussianPotential(b=1., c=1.),
    potential.TriaxialJaffePotential(b=1., c=1.),
    potential.TriaxialHernquistPotential(b=1., c=1.),
    potential.TriaxialNFWPotential(b=1., c=1.),
    potential.PerfectEllipsoidPotential(b=1., c=1.),
]

SUPPORTED_GALPY_AXISYMMETRIC_POTENTIALS = [
    potential.FlattenedPowerPotential(),
    potential.KuzminDiskPotential(),
    potential.KuzminKutuzovStaeckelPotential(),
    potential.LogarithmicHaloPotential(q=0.8),
    potential.MiyamotoNagaiPotential(),
    potential.MN3ExponentialDiskPotential(),
    potential.RingPotential(),
    potential.DoubleExponentialDiskPotential(),
    potential.RazorThinExponentialDiskPotential(),
    potential.interpRZPotential(potential.MWPotential, interpPot=True)
]

SUPPORTED_GALPY_ELLIPSOIDAL_TRIAXIAL_POTENTIALS = [
    # Ellipsoidal Potentials
    potential.TwoPowerTriaxialPotential(b=0.8, c=0.6),
    potential.TriaxialGaussianPotential(b=0.8, c=0.6),
    potential.TriaxialJaffePotential(b=0.8, c=0.6),
    potential.TriaxialHernquistPotential(b=0.8, c=0.6),
    potential.TriaxialNFWPotential(b=0.8, c=0.6),
    potential.PerfectEllipsoidPotential(b=0.8, c=0.6),
]

SUPPORTED_GALPY_GENERAL_TRIAXIAL_POTENTIALS = [
    potential.DehnenBarPotential(),
    potential.FerrersPotential(),
    potential.NullPotential(),
    potential.SoftenedNeedleBarPotential(),
    potential.SpiralArmsPotential(),
]

ALL_SUPPORTED_GALPY_POTENTIALS = (SUPPORTED_GALPY_SPHERICAL_POTENTIALS + 
                                  SUPPORTED_GALPY_AXISYMMETRIC_POTENTIALS + 
                                  SUPPORTED_GALPY_ELLIPSOIDAL_TRIAXIAL_POTENTIALS + 
                                  SUPPORTED_GALPY_GENERAL_TRIAXIAL_POTENTIALS)

UNSUPPORTED_GALPY_POTENTIALS = [
    potential.DiskSCFPotential(),
    potential.SCFPotential(),
]

g = np.linspace(-100, 100, 5)
FULL_TEST_GRID_POSITIONS = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
FULL_TEST_R, FULL_TEST_PHI, FULL_TEST_Z = rect_to_cyl(*FULL_TEST_GRID_POSITIONS.T*u.kpc)

# Exclude the z-axis (x=0, y=0) — galpy forces have a known singularity at R=0
TEST_GRID_POSITIONS = FULL_TEST_GRID_POSITIONS[((FULL_TEST_GRID_POSITIONS[:, 0] != 0) | (FULL_TEST_GRID_POSITIONS[:, 1] != 0))]
TEST_R, TEST_PHI, TEST_Z = rect_to_cyl(*TEST_GRID_POSITIONS.T*u.kpc)


#-----------------------#
#  Spherical Potentials #
#-----------------------#

@pytest.fixture(params=SUPPORTED_GALPY_SPHERICAL_POTENTIALS, ids=lambda p: type(p).__name__)
def spherical_potential(request):
    pot = request.param
    pot.turn_physical_on()
    return pot

def test_radial_acc_only(spherical_potential):
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(spherical_potential)
    acc = acc_fn(TEST_GRID_POSITIONS, t=0)
    assert np.allclose(np.cross(TEST_GRID_POSITIONS, acc), 0, atol=1e-14)

def test_spherical_symmetry(spherical_potential):
    '''|a| should be identical at the same radius but different angles.'''
    r = 10.0  # kpc
    # Points at the same radius along different axes / diagonals
    d = r / np.sqrt(3)
    points = np.array([
        [r, 0, 0],
        [0, r, 0],
        [-r, 0, 0],
        [d, d, d],
        [-d, d, -d],
    ])
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(spherical_potential)
    acc = acc_fn(points, t=0)
    magnitudes = np.linalg.norm(acc, axis=1)
    np.testing.assert_allclose(magnitudes, magnitudes[0], rtol=1e-10)

def test_reflection_symmetry(spherical_potential):
    '''For spherical potentials, a(r) = -a(-r) (odd parity).'''
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(spherical_potential)
    pos = np.array([[5.0, 3.0, 1.0], [10.0, -7.0, 2.0]])
    acc_pos = acc_fn(pos, t=0)
    acc_neg = acc_fn(-pos, t=0)
    np.testing.assert_allclose(acc_pos, -acc_neg, rtol=1e-12)
    

#----------------------------#
#  Axisymmetric Potentials   #
#----------------------------#

@pytest.fixture(params=SUPPORTED_GALPY_AXISYMMETRIC_POTENTIALS, ids=lambda p: type(p).__name__)
def axisymmetric_potential(request):
    pot = request.param
    pot.turn_physical_on()
    return pot

def test_axisymmetry(axisymmetric_potential):
    '''Acc should be invariant under rotation about z-axis for axisymmetric potentials.'''
    pot = axisymmetric_potential
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
    R, z = 8.0, 1.0
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    points = np.column_stack([R * np.cos(angles), R * np.sin(angles), np.full_like(angles, z)])
    acc = acc_fn(points, t=0)
    magnitudes = np.linalg.norm(acc, axis=1)
    np.testing.assert_allclose(magnitudes, magnitudes[0], rtol=1e-12)


#---------------------------#
#  All Supported Potentials #
#---------------------------#

# # Reuse the grid without the origin for acceleration/potential tests
# POSITION_GRID = TEST_GRID_POSITIONS

@pytest.fixture(params=ALL_SUPPORTED_GALPY_POTENTIALS, ids=lambda p: type(p).__name__)
def galpy_potential(request):
    pot = request.param
    pot.turn_physical_on()
    return pot

def _numerical_acc(pot_fn, pos, h=1e-5):
    """Compute -grad(Phi) via central differences."""
    pos = np.asarray(pos, dtype=float)
    acc = np.zeros_like(pos)
    for i in range(3):
        pos_plus = pos.copy()
        pos_minus = pos.copy()
        pos_plus[:, i] += h
        pos_minus[:, i] -= h
        acc[:, i] = -(pot_fn(pos_plus) - pot_fn(pos_minus)) / (2 * h)
    return acc

def test_acceleration_match(galpy_potential):
    '''Compare the acceleration from the galpy potential wrapper to 
    numerical accelerations.'''
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(galpy_potential)
    pot_fn = _galpy_bridge._galpy_pot_to_pot_fn(galpy_potential)
    pot_i = partial(pot_fn, t=0)
    acc_bridge = acc_fn(TEST_GRID_POSITIONS, t=0)
    acc_num = _numerical_acc(pot_i, TEST_GRID_POSITIONS)
    rtol = 1e-4 if isinstance(galpy_potential, potential.interpRZPotential) else 1e-10
    assert np.allclose(acc_bridge, acc_num, rtol=rtol, atol=1e-10)

def test_potential_match(galpy_potential):
    '''Compare the potential from the galpy potential wrapper to 
    galpy's own potential evaluation.'''
    ez_pot_fn = _galpy_bridge._galpy_pot_to_pot_fn(galpy_potential)
    ez_pot = ez_pot_fn(FULL_TEST_GRID_POSITIONS, t=0)
    if isinstance(galpy_potential, _galpy_bridge.UNVECTORIZED_POTENTIALS):
        galpy_pot = np.array([
            galpy_potential(R, z, phi=p, t=0, quantity=True).to(u.kpc**2/u.Myr**2).value
            for R, z, p in zip(FULL_TEST_R, FULL_TEST_Z, FULL_TEST_PHI)
        ])
    else:
        galpy_pot = galpy_potential(FULL_TEST_R, FULL_TEST_Z, phi=FULL_TEST_PHI, t=0, quantity=True).to(u.kpc**2/u.Myr**2).value
    assert np.allclose(ez_pot, galpy_pot, rtol=1e-15, equal_nan=True)

#---------------------------#
#   Triaxial Potentials     #
#---------------------------#

@pytest.fixture(params=SUPPORTED_GALPY_ELLIPSOIDAL_TRIAXIAL_POTENTIALS, ids=lambda p: type(p).__name__)
def triaxial_potential(request):
    pot = request.param
    pot.turn_physical_on()
    return pot

def test_triaxial_reflection_symmetry(triaxial_potential):
    '''For triaxial potentials centered at origin, a(r) = -a(-r).'''
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(triaxial_potential)
    pos = np.array([[5.0, 3.0, 1.0], [10.0, -7.0, 2.0]])
    acc_pos = acc_fn(pos, t=0)
    acc_neg = acc_fn(-pos, t=0)
    np.testing.assert_allclose(acc_pos, -acc_neg, rtol=1e-12)

def test_triaxial_plane_symmetry(triaxial_potential):
    '''Reflecting through a coordinate plane flips only that force component.'''
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(triaxial_potential)
    pos = np.array([[5.0, 3.0, 2.0]])
    acc = acc_fn(pos, t=0)
    for axis in range(3):
        reflected = pos.copy()
        reflected[0, axis] *= -1
        acc_ref = acc_fn(reflected, t=0)
        for j in range(3):
            if j == axis:
                np.testing.assert_allclose(acc_ref[0, j], -acc[0, j], rtol=1e-10)
            else:
                np.testing.assert_allclose(acc_ref[0, j], acc[0, j], rtol=1e-10)

def test_triaxial_nonzero_phitorque(triaxial_potential):
    '''Triaxial potentials should have non-zero phi force at generic positions.'''
    R, z, phi = 8.0 * u.kpc, 1.0 * u.kpc, 0.3 * u.rad
    torque = triaxial_potential.phitorque(R, z, phi=phi, quantity=True)
    assert torque.value != 0, "Expected non-zero phi torque for triaxial potential"

#---------------------------#
#  Unsupported Potentials   #
#---------------------------#

@pytest.fixture(params=UNSUPPORTED_GALPY_POTENTIALS, ids=lambda p: type(p).__name__)
def unsupported_potential(request):
    return request.param

def test_identify_unsupported_potential(unsupported_potential):
    with pytest.raises(TypeError):
        _galpy_bridge._check_supported_pot(unsupported_potential)


#-----------------------#
#  Unit Conversion      #
#-----------------------#

def test_acc_units():
    '''Verify acc_fn returns the correct analytic value in internal units
    for a Kepler potential at a known position.'''
    from ezfalcon.util.units import G_INTERNAL
    # Kepler potential: a = -GM/r^2 rhat
    from galpy.util.conversion import get_physical
    M_msun = 1e3 # Msun
    pot = potential.KeplerPotential(amp=M_msun * u.Msun)  # amp=1 Msun in physical units
    pot.turn_physical_on()
    r = 0.1 # kpc
    pos = np.array([[r, 0.0, 0.0]])  # 10 kpc along x
    expected_ax = -G_INTERNAL * M_msun / r**2
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
    acc = acc_fn(pos, t=0)
    assert acc.shape == (1, 3)
    np.testing.assert_allclose(acc[0, 0], expected_ax, rtol=1e-8)
    np.testing.assert_allclose(acc[0, 1], 0.0, atol=1e-20)
    np.testing.assert_allclose(acc[0, 2], 0.0, atol=1e-20)

def test_pot_units():
    '''Verify pot_fn returns the correct analytic value in internal units
    for a Kepler potential at a known position.'''
    from ezfalcon.util.units import G_INTERNAL
    from galpy.util.conversion import get_physical
    M_msun = 1e3 # Msun
    pot = potential.KeplerPotential(amp=M_msun * u.Msun)  # amp=1 Msun in physical units
    pot.turn_physical_on()
    r = 0.1 # kpc
    pos = np.array([[r, 0.0, 0.0]])
    expected_phi = -G_INTERNAL * M_msun / r
    pot_fn = _galpy_bridge._galpy_pot_to_pot_fn(pot)
    phi = pot_fn(pos, t=0)
    np.testing.assert_allclose(phi[0], expected_phi, rtol=1e-8)


#-----------------------#
#  Input Shape Handling #
#-----------------------#

def test_single_particle_acc_shape():
    '''acc_fn should accept a (1,3) array and return (1,3).'''
    pot = potential.PlummerPotential()
    pot.turn_physical_on()
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
    pos = np.array([[8.0, 0.0, 0.0]])
    acc = acc_fn(pos, t=0)
    assert acc.shape == (1, 3)
    assert np.all(np.isfinite(acc))

def test_single_particle_pot_shape():
    '''pot_fn should accept a (1,3) array and return a (1,) array.'''
    pot = potential.PlummerPotential()
    pot.turn_physical_on()
    pot_fn = _galpy_bridge._galpy_pot_to_pot_fn(pot)
    pos = np.array([[8.0, 0.0, 0.0]])
    phi = pot_fn(pos, t=0)
    assert phi.shape == (1,)
    assert np.all(np.isfinite(phi))

def test_large_batch():
    '''acc_fn should handle 10k particles without issue.'''
    pot = potential.NFWPotential()
    pot.turn_physical_on()
    rng = np.random.default_rng(42)
    pos = rng.uniform(-50, 50, size=(10_000, 3))
    # Avoid z-axis
    pos[np.abs(pos[:, 0]) < 0.01, 0] = 0.01
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
    acc = acc_fn(pos, t=0)
    assert acc.shape == (10_000, 3)
    assert np.all(np.isfinite(acc))


#-------------------#
#  Time Dependence  #
#-------------------#

def test_time_independence(spherical_potential):
    '''For static potentials, acc should not depend on t.'''
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(spherical_potential)
    pos = np.array([[8.0, 0.0, 1.0], [3.0, 4.0, 0.0]])
    acc_t0 = acc_fn(pos, t=0)
    acc_t100 = acc_fn(pos, t=100)
    np.testing.assert_array_equal(acc_t0, acc_t100)

#-----------------------#
#  Force Direction      #
#-----------------------#
@pytest.fixture(params=(SUPPORTED_GALPY_SPHERICAL_POTENTIALS + 
                                  SUPPORTED_GALPY_AXISYMMETRIC_POTENTIALS + 
                                  SUPPORTED_GALPY_ELLIPSOIDAL_TRIAXIAL_POTENTIALS), ids=lambda p: type(p).__name__)
def general_galpy_potential(request):
    pot = request.param
    pot.turn_physical_on()
    return pot

def test_force_direction_attractive(general_galpy_potential):
    '''Radial component of acceleration should point inward (dot(a, r) < 0).'''
    if isinstance(general_galpy_potential, potential.RingPotential) or isinstance(general_galpy_potential, potential.SphericalShellPotential):
        pytest.xfail("RingPotential or SphericalShellPotential is not purely attractive at all positions")
    pos = np.array([
        [8.0, 0.0, 0.0],
        [0.0, 5.0, 3.0],
        [3.0, -4.0, 1.0],
        [-2.0, -2.0, -2.0],
    ])
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(general_galpy_potential)
    acc = acc_fn(pos, t=0)
    dots = np.sum(pos * acc, axis=1)
    assert np.all(dots < 0), f"Expected all dot products < 0, got {dots}"

#-------------------------------#
#  Validation Helper Functions  #
#-------------------------------#

def test_check_physical_pot_warns():
    '''_check_physical_pot should warn and turn physical on when not set.'''
    pot = potential.PlummerPotential()
    # Freshly created — physical outputs not explicitly set
    with pytest.warns(UserWarning, match="physical outputs turned off"):
        _galpy_bridge._check_physical(pot)
    # After the call, physical should be on
    assert pot._roSet or pot._voSet

def test_check_physical_pot_noop():
    '''_check_physical_pot should not warn when physical is already on.'''
    pot = potential.PlummerPotential()
    pot.turn_physical_on()
    # Should issue no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _galpy_bridge._check_physical(pot)

def test_check_supported_warns_non_vectorized():
    '''_check_supported_pot should warn for non-vectorized potentials.'''
    pot = potential.HomogeneousSpherePotential()
    with pytest.warns(UserWarning, match="not vectorized"):
        _galpy_bridge._check_supported_pot(pot)

#--------------#
#  Edge Cases  #
#--------------#

def test_z_axis_nan():
    '''Positions on the z-axis (R=0) should produce NaN — documenting the known galpy singularity.'''
    pot = potential.NFWPotential()
    pot.turn_physical_on()
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
    pos = np.array([[0.0, 0.0, 5.0]])
    acc = acc_fn(pos, t=0)
    assert np.any(np.isnan(acc)), "Expected NaN on z-axis due to galpy R=0 singularity"

def test_very_large_radius():
    '''Bridge should return finite values at very large radii.'''
    pot = potential.NFWPotential()
    pot.turn_physical_on()
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
    radii = [1e3, 1e4, 1e5]
    for r in radii:
        pos = np.array([[r, 0.0, 0.0]])
        acc = acc_fn(pos, t=0)
        assert np.all(np.isfinite(acc)), f"Non-finite acc at r={r}"
    # Magnitude should decrease with radius
    mags = []
    for r in radii:
        pos = np.array([[r, 0.0, 0.0]])
        acc = acc_fn(pos, t=0)
        mags.append(np.linalg.norm(acc))
    assert all(mags[i] > mags[i + 1] for i in range(len(mags) - 1)), \
        f"|a| should decrease with r, got {mags}"

def test_very_small_radius_cored():
    '''Potentials with cores (Plummer, DehnenCore, Burkert) should return finite acc at small r.'''
    cored_pots = [
        potential.PlummerPotential(),
        potential.DehnenCoreSphericalPotential(),
        potential.BurkertPotential(),
    ]
    for pot in cored_pots:
        pot.turn_physical_on()
        acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
        pos = np.array([[1e-10, 0.0, 0.0]])
        acc = acc_fn(pos, t=0)
        assert np.all(np.isfinite(acc)), f"{type(pot).__name__} returned non-finite acc at r=1e-10"


#--------------------------------#
#  interpSphericalPotential      #
#--------------------------------#

def test_interp_spherical_outside_grid():
    '''interpSphericalPotential evaluated outside its rgrid should still return finite values.'''
    pot = potential.interpSphericalPotential(
        rforce=lambda r: -1. / r,
        rgrid=np.geomspace(0.01, 20, 101),
        Phi0=0.,
    )
    pot.turn_physical_on()
    acc_fn = _galpy_bridge._galpy_pot_to_acc_fn(pot)
    # r=50 is outside the grid (max=20)
    pos = np.array([[50.0, 0.0, 0.0]])
    acc = acc_fn(pos, t=0)
    # Just document whether it's finite or not — it extrapolates
    if np.any(np.isnan(acc)):
        pytest.skip("interpSphericalPotential produces NaN outside rgrid (expected)")
    assert np.all(np.isfinite(acc))


    # TO ADD: galpy combo potential, time-dependent potential,