"""Tests for the boundedness diagnostics, focused on the tidal Jacobi criterion.

The ``'energy'`` criterion is the self-binding baseline; ``'jacobi'`` adds the
linear tidal field. These tests pin down the tidal-tensor sign convention (the
one convention most likely to be silently wrong), verify the Roche/Jacobi
criterion strips particles the energy criterion wrongly keeps, and check the
input validation on the new parameter.
"""

import numpy as np
import pytest
import astropy.units as u
from galpy.potential import KeplerPotential

from tambora.dynamics.diagnostics import bound_mask
from tambora.dynamics.forces.external_force import TidalTensorGalpyForce
from tambora.dynamics.hooks import BoundednessHook, BoundKinematics
from tambora.tools.util.units import G_INTERNAL


# Point-mass host: analytic tidal tensor at galactocentric radius R along +x is
# diag(2GM/R^3, -GM/R^3, -GM/R^3) -- radial stretching, transverse compression.
M_GAL = 1e11      # Msun
R_GAL = 20.0      # kpc


def _kepler_force(R=R_GAL, M=M_GAL):
    pot = KeplerPotential(amp=M * u.Msun)
    pot.turn_physical_on()
    return TidalTensorGalpyForce(pot, center=[R, 0.0, 0.0])


def _cold_blob(m_cl, r_t, N=300, n_far=2, seed=0):
    """A compact cold cluster at (R,0,0) plus `n_far` tracers escaped along +/-x.

    Velocities are uniform (so ``v - v_com == 0``): boundedness is decided by
    potential alone, isolating the tidal term.
    """
    rng = np.random.default_rng(seed)
    core = rng.normal(scale=r_t / 4, size=(N, 3))
    far = np.array([[3 * r_t, 0.0, 0.0], [-3 * r_t, 0.0, 0.0]])[:n_far]
    pos = np.vstack([core, far]) + np.array([R_GAL, 0.0, 0.0])
    vel = np.zeros_like(pos)
    mass = np.full(len(pos), m_cl / len(pos))
    return pos, vel, mass


# --- sign convention (the thing most likely to be silently backwards) ---------

def test_tidal_tensor_sign_convention_kepler():
    """Largest eigenvalue is the +2GM/R^3 radial stretching mode."""
    force = _kepler_force()
    T = force.tidal_tensor(np.array([[R_GAL, 0.0, 0.0]]))
    lam = np.linalg.eigvalsh(T)                      # ascending
    np.testing.assert_allclose(lam[-1], 2 * G_INTERNAL * M_GAL / R_GAL**3, rtol=1e-6)
    np.testing.assert_allclose(lam[:2], -G_INTERNAL * M_GAL / R_GAL**3, rtol=1e-6)


def test_tidal_radius_matches_king_formula():
    """r_t = (G m / lam1)^(1/3) reproduces R (m / 2M)^(1/3)."""
    m_cl = 1e6
    lam1 = 2 * G_INTERNAL * M_GAL / R_GAL**3
    r_t = (G_INTERNAL * m_cl / lam1) ** (1 / 3)
    np.testing.assert_allclose(r_t, R_GAL * (m_cl / (2 * M_GAL)) ** (1 / 3), rtol=1e-12)


# --- the physics the criterion exists for -------------------------------------

def test_jacobi_strips_escaped_tracers_that_energy_keeps():
    """Tracers beyond the tidal radius are bound by 'energy' but not 'jacobi'."""
    m_cl = 1e6
    lam1 = 2 * G_INTERNAL * M_GAL / R_GAL**3
    r_t = (G_INTERNAL * m_cl / lam1) ** (1 / 3)
    force = _kepler_force()
    pos, vel, mass = _cold_blob(m_cl, r_t)

    b_energy = bound_mask(pos, vel, mass, eps=0.01, method='direct',
                          criterion='energy')
    b_jacobi = bound_mask(pos, vel, mass, eps=0.01, method='direct',
                          criterion='jacobi', tidal_force=force)

    # The far tracers (last two) are the escaped ones.
    assert b_energy[-2:].all()          # energy criterion is fooled
    assert not b_jacobi[-2:].any()      # jacobi correctly unbinds them
    assert b_jacobi.sum() < b_energy.sum()


def test_jacobi_robust_to_offset_com_from_stream():
    """A one-sided stream drags the global mean far off the core.

    Regression: seeding the tidal cut from the global COM strips *everything* on
    the first iteration (the sub-kpc r_t gate sees no particles near a COM that
    sits kpc away in the tail). Energy-first seeding must recover the core.
    """
    m_cl = 1e6
    lam1 = 2 * G_INTERNAL * M_GAL / R_GAL**3
    r_t = (G_INTERNAL * m_cl / lam1) ** (1 / 3)
    force = _kepler_force()

    rng = np.random.default_rng(1)
    N_core = 300
    core = rng.normal(scale=r_t / 4, size=(N_core, 3)) + np.array([R_GAL, 0.0, 0.0])
    # Long one-sided tail strung out to +x, well beyond r_t -- the mean lands here.
    tail_x = np.linspace(R_GAL + 1.0, R_GAL + 15.0, 300)
    tail = np.column_stack([tail_x, np.zeros_like(tail_x), np.zeros_like(tail_x)])
    pos = np.vstack([core, tail])
    vel = np.zeros_like(pos)
    vel[N_core:, 0] = 10.0                       # tail streaming outward -> energy-unbound
    mass = np.full(len(pos), m_cl / len(pos))

    assert abs(pos.mean(0)[0] - R_GAL) > r_t    # COM really is dragged off the core

    jm = bound_mask(pos, vel, mass, eps=0.01, method='direct',
                    criterion='jacobi', tidal_force=force)
    assert jm.sum() > 0                         # core survives (was 0 before the fix)
    assert jm[:N_core].mean() > 0.5             # mostly the core
    assert not jm[N_core:].any()               # none of the escaped tail


def test_jacobi_reduces_to_energy_when_tide_negligible():
    """With a very distant host the tide vanishes and the masks coincide."""
    m_cl = 1e6
    # r_t computed at the true R so the blob is compact relative to it.
    lam1 = 2 * G_INTERNAL * M_GAL / R_GAL**3
    r_t = (G_INTERNAL * m_cl / lam1) ** (1 / 3)
    pos, vel, mass = _cold_blob(m_cl, r_t, n_far=0)   # no escaped tracers

    # Put the host 1000x farther: tidal field ~1e-9 weaker, r_t enormous.
    far_force = _kepler_force(R=1000 * R_GAL)
    b_energy = bound_mask(pos, vel, mass, eps=0.01, method='direct',
                          criterion='energy')
    b_jacobi = bound_mask(pos, vel, mass, eps=0.01, method='direct',
                          criterion='jacobi', tidal_force=far_force)
    np.testing.assert_array_equal(b_energy, b_jacobi)


# --- input validation ---------------------------------------------------------

def test_jacobi_without_tidal_force_raises():
    pos, vel, mass = _cold_blob(1e6, 0.3, N=10, n_far=0)
    with pytest.raises(ValueError, match="requires a tidal_force"):
        bound_mask(pos, vel, mass, eps=0.01, method='direct', criterion='jacobi')


def test_unknown_criterion_raises():
    pos, vel, mass = _cold_blob(1e6, 0.3, N=10, n_far=0)
    with pytest.raises(ValueError, match="Unknown criterion"):
        bound_mask(pos, vel, mass, eps=0.01, method='direct', criterion='roche')


def test_purely_compressive_tensor_rejected():
    """A location with no stretching (positive) eigenvalue has no escape surface."""
    class _CompressiveTide:                 # negative-definite: fully compressive
        def tidal_tensor(self, pos):
            return -np.eye(3)
    pos, vel, mass = _cold_blob(1e6, 0.3, N=10, n_far=0)
    with pytest.raises(ValueError, match="no positive .*stretching"):
        bound_mask(pos, vel, mass, eps=0.01, method='direct',
                   criterion='jacobi', tidal_force=_CompressiveTide())


@pytest.mark.parametrize("hook_cls", [BoundednessHook, BoundKinematics])
def test_hook_jacobi_requires_tidal_force(hook_cls):
    with pytest.raises(ValueError, match="requires a tidal_force"):
        hook_cls("sat", eps=0.01, criterion='jacobi')
