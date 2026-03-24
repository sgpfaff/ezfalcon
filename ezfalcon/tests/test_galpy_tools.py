import pytest

from galpy.df import isotropicHernquistdf
from galpy.potential import HernquistPotential
from ezfalcon.util import galpydfsampler, galpy_orbit_to_ezfalcon
import numpy as np
import astropy.units as u

@pytest.fixture
def hernquist_df():
    pot = HernquistPotential()
    pot.turn_physical_on()
    df = isotropicHernquistdf(pot=pot)
    df.turn_physical_on()
    return df

def test_sampler_shapes(hernquist_df):
    pos, vel, masses = galpydfsampler(hernquist_df, n=100, m_total=1e6)
    assert pos.shape == (100, 3)
    assert vel.shape == (100, 3)
    assert masses.shape == (100,)

def test_sampler_masses(hernquist_df):
    pos, vel, masses = galpydfsampler(hernquist_df, n=200, m_total=1e6)
    np.testing.assert_allclose(masses, 1e6 / 200)

def test_sampler_center_offset(hernquist_df):
    pos, vel, masses = galpydfsampler(hernquist_df, n=500, m_total=1e6,
                                       center_pos=[100, 0, 0])
    assert np.mean(pos[:, 0]) > 50  # shifted well away from origin

def test_orbit_to_ezfalcon_vel_units():
    from galpy.orbit import Orbit
    # Orbit at R=8 kpc, vR=0, vT=220 km/s, z=0, vz=0, phi=0
    o = Orbit([8., 0., 220., 0., 0., 0.], ro=8., vo=220.)
    o.turn_physical_on()
    pos, vel = galpy_orbit_to_ezfalcon(o)
    expected_vT_kpcmyr = (220 * u.km/u.s).to(u.kpc/u.Myr).value
    assert np.max(np.abs(vel)) < 1.0  # kpc/Myr are small numbers (~0.2)
    assert np.max(np.abs(vel)) > 0.01  # but not zero

def test_orbit_to_ezfalcon_shapes():
    from galpy.orbit import Orbit
    o = Orbit.from_name("LMC")
    o.turn_physical_on()
    pos, vel = galpy_orbit_to_ezfalcon(o)
    assert pos.shape == (1, 3)
    assert vel.shape == (1, 3)