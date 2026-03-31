import pytest
import numpy as np


#### Test leapfrog functions ####

from ezfalcon.dynamics.integration import leapfrog
def test_leapfrog_kick():
    vel = np.array([[1.0, 0.0, 0.0]])
    acc = np.array([[0.0, 1.0, 0.0]])
    dt = 1.0
    new_vel = leapfrog.leapfrog_kick(vel, acc, dt)
    assert np.allclose(new_vel, [[1.0, 1.0, 0.0]])

def test_leapfrog_drift():
    pos = np.array([[0.0, 0.0, 0.0]])
    vel = np.array([[1.0, 0.0, 0.0]])
    dt = 1.0
    new_pos = leapfrog.leapfrog_drift(pos, vel, dt)
    assert np.allclose(new_pos, [[1.0, 0.0, 0.0]])

#### Test integration function ####

from ezfalcon.dynamics.integration import _integrate
from galpy.util.coords import cyl_to_rect, cyl_to_rect_vec
from ezfalcon.util import _galpy_pot_to_acc_fn, _galpy_pot_to_pot_fn
from ezfalcon.simulation import Sim
from galpy.potential import NFWPotential
from galpy.orbit import Orbit
import astropy.units as u

t_end = 10 * u.Myr # INCREASE TO 100 Myr FOR REAL TESTS
dt = 1/128*u.Myr
ts = np.arange(0, t_end.value + dt.value, dt.value) * u.Myr
pot = NFWPotential(amp=1e13*u.Msun, a=20*u.kpc)
R, vR, vT, z, vz, phi = 8., 0.1, 220.0, 0., 0.5, 0.
pot.turn_physical_on()
pos = np.array([cyl_to_rect(R, phi, z)])
vel = np.array([(cyl_to_rect_vec(vR, vT, vz, phi) * u.km/u.s).to(u.kpc/u.Myr).value])
acc_fn = _galpy_pot_to_acc_fn(pot)
pos_out, vel_out, _, _, ts_out = _integrate(pos, vel, np.array([1.]), False, [acc_fn], t_end.value, dt.value, dt.value, eps=0.0)

pot_fn = _galpy_pot_to_pot_fn(pot)
nsnaps, npart = vel_out.shape[:2]
KE = 0.5 * np.sum(vel_out**2, axis=-1)  # (nsnaps, N)
PE = pot_fn(pos_out.reshape(-1, 3), t=0).reshape(nsnaps, npart)  # flatten → eval → reshape
E_out = ((KE + PE).squeeze()* u.kpc**2/u.Myr**2).to(u.km**2/u.s**2).value
Lz_out = (pos_out[...,0]*vel_out[...,1] - pos_out[...,1]*vel_out[...,0]).squeeze()

def test_output_times():
    np.testing.assert_allclose(ts_out, ts.value)

def test_conserves_energy():
    np.testing.assert_allclose(E_out, E_out[0], rtol=1e-8)

def test_conserves_angular_momentum():
    np.testing.assert_allclose(Lz_out, Lz_out[0], rtol=1e-8)

### Test against galpy

o_galpy = Orbit([R*u.kpc, vR*u.km/u.s, vT*u.km/u.s, z*u.kpc, vz*u.km/u.s, phi*u.rad])
o_galpy.integrate(ts, pot, method='leapfrog_c', dt=dt)
o_galpy.turn_physical_on()

def test_x_against_galpy():
    np.testing.assert_allclose(pos_out[...,0][...,0], o_galpy.x(ts).T, rtol=1e-9), "x position does not match galpy output."
    
def test_vx_against_galpy():
    np.testing.assert_allclose(vel_out[...,0][...,0], (o_galpy.vx(ts).T * u.km/u.s).to(u.kpc/u.Myr).value, rtol=1e-9), "Velocity does not match galpy output."

def test_y_against_galpy():
    np.testing.assert_allclose(pos_out[...,1][...,0], o_galpy.y(ts).T, rtol=1e-9), "y position does not match galpy output."

def test_vy_against_galpy():
    np.testing.assert_allclose(vel_out[...,1][...,0], (o_galpy.vy(ts).T * u.km/u.s).to(u.kpc/u.Myr).value, rtol=1e-9), "y velocity does not match galpy output."

def test_z_against_galpy():
    np.testing.assert_allclose(pos_out[...,2][...,0], o_galpy.z(ts).T, rtol=1e-9), "z position does not match galpy output."

def test_vz_against_galpy():
    np.testing.assert_allclose(vel_out[...,2][...,0], (o_galpy.vz(ts).T * u.km/u.s).to(u.kpc/u.Myr).value, rtol=1e-9), "z velocity does not match galpy output."

def test_energy_against_galpy():
    energy_galpy = o_galpy.E(ts, quantity=True).to(u.km**2/u.s**2).value
    np.testing.assert_allclose(E_out, energy_galpy, rtol=1e-9), "Energy does not match galpy output."

def test_Lz_against_galpy():
    Lz_galpy = o_galpy.Lz(ts, quantity=True).to(u.kpc*u.kpc/u.Myr).value
    np.testing.assert_allclose(Lz_out, Lz_galpy, rtol=1e-9), "Angular momentum does not match galpy output."