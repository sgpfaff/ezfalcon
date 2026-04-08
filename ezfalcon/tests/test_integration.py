import pytest
import numpy as np


# --- leapfrog functions ---------------------------------------------------------- #

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

# --- integration function ----------------------------------------------------------#

from ezfalcon.dynamics.integration import _integrate
from galpy.util.coords import cyl_to_rect, cyl_to_rect_vec
from ezfalcon.util import _galpy_pot_to_acc_fn, _galpy_pot_to_pot_fn
from ezfalcon.simulation import Sim
from galpy.potential import NFWPotential
from galpy.orbit import Orbit
import astropy.units as u

t_end = 10 * u.Myr # INCREASE TO 100 Myr FOR REAL TESTS
dt = 1/200*u.Myr
ts = np.arange(0, t_end.value + dt.value, dt.value) * u.Myr
pot = NFWPotential(amp=1e13*u.Msun, a=20*u.kpc)
R, vR, vT, z, vz, phi = 8., 0.1, 220.0, 0., 0.5, 0.
pot.turn_physical_on()
pos = np.array([cyl_to_rect(R, phi, z)])
vel = np.array([(cyl_to_rect_vec(vR, vT, vz, phi) * u.km/u.s).to(u.kpc/u.Myr).value])
acc_fn = _galpy_pot_to_acc_fn(pot)
pos_out, vel_out, ts_out, _, _ = _integrate(pos, vel, np.array([1.]), False, None, [acc_fn], 
                                      t_end.value, dt.value, dt.value, eps=0.0,
                                      return_self_potential=False, return_self_gravity=False)

pot_fn = _galpy_pot_to_pot_fn(pot)
nsnaps, npart = vel_out.shape[:2]
KE = 0.5 * np.sum(vel_out**2, axis=-1)  # (nsnaps, N)
PE = pot_fn(pos_out.reshape(-1, 3), t=0).reshape(nsnaps, npart)  # flatten → eval → reshape
E_out = ((KE + PE).squeeze()* u.kpc**2/u.Myr**2).to(u.km**2/u.s**2).value
Lz_out = (pos_out[...,0]*vel_out[...,1] - pos_out[...,1]*vel_out[...,0]).squeeze()

def test_output_times():
    np.testing.assert_allclose(ts_out, ts.value)

def test_conserves_energy():
    '''
    Test that total energy is conserved when integrating an orbit in an 
    external potential. Ensures that the integrator is working correctly.
    
    Does not test how the integrator handles self-gravity, which is tested separately.
    '''
    np.testing.assert_allclose(E_out, E_out[0], rtol=1e-10)

def test_conserves_angular_momentum():
    np.testing.assert_allclose(Lz_out, Lz_out[0], rtol=1e-10)

# --- test single orbit integration against galpy --------------------------------------------------------------------------------#

o_galpy = Orbit([R*u.kpc, vR*u.km/u.s, vT*u.km/u.s, z*u.kpc, vz*u.km/u.s, phi*u.rad])
o_galpy.integrate(ts, pot, method='leapfrog_c', dt=dt)
o_galpy.turn_physical_on()

def test_x_against_galpy():
    np.testing.assert_allclose(pos_out[...,0][...,0], o_galpy.x(ts).T, rtol=1e-8), "x position does not match galpy output."
    
def test_vx_against_galpy():
    np.testing.assert_allclose(vel_out[...,0][...,0], (o_galpy.vx(ts).T * u.km/u.s).to(u.kpc/u.Myr).value, rtol=1e-6), "Velocity does not match galpy output."

def test_y_against_galpy():
    np.testing.assert_allclose(pos_out[...,1][...,0], o_galpy.y(ts).T, rtol=1e-6), "y position does not match galpy output."

def test_vy_against_galpy():
    np.testing.assert_allclose(vel_out[...,1][...,0], (o_galpy.vy(ts).T * u.km/u.s).to(u.kpc/u.Myr).value, rtol=1e-6), "y velocity does not match galpy output."

def test_z_against_galpy():
    np.testing.assert_allclose(pos_out[...,2][...,0], o_galpy.z(ts).T, rtol=1e-6), "z position does not match galpy output."

def test_vz_against_galpy():
    np.testing.assert_allclose(vel_out[...,2][...,0], (o_galpy.vz(ts).T * u.km/u.s).to(u.kpc/u.Myr).value, rtol=1e-6), "z velocity does not match galpy output."

def test_energy_against_galpy():
    energy_galpy = o_galpy.E(ts, quantity=True).to(u.km**2/u.s**2).value
    np.testing.assert_allclose(E_out, energy_galpy, rtol=1e-6), "Energy does not match galpy output."

def test_Lz_against_galpy():
    Lz_galpy = o_galpy.Lz(ts, quantity=True).to(u.kpc*u.kpc/u.Myr).value
    np.testing.assert_allclose(Lz_out, Lz_galpy, rtol=1e-6), "Angular momentum does not match galpy output."

# --- vectorized integration --------------------------------------------------------------------------------#

from galpy.util.coords import rect_to_cyl, rect_to_cyl_vec

pos = np.array([[8., 0., 0.], [8., 0., 0.]])
vel = np.array([[0.1, 220.0, 0.5], [0.2, 220.0, 0.5]])
vel_internal = (vel * u.km/u.s).to(u.kpc/u.Myr).value
mass = np.array([1., 1.])
(pos_out, vel_out, ts_out, 
self_acc_out, self_pot_out)  = _integrate(pos, vel_internal, mass, False, None, [acc_fn], 
                                    t_end.value, dt.value, dt.value, eps=0.0,
                                    return_self_potential=True, return_self_gravity=True)
R, phi, z = rect_to_cyl(*pos.T)
vR, vT, vz = rect_to_cyl_vec(*vel.T, *pos.T)
o_galpy_many = Orbit([R*u.kpc, vR*u.km/u.s, vT*u.km/u.s, z*u.kpc, vz*u.km/u.s, phi*u.rad])
o_galpy_many.integrate(ts, pot, method='leapfrog_c', dt=dt)
o_galpy_many.turn_physical_on()

def test_integrate_multiple_orbits_output_pos_shape():
    '''
    Test that the output positions have the correct shape when integrating multiple orbits.
    '''
    assert pos_out.shape == (len(ts_out), 2, 3)

def test_integrate_multiple_orbits_output_vel_shape():
    '''
    Test that the output velocities have the correct shape when integrating multiple orbits.
    '''
    assert vel_out.shape == (len(ts_out), 2, 3)

def test_integrate_multiple_orbits_output_self_acc_shape():
    '''
    Test that the output self-accelerations have the correct shape when integrating multiple orbits.
    '''
    assert self_acc_out.shape == (len(ts_out), 2, 3)

def test_integrate_multiple_orbits_output_self_pot_shape():
    '''
    Test that the output self-potentials have the correct shape when integrating multiple orbits.
    '''
    assert self_pot_out.shape == (len(ts_out), 2)

# --- test against galpy for multiple orbits --------------------------------------------------------------------------------#

def test_integrate_multiple_orbits_x_match_galpy():
    '''
    Test that integrating multiple orbits in an external potential matches galpy x position output.
    '''
    np.testing.assert_allclose(pos_out[...,0], o_galpy_many.x(ts).T, rtol=1e-8), "x position does not match galpy output."

def test_integrate_multiple_orbits_vx_match_galpy():
    '''    
    Test that integrating multiple orbits in an external potential matches galpy vx velocity output.
    '''
    np.testing.assert_allclose(vel_out[...,0], (o_galpy_many.vx(ts).T * u.km/u.s).to(u.kpc/u.Myr).value, rtol=1e-6), "Velocity does not match galpy output."

def test_integrate_multiple_orbits_y_match_galpy():
    '''
    Test that integrating multiple orbits in an external potential matches galpy y position output.
    '''
    np.testing.assert_allclose(pos_out[...,1], o_galpy_many.y(ts).T, rtol=1e-6), "y position does not match galpy output."

def test_integrate_multiple_orbits_vy_match_galpy():
    '''
    Test that integrating multiple orbits in an external potential matches galpy vy velocity output.
    '''
    np.testing.assert_allclose(vel_out[...,1], (o_galpy_many.vy(ts).T * u.km/u.s).to(u.kpc/u.Myr).value, rtol=1e-6), "y velocity does not match galpy output."

def test_integrate_multiple_orbits_z_match_galpy():
    '''
    Test that integrating multiple orbits in an external potential matches galpy z position output.
    '''
    np.testing.assert_allclose(pos_out[...,2], o_galpy_many.z(ts).T, rtol=1e-6), "z position does not match galpy output."

def test_integrate_multiple_orbits_vz_match_galpy():
    '''
    Test that integrating multiple orbits in an external potential matches galpy vz velocity output.
    '''
    np.testing.assert_allclose(vel_out[...,2], (o_galpy_many.vz(ts).T * u.km/u.s).to(u.kpc/u.Myr).value, rtol=1e-6), "z velocity does not match galpy output."

# --- return option --------------------------------------------------------------------------------

def test_return_self_potential_only():
    '''
    Test that we can return just the self-potential without the self-acceleration.
    '''
    out = _integrate(pos, vel, mass, True, 'direct', [acc_fn], 
                                      t_end.value, dt.value*10, dt.value*10, eps=0.0,
                                      return_self_potential=True, return_self_gravity=False)
    assert out[-1].shape == (len(out[2]), mass.shape[0])
    assert out[-2] is None

def test_return_self_acceleration_only():
    '''
    Test that we can return just the self-acceleration without the self-potential.
    '''

    out = _integrate(pos, vel, mass, True, 'direct', [acc_fn], 
                                      t_end.value, dt.value*10, dt.value*10, eps=0.0,
                                      return_self_potential=False, return_self_gravity=True)
    assert out[-2].shape == (len(out[2]), mass.shape[0], 3)
    assert out[-1] is None

def test_return_self_acceleration_and_potential():
    '''
    Test that we can return both the self-acceleration and self-potential.
    '''
    out = _integrate(pos, vel, mass, True, 'direct', [acc_fn], 
                                      t_end.value, dt.value*10, dt.value*10, eps=0.0,
                                      return_self_potential=True, return_self_gravity=True)
    assert out[-2].shape == (len(out[2]), mass.shape[0], 3)
    assert out[-1].shape == (len(out[2]), mass.shape[0])

def test_return_neither_self_acceleration_nor_potential():
    '''
    Test that we can return neither the self-acceleration nor self-potential.
    '''
    out = _integrate(pos, vel, mass, True, 'direct', [acc_fn], 
                                      t_end.value, dt.value*10, dt.value*10, eps=0.0,
                                      return_self_potential=False, return_self_gravity=False)
    assert out[-1] is None
    assert out[-2] is None


# Array eps