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

t_end = 0.01 * u.Gyr # INCREASE TO 0.1 Gyr FOR REAL TESTS
dt = 5e-6 * u.Gyr
ts = np.arange(0, t_end.value + dt.value, dt.value) * u.Gyr
pot = NFWPotential(amp=1e13*u.Msun, a=20*u.kpc)
R, vR, vT, z, vz, phi = 8., 0.1, 220.0, 0., 0.5, 0.
pot.turn_physical_on()
pos = np.array([cyl_to_rect(R, phi, z)])
vel = np.array([(cyl_to_rect_vec(vR, vT, vz, phi) * u.km/u.s).to(u.kpc/u.Gyr).value])
acc_fn = _galpy_pot_to_acc_fn(pot)
pos_out, vel_out, ts_out, _, _ = _integrate(pos, vel, np.array([1.]), False, None, [acc_fn], 
                                      t_end.value, dt.value, dt.value, eps=0.0,
                                      return_self_potential=False, return_self_gravity=False)

pot_fn = _galpy_pot_to_pot_fn(pot)
nsnaps, npart = vel_out.shape[:2]
KE = 0.5 * np.sum(vel_out**2, axis=-1)  # (nsnaps, N)
PE = pot_fn(pos_out.reshape(-1, 3), t=0).reshape(nsnaps, npart)  # flatten → eval → reshape
E_out = ((KE + PE).squeeze()* u.kpc**2/u.Gyr**2).to(u.km**2/u.s**2).value
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
    np.testing.assert_allclose(vel_out[...,0][...,0], (o_galpy.vx(ts).T * u.km/u.s).to(u.kpc/u.Gyr).value, rtol=1e-6), "Velocity does not match galpy output."

def test_y_against_galpy():
    np.testing.assert_allclose(pos_out[...,1][...,0], o_galpy.y(ts).T, rtol=1e-6), "y position does not match galpy output."

def test_vy_against_galpy():
    np.testing.assert_allclose(vel_out[...,1][...,0], (o_galpy.vy(ts).T * u.km/u.s).to(u.kpc/u.Gyr).value, rtol=1e-6), "y velocity does not match galpy output."

def test_z_against_galpy():
    np.testing.assert_allclose(pos_out[...,2][...,0], o_galpy.z(ts).T, rtol=1e-6), "z position does not match galpy output."

def test_vz_against_galpy():
    np.testing.assert_allclose(vel_out[...,2][...,0], (o_galpy.vz(ts).T * u.km/u.s).to(u.kpc/u.Gyr).value, rtol=1e-6), "z velocity does not match galpy output."

def test_energy_against_galpy():
    energy_galpy = o_galpy.E(ts, quantity=True).to(u.km**2/u.s**2).value
    np.testing.assert_allclose(E_out, energy_galpy, rtol=1e-6), "Energy does not match galpy output."

def test_Lz_against_galpy():
    Lz_galpy = o_galpy.Lz(ts, quantity=True).to(u.kpc*u.kpc/u.Gyr).value
    np.testing.assert_allclose(Lz_out, Lz_galpy, rtol=1e-6), "Angular momentum does not match galpy output."

# --- vectorized integration --------------------------------------------------------------------------------#

from galpy.util.coords import rect_to_cyl, rect_to_cyl_vec

pos = np.array([[8., 0., 0.], [8., 0., 0.]])
vel = np.array([[0.1, 220.0, 0.5], [0.2, 220.0, 0.5]])
vel_internal = (vel * u.km/u.s).to(u.kpc/u.Gyr).value
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
    np.testing.assert_allclose(vel_out[...,0], (o_galpy_many.vx(ts).T * u.km/u.s).to(u.kpc/u.Gyr).value, rtol=1e-6), "Velocity does not match galpy output."

def test_integrate_multiple_orbits_y_match_galpy():
    '''
    Test that integrating multiple orbits in an external potential matches galpy y position output.
    '''
    np.testing.assert_allclose(pos_out[...,1], o_galpy_many.y(ts).T, rtol=1e-6), "y position does not match galpy output."

def test_integrate_multiple_orbits_vy_match_galpy():
    '''
    Test that integrating multiple orbits in an external potential matches galpy vy velocity output.
    '''
    np.testing.assert_allclose(vel_out[...,1], (o_galpy_many.vy(ts).T * u.km/u.s).to(u.kpc/u.Gyr).value, rtol=1e-6), "y velocity does not match galpy output."

def test_integrate_multiple_orbits_z_match_galpy():
    '''
    Test that integrating multiple orbits in an external potential matches galpy z position output.
    '''
    np.testing.assert_allclose(pos_out[...,2], o_galpy_many.z(ts).T, rtol=1e-6), "z position does not match galpy output."

def test_integrate_multiple_orbits_vz_match_galpy():
    '''
    Test that integrating multiple orbits in an external potential matches galpy vz velocity output.
    '''
    np.testing.assert_allclose(vel_out[...,2], (o_galpy_many.vz(ts).T * u.km/u.s).to(u.kpc/u.Gyr).value, rtol=1e-6), "z velocity does not match galpy output."

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

# --- time-dependent external potential --------------------------------------------------------- #

from galpy.potential import DehnenSmoothWrapperPotential
from ezfalcon.util.units import KMS_TO_KPCGYR

def test_time_dependent_potential_matches_galpy():
    '''
    Integrate an orbit in a DehnenSmoothWrapperPotential (NFW that grows from
    zero to full strength) and compare the result against galpy's own orbit
    integration.  This verifies that the integration time is correctly
    forwarded to external force functions.
    '''
    # Static NFW wrapped in a smooth growth envelope.
    nfw_td = NFWPotential(amp=1e13 * u.Msun, a=20 * u.kpc)
    smooth_pot = DehnenSmoothWrapperPotential(pot=nfw_td, tform=0., tsteady=0.02*u.Gyr)
    smooth_pot.turn_physical_on()

    # Integration parameters
    td_t_end = 0.01 * u.Gyr
    td_dt = 5e-6 * u.Gyr
    td_ts = np.arange(0, td_t_end.value + td_dt.value, td_dt.value) * u.Gyr

    # Initial conditions: particle at R=8 kpc with some velocity
    td_R, td_vR, td_vT, td_z, td_vz, td_phi = 8., 0.1, 220.0, 0., 0.5, 0.
    td_pos = np.array([cyl_to_rect(td_R, td_phi, td_z)])
    td_vel = np.array([(cyl_to_rect_vec(td_vR, td_vT, td_vz, td_phi) * u.km / u.s).to(u.kpc / u.Gyr).value])

    # ezfalcon integration
    td_acc_fn = _galpy_pot_to_acc_fn(smooth_pot)
    td_pos_out, td_vel_out, td_ts_out, _, _ = _integrate(
        td_pos, td_vel, np.array([1.0]),
        False, None, [td_acc_fn],
        td_t_end.value, td_dt.value, td_dt.value,
        eps=0.0, return_self_potential=False, return_self_gravity=False,
    )

    # galpy reference integration
    td_o = Orbit([td_R * u.kpc, td_vR * u.km / u.s, td_vT * u.km / u.s,
                  td_z * u.kpc, td_vz * u.km / u.s, td_phi * u.rad])
    td_o.integrate(td_ts, smooth_pot, method='leapfrog_c', dt=td_dt)
    td_o.turn_physical_on()

    np.testing.assert_allclose(
        td_pos_out[..., 0].squeeze(), td_o.x(td_ts).T.squeeze(),
        rtol=1e-8,
        err_msg="x position does not match galpy for time-dependent potential.",
    )
    np.testing.assert_allclose(
        td_pos_out[..., 1].squeeze(), td_o.y(td_ts).T.squeeze(),
        rtol=1e-6,
        err_msg="y position does not match galpy for time-dependent potential.",
    )
    np.testing.assert_allclose(
        td_pos_out[..., 2].squeeze(), td_o.z(td_ts).T.squeeze(),
        rtol=1e-6,
        err_msg="z position does not match galpy for time-dependent potential.",
    )


def test_time_dependent_potential_differs_from_static():
    '''
    Verify that a time-dependent potential actually produces different
    trajectories than a static one.  This catches the case where t=0 is
    silently passed — the growing potential at t=0 has zero force, so the
    particle would drift in a straight line.
    '''
    nfw_static = NFWPotential(amp=1e13 * u.Msun, a=20 * u.kpc)
    nfw_growing = DehnenSmoothWrapperPotential(pot=nfw_static, tform=0., tsteady=0.02*u.Gyr)
    nfw_growing.turn_physical_on()

    td2_t_end = 0.01 * u.Gyr
    td2_dt = 5e-6 * u.Gyr

    td2_R, td2_phi, td2_z = 8., 0., 0.
    td2_vR, td2_vT, td2_vz = 0.1, 220.0, 0.5
    td2_pos = np.array([cyl_to_rect(td2_R, td2_phi, td2_z)])
    td2_vel = np.array([(cyl_to_rect_vec(td2_vR, td2_vT, td2_vz, td2_phi) * u.km / u.s).to(u.kpc / u.Gyr).value])

    # Integrate in growing potential
    acc_growing = _galpy_pot_to_acc_fn(nfw_growing)
    pos_growing, _, _, _, _ = _integrate(
        td2_pos.copy(), td2_vel.copy(), np.array([1.0]),
        False, None, [acc_growing],
        td2_t_end.value, td2_dt.value, td2_dt.value,
        eps=0.0, return_self_potential=False, return_self_gravity=False,
    )

    # Integrate in static potential
    nfw_static.turn_physical_on()
    acc_static = _galpy_pot_to_acc_fn(nfw_static)
    pos_static, _, _, _, _ = _integrate(
        td2_pos.copy(), td2_vel.copy(), np.array([1.0]),
        False, None, [acc_static],
        td2_t_end.value, td2_dt.value, td2_dt.value,
        eps=0.0, return_self_potential=False, return_self_gravity=False,
    )

    # Trajectories must differ (growing starts weaker)
    final_diff = np.linalg.norm(pos_growing[-1] - pos_static[-1])
    assert final_diff > 1e-6, (
        f"Growing and static potentials produced the same trajectory (diff={final_diff}). "
        "Time is likely not being forwarded to the external force function."
    )


from ezfalcon.util._galpy_bridge import _galpy_pot_to_pot_fn

def test_time_dependent_potential_energy_matches_galpy():
    '''
    Compare the total energy trajectory E(t) = KE + PE from ezfalcon
    against galpy for a time-dependent potential.

    In a time-dependent potential total energy is NOT conserved — it
    changes as dE/dt = dPhi/dt.  The correct check is that both
    integrators agree on the energy trajectory, confirming that the
    time argument is threaded correctly through the force evaluation.
    '''
    nfw_e = NFWPotential(amp=1e13 * u.Msun, a=20 * u.kpc)
    smooth_e = DehnenSmoothWrapperPotential(pot=nfw_e, tform=0., tsteady=2. * u.Gyr)
    smooth_e.turn_physical_on()

    # Short enough integration with fine timestep for close agreement
    e_t_end = 0.1 * u.Gyr
    e_dt = 1e-3 * u.Gyr
    e_dt_out = e_dt
    e_ts = np.arange(0, e_t_end.value + e_dt.value, e_dt.value) * u.Gyr

    e_R, e_vR, e_vT, e_z, e_vz, e_phi = 8., 0.1, 220.0, 0., 0.5, 0.
    e_pos = np.array([cyl_to_rect(e_R, e_phi, e_z)])
    e_vel = np.array([(cyl_to_rect_vec(e_vR, e_vT, e_vz, e_phi)
                       * u.km / u.s).to(u.kpc / u.Gyr).value])

    # --- ezfalcon integration ---
    e_acc_fn = _galpy_pot_to_acc_fn(smooth_e)
    e_pot_fn = _galpy_pot_to_pot_fn(smooth_e)
    e_pos_out, e_vel_out, e_ts_out, _, _ = _integrate(
        e_pos, e_vel, np.array([1.0]),
        False, None, [e_acc_fn],
        e_t_end.value, e_dt.value, e_dt_out.value,
        eps=0.0, return_self_potential=False, return_self_gravity=False,
    )
    nsnaps = e_pos_out.shape[0]
    ez_KE = 0.5 * np.sum(e_vel_out ** 2, axis=-1).squeeze()         # (nsnaps,)
    ez_PE = np.array([e_pot_fn(e_pos_out[i], t=e_ts_out[i])
                      for i in range(nsnaps)]).squeeze()              # (nsnaps,)
    ez_E = ((ez_KE + ez_PE) * u.kpc ** 2 / u.Gyr ** 2).to(u.km ** 2 / u.s ** 2).value

    # --- galpy reference ---
    e_o = Orbit([e_R * u.kpc, e_vR * u.km / u.s, e_vT * u.km / u.s,
                 e_z * u.kpc, e_vz * u.km / u.s, e_phi * u.rad])
    e_o.integrate(e_ts, smooth_e, method='leapfrog_c', dt=e_dt)
    e_o.turn_physical_on()
    galpy_E = e_o.E(e_ts, pot=smooth_e, quantity=True).to(u.km ** 2 / u.s ** 2).value

    # Energy trajectories should agree closely
    np.testing.assert_allclose(
        ez_E, galpy_E, rtol=1e-6,
        err_msg="Energy trajectory does not match galpy for time-dependent potential.",
    )

    # Sanity: energy should NOT be constant (since potential is time-dependent)
    energy_range = np.ptp(ez_E)
    assert energy_range > 1.0, (
        f"Energy barely changed (range={energy_range:.2e} km^2/s^2) — "
        "time-dependent potential may not be evolving."
    )
