"""Find how many substeps galpy's leapfrog uses internally."""
import numpy as np
import astropy.units as u
from galpy.potential import NFWPotential
from galpy.orbit import Orbit
from galpy.util.symplecticode import _leapfrog_estimate_step, leapfrog

pot = NFWPotential(amp=1e13*u.Msun, a=20*u.kpc)
R, vR, vT, z, vz, phi = 8., 0.1, 220.0, 0., 0.5, 0.

# galpy works in natural units internally (ro=8, vo=220)
from galpy.util.conversion import get_physical
phys = get_physical(pot)
ro = phys['ro']
vo = phys['vo']
to = ro / vo  # natural time unit in Gyr
print(f"galpy natural units: ro={ro} kpc, vo={vo} km/s, to={to:.6f} Gyr = {to*1000:.3f} Myr")

# Convert dt=1 Myr to galpy natural units
dt_phys = 1.0  # Myr
dt_nat = dt_phys / 1000.0 / to  # Myr -> Gyr -> natural
print(f"dt=1 Myr in natural units: {dt_nat:.8f}")

# Get galpy's internal representation of the orbit
o = Orbit([R*u.kpc, vR*u.km/u.s, vT*u.km/u.s, z*u.kpc, vz*u.km/u.s, phi*u.rad])

# In natural units: R and z in units of ro, velocities in units of vo
R_nat = R / ro
vR_nat = vR / vo
vT_nat = vT / vo
z_nat = z / ro
vz_nat = vz / vo

# Full orbit state: [R, vR, vT, z, vz, phi] -> q=[R, z, phi], p=[vR, vT, vz] (roughly)
# Actually galpy uses yo = [R, vR, vT, z, vz, phi] for full orbit
# Split: q = [R, z, phi], p = [vR, vT, vz]... but let me check the actual call

# The func for full orbit integration is _EOM which takes q and returns forces
# Let me just call _leapfrog_estimate_step directly with approximate values
qo = np.array([R_nat, z_nat, phi])
po = np.array([vR_nat, vT_nat, vz_nat])

# The actual leapfrog call uses yo = [vR, vT, vz, R, phi, z] for full orbit
# See integrateFullOrbit.py. Let me just measure empirically instead.

# Empirical: if galpy uses ndt substeps, then 
# dE_galpy ≈ dE_ezfalcon / ndt^2  (both 2nd order)
# So ndt ≈ sqrt(dE_ezfalcon / dE_galpy)

dE_ezfalcon = 1.610726e-04
dE_galpy = 1.560222e-07
ndt_est = np.sqrt(dE_ezfalcon / dE_galpy)
print(f"\nEstimated substeps per output: ndt ≈ {ndt_est:.1f}")
print(f"So galpy is using ~{ndt_est:.0f} internal steps per 1 Myr output interval")
print(f"Effective galpy dt ≈ {1.0/ndt_est:.4f} Myr = {1000.0/ndt_est:.1f} kyr")

# Let's also verify: if ezfalcon uses dt=1/32 Myr, does it match galpy?
print(f"\nIf we run ezfalcon with dt = 1/{int(ndt_est)} Myr:")
from galpy.util.coords import cyl_to_rect, cyl_to_rect_vec
from ezfalcon.simulation import Sim

pos_init = cyl_to_rect(R, phi, z)
vel_init = (cyl_to_rect_vec(vR, vT, vz, phi) * u.km/u.s).to(u.kpc/u.Myr).value

# Test with dt = 1/32 Myr
dt_fine = 1.0 / 32.0
o_f = Sim()
o_f.turn_self_gravity_off()
o_f.add_external_pot('nfw', pot)
o_f.add_particles('o', pos_init, vel_init, np.array([1.]))
o_f.run(100, dt_fine, 1.0, eps=0.0)
dE_fine = o_f.dE()
print(f"ezfalcon dt={dt_fine:.4f} Myr: max dE = {np.max(dE_fine[1:]):.6e}")
print(f"galpy: max dE = {dE_galpy:.6e}")
print(f"ratio: {np.max(dE_fine[1:])/dE_galpy:.2f}")
