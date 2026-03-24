import numpy as np
from galpy import potential
from galpy.util.coords import rect_to_cyl
import astropy.units as u
from ezfalcon.util import _galpy_bridge

g = np.linspace(-100, 100, 5)
FULL = np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
R, PHI, Z = rect_to_cyl(*FULL.T*u.kpc)

def check(name, pot, unvectorized=False):
    pot.turn_physical_on()
    ez = _galpy_bridge._galpy_pot_to_pot_fn(pot)(FULL, t=0)
    if unvectorized:
        gp = np.array([
            pot(Ri, Zi, phi=Pi, t=0, quantity=True).to(u.kpc**2/u.Myr**2).value
            for Ri, Zi, Pi in zip(R, Z, PHI)
        ])
    else:
        gp = pot(R, Z, phi=PHI, t=0, quantity=True).to(u.kpc**2/u.Myr**2).value

    # Check for NaN mismatches
    nan_ez = np.isnan(ez)
    nan_gp = np.isnan(gp)
    nan_mismatch = np.sum(nan_ez != nan_gp)

    # Compare non-NaN values
    mask = ~nan_ez & ~nan_gp
    if mask.any():
        diff = np.abs(ez[mask] - gp[mask])
        reldiff = diff / (np.abs(gp[mask]) + 1e-30)
        max_abs = np.max(diff)
        max_rel = np.max(reldiff)
    else:
        max_abs = max_rel = 0

    print(f"{name}:")
    print(f"  NaN count — ez: {nan_ez.sum()}, gp: {nan_gp.sum()}, mismatch: {nan_mismatch}")
    print(f"  max abs diff: {max_abs:.3e}")
    print(f"  max rel diff: {max_rel:.3e}")
    for tol in [1e-10, 1e-13, 1e-14, 1e-15]:
        print(f"  rtol={tol:.0e}: {np.allclose(ez, gp, rtol=tol, equal_nan=True)}")
    print()

check("BurkertPotential", potential.BurkertPotential())
check("RingPotential", potential.RingPotential())
check("DoubleExponentialDiskPotential", potential.DoubleExponentialDiskPotential(), unvectorized=True)
check("RazorThinExponentialDiskPotential", potential.RazorThinExponentialDiskPotential(), unvectorized=True)
