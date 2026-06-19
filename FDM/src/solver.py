from galpy.util import conversion
import astropy.units as u
from astropy.constants import G, hbar, c
from .radial_functions import solve_radial_equation
import numpy as np
from .potential import HaloModel
from .helper_functions import create_r_array, Solution
from .wavefunction import EigenmodeBasis, WaveFunction
import time


def build_wavefunction(hm, r_array, m_boson, ro, vo, E_max):
    """Build the eigenmode basis, draw one realization of a_nlm, return wf."""
    eps = (hbar / (m_boson * vo*(u.km/u.s) * ro*(u.kpc))).decompose()
    r_nat = r_array / ro
    R_max = r_array[-1]


    # This solves for the radial wavefunctions and energies of the FDM halo,
    # and puts the solutions into the solutions_dict, which is indexed by (n,l).
    solutions_dict = {}
    num_solns = 0
    l=0 
    while True:
        E_nl, R_nl = solve_radial_equation(r_nat, l=l, V=lambda r: hm.potential(r), eps=eps, E_max=E_max)
        # print(np.trapz(R_nl[:,0]**2*r_nat**2, r_nat)) #orthonormality check    
        if len(E_nl) == 0:
            break
        else:
            for n in range(len(E_nl)):
                solutions_dict[(n,l)] = (E_nl[n], R_nl[:,n])
        print(f"Number of radial solutions for l = {l}:", R_nl.shape[1])
        num_solns += R_nl.shape[1]
        l+=1
    print(f"Total number of solutions found: {num_solns}")
    # Create and save solution object (save yet to be implemented)
    solution = Solution(solution_dict=solutions_dict, HaloModel=hm, r_array=r_array)

    eigenmode_dict = {}
    # Spherical harmonics to get the full 3D wavefuntion
    from scipy.special import sph_harm_y
    for n,l in solutions_dict.keys():
        for m in range(-l,l+1):
            Y_lm = lambda theta, phi, m=m,l=l: sph_harm_y(l, m, theta, phi)
            eigenmode_dict[n,l,m] = (solutions_dict[n,l][0], solutions_dict[n,l][1], Y_lm) #E_nl, R_nl, Y_lm


    # Calculate the Coefficients a_nlm = C*sqrt(f(E_nl))*N_nlm, where C is a normalization constant
    # f is the distribution function, and N_nlm is a random complex number from distribution
    # with unit variance.
    #
    # We fix C by matching the *ensemble-averaged* reconstructed density to the
    # target halo density, rather than the total enclosed mass. Averaging |psi|^2
    # over the unit-variance N_nlm and summing |Y_lm|^2 over m (which collapses to
    # (2l+1)/(4pi)) gives
    #     <rho(r)> = (m_boson / r0^3) * C^2 * sum_{n,l} f(E_nl) (2l+1)/(4pi) R_nl(r)^2.
    # Define rho_shape(r) as this sum at C = 1; then C^2 = rho_target / rho_shape.
    # We take the median of that ratio over a radial window where the granular
    # wave interference has averaged out in the mean: beyond a few de Broglie
    # wavelengths (the granule scale) and inside half the box (away from the
    # truncation edge).
    acc = np.zeros_like(r_nat)
    for (n, l), (E_nl, R_nl) in solutions_dict.items():
        f_E_nl = solution.HaloModel.df(E_nl)
        acc += f_E_nl * (2 * l + 1) / (4 * np.pi) * R_nl ** 2
    rho_shape = (m_boson / (ro * u.kpc) ** 3) * acc                  # = <rho_recon> / C^2
    rho_target = hm.density(r_nat) * conversion.dens_in_msolpc3(vo, ro) * u.Msun / u.pc ** 3
    ratio = (rho_target / rho_shape).decompose().value              # dimensionless

    # de Broglie wavelength in natural (r0) units: lambda_dB = hbar/(m_boson vo) / r0 = eps.
    lam_dB = float(eps)
    mask = (r_nat > 5 * lam_dB) & (r_nat < 0.3 * r_nat[-1])
    if not np.any(mask):
        mask = np.ones_like(r_nat, dtype=bool)
    ratio_win = ratio[mask]
    C = float(np.sqrt(np.median(ratio_win)))
    print(f"Density-ratio normalization: window "
          f"{5 * lam_dB * ro:.1f}-{0.3 * r_nat[-1] * ro:.1f} kpc ({mask.sum()} pts), "
          f"ratio median={np.median(ratio_win):.4g}, "
          f"16-84%=[{np.percentile(ratio_win, 16):.3g}, {np.percentile(ratio_win, 84):.3g}]")

    # Total enclosed mass at R_max, kept only as a diagnostic comparison for the |a_nlm|^2 sum.
    M_enclosed = hm.mass_enclosed(R_max/ro) * conversion.mass_in_msol(vo, ro) * u.Msun


    # Assembling the full a_nlm:
    for n,l,m in eigenmode_dict.keys():
        E_nl = eigenmode_dict[n,l,m][0]
        f_E_nl = solution.HaloModel.df(E_nl)
        N_nlm = (np.random.normal(0,1) + 1j*np.random.normal(0,1)) / np.sqrt(2)
        a_nlm = C * np.sqrt(f_E_nl) * N_nlm
        eigenmode_dict[n,l,m] = (eigenmode_dict[n,l,m][0], eigenmode_dict[n,l,m][1], eigenmode_dict[n,l,m][2], a_nlm) #E_nl, R_nl, Y_lm, a_nlm
    check_sum = 0
    for n,l,m in eigenmode_dict.keys():
        a_nlm = eigenmode_dict[n,l,m][3]
        check_sum += np.abs(a_nlm)**2
    print("Sum over |a_nlm|^2:", check_sum, "M/m_boson:", (M_enclosed/m_boson).decompose())

    # Assemble the full wavefunction via the EigenmodeBasis / WaveFunction classes.
    # psi(x, t) = sum_{n,l,m} a_nlm exp(-i E_nl t / eps) R_nl(r) Y_lm(theta, phi)
    basis = EigenmodeBasis.from_dict(eigenmode_dict, r_grid=r_nat, eps=float(eps.value))
    wf = WaveFunction.from_eigenmode_dict(basis, eigenmode_dict, m_boson=m_boson,
                                        r0=ro*u.kpc)
    

    #frequency vs wavenumber graph:
    import matplotlib.pyplot as plt
    omegas = []
    wavenums = []
    for n,l,m in eigenmode_dict.keys():
        E_nl = eigenmode_dict[n,l,m][0]
        omega_physical_nl = -(E_nl / eps * vo * u.km / u.s / (ro *u.kpc)).to_value(1/u.Gyr)
        wavenum = n + l
        omegas.append(omega_physical_nl)
        wavenums.append(wavenum)
    plt.scatter(wavenums, omegas, s=0.2)
    plt.xlabel("Wavenumber (n+l)")
    plt.ylabel(r"Frequency (Gyr$^-1$)")
    plt.show()

    return wf