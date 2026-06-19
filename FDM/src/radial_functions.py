import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt 

def diagonal_functions(r_array, l, V, eps):
    '''
    r_array: array of radii at which to evaluate the functions
    l: angular momentum quantum number
    V: potential function, V(r)
    eps: = hbar / (m_b * v_o *r_0), proportional to the de Broglie wavelength of the FDM particle.
    '''
    delta_r = r_array[1] - r_array[0]
    return eps**2/delta_r**2 + V(r_array) + eps**2 * l*(l+1)/(2*r_array**2)

def off_diagonal_functions(r_array, eps):
    '''
    r_array: array of radii at which to evaluate the functions
    eps: = hbar / (m_b * v_o *r_0), proportional to the de Broglie wavelength of the FDM particle.
    '''
    delta_r = r_array[1] - r_array[0]
    return -eps**2/(2*delta_r**2) * np.ones_like(r_array[:-1])

def solve_radial_equation(r_array, l, V, eps, E_max=np.inf):
    '''
    Returns the eigenvalues and eigenvectors (energy E_nl and radial functions f_nl)
    of from equation [14] of Lin et al.  We assume a linearly spaced grid in r.

    r_array: array of radii at which to evaluate the functions (linearly spaced)
    l: angular momentum quantum number
    V: potential function, V(r)
    E_max: maximum energy to consider (should typically set this to V(R_max))

    Returns:
    E_nl: the energy eigenvalues (energy per particle mass, in units of v_o**2)
    R_nl: the radial functions.
    '''
    d = diagonal_functions(r_array, l, V, eps)
    e = off_diagonal_functions(r_array, eps)

    E_nl,u_nl = la.eigh_tridiagonal(d,e, select='v', select_range=(-np.inf, E_max))
    u_nl = u_nl/(r_array[1] - r_array[0])**0.5  # Normalize the eigenvectors
    R_nl = u_nl/r_array[:,None]
    return E_nl,R_nl


# V  = lambda r: 0.5 * 1**2 * r**2


# R_max = 20.0
# N = 16000

# dr = R_max / N
# r  = (np.arange(N) + 0.5) * dr   # half-integer grid: r[0] = dr/2

# plt.plot(r, V(r))
# plt.show()


# # E_nl,R_nl = solve_radial_equation(r, l=1, V=lambda r: 0.5 * r**2, E_max=12.0)
# E_nl,R_nl = solve_radial_equation(r, l=1, V=V, E_max=V(R_max), eps=1)


# print(R_nl.shape)
# for i in range(R_nl.shape[1]):
#     plt.plot(r, R_nl[:,i], label=f'n_r={i}')

# plt.xlabel('r')
# plt.ylabel('R_nl(r)')
# plt.title('Radial Functions for 3D Harmonic Oscillator (l=1)')
# plt.legend()
# plt.show()


# """
# Test suite for the radial Schrodinger eigensolver.
 
# Solves the dimensionless equation
 
#     -(eps^2 / 2) u''(r) + V(r) u(r) + eps^2 * l(l+1)/(2 r^2) u(r) = E u(r)
 
# with u(0) = u(R_max) = 0, on a uniform grid (half-integer offset to avoid r=0).
 
# Tests:
#     1.  Harmonic oscillator eigenvalues  (algebraic correctness)
#     2.  Eigenfunction orthonormality     (normalization)
#     3.  Node counting                    (sign / Sturm-Liouville)
#     4.  Virial theorem                   (r^2 measure correctness)
#     5.  Epsilon scaling                  (validates eps on BOTH kinetic
#                                           and centrifugal terms)
 
# Run:  python test_radial_suite.py
# """
 
# import numpy as np
# import scipy.linalg as la
 
 
# # ─────────────────────────────────────────────────────────────────────────────
# # Solver  (mirrors the version from our discussion)
# # ─────────────────────────────────────────────────────────────────────────────
 
# def solve_radial_equation(r_array, l, V, epsilon, E_max=None):
#     """
#     Solve  -(eps^2/2) u'' + [V(r) + eps^2 l(l+1)/(2r^2)] u = E u
#     on a uniform grid with u(0) = u(R_max) = 0.
 
#     Returns
#     -------
#     E_nl : (K,) eigenvalues, ascending
#     R_nl : (N, K) radial functions, R_nl[:, k] = u_nl(r)/r,
#            normalised so  sum_i |R_nl[i,k]|^2 * r[i]^2 * dr = 1
#     u_nl : (N, K) radial functions u_nl = r * R_nl,
#            normalised so  sum_i |u_nl[i,k]|^2 * dr = 1
#     """
#     if epsilon <= 0:
#         raise ValueError(f"epsilon must be positive, got {epsilon}")
#     if r_array[0] <= 0:
#         raise ValueError("r_array must not include r=0")
#     dr = r_array[1] - r_array[0]
#     if not np.allclose(np.diff(r_array), dr, rtol=1e-10):
#         raise ValueError("r_array must be uniformly spaced")
 
#     d = (epsilon**2 / dr**2
#          + V(r_array)
#          + epsilon**2 * l * (l + 1) / (2.0 * r_array**2))
#     e = np.full(len(r_array) - 1, -epsilon**2 / (2.0 * dr**2))
 
#     if E_max is None:
#         E_nl, u_nl = la.eigh_tridiagonal(d, e)
#     else:
#         E_min = np.min(d) - abs(np.min(d)) - 1.0
#         if E_min >= E_max:
#             raise ValueError(f"E_min={E_min:.3g} >= E_max={E_max:.3g}")
#         E_nl, u_nl = la.eigh_tridiagonal(
#             d, e, select='v', select_range=(E_min, E_max)
#         )
 
#     # scipy returns l2-normalised eigenvectors:  sum |u_i|^2 = 1
#     # we want integral |u|^2 dr = 1  =>  divide by sqrt(dr)
#     u_nl = u_nl / np.sqrt(dr)
#     R_nl = u_nl / r_array[:, None]
 
#     return E_nl, R_nl, u_nl
 
 
# # ─────────────────────────────────────────────────────────────────────────────
# # Helper: standard harmonic oscillator grid
# # ─────────────────────────────────────────────────────────────────────────────
 
# def ho_grid(N=8000, R_max=20.0):
#     """Half-integer grid: r[0] = dr/2,  r[-1] = R_max - dr/2."""
#     dr = R_max / N
#     r = (np.arange(N) + 0.5) * dr
#     return r, dr
 
 
# def V_ho(r, omega=1.0):
#     return 0.5 * omega**2 * r**2
 
 
# # ─────────────────────────────────────────────────────────────────────────────
# # TEST 1.  Harmonic oscillator eigenvalues
# # ─────────────────────────────────────────────────────────────────────────────
# #
# # Exact spectrum for V = omega^2 r^2 / 2, eps = 1, hbar = m = 1:
# #     E(n_r, l) = omega * (2*n_r + l + 3/2)
# # Errors at l = 0 are dr^2-limited (centrifugal barrier absent, wavefunction
# # probes the coarse grid near r=0); l >= 1 errors are at machine precision.
 
# def test_1_ho_eigenvalues():
#     print("=" * 70)
#     print("TEST 1  —  Harmonic oscillator eigenvalues")
#     print("=" * 70)
 
#     r, dr = ho_grid()
#     omega = 1.0
#     failures = []
 
#     for l in range(4):
#         E_nl, _, _ = solve_radial_equation(r, l, V_ho, epsilon=1.0, E_max=12.0)
#         tol = 2e-3 if l == 0 else 1e-4    # l=0 is dr^2 limited
 
#         for n_r in range(3):
#             E_exact = omega * (2*n_r + l + 1.5)
#             if n_r >= len(E_nl):
#                 failures.append(f"l={l} n_r={n_r}: missing eigenvalue")
#                 continue
#             err = abs(E_nl[n_r] - E_exact) / E_exact
#             tag = "PASS" if err < tol else "FAIL"
#             if err >= tol:
#                 failures.append(
#                     f"l={l} n_r={n_r}: err={err:.2e} (tol={tol:.0e})"
#                 )
#             print(f"  l={l}  n_r={n_r}:  E_exact={E_exact:6.3f}  "
#                   f"E_num={E_nl[n_r]:9.6f}  err={err:.2e}  [{tag}]")
 
#     print()
#     return _report(failures)
 
 
# # ─────────────────────────────────────────────────────────────────────────────
# # TEST 2.  Eigenfunction orthonormality
# # ─────────────────────────────────────────────────────────────────────────────
# #
# # For fixed l, eigenfunctions of distinct n must be orthonormal under
# #     <u_{nl} | u_{n'l}> = integral u_{nl}(r) u_{n'l}(r) dr  =  delta_{nn'}
# # i.e. the matrix u_nl.T @ u_nl * dr must equal the identity.
# # Diagonal alone tests normalization; off-diagonals catch broadcasting /
# # inner-product bugs that pure diagonal checks would miss.
 
# def test_2_orthonormality():
#     print("=" * 70)
#     print("TEST 2  —  Eigenfunction orthonormality (u-basis)")
#     print("=" * 70)
 
#     r, dr = ho_grid()
#     failures = []
 
#     for l in [0, 2]:                              # spot-check two l values
#         _, _, u_nl = solve_radial_equation(r, l, V_ho, epsilon=1.0, E_max=8.0)
#         K = u_nl.shape[1]
#         if K < 3:
#             failures.append(f"l={l}: only {K} states found, need >= 3")
#             continue
 
#         # Restrict to the first few states for a cleaner printout
#         K_use = min(K, 4)
#         M = (u_nl[:, :K_use].T @ u_nl[:, :K_use]) * dr
#         I = np.eye(K_use)
 
#         diag_err    = np.max(np.abs(np.diag(M) - 1.0))
#         offdiag_err = np.max(np.abs(M - np.diag(np.diag(M))))
 
#         tag = "PASS" if (diag_err < 1e-8 and offdiag_err < 1e-8) else "FAIL"
#         if tag == "FAIL":
#             failures.append(
#                 f"l={l}: diag_err={diag_err:.2e}, off-diag={offdiag_err:.2e}"
#             )
#         print(f"  l={l}  K={K_use}:  max|diag-1|={diag_err:.2e}  "
#               f"max|off-diag|={offdiag_err:.2e}  [{tag}]")
 
#     print()
#     return _report(failures)
 
 
# # ─────────────────────────────────────────────────────────────────────────────
# # TEST 3.  Node counting
# # ─────────────────────────────────────────────────────────────────────────────
# #
# # Sturm-Liouville theorem: the k-th eigenfunction (k = 0, 1, 2, ...) for
# # fixed l has exactly k interior nodes (sign changes).  A wrong-sign kinetic
# # operator or a misordered spectrum can produce "OK" eigenvalues with the
# # wrong node structure.
 
# def count_interior_nodes(u, rel_threshold=1e-6):
#     """
#     Count physical sign changes in u.
 
#     Naive np.diff(np.sign(u)) is brittle: in the classically forbidden
#     region (e.g. the Gaussian tail of an HO eigenstate) u decays to values
#     like 1e-40, and floating-point noise produces spurious sign flips.
#     We mask out samples below `rel_threshold * max|u|` (treating them as
#     zero) and count transitions between actual + and - regions only.
#     """
#     thresh = rel_threshold * np.max(np.abs(u))
#     masked = np.where(np.abs(u) < thresh, 0.0, np.sign(u))
 
#     count = 0
#     prev = 0
#     for s in masked:
#         if s != 0 and prev != 0 and s != prev:
#             count += 1
#         if s != 0:
#             prev = s
#     return count
 
 
# def test_3_node_counting():
#     print("=" * 70)
#     print("TEST 3  —  Node counting (Sturm-Liouville)")
#     print("=" * 70)
 
#     r, _ = ho_grid()
#     failures = []
 
#     for l in [0, 1, 2]:
#         _, _, u_nl = solve_radial_equation(r, l, V_ho, epsilon=1.0, E_max=10.0)
#         K = min(u_nl.shape[1], 4)
#         for k in range(K):
#             n_nodes = count_interior_nodes(u_nl[:, k])
#             tag = "PASS" if n_nodes == k else "FAIL"
#             if n_nodes != k:
#                 failures.append(f"l={l} k={k}: nodes={n_nodes}, expected {k}")
#             print(f"  l={l}  k={k}:  nodes={n_nodes}  expected={k}  [{tag}]")
#         print()
 
#     return _report(failures)
 
 
# # ─────────────────────────────────────────────────────────────────────────────
# # TEST 4.  Virial theorem
# # ─────────────────────────────────────────────────────────────────────────────
# #
# # For V = omega^2 r^2 / 2 (harmonic oscillator),  <T> = <V> = E/2.
# # We compute <V> = integral |R|^2 V(r) r^2 dr directly; if the r^2 Jacobian
# # is wrong anywhere, <V> will be far from E/2 even when eigenvalues look fine.
# # The harmonic oscillator is well suited here because its eigenfunctions
# # decay rapidly (Gaussian envelope), so R_max truncation error is negligible.
 
# def test_4_virial():
#     print("=" * 70)
#     print("TEST 4  —  Virial theorem: <V> = E/2 for harmonic oscillator")
#     print("=" * 70)
 
#     r, dr = ho_grid()
#     E_nl, R_nl, _ = solve_radial_equation(r, 0, V_ho, epsilon=1.0, E_max=10.0)
 
#     failures = []
#     K = min(R_nl.shape[1], 4)
#     Vvals = V_ho(r)
 
#     for k in range(K):
#         # <V> = integral |R|^2 V(r) r^2 dr
#         V_exp = np.sum(R_nl[:, k]**2 * Vvals * r**2) * dr
#         ratio = V_exp / (E_nl[k] / 2.0)
#         err = abs(ratio - 1.0)
 
#         tag = "PASS" if err < 2e-3 else "FAIL"
#         if err >= 2e-3:
#             failures.append(f"k={k}: <V>/<E/2>={ratio:.5f}")
#         print(f"  k={k}:  E={E_nl[k]:.5f}  <V>={V_exp:.5f}  "
#               f"<V>/(E/2)={ratio:.5f}  [{tag}]")
 
#     print()
#     return _report(failures)
 
 
# # ─────────────────────────────────────────────────────────────────────────────
# # TEST 5.  Epsilon scaling
# # ─────────────────────────────────────────────────────────────────────────────
# #
# # For V = omega^2 r^2 / 2, the dimensionless equation
# #     -(eps^2/2) u'' + eps^2 l(l+1)/(2r^2) u + omega^2 r^2 / 2 u = E u
# # admits the substitution r = sqrt(eps) * r_tilde, which rescales it to the
# # eps=1 form.  This implies eigenvalues scale linearly with eps:
# #     E_nl(eps) = eps * (2 n_r + l + 3/2)
# #
# # This test is THE check that epsilon is wired into both the kinetic stencil
# # AND the centrifugal term consistently — if eps were missing from the
# # centrifugal term, the scaling would hold for l=0 but break for l >= 1.
 
# def test_5_eps_scaling():
#     print("=" * 70)
#     print("TEST 5  —  Epsilon scaling (eigenvalues ∝ eps for HO)")
#     print("=" * 70)
 
#     failures = []
 
#     for eps in [1.0, 0.5, 0.25]:
#         # Grid: ground-state width scales like sqrt(eps), so shrink R_max with eps.
#         # Doing this keeps the relative resolution constant and the grid honest.
#         R_max = 20.0 * np.sqrt(eps)
#         N     = 8000
#         dr    = R_max / N
#         r     = (np.arange(N) + 0.5) * dr
 
#         # Need E_max large enough to capture the states we test:
#         # E_max(eps) = eps * (2*2 + 3 + 1.5) = 8.5*eps  for n_r<=2, l<=3
#         E_nl_l0, _, _ = solve_radial_equation(r, 0, V_ho, eps, E_max=20.0 * eps)
#         E_nl_l2, _, _ = solve_radial_equation(r, 2, V_ho, eps, E_max=20.0 * eps)
 
#         print(f"  eps = {eps}")
#         for n_r in range(3):
#             for l, E_nl in [(0, E_nl_l0), (2, E_nl_l2)]:
#                 E_exact = eps * (2*n_r + l + 1.5)
#                 tol     = 5e-3 if l == 0 else 1e-3   # l=0 still dr^2 limited
#                 err     = abs(E_nl[n_r] - E_exact) / E_exact
#                 tag     = "PASS" if err < tol else "FAIL"
#                 if err >= tol:
#                     failures.append(
#                         f"eps={eps} l={l} n_r={n_r}: err={err:.2e}"
#                     )
#                 print(f"    l={l} n_r={n_r}:  E_exact={E_exact:8.5f}  "
#                       f"E_num={E_nl[n_r]:9.6f}  err={err:.2e}  [{tag}]")
#         print()
 
#     return _report(failures)
 
 
# # ─────────────────────────────────────────────────────────────────────────────
# # Reporting
# # ─────────────────────────────────────────────────────────────────────────────
 
# def _report(failures):
#     if not failures:
#         print("  ✓ all checks passed in this test")
#     else:
#         print(f"  ✗ {len(failures)} failure(s):")
#         for f in failures:
#             print(f"      • {f}")
#     print()
#     return len(failures) == 0
 
 
# def run_all():
#     results = [
#         ("HO eigenvalues",    test_1_ho_eigenvalues()),
#         ("orthonormality",    test_2_orthonormality()),
#         ("node counting",     test_3_node_counting()),
#         ("virial theorem",    test_4_virial()),
#         ("epsilon scaling",   test_5_eps_scaling()),
#     ]
 
#     print("=" * 70)
#     print("SUMMARY")
#     print("=" * 70)
#     for name, ok in results:
#         tag = "PASS" if ok else "FAIL"
#         print(f"  [{tag}]  {name}")
#     n_fail = sum(1 for _, ok in results if not ok)
#     print()
#     if n_fail == 0:
#         print("ALL TESTS PASSED")
#     else:
#         print(f"{n_fail} TEST(S) FAILED")
#     print("=" * 70)
#     return n_fail == 0
 
 
# if __name__ == "__main__":
#     import sys
#     sys.exit(0 if run_all() else 1)



