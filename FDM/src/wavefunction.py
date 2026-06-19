"""
FDM wavefunction assembled from spherically-symmetric eigenmodes.

Implements the construction of Dalal, Bovy, Hui, Li (arXiv:2011.13141) Eqns. (1)
and (3):

    psi(x, t) = sum_{n,l,m}  a_nlm  exp(-i E_nl t / eps)  R_nl(r)  Y_lm(theta, phi)

where (r, theta, phi) are spherical coordinates (r = |x|, theta = polar angle
from +z, phi = azimuthal) and a_nlm = C sqrt(f(E_nl)) N_nlm with N_nlm
independent unit-variance complex Gaussians.

External API (psi / density) uses **galpy cylindrical** coordinates (R, z, phi):
    R   = sqrt(x^2 + y^2)
    z   = z
    phi = atan2(y, x)
internally converted to spherical via r = sqrt(R^2 + z^2), theta = arccos(z/r).

Conventions: everything is in galpy natural units. lengths in units of r0,
energies in units of v0^2, time in r0/v0, and eps = hbar/(m_b v0 r0) plays the
role of hbar/m.
"""

from dataclasses import dataclass
import numpy as np
from scipy.special import sph_harm_y


@dataclass
class EigenmodeBasis:
    """Flattened, vectorisable view of an eigenmode set.

    Parallel arrays of length N_modes label each (n_r, l, m) state. The radial
    functions R live on a shared natural-unit grid r_grid.
    """

    r_grid: np.ndarray   # (n_r,) natural-unit grid the radial solver used
    eps: float           # dimensionless hbar/(m_b v0 r0)
    n_idx: np.ndarray    # (N,) int — radial quantum number
    l_idx: np.ndarray    # (N,) int — angular momentum
    m_idx: np.ndarray    # (N,) int — magnetic quantum number
    E: np.ndarray        # (N,) float — eigenvalues in units of v0^2
    R: np.ndarray        # (N, n_r) float — radial functions on r_grid

    @classmethod
    def from_dict(cls, eigenmode_dict, r_grid, eps):
        """Build from a dict keyed by (n, l, m) with values (E, R, ...).

        Any extra entries beyond (E, R) in each value (e.g. a stored Y_lm or
        a_nlm tuple element) are ignored — this class holds only the basis.
        """
        keys = list(eigenmode_dict.keys())
        if not keys:
            raise ValueError("eigenmode_dict is empty")
        n = np.array([k[0] for k in keys], dtype=int)
        l = np.array([k[1] for k in keys], dtype=int)
        m = np.array([k[2] for k in keys], dtype=int)
        E = np.array([eigenmode_dict[k][0] for k in keys], dtype=float)
        R = np.stack([np.asarray(eigenmode_dict[k][1], dtype=float) for k in keys])
        return cls(np.asarray(r_grid, dtype=float), float(eps), n, l, m, E, R)

    @property
    def n_modes(self):
        return self.E.size


def _interp_radial(R, r_grid, r):
    """Batch-interpolate R[i, :] at radii r.

    R       : (N, n_r)
    r_grid  : (n_r,) sorted ascending
    r       : scalar or array, any shape S

    Returns shape (N,) if r is scalar, else (N, *S).
    Values outside r_grid are clipped to the endpoint values.
    """
    r_arr = np.atleast_1d(np.asarray(r, dtype=float))
    flat = r_arr.ravel()
    # np.interp doesn't broadcast across the leading axis of R, so loop in
    # the mode dimension. The mode count (~1e3-1e5) is the bottleneck only
    # for very large modal sets; n_eval is usually small.
    out = np.empty((R.shape[0], flat.size), dtype=R.dtype)
    for i in range(R.shape[0]):
        out[i] = np.interp(flat, r_grid, R[i])
    if np.isscalar(r) or np.ndim(r) == 0:
        return out[:, 0]
    return out.reshape((R.shape[0],) + r_arr.shape)


def _cyl_to_spherical(R, z):
    """Convert cylindrical (R, z) to spherical (r, theta).

    R, z may be scalars or arrays of the same shape.
    Uses theta = arctan2(R, z), so theta = 0 along +z axis (galpy convention).
    The r = 0 case gives theta = 0 (the arctan2 picks 0 for (0,0) inputs).
    """
    r = np.hypot(R, z)
    theta = np.arctan2(R, z)
    return r, theta


class WaveFunction:
    """One realization of the FDM wavefunction on a fixed EigenmodeBasis.

    Holds the per-mode complex coefficients a_nlm and exposes two flavors of
    evaluator, both in **galpy cylindrical coordinates** (R, z, phi):

      - psi(R, z, phi, t) / density(R, z, phi, t) — scattered-point. R, z,
        phi are arrays of the same shape; one psi per element. Use for
        test-particle work, scattered samples, etc.
      - psi_grid(R, z, phi, t) / density_grid(R, z, phi, t) — fast outer-product
        grid evaluator. R, z, phi are three 1-D arrays; the result is on the
        outer-product grid of shape (len(R), len(z), len(phi)). Exploits the
        spherical-harmonic separability of Y_lm to avoid materialising any
        (N_modes × N_pts) intermediate; uses one (m, l) lexsort cache and two
        np.add.reduceat calls under the hood.

    Note on units. The radial functions in `basis.R` live on the natural-unit
    grid r_nat = r_phys / r0, with ∫|R|² r_nat² dr_nat = 1. The corresponding
    "physical" eigenmode is F_nlm(r_phys) = R_nl(r_phys/r0) Y_lm / r0^(3/2),
    so |psi_phys|² = |psi_natural|² / r0³ . Pass `r0` (an astropy Quantity
    with length units) to get density()/density_grid() in proper mass/volume
    units.
    """

    def __init__(self, basis: EigenmodeBasis, a, m_boson, r0=None):
        self.basis = basis
        self.a = np.asarray(a, dtype=complex)
        if self.a.shape != (basis.n_modes,):
            raise ValueError(
                f"a has shape {self.a.shape}, expected ({basis.n_modes},)"
            )
        self.m_boson = m_boson
        self.r0 = r0    # astropy Quantity with length units, or None

    @classmethod
    def from_eigenmode_dict(cls, basis, eigenmode_dict, m_boson, r0=None, a_index=3):
        """Construct a WaveFunction using a_nlm already stored in eigenmode_dict.

        Pulls eigenmode_dict[(n,l,m)][a_index] for each mode in the basis's
        index order (basis.n_idx/l_idx/m_idx), so it works regardless of how
        the dict was iterated when the basis was built.

        If the coefficients are astropy Quantities, their `.value` is taken;
        units are not preserved on the wavefunction (psi/density use r0 and
        m_boson to attach physical units at the output).
        """
        keys = list(zip(basis.n_idx.tolist(), basis.l_idx.tolist(), basis.m_idx.tolist()))
        raw = [eigenmode_dict[k][a_index] for k in keys]
        # strip astropy Quantity units if present
        a = np.array([getattr(x, "value", x) for x in raw], dtype=complex)
        return cls(basis, a, m_boson, r0=r0)

    def psi(self, R, z, phi, t=0.0):
        """Evaluate ψ at the galpy cylindrical point (R, z, phi) at time t.

        R, z, phi may be scalars or arrays of a common broadcast-compatible
        shape S. All in natural units (R = R_phys / r0, z = z_phys / r0,
        phi in radians, t in r0/v0).

        Returns a complex array of shape S (scalar if all inputs were scalars).
        """
        R_arr = np.asarray(R, dtype=float)
        z_arr = np.asarray(z, dtype=float)
        ph_arr = np.asarray(phi, dtype=float)
        if R_arr.shape != z_arr.shape or R_arr.shape != ph_arr.shape:
            raise ValueError(
                f"R, z, phi must share shape; got {R_arr.shape}, "
                f"{z_arr.shape}, {ph_arr.shape}"
            )

        # Galpy cylindrical -> spherical (r, theta) needed by the eigenmode basis.
        r_arr, th_arr = _cyl_to_spherical(R_arr, z_arr)

        basis = self.basis
        # Radial: shape (N,) if scalar else (N, *S)
        R_at = _interp_radial(basis.R, basis.r_grid, r_arr)

        # Angular: sph_harm_y broadcasts over modes if we add a trailing axis
        # to (theta, phi). We want output shape (N, *S).
        if r_arr.ndim == 0:
            Y = sph_harm_y(basis.l_idx, basis.m_idx, float(th_arr), float(ph_arr))
        else:
            l_b = basis.l_idx.reshape((-1,) + (1,) * r_arr.ndim)
            m_b = basis.m_idx.reshape((-1,) + (1,) * r_arr.ndim)
            Y = sph_harm_y(l_b, m_b, th_arr[None, ...], ph_arr[None, ...])

        # Time-phase: exp(-i E t / eps), shape (N,)
        phase = np.exp(-1j * basis.E * (t / basis.eps))

        # Combine: sum over modes
        coeff = self.a * phase                       # (N,)
        if r_arr.ndim == 0:
            return np.sum(coeff * R_at * Y)
        return np.einsum("i,i...,i...->...", coeff, R_at, Y)

    def density(self, R, z, phi, t=0.0):
        """Mass density at the galpy cylindrical point (R, z, phi, t).

        If r0 was provided at construction, returns m_boson · |psi|² / r0³
        in proper mass/volume units. Otherwise returns m_boson · |psi|²
        (mass per natural-unit volume — multiply by 1/r0³ yourself to get
        physical density).
        """
        rho_natural = self.m_boson * np.abs(self.psi(R, z, phi, t)) ** 2
        if self.r0 is None:
            return rho_natural
        return rho_natural / self.r0 ** 3

    def total_particles(self):
        """Sum |a_nlm|² — equals M_total/m_boson in expectation."""
        return float(np.sum(np.abs(self.a) ** 2))

    # ─────────────────────────────────────────────────────────────────────────
    # Fast 3-D grid evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def _build_lm_sort_cache(self):
        """Lazy-build sorting structures needed by psi_grid / density_grid.

        After the first call this object caches:
          self._sort_perm     : (N_modes,) lex-sort permutation by (m, l)
                                 — m primary key, l secondary. This is the
                                 right ordering for *both* reductions below.
          self._group_starts_lm : (n_unique_lm,) start indices of contiguous
                                 (l, m) blocks in the sorted mode axis.
          self._l_unique, self._m_unique : (n_unique_lm,) labels of those
                                 blocks. m_unique is non-decreasing.
          self._m_group_starts  : (n_m_unique,) start indices of contiguous
                                 m blocks in the *shrunken* (n_unique_lm,)
                                 axis — for the m-reduceat in step 3.
          self._m_unique_values : (n_m_unique,) the distinct azimuthal orders.
          self._R_sorted, self._E_sorted, self._a_sorted : basis arrays
                                 reordered by self._sort_perm.

        None of these depend on (R, z, phi, t), so they're computed once
        per WaveFunction.
        """
        if getattr(self, "_sort_perm", None) is not None:
            return
        basis = self.basis

        # m primary key, l secondary (np.lexsort uses the LAST tuple element
        # as the primary key).
        sort_perm = np.lexsort((basis.l_idx, basis.m_idx))
        l_sorted = basis.l_idx[sort_perm]
        m_sorted = basis.m_idx[sort_perm]

        # (l, m) group starts in the sorted mode axis.
        is_new_lm = np.empty(l_sorted.size, dtype=bool)
        is_new_lm[0] = True
        is_new_lm[1:] = (l_sorted[1:] != l_sorted[:-1]) | (m_sorted[1:] != m_sorted[:-1])
        group_starts_lm = np.flatnonzero(is_new_lm)
        l_unique = l_sorted[group_starts_lm]
        m_unique = m_sorted[group_starts_lm]   # non-decreasing because m was primary

        # m group starts in the (n_unique_lm,) axis (i.e. on m_unique itself).
        is_new_m = np.empty(m_unique.size, dtype=bool)
        is_new_m[0] = True
        is_new_m[1:] = m_unique[1:] != m_unique[:-1]
        m_group_starts = np.flatnonzero(is_new_m)
        m_unique_values = m_unique[m_group_starts]

        # Sorted basis arrays. R_sorted is the big one — contiguous-copy it so
        # subsequent fancy-indexing along the radial axis is cache-friendly.
        R_sorted = np.ascontiguousarray(basis.R[sort_perm])
        E_sorted = basis.E[sort_perm]
        a_sorted = self.a[sort_perm]

        self._sort_perm = sort_perm
        self._group_starts_lm = group_starts_lm
        self._l_unique = l_unique
        self._m_unique = m_unique
        self._m_group_starts = m_group_starts
        self._m_unique_values = m_unique_values
        self._R_sorted = R_sorted
        self._E_sorted = E_sorted
        self._a_sorted = a_sorted

    def psi_grid(self, R, z, phi, t=0.0, chunk_n_R=None):
        """ψ on the outer-product 3-D grid {R_i} × {z_j} × {φ_k}.

        Parameters
        ----------
        R, z, phi : 1-D ndarrays
            Cylindrical (galpy) coordinate axes, in natural units (R/r0, z/r0)
            and radians. The full evaluation grid is the outer product.
        t : float
            Time in natural units (r0 / v0).
        chunk_n_R : int or None
            How many R values to process per inner pass. Controls peak memory.
            None = process all of R in one pass. Pick a small value if the
            (N_modes, chunk_n_R, n_z) buffer doesn't fit in RAM.

        Returns
        -------
        psi : ndarray of shape (len(R), len(z), len(phi)), complex.

        Algorithm (no approximations — three nested contractions of the exact
        eigenmode sum):

          ψ(R_i, z_j, φ_k, t) = Σ_n A_n R_n(r_ij) Y_{l_n m_n}(θ_ij, φ_k)
                            = Σ_m exp(im φ_k) Σ_l c_{lm} P_l^|m|(cos θ_ij)
                                                  · Σ_{n_r} A_{n_r l m} R_{n_r l}(r_ij)

        where r_ij = √(R_i²+z_j²), θ_ij = arctan2(R_i, z_j). The three Σs
        become two reduceat calls and one matmul-shaped einsum after the
        (m, l) lexsort cache is built.
        """
        R = np.asarray(R, dtype=float)
        z = np.asarray(z, dtype=float)
        phi = np.asarray(phi, dtype=float)
        if R.ndim != 1 or z.ndim != 1 or phi.ndim != 1:
            raise ValueError("R, z, phi must each be 1-D arrays")
        n_R, n_z, n_phi = R.size, z.size, phi.size

        self._build_lm_sort_cache()
        basis = self.basis
        R_sorted = self._R_sorted
        E_sorted = self._E_sorted
        a_sorted = self._a_sorted
        group_starts_lm = self._group_starts_lm
        l_unique = self._l_unique
        m_unique = self._m_unique
        m_group_starts = self._m_group_starts
        m_unique_values = self._m_unique_values

        # Uniform r-grid linear-interp helpers
        r_grid = basis.r_grid
        dr = r_grid[1] - r_grid[0]
        r_min = r_grid[0]
        r_max_grid = r_grid[-1]
        n_grid = r_grid.size

        # Time-evolved coefficient.
        A = a_sorted * np.exp(-1j * E_sorted * (t / basis.eps))  # (N_modes,)

        # Precompute the θ-dependent Y_lm factor for the full (R, z) plane:
        #     P_at[k, i, j] = c_{l_k m_k} · P_{l_k}^{|m_k|}(cos θ_ij)
        # We use sph_harm_y(l, m, θ, 0), which returns exactly this real factor
        # (the e^{im·0} = 1 contribution is trivial). Doing it once outside the
        # R-chunk loop avoids re-evaluating Legendre functions per chunk.
        r_full = np.sqrt(R[:, None] ** 2 + z[None, :] ** 2)         # (n_R, n_z)
        theta_full = np.arctan2(R[:, None], z[None, :])             # (n_R, n_z)
        P_at_full = sph_harm_y(
            l_unique[:, None, None],
            m_unique[:, None, None],
            theta_full[None, :, :],
            0.0,
        ).real  # imaginary part is ~0 to machine precision

        # Radial-interp prep on the full plane (cheap).
        idx_f_full = (r_full - r_min) / dr
        idx_lo_full = np.clip(np.floor(idx_f_full).astype(int), 0, n_grid - 2)
        frac_full = idx_f_full - idx_lo_full
        outside_full = r_full > r_max_grid

        # exp(i m φ) lookup table: (n_m_unique, n_phi)
        E_phi = np.exp(1j * m_unique_values[:, None] * phi[None, :])

        if chunk_n_R is None or chunk_n_R >= n_R:
            chunk_n_R = n_R

        psi = np.empty((n_R, n_z, n_phi), dtype=complex)

        for i0 in range(0, n_R, chunk_n_R):
            i1 = min(i0 + chunk_n_R, n_R)
            idx_lo = idx_lo_full[i0:i1]            # (chunk, n_z)
            frac = frac_full[i0:i1]                # (chunk, n_z)
            outside = outside_full[i0:i1]
            P_at = P_at_full[:, i0:i1, :]          # (n_unique_lm, chunk, n_z)

            # Vectorised linear interp of R on the uniform grid.
            # Fancy-indexing R_sorted of shape (N, n_grid) with idx_lo (chunk, n_z)
            # produces shape (N, chunk, n_z).
            R_at = R_sorted[:, idx_lo] * (1.0 - frac) + R_sorted[:, idx_lo + 1] * frac
            if outside.any():
                R_at[:, outside] = 0.0

            # n_r reduction: T[k, i, j] = Σ_{n_r in group k} A_n R_n(r_ij)
            AR = A[:, None, None] * R_at                           # (N, chunk, n_z) complex
            T = np.add.reduceat(AR, group_starts_lm, axis=0)       # (n_unique_lm, chunk, n_z)
            del AR, R_at

            # Multiply by the angular (l, m) factor c_lm·P_l^|m|(cos θ).
            S = T * P_at                                           # (n_unique_lm, chunk, n_z)
            del T

            # m reduction: S_m[mu, i, j] = Σ_{k with m_k=mu} S[k, i, j].
            # m_unique is non-decreasing (m was primary sort key), so m groups
            # are already contiguous on axis 0.
            S_m = np.add.reduceat(S, m_group_starts, axis=0)       # (n_m_unique, chunk, n_z)
            del S

            # Final φ contraction: ψ[i, j, k] = Σ_m S_m[m, i, j] · E_phi[m, k].
            psi[i0:i1] = np.einsum("mij,mk->ijk", S_m, E_phi, optimize=True)

        return psi

    def density_grid(self, R, z, phi, t=0.0, chunk_n_R=None):
        """Mass density on the outer-product cylindrical grid (R, z, phi).

        Equivalent to ``m_boson · |psi_grid|² / r0³`` but evaluated in one
        call. See :py:meth:`psi_grid` for the algorithm and the meaning of
        ``chunk_n_R``.
        """
        psi_arr = self.psi_grid(R, z, phi, t=t, chunk_n_R=chunk_n_R)
        rho_natural = self.m_boson * np.abs(psi_arr) ** 2
        if self.r0 is None:
            return rho_natural
        return rho_natural / self.r0 ** 3

    # ─────────────────────────────────────────────────────────────────────────
    # Polar-grid + time evaluator (for galpy multipole quadrature)
    # ─────────────────────────────────────────────────────────────────────────
    #
    # galpy's MultipoleExpansionPotential projects the density on a *polar*
    # (r, θ) grid × φ, with a separate time axis. That is exactly the
    # separable structure psi_grid exploits, so the methods below reuse the
    # same (m, l) lexsort cache but take θ as an explicit scalar and carry a
    # time axis. The θ-independent radial/temporal core D[group, r, t] is
    # cached so galpy's repeated per-θ calls build it only once.

    def _interp_R_sorted_uniform(self, r):
        """Linear-interp the (m, l)-sorted radial functions onto radii ``r``.

        The source grid ``basis.r_grid`` is uniform (the radial solver uses a
        linearly spaced grid), so a single index arithmetic handles arbitrary
        query radii ``r`` (1-D). Radii beyond the outer grid edge get zero
        (the halo support ends there); in normal use the smallest query equals
        ``r_grid[0]`` so there is no inward extrapolation.

        Returns (N_modes, len(r)).
        """
        self._build_lm_sort_cache()
        r_grid = self.basis.r_grid
        dr = r_grid[1] - r_grid[0]
        n_grid = r_grid.size
        r = np.asarray(r, dtype=float)
        idx_f = (r - r_grid[0]) / dr
        idx_lo = np.clip(np.floor(idx_f).astype(int), 0, n_grid - 2)
        frac = idx_f - idx_lo
        Rs = self._R_sorted
        R_at = Rs[:, idx_lo] * (1.0 - frac) + Rs[:, idx_lo + 1] * frac
        outside = r > r_grid[-1]
        if outside.any():
            R_at[:, outside] = 0.0
        return R_at

    def _modal_radial_temporal(self, r_1d, t_1d, chunk_n_r=None):
        """θ-independent radial/temporal core, grouped by (l, m).

            D[k, i, j] = Σ_{n_r in group k} a_{n_r l m} R_{n_r l}(r_i)
                                            · exp(-i E_{n_r l} t_j / eps)

        Shape (n_unique_lm, len(r_1d), len(t_1d)), complex. The (l, m) groups
        match ``self._group_starts_lm`` / ``self._l_unique`` / ``self._m_unique``.

        The result is cached on the *values* of (r_1d, t_1d); galpy reuses the
        same rgrid/tgrid for every θ-node, so this expensive (N_modes-scaling)
        step is paid once per build. ``chunk_n_r`` bounds the peak
        (N_modes, chunk, n_t) intermediate.
        """
        self._build_lm_sort_cache()
        r_1d = np.asarray(r_1d, dtype=float)
        t_1d = np.atleast_1d(np.asarray(t_1d, dtype=float))

        cache = getattr(self, "_D_cache", None)
        if cache is not None:
            r_c, t_c, D_c = cache
            if (
                r_c.shape == r_1d.shape
                and t_c.shape == t_1d.shape
                and np.array_equal(r_c, r_1d)
                and np.array_equal(t_c, t_1d)
            ):
                return D_c

        n_r, n_t = r_1d.size, t_1d.size
        n_lm = self._group_starts_lm.size
        # Per-mode time-evolved coefficient a · exp(-i E t / eps): (N_modes, n_t)
        a_phase = self._a_sorted[:, None] * np.exp(
            -1j * self._E_sorted[:, None] * (t_1d[None, :] / self.basis.eps)
        )
        D = np.empty((n_lm, n_r, n_t), dtype=complex)
        if chunk_n_r is None or chunk_n_r >= n_r:
            chunk_n_r = n_r
        for i0 in range(0, n_r, chunk_n_r):
            i1 = min(i0 + chunk_n_r, n_r)
            R_at = self._interp_R_sorted_uniform(r_1d[i0:i1])     # (N, chunk)
            AR = R_at[:, :, None] * a_phase[:, None, :]           # (N, chunk, n_t)
            D[:, i0:i1, :] = np.add.reduceat(AR, self._group_starts_lm, axis=0)
        self._D_cache = (r_1d.copy(), t_1d.copy(), D)
        return D

    def psi_polar_grid(self, r_1d, theta, phi_1d, t_1d, chunk_n_r=None):
        """ψ on the outer-product grid {r_i} × {t_j} × {φ_k} at fixed polar θ.

        Parameters
        ----------
        r_1d : 1-D ndarray
            Spherical radii (natural units, r / r0).
        theta : float
            Polar angle (galpy convention, θ = 0 along +z), held fixed.
        phi_1d : 1-D ndarray
            Azimuths (radians).
        t_1d : float or 1-D ndarray
            Times (natural units r0 / v0).
        chunk_n_r : int or None
            Radial chunk size for the (N_modes, chunk, n_t) intermediate.

        Returns
        -------
        psi : ndarray, shape (len(r_1d), len(t_1d), len(phi_1d)), complex.
        """
        self._build_lm_sort_cache()
        r_1d = np.asarray(r_1d, dtype=float)
        phi_1d = np.asarray(phi_1d, dtype=float)
        t_1d = np.atleast_1d(np.asarray(t_1d, dtype=float))

        D = self._modal_radial_temporal(r_1d, t_1d, chunk_n_r=chunk_n_r)
        # Angular (l, m) factor c_lm · P_l^|m|(cos θ) for this θ, per group.
        # sph_harm_y(l, m, θ, 0) returns exactly this real value (e^{im·0}=1).
        A = np.real(sph_harm_y(self._l_unique, self._m_unique, float(theta), 0.0))
        S = D * A[:, None, None]                                  # (n_lm, n_r, n_t)
        # Sum l within each m (m groups already contiguous on axis 0).
        S_m = np.add.reduceat(S, self._m_group_starts, axis=0)    # (n_m, n_r, n_t)
        E_phi = np.exp(1j * self._m_unique_values[:, None] * phi_1d[None, :])
        # optimize=True routes the m-contraction through BLAS (tensordot); the
        # default einsum path is ~100x slower here and dominates the build.
        return np.einsum("mrt,mp->rtp", S_m, E_phi, optimize=True)

    def density_polar_grid(self, r_1d, theta, phi_1d, t_1d, chunk_n_r=None):
        """Mass density on the polar outer-product grid (see psi_polar_grid).

        Returns m_boson · |psi|² (/ r0³ if r0 was provided).
        """
        psi_arr = self.psi_polar_grid(r_1d, theta, phi_1d, t_1d, chunk_n_r=chunk_n_r)
        rho_natural = self.m_boson * np.abs(psi_arr) ** 2
        if self.r0 is None:
            return rho_natural
        return rho_natural / self.r0 ** 3

    def psi_points(self, R, z, phi, t=0.0, chunk=100_000):
        """ψ at scattered galpy-cylindrical points with per-point time.

        Broadcast-tolerant counterpart of :py:meth:`psi`: R, z, phi, t may be
        any broadcast-compatible shapes (including a per-point ``t``). Used as
        the fallback for galpy's scalar probes and for verification, so it
        favours generality over speed (a plain O(N_modes × N_pts) sum, chunked
        over points to bound memory).

        Returns a complex array of the broadcast shape.
        """
        Rb, zb, phb, tb = np.broadcast_arrays(
            *(np.asarray(x, dtype=float) for x in (R, z, phi, t))
        )
        shape = Rb.shape
        Rf, zf, phf, tf = (a.ravel() for a in (Rb, zb, phb, tb))
        r = np.hypot(Rf, zf)
        theta = np.arctan2(Rf, zf)
        b = self.basis
        out = np.empty(Rf.size, dtype=complex)
        for j0 in range(0, Rf.size, max(chunk, 1)):
            sl = slice(j0, min(j0 + chunk, Rf.size))
            R_at = _interp_radial(b.R, b.r_grid, r[sl])               # (N, P)
            ph = np.exp(-1j * b.E[:, None] * (tf[sl][None, :] / b.eps))  # (N, P)
            Y = sph_harm_y(
                b.l_idx[:, None], b.m_idx[:, None],
                theta[sl][None, :], phf[sl][None, :],
            )                                                        # (N, P)
            out[sl] = np.einsum("i,ip,ip,ip->p", self.a, R_at, ph, Y)
        return out.reshape(shape)

    def density_points(self, R, z, phi, t=0.0, chunk=100_000):
        """Mass density at scattered points (see psi_points)."""
        psi_arr = self.psi_points(R, z, phi, t, chunk=chunk)
        rho_natural = self.m_boson * np.abs(psi_arr) ** 2
        if self.r0 is None:
            return rho_natural
        return rho_natural / self.r0 ** 3
