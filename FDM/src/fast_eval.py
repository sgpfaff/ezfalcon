"""
Vectorized, numba-compiled force evaluator for a *snapshot*
:class:`fast_multipole.FastMultipoleExpansionPotential` — bypasses galpy's
per-point Python force loop entirely.

galpy evaluates multipole forces one point at a time
(``SphericalHarmonicPotentialMixin._evaluate_cyl_force`` runs a Python
``for idx in ndindex`` over the points, calling ``_compute_spher_forces_at_point``
per particle), and tambora's bridge calls that three times (R/z/phi force). For
an N-particle simulation this is O(N · L²) Python-level work and dominates the
cost (~1.4 ms/particle at L=4, far worse at high L).

This module extracts the radial multipole coefficients from a built (frozen,
time-constant) potential once, then evaluates the forces for *all* particles at
once: the angular Legendre factors come from a single vectorized
``scipy.special.assoc_legendre_p_all`` call (the exact function galpy uses, so
the convention matches), and the radial quintic assembly + (l,m) sum run in a
``numba`` kernel parallelised over particles. Same math as galpy's
``_compute_spher_forces_at_point`` (validated to ~1e-12), but ~N×L²/threads
faster.

Only first derivatives (forces) are provided — that is all an external-force
sim needs. The potential value / second derivatives still go through galpy.
"""

import numpy as np
from numba import njit, prange


# ─────────────────────────────────────────────────────────────────────────────
# Radial-table extraction (once per built snapshot)
# ─────────────────────────────────────────────────────────────────────────────

def build_force_tables(pot, t=0.0):
    """Extract the per-(l,m) radial quintic coefficients from a built
    ``FastMultipoleExpansionPotential`` snapshot, for vectorized force eval.

    Returns a dict of numpy arrays consumed by :func:`forces_cyl`. ``pot`` must
    be time-dependent (built via ``from_density`` with a tgrid — the FDM snapshot
    path); the coefficients are read at time ``t`` (constant for a snapshot).
    """
    if not getattr(pot, "_tdep", False):
        raise ValueError("fast_eval requires a time-dependent (tgrid) build; "
                         "the FDM snapshot path provides this.")
    L, M = pot._L, pot._M
    rgrid = np.ascontiguousarray(pot._rgrid, dtype=float)
    Nr = len(rgrid)

    slots = [(l, m) for l in range(L) for m in range(min(l + 1, M))
             if pot._I_inner_cos_interp[l][m] is not None]
    n_lm = len(slots)
    l_arr = np.array([s[0] for s in slots], dtype=np.int64)
    m_arr = np.array([s[1] for s in slots], dtype=np.int64)

    inner_cos = np.zeros((n_lm, Nr - 1, 6))
    outer_cos = np.zeros((n_lm, Nr - 1, 6))
    inner_sin = np.zeros((n_lm, Nr - 1, 6))
    outer_sin = np.zeros((n_lm, Nr - 1, 6))

    for s, (l, m) in enumerate(slots):
        inner_cos[s] = np.asarray(pot._I_inner_cos_interp[l][m](t)).reshape(Nr - 1, 6)
        outer_cos[s] = np.asarray(pot._I_outer_cos_interp[l][m](t)).reshape(Nr - 1, 6)
        if m > 0 and pot._I_inner_sin_interp[l][m] is not None:
            inner_sin[s] = np.asarray(pot._I_inner_sin_interp[l][m](t)).reshape(Nr - 1, 6)
            outer_sin[s] = np.asarray(pot._I_outer_sin_interp[l][m](t)).reshape(Nr - 1, 6)

    return {
        "L": L, "M": M, "rgrid": rgrid,
        "l_arr": l_arr, "m_arr": m_arr,
        "inner_cos": np.ascontiguousarray(inner_cos),
        "outer_cos": np.ascontiguousarray(outer_cos),
        "inner_sin": np.ascontiguousarray(inner_sin),
        "outer_sin": np.ascontiguousarray(outer_sin),
    }


# ─────────────────────────────────────────────────────────────────────────────
# numba kernel: radial assembly + (l,m) sum, parallel over particles
# ─────────────────────────────────────────────────────────────────────────────

@njit(parallel=True, fastmath=True, cache=True)
def _grad_kernel(r, costheta, sintheta, phi, rgrid, Lmax, Mmax,
                 l_arr, m_arr, inner_cos, outer_cos, inner_sin, outer_sin,
                 gdr, gdth, gdph):
    Nr = rgrid.shape[0]
    rmin = rgrid[0]
    rmax = rgrid[-1]
    n_lm = l_arr.shape[0]
    N = r.shape[0]
    for i in prange(N):
        ri = r[i]
        if ri == 0.0 or not np.isfinite(ri):
            gdr[i] = 0.0; gdth[i] = 0.0; gdph[i] = 0.0
            continue
        # clamp cos(theta) off the exact poles so the associated-Legendre
        # derivative recurrence (1/(x^2-1)) stays finite; only affects
        # particles within ~5e-4 rad of the z-axis.
        xx = costheta[i]
        if xx > 1.0 - 1e-7:
            xx = 1.0 - 1e-7
        elif xx < -1.0 + 1e-7:
            xx = -1.0 + 1e-7
        st = np.sqrt(1.0 - xx * xx)
        ph = phi[i]
        # ---- associated Legendre P_l^m(xx) and dP/dx, upward recurrence ----
        P = np.zeros((Lmax, Mmax))
        dP = np.zeros((Lmax, Mmax))
        denom = xx * xx - 1.0
        pmm = 1.0
        for m in range(Mmax):
            if m > 0:
                pmm = -pmm * (2 * m - 1) * st      # (-1)^m (2m-1)!! sin^m (Condon-Shortley)
            P[m, m] = pmm
            if m + 1 < Lmax:
                P[m + 1, m] = xx * (2 * m + 1) * pmm
            for l in range(m + 2, Lmax):
                P[l, m] = ((2 * l - 1) * xx * P[l - 1, m]
                           - (l + m - 1) * P[l - 2, m]) / (l - m)
            for l in range(m, Lmax):
                plm1 = P[l - 1, m] if l - 1 >= m else 0.0
                dP[l, m] = (l * xx * P[l, m] - (l + m) * plm1) / denom
        # Precompute r^l (l=0..Lmax) and cos(m phi)/sin(m phi) (m=0..Mmax-1)
        # once per particle by recurrence -- avoids ~n_lm pow/cos/sin calls.
        powr = np.empty(Lmax + 1)
        powr[0] = 1.0
        for l in range(1, Lmax + 1):
            powr[l] = powr[l - 1] * ri
        cph = np.cos(ph); sph = np.sin(ph)
        cosm = np.empty(Mmax); sinm = np.empty(Mmax)
        cosm[0] = 1.0; sinm[0] = 0.0
        for m in range(1, Mmax):
            cosm[m] = cosm[m - 1] * cph - sinm[m - 1] * sph
            sinm[m] = sinm[m - 1] * cph + cosm[m - 1] * sph
        # locate radial interval once (shared by all (l,m))
        in_grid = (ri >= rmin) and (ri <= rmax)
        i_r = 0
        drr = 0.0
        if in_grid:
            lo = 0; hi = Nr - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if rgrid[mid] <= ri:
                    lo = mid
                else:
                    hi = mid
            i_r = lo
            drr = ri - rgrid[i_r]
        dr_max = rgrid[Nr - 1] - rgrid[Nr - 2]
        acc_dr = 0.0; acc_dth = 0.0; acc_dph = 0.0
        for s in range(n_lm):
            l = l_arr[s]; m = m_arr[s]
            rnl = 1.0 / powr[l + 1]
            rl = powr[l]
            # ---- cos coefficient radial value R and dR/dr ----
            if in_grid:
                ic = inner_cos[s, i_r]; oc = outer_cos[s, i_r]
                Ii = ((((ic[0]*drr+ic[1])*drr+ic[2])*drr+ic[3])*drr+ic[4])*drr+ic[5]
                dIi = (((5*ic[0]*drr+4*ic[1])*drr+3*ic[2])*drr+2*ic[3])*drr+ic[4]
                Io = ((((oc[0]*drr+oc[1])*drr+oc[2])*drr+oc[3])*drr+oc[4])*drr+oc[5]
                dIo = (((5*oc[0]*drr+4*oc[1])*drr+3*oc[2])*drr+2*oc[3])*drr+oc[4]
                Rc = rnl * Ii + rl * Io
                dRc = -(l+1)*rnl/ri*Ii + rnl*dIi + l*rl/ri*Io + rl*dIo
            elif ri > rmax:
                icl = inner_cos[s, Nr - 2]
                Iim = ((((icl[0]*dr_max+icl[1])*dr_max+icl[2])*dr_max+icl[3])*dr_max+icl[4])*dr_max+icl[5]
                Rc = Iim * rnl
                dRc = -(l+1) * Iim * rnl / ri
            else:  # ri < rmin: smooth inward extrapolation (galpy formula)
                ic = inner_cos[s, 0]; oc = outer_cos[s, 0]
                dIi0 = ic[4]                      # dI_inner at rmin (dr=0)
                Io0 = oc[5]                       # I_outer at rmin (dr=0)
                P_rho0 = dIi0 / rmin ** (l + 2)
                I_in_ext = P_rho0 / (l + 3) * ri ** (l + 3)
                if l == 2:
                    extra = P_rho0 * np.log(rmin / ri)
                else:
                    extra = P_rho0 / (2 - l) * (rmin ** (2 - l) - ri ** (2 - l))
                I_out_ext = Io0 + extra
                Rc = rnl * I_in_ext + rl * I_out_ext
                dRc = -(l+1)*rnl/ri*I_in_ext + rnl*(P_rho0*ri**(l+2)) \
                    + l*rl/ri*I_out_ext + rl*(-P_rho0*ri**(-(l+1)))
            Plm = P[l, m]; dPlm = dP[l, m]
            cmf = cosm[m]; smf = sinm[m]
            acc_dr += Plm * cmf * dRc
            acc_dth += dPlm * (-st) * cmf * Rc
            acc_dph += Plm * (-m * smf) * Rc
            if m > 0:
                ic = inner_sin[s, i_r] if in_grid else inner_sin[s, Nr - 2]
                if in_grid:
                    isc = inner_sin[s, i_r]; osc = outer_sin[s, i_r]
                    Iis = ((((isc[0]*drr+isc[1])*drr+isc[2])*drr+isc[3])*drr+isc[4])*drr+isc[5]
                    dIis = (((5*isc[0]*drr+4*isc[1])*drr+3*isc[2])*drr+2*isc[3])*drr+isc[4]
                    Ios = ((((osc[0]*drr+osc[1])*drr+osc[2])*drr+osc[3])*drr+osc[4])*drr+osc[5]
                    dIos = (((5*osc[0]*drr+4*osc[1])*drr+3*osc[2])*drr+2*osc[3])*drr+osc[4]
                    Rs = rnl * Iis + rl * Ios
                    dRs = -(l+1)*rnl/ri*Iis + rnl*dIis + l*rl/ri*Ios + rl*dIos
                elif ri > rmax:
                    isc = inner_sin[s, Nr - 2]
                    Iism = ((((isc[0]*dr_max+isc[1])*dr_max+isc[2])*dr_max+isc[3])*dr_max+isc[4])*dr_max+isc[5]
                    Rs = Iism * rnl
                    dRs = -(l+1) * Iism * rnl / ri
                else:
                    isc = inner_sin[s, 0]; osc = outer_sin[s, 0]
                    dIis0 = isc[4]; Ios0 = osc[5]
                    P_rho0s = dIis0 / rmin ** (l + 2)
                    I_in_exts = P_rho0s / (l + 3) * ri ** (l + 3)
                    if l == 2:
                        extras = P_rho0s * np.log(rmin / ri)
                    else:
                        extras = P_rho0s / (2 - l) * (rmin ** (2 - l) - ri ** (2 - l))
                    I_out_exts = Ios0 + extras
                    Rs = rnl * I_in_exts + rl * I_out_exts
                    dRs = -(l+1)*rnl/ri*I_in_exts + rnl*(P_rho0s*ri**(l+2)) \
                        + l*rl/ri*I_out_exts + rl*(-P_rho0s*ri**(-(l+1)))
                acc_dr += Plm * smf * dRs
                acc_dth += dPlm * (-st) * smf * Rs
                acc_dph += Plm * (m * cmf) * Rs
        gdr[i] = acc_dr; gdth[i] = acc_dth; gdph[i] = acc_dph


def forces_cyl(tables, R, z, phi, chunk=200_000):
    """Vectorized galpy-internal-unit cylindrical forces (Rforce, zforce,
    phitorque) for arrays of cylindrical (R, z, phi), bypassing galpy.

    Matches ``evaluateRforces`` / ``evaluatezforces`` / ``evaluatephitorques``
    on the snapshot potential ``tables`` came from.
    """
    R = np.ascontiguousarray(R, dtype=float).ravel()
    z = np.ascontiguousarray(z, dtype=float).ravel()
    phi = np.ascontiguousarray(np.broadcast_to(phi, R.shape), dtype=float)
    L, M = tables["L"], tables["M"]
    rgrid = tables["rgrid"]
    N = R.size
    Rforce = np.empty(N); zforce = np.empty(N); phitorque = np.empty(N)
    for c0 in range(0, N, chunk):
        c1 = min(c0 + chunk, N)
        Rc = R[c0:c1]; zc = z[c0:c1]; pc = np.ascontiguousarray(phi[c0:c1])
        r = np.hypot(Rc, zc)
        with np.errstate(invalid="ignore", divide="ignore"):
            costheta = np.where(r > 0, zc / r, 1.0)
        sintheta = np.where(r > 0, Rc / r, 0.0)
        n = r.size
        gdr = np.empty(n); gdth = np.empty(n); gdph = np.empty(n)
        _grad_kernel(np.ascontiguousarray(r), np.ascontiguousarray(costheta),
                     np.ascontiguousarray(sintheta), pc, rgrid, L, M,
                     tables["l_arr"], tables["m_arr"],
                     tables["inner_cos"], tables["outer_cos"],
                     tables["inner_sin"], tables["outer_sin"],
                     gdr, gdth, gdph)
        # forces = -gradient; chain rule to cylindrical (galpy convention)
        Fr = -gdr; Fth = -gdth
        with np.errstate(invalid="ignore", divide="ignore"):
            dr_dR = np.where(r > 0, Rc / r, 0.0)
            dth_dR = np.where(r > 0, zc / r**2, 0.0)
            dr_dz = np.where(r > 0, zc / r, 0.0)
            dth_dz = np.where(r > 0, -Rc / r**2, 0.0)
        Rforce[c0:c1] = dr_dR * Fr + dth_dR * Fth
        zforce[c0:c1] = dr_dz * Fr + dth_dz * Fth
        phitorque[c0:c1] = -gdph
    return Rforce, zforce, phitorque
