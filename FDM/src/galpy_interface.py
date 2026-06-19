"""
Bridge between an FDM :class:`wavefunction.WaveFunction` realization and galpy's
:class:`galpy.potential.MultipoleExpansionPotential`.

``MultipoleExpansionPotential.from_density`` projects a density callable
``dens(R, z, phi, t)`` onto real spherical harmonics on a radial grid, then
solves the radial Poisson equation per (l, m) and interpolates in time. It
samples the density on a *polar* (r, θ) grid × φ, with one broadcast call per
Gauss-Legendre θ-node of shape ``(Nr, Nt, phi_order)``.

:class:`FDMDensity` is the adapter we hand to ``from_density``. It:

  * returns the density in **galpy internal units** as plain floats (astropy
    Quantities are rejected on the time-dependent path), via a precomputed
    scalar ``K`` with ρ_internal = K · |ψ_natural|²;
  * routes galpy's factored per-θ broadcast call to
    :py:meth:`WaveFunction.psi_polar_grid` (fast, separable, the θ-independent
    radial/temporal core is cached so the N_modes-scaling cost is paid once);
  * falls back to :py:meth:`WaveFunction.psi_points` for scalar probes and
    arbitrary verification points.

Coordinates need no conversion: galpy internal (R, z) are in units of r0, which
is exactly the natural unit the wavefunction uses, and galpy internal time is
r0 / v0, the unit of the wavefunction's exp(-i E t / eps) phases.
"""

import pickle

import astropy.units as u
import numpy as np
from galpy.potential import MultipoleExpansionPotential
from galpy.util import conversion

from . import galpy_patches # noqa: F401  -- applies galpy 1.12.0.dev0 vectorized-multipole integration fix


class FDMDensity:
    """Callable density adapter for ``MultipoleExpansionPotential.from_density``.

    Parameters
    ----------
    wf : wavefunction.WaveFunction
        The FDM realization. Must have been built with ``r0`` and ``m_boson``
        set (so units are well defined).
    ro, vo : float
        galpy distance (kpc) and velocity (km/s) scales.
    rgrid, tgrid : 1-D ndarray
        The exact radial (natural units) and time (r0/v0) grids passed to
        ``from_density``. The radial axis of every factored galpy call is taken
        to be ``rgrid``, so it must match.
    chunk_n_r : int or None
        Radial chunk size forwarded to the separable evaluator to bound memory.
    freeze_t : float or None
        If set, ignore the time(s) galpy passes and evaluate the density at this
        single wavefunction time, broadcasting the result across the requested
        time axis. Used to build a *frozen single snapshot* (constant-in-time)
        potential through the time-dependent machinery — see
        :func:`build_fdm_multipole_snapshot`.
    """

    def __init__(self, wf, ro, vo, rgrid, tgrid, chunk_n_r=None, freeze_t=None):
        self.wf = wf
        self.rgrid = np.asarray(rgrid, dtype=float)
        self.tgrid = np.atleast_1d(np.asarray(tgrid, dtype=float))
        self.chunk_n_r = chunk_n_r
        self.freeze_t = None if freeze_t is None else float(freeze_t)
        # ρ_physical[Msun/pc³] = |ψ_nat|² · (m_boson / r0³); divide by the
        # internal->Msun/pc³ factor to land in galpy internal density units.
        K_phys = (wf.m_boson / wf.r0 ** 3).to(u.Msun / u.pc ** 3).value
        self._K = K_phys / conversion.dens_in_msolpc3(vo, ro)

    def _is_factored(self, R, z, phi, t):
        """True for galpy's per-θ broadcast call: R,z=(Nr,1,1), phi=(1,1,Nphi),
        t=(1,Nt,1)."""
        Nr, Nt = self.rgrid.size, self.tgrid.size
        return (
            R.ndim == 3
            and z.ndim == 3
            and R.shape == (Nr, 1, 1)
            and z.shape == (Nr, 1, 1)
            and phi.ndim == 3
            and phi.shape[0] == 1
            and phi.shape[1] == 1
            and t.ndim == 3
            and t.shape == (1, Nt, 1)
        )

    def __call__(self, R, z, phi, t=0.0):
        R = np.asarray(R, dtype=float)
        z = np.asarray(z, dtype=float)
        phi = np.asarray(phi, dtype=float)
        t = np.asarray(t, dtype=float)

        if self._is_factored(R, z, phi, t):
            # θ = arctan2(R, z) is constant across the polar grid at one node;
            # the radial axis is rgrid (galpy's R came from rgrid * sinθ).
            theta = float(np.arctan2(R, z).ravel()[0])
            t_eval = np.ravel(t) if self.freeze_t is None else \
                np.array([self.freeze_t], dtype=float)
            psi = self.wf.psi_polar_grid(
                self.rgrid,
                theta,
                np.ravel(phi),
                t_eval,
                chunk_n_r=self.chunk_n_r,
            )  # (Nr, Nt_eval, Nphi)
            dens = self._K * np.abs(psi) ** 2
            if self.freeze_t is not None:
                # Frozen snapshot: identical density at every requested time node
                # (Nt_eval == 1 -> broadcast across galpy's Nt time axis).
                dens = np.broadcast_to(dens, (dens.shape[0], t.shape[1],
                                              dens.shape[2]))
            return dens

        # Fallback: scalar probes / arbitrary verification points.
        t_pts = t if self.freeze_t is None else self.freeze_t
        psi = self.wf.psi_points(R, z, phi, t_pts)
        return self._K * np.abs(psi) ** 2


def build_fdm_multipole_potential(
    wf,
    ro,
    vo,
    tgrid,
    L=None,
    rgrid=None,
    Nr=600,
    chunk_n_r=None,
    fast=True,
    _freeze_t=None,
    **from_density_kwargs,
):
    """Build a time-dependent, non-axisymmetric MultipoleExpansionPotential.

    Parameters
    ----------
    wf : wavefunction.WaveFunction
        The FDM realization.
    ro, vo : float
        galpy distance (kpc) and velocity (km/s) scales.
    tgrid : 1-D ndarray
        Time grid (natural units r0/v0) spanning the intended integration
        window. Its spacing sets the temporal resolution of the granule
        flicker captured by the potential.
    L : int or None
        Max spherical-harmonic degree + 1. Defaults to ``2*l_max + 1`` where
        ``l_max`` is the highest l in the eigenmode set — enough to capture the
        full angular content of ρ ~ |ψ|². The build's dominant cost (galpy's
        radial Poisson integrals) scales as ``L²``, so reducing L is the main
        speed lever. The granular speckle of ρ lives in the high-l tail and
        barely affects the *forces*, so L can usually be cut well below
        ``2*l_max+1`` (choose it from the angular power spectrum) for a large
        speed-up at sub-percent acceleration error — e.g. for the standard case
        l_max=35, L=35 gives a ~3.4x faster build with <0.5% force error. The
        angular quadrature is always sized to the full ``2*l_max`` bandwidth, so
        a reduced L truncates cleanly without aliasing.
    rgrid : 1-D ndarray or None
        Radial grid (natural units). Defaults to a geomspace spanning the
        eigenmode support ``[r_grid[0], r_grid[-1]]`` with ``Nr`` points.
    Nr : int
        Number of radial points for the default rgrid.
    chunk_n_r : int or None
        Radial chunk size for the separable density evaluator (memory control).
    fast : bool
        Use :class:`fast_multipole.FastMultipoleExpansionPotential`, which
        replaces galpy's per-radius FITPACK radial-integral loop (the dominant
        build cost, ~L²·Nt·Nr spline-integral calls) with an exact vectorized
        equivalent (directional cumulative sums of per-interval cubic
        integrals). This is both ~3x faster to build AND more accurate than
        stock galpy 1.12.0.dev0: stock's outer radial integrals suffer a
        catastrophic-cancellation noise floor that corrupts high-l terms
        (~0.5% median / ~4% max force error on the standard L=35 FDM case;
        established by mpmath arbitration in ``validate_fdm_potential.py``).
        Leave True. ``fast=False`` (stock) is retained only for comparison.
    **from_density_kwargs
        Forwarded to ``MultipoleExpansionPotential.from_density`` (e.g.
        ``costheta_order``, ``phi_order``, ``amp``, ``normalize``).

    Returns
    -------
    galpy.potential.MultipoleExpansionPotential
        C-accelerated, time-interpolated FDM potential.
    """
    rgrid_basis = wf.basis.r_grid
    if rgrid is None:
        rgrid = np.geomspace(rgrid_basis[0], rgrid_basis[-1], Nr)
    l_max = int(wf.basis.l_idx.max())
    if L is None:
        L = 2 * l_max + 1
    # rho = |psi|^2 has angular content up to 2*l_max regardless of the (possibly
    # truncated) expansion order L. Size the angular quadrature to that *full*
    # bandwidth so reducing L cleanly truncates the high-l multipoles instead of
    # aliasing their power back into the retained low-l coefficients. (At the
    # default L = 2*l_max+1 these match galpy's own L-based defaults.)
    from_density_kwargs.setdefault("costheta_order", max(20, 2 * l_max + 2))
    from_density_kwargs.setdefault("phi_order", max(20, 4 * l_max + 1))

    tgrid = np.atleast_1d(np.asarray(tgrid, dtype=float))
    dens = FDMDensity(wf, ro, vo, rgrid=rgrid, tgrid=tgrid,
                      chunk_n_r=chunk_n_r, freeze_t=_freeze_t)
    if fast:
        from .fast_multipole import FastMultipoleExpansionPotential as _cls
    else:
        _cls = MultipoleExpansionPotential
    return _cls.from_density(
        dens,
        L=L,
        rgrid=rgrid,
        tgrid=tgrid,
        symmetry=None,
        ro=ro,
        vo=vo,
        **from_density_kwargs,
    )


def build_fdm_multipole_snapshot(wf, ro, vo, t0=0.0, **kwargs):
    """Build a *single-snapshot* FDM MultipoleExpansionPotential at time ``t0``.

    Turns the instantaneous density ``|ψ(·, t0)|²`` into a galpy potential that
    is **constant in time**, so you can evaluate forces / integrate orbits in the
    frozen field at that snapshot. Evaluate at any ``t`` (the default ``t=0``
    works); the field does not vary.

    This deliberately routes through the *time-dependent* multipole machinery
    (with the density frozen at ``t0`` and a minimal 4-node time grid) rather
    than galpy's static ``from_density`` path, because the static path is both:

      * **slower** — its angular projection (``_compute_rho_lm``) is a pure-Python
        ``Nr × costheta_order`` loop, while the time-dependent projection
        (``_compute_rho_lm_timedep``) is fully vectorized and pairs with the fast
        separable ``psi_polar_grid`` evaluator; and
      * **less accurate** — only the time-dependent radial precompute is fixed by
        :class:`fast_multipole.FastMultipoleExpansionPotential` (the high-l
        catastrophic-cancellation bug, see :func:`build_fdm_multipole_potential`);
        galpy's static ``_precompute_radial_integrals`` still carries it.

    Freezing the density at ``t0`` makes all 4 (arbitrary) time nodes identical,
    so galpy's cubic time-spline collapses to a constant — a genuine snapshot at
    no extra density cost (``psi`` is evaluated at the single time ``t0`` and
    broadcast). The 4 nodes are the minimum for galpy's ``k=3`` time spline.

    Parameters
    ----------
    wf, ro, vo
        As in :func:`build_fdm_multipole_potential`.
    t0 : float
        Wavefunction time (natural units r0/v0) of the snapshot. Default 0.
    **kwargs
        Forwarded to :func:`build_fdm_multipole_potential` (``L``, ``rgrid``,
        ``Nr``, ``chunk_n_r``, ``fast``, and ``from_density`` kwargs).

    Returns
    -------
    galpy.potential.MultipoleExpansionPotential
        Time-constant (frozen) FDM potential at the ``t0`` snapshot.
    """
    tgrid = float(t0) + np.arange(4, dtype=float)  # >= k+1=4 nodes for k=3 spline
    return build_fdm_multipole_potential(
        wf, ro, vo, tgrid=tgrid, _freeze_t=float(t0), **kwargs
    )


def save_multipole(mp, path):
    """Pickle a built ``MultipoleExpansionPotential`` for later reuse (skip the rebuild).

    The time-dependent object stores two *build-only* lambda attributes
    (``_rho_cos_funcs`` / ``_rho_sin_funcs``, the ``f(r, t)`` density callables) that
    stdlib ``pickle`` cannot serialize. They are used only by the one-time radial-integral
    precompute (``_precompute_radial_integrals_2d``), never by evaluation or the C
    integration path (``_serialize_for_c`` reads only the ``CubicSpline`` interpolators and
    the r/t grids). So we null them for the dump and restore them afterward, leaving the live
    object unchanged. The reloaded potential evaluates forces / integrates orbits with no
    recompute.

    Note: the object is a dense (l, m, r, t) coefficient tensor — the file is large
    (~2 GB at L=35, scaling as L^2 * Nt * Nr). Pickle is tied to the scipy/numpy version;
    for cross-version archival, persist the raw spline arrays instead.
    """
    has_cos, has_sin = hasattr(mp, "_rho_cos_funcs"), hasattr(mp, "_rho_sin_funcs")
    stash = (getattr(mp, "_rho_cos_funcs", None), getattr(mp, "_rho_sin_funcs", None))
    if has_cos:
        mp._rho_cos_funcs = None
    if has_sin:
        mp._rho_sin_funcs = None
    try:
        with open(path, "wb") as f:
            pickle.dump(mp, f, protocol=pickle.HIGHEST_PROTOCOL)
    finally:  # restore so the in-memory object is untouched by saving
        if has_cos:
            mp._rho_cos_funcs = stash[0]
        if has_sin:
            mp._rho_sin_funcs = stash[1]
    return path


def load_multipole(path):
    """Load a ``MultipoleExpansionPotential`` saved by :func:`save_multipole`.

    Ready to evaluate forces / integrate orbits (C path) immediately, with no recompute.
    The build-only ``_rho_cos_funcs`` / ``_rho_sin_funcs`` are ``None`` on the loaded object,
    which is harmless unless you try to rebuild it.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
