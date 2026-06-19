"""
FastMultipoleExpansionPotential: drop-in galpy MultipoleExpansionPotential whose
*time-dependent* radial-integral precompute is vectorized — and, unlike the first
attempt at this (see HISTORY below), numerically faithful to stock galpy.

What stock galpy does (``_precompute_radial_integrals_2d``)
-----------------------------------------------------------
For every (l, m, cos/sin, t) it builds cubic interpolants of
``f_inner = r^(l+2) rho_lm(r)`` and ``f_outer = r^(1-l) rho_lm(r)`` and evaluates

    I_inner(r_i) = splint(r_0, r_i)                       [FITPACK, per radius]
    I_outer(r_i) = splint(r_0, r_max) - splint(r_0, r_i)  [FITPACK, per radius]

That is ~``L^2 * Nt * 2 * Nr`` FITPACK calls (~10M at L=35, Nt=8, Nr=600) and
dominates the build (~150 s).

Why the per-radius FITPACK loop is *stable*, and how to keep that while
vectorizing
-----------------------------------------------------------------------
The high-l integrands span a huge dynamic range (~1e77 across the grid). FITPACK's
``splint(a, b)`` is a linear combination of the B-spline coefficients with weights
``int_a^b B_j``; for any ``b`` beyond the support of ``B_j`` that weight is the
*identical float*, so in ``splint(r0, rmax) - splint(r0, r)`` every contribution
from intervals left of ``r`` cancels *exactly*, term by term. The result is always
at the local scale of ``[r, rmax]`` — no catastrophic cancellation.

The equivalent vectorized construction is to integrate the *same* cubic
interpolant exactly on each grid interval (a 4-term polynomial formula) and then

    I_inner = forward cumulative sum of the interval integrals   (starts at 0)
    I_outer = reverse  cumulative sum of the interval integrals  (ends   at 0)

Each cumulative value is built only from intervals it actually covers, so it is
computed at its own local scale, exactly like FITPACK — but in two ``cumsum``
calls instead of ~10M FITPACK calls. The interpolant itself is identical:
``InterpolatedUnivariateSpline(k=3, s=0)`` is the not-a-knot cubic interpolant
(FITPACK places no knots at x_1 and x_{n-2}), i.e. the same function scipy's
``CubicSpline`` constructs, so a single multi-RHS ``CubicSpline`` solve replaces
all per-(l,m,t) spline constructions.

Everything downstream is byte-compatible with stock: the same
``_quintic_hermite_ppoly_coeffs`` call (galpy's, already batched), the same
per-(l,m) ``CubicSpline(tgrid, ...)`` time-interpolator objects, so evaluation,
``_serialize_for_c`` (C orbit integration) and pickling are inherited unchanged.

Accuracy (measured by ``validate_fdm_potential.py``, mpmath 50-digit arbitration):
the I-tables produced here match exact integration of the interpolant to
~2e-16 *everywhere*, including 280-decade dynamic ranges at l=70. Stock galpy
does NOT: its outer integral ``splint(r0,rmax) - splint(r0,r)`` carries an
absolute noise floor of ~1 ulp of I_outer(r0), so wherever the true I_outer(r)
has decayed below ~1e-16 of its peak — generic for any radially-localized
high-l density component — the stored value is garbage (often exactly 0.0), and
at large r the ``r^l`` assembly amplifies the noise into the potential. On the
standard FDM case (L=35) stock's force error vs this implementation is ~0.5%
median / ~4% max; on synthetic high-l tests it reaches order unity. So this
class is both faster *and strictly more accurate* than stock galpy 1.12.0.dev0
(bug worth reporting upstream; the same flaw exists in galpy's *static*
``_precompute_radial_integrals``, which this class does NOT override — always
pass ``tgrid`` so the corrected time-dependent path is used).

HISTORY — why the previous "fast" version was also wrong
--------------------------------------------------------
The first attempt evaluated a *global antiderivative* PPoly and formed
``cum = F(r) - F(r0)`` and ``I_outer = cum[-1] - cum``: the antiderivative's
running constant carries the full dynamic-range scale, so the subtraction at
large r cancels catastrophically — same disease as stock, different vector.
The directional cumsums above never subtract large numbers.

NOTE: overrides a galpy internal; developed against galpy 1.12.0.dev0.
"""

import numpy
from scipy.interpolate import CubicSpline, PPoly

from galpy.potential import MultipoleExpansionPotential


def _eval_rho_func_all_t(rho_func, rgrid, tgrid):
    """Evaluate a galpy time-dependent density callable ``f(r, t)`` at all
    times, returning shape (Nt, Nr).

    galpy's ``from_density`` builds these callables as
    ``lambda r, t: spline_over_t(t)`` (the radial grid is baked in), which
    vectorizes over ``t`` directly; fall back to a per-``t`` loop for
    user-supplied callables that don't.
    """
    Nt, Nr = len(tgrid), len(rgrid)
    try:
        vals = numpy.asarray(rho_func(rgrid, tgrid))
        if vals.shape == (Nt, Nr):
            return vals
    except (TypeError, ValueError):
        pass
    out = numpy.empty((Nt, Nr))
    for it, t in enumerate(tgrid):
        out[it] = rho_func(rgrid, t)
    return out


class FastMultipoleExpansionPotential(MultipoleExpansionPotential):
    """MultipoleExpansionPotential with a vectorized, numerically faithful
    time-dependent radial precompute (see module docstring).

    Identical public behavior to the stock class; only
    ``_precompute_radial_integrals_2d`` is overridden.
    """

    # Cap on simultaneous spline-solve columns (chunk_S * Nt); bounds peak
    # memory at ~a few hundred MB regardless of L.
    _CHUNK_COLS = 4096

    @staticmethod
    def _const_time_interp(value, tgrid):
        """A ``PPoly`` that is constant in t (equal to ``value``) on every
        interval of ``tgrid``.

        This is *exactly* the object a ``CubicSpline(tgrid, data)`` produces when
        ``data`` is identical at every node (the cubic-in-t coefficients collapse
        to c=[0,0,0,value]), but it is built straight from the coefficients with
        no banded solve — the per-(l,m) spline solves are the dominant high-L
        build cost, and for a frozen snapshot they are all redundant. ``PPoly`` is
        a ``CubicSpline`` superclass, so ``.c`` / ``.x`` / ``__call__`` satisfy the
        same contract galpy's evaluation (``_fused_ppoly_eval`` reads ``.c``) and
        its monopole/non-axi probes (``interp(tgrid)``) rely on.
        """
        n_int = max(len(tgrid) - 1, 1)
        c = numpy.zeros((4, n_int) + numpy.shape(value), dtype=float)
        c[3] = value  # broadcast the constant across all t-intervals
        return PPoly(c, tgrid, extrapolate=True)

    def _precompute_radial_integrals_2d(self):
        rgrid = numpy.asarray(self._rgrid, dtype=float)
        tgrid = numpy.asarray(self._tgrid, dtype=float)
        L, M = self._L, self._M
        Nt, Nr = len(tgrid), len(rgrid)
        dx = numpy.diff(rgrid)

        self._I_inner_cos_interp = [[None for _ in range(M)] for _ in range(L)]
        self._I_inner_sin_interp = [[None for _ in range(M)] for _ in range(L)]
        self._I_outer_cos_interp = [[None for _ in range(M)] for _ in range(L)]
        self._I_outer_sin_interp = [[None for _ in range(M)] for _ in range(L)]
        self._rho_cos_interp = [[None for _ in range(M)] for _ in range(L)]
        self._rho_sin_interp = [[None for _ in range(M)] for _ in range(L)]

        # ----- flatten the (l, m, kind) loop into a slot list -----------------
        slots = []
        for l in range(L):
            for m in range(min(l + 1, M)):
                slots.append((l, m, "cos"))
                if m > 0:
                    slots.append((l, m, "sin"))
        S = len(slots)
        l_of_slot = numpy.array([s[0] for s in slots])

        # ----- per-l radial power tables (same int-exponent path as stock) ----
        # pow_*[l] are exactly ``rgrid ** (l+2)`` etc. as stock computes them.
        pow_in = numpy.empty((L, Nr))    # r^(l+2)
        pow_in_d = numpy.empty((L, Nr))  # r^(l+1)
        pow_out = numpy.empty((L, Nr))   # r^(1-l)
        pow_out_d = numpy.empty((L, Nr))  # r^(-l)
        for l in range(L):
            pow_in[l] = rgrid ** (l + 2)
            pow_in_d[l] = rgrid ** (l + 1)
            pow_out[l] = rgrid ** (1 - l)
            pow_out_d[l] = rgrid ** (-l)
        pref_of_l = -4.0 * numpy.pi / (2 * numpy.arange(L) + 1)

        # ----- density coefficients for every slot: (S, Nt, Nr) ---------------
        rho_all = numpy.empty((S, Nt, Nr))
        for s, (l, m, kind) in enumerate(slots):
            f = (self._rho_cos_funcs[l][m] if kind == "cos"
                 else self._rho_sin_funcs[l][m])
            rho_all[s] = _eval_rho_func_all_t(f, rgrid, tgrid)

        # ----- frozen-snapshot fast path --------------------------------------
        # A snapshot build (build_fdm_multipole_snapshot) freezes the density at
        # one time and broadcasts it across the Nt(>=4) time nodes galpy's cubic
        # time-spline requires, so every node is bit-identical. In that case the
        # entire t-axis is redundant: the radial integrals only need ONE slice,
        # and the per-(l,m) time-interpolators are constants (built solve-free via
        # _const_time_interp). This collapses the dominant high-L cost (~L^2
        # CubicSpline solves over t) and the Nt-fold radial work. Detection is
        # exact equality (broadcast => identical floats); any genuinely
        # time-varying build falls through to the unchanged general path.
        frozen = Nt > 1 and all(
            numpy.array_equal(rho_all[:, it], rho_all[:, 0]) for it in range(1, Nt)
        )
        Nt_w = 1 if frozen else Nt
        rho_work = rho_all[:, :Nt_w]

        # ----- chunked vectorized radial work ---------------------------------
        chunk_S = max(1, self._CHUNK_COLS // max(Nt_w, 1))
        for s0 in range(0, S, chunk_S):
            s1 = min(s0 + chunk_S, S)
            nS = s1 - s0
            lc = l_of_slot[s0:s1]
            rho_c = rho_work[s0:s1]                     # (nS, Nt_w, Nr)
            B = nS * Nt_w

            # One not-a-knot cubic solve for all (slot, t) density rows: equals
            # stock's per-row InterpolatedUnivariateSpline(k=3) interpolant.
            cs_rho = CubicSpline(rgrid, rho_c.reshape(B, Nr).T)  # c:(4,Nr-1,B)
            # First derivative at the nodes (left coefficients; last node from
            # the final interval's polynomial).
            c = cs_rho.c
            rho_deriv = numpy.empty((B, Nr))
            rho_deriv[:, :-1] = c[2].T
            h_last = dx[-1]
            rho_deriv[:, -1] = (3.0 * c[0, -1] * h_last**2
                                + 2.0 * c[1, -1] * h_last + c[2, -1])
            rho_deriv = rho_deriv.reshape(nS, Nt_w, Nr)

            # Weighted integrands (same product-then-interpolate as stock).
            f_inner = rho_c * pow_in[lc][:, None, :]    # (nS, Nt_w, Nr)
            f_outer = rho_c * pow_out[lc][:, None, :]

            I_inner = self._cumulative_integrals(rgrid, dx, f_inner, "inner")
            I_outer = self._cumulative_integrals(rgrid, dx, f_outer, "outer")

            # Analytic second derivatives of the integrals (stock formulas).
            d2I_inner = (lc[:, None, None] + 2) * pow_in_d[lc][:, None, :] \
                * rho_c + pow_in[lc][:, None, :] * rho_deriv
            d2I_outer = -(1 - lc[:, None, None]) * pow_out_d[lc][:, None, :] \
                * rho_c - pow_out[lc][:, None, :] * rho_deriv

            # Batched quintic-Hermite PPoly coefficients (galpy's own helper).
            pref = pref_of_l[lc][:, None, None]
            inner_flat = self._quintic_hermite_ppoly_coeffs(
                pref * I_inner, pref * f_inner, pref * d2I_inner, dx
            )                                           # (nS, Nt_w, 6*(Nr-1))
            outer_flat = self._quintic_hermite_ppoly_coeffs(
                pref * I_outer, pref * (-f_outer), pref * d2I_outer, dx
            )

            # Per-(l,m) time-interpolators. Frozen: constant PPolys built without
            # a solve (the radial work above ran on the single slice Nt_w=1).
            # General: stock's CubicSpline-over-t, byte-identical to galpy.
            for j in range(nS):
                l, m, kind = slots[s0 + j]
                if frozen:
                    I_inner_interp = self._const_time_interp(inner_flat[j, 0], tgrid)
                    I_outer_interp = self._const_time_interp(outer_flat[j, 0], tgrid)
                    rho_interp = self._const_time_interp(rho_c[j, 0], tgrid)
                else:
                    I_inner_interp = CubicSpline(tgrid, inner_flat[j])
                    I_outer_interp = CubicSpline(tgrid, outer_flat[j])
                    rho_interp = CubicSpline(tgrid, rho_c[j])
                if kind == "cos":
                    self._I_inner_cos_interp[l][m] = I_inner_interp
                    self._I_outer_cos_interp[l][m] = I_outer_interp
                    self._rho_cos_interp[l][m] = rho_interp
                else:
                    self._I_inner_sin_interp[l][m] = I_inner_interp
                    self._I_outer_sin_interp[l][m] = I_outer_interp
                    self._rho_sin_interp[l][m] = rho_interp

    @staticmethod
    def _cumulative_integrals(rgrid, dx, f, which):
        """Cumulative integrals of the not-a-knot cubic interpolant of ``f``.

        Parameters
        ----------
        f : ndarray (..., Nr)
            Integrand values at the grid nodes.
        which : 'inner' | 'outer'
            'inner' -> I(r_i) = int_{r_0}^{r_i} (forward cumsum, I[0] = 0);
            'outer' -> I(r_i) = int_{r_i}^{r_max} (reverse cumsum, I[-1] = 0).

        Each grid interval's integral is evaluated exactly from the local cubic
        (so every value is formed at its own scale; no large-scale subtraction).
        """
        shape = f.shape
        Nr = shape[-1]
        B = int(numpy.prod(shape[:-1], dtype=int))
        cs = CubicSpline(rgrid, f.reshape(B, Nr).T)
        c = cs.c                                          # (4, Nr-1, B)
        d = dx[:, None]
        # int_0^d (c0 u^3 + c1 u^2 + c2 u + c3) du, Horner form
        seg = d * (c[3] + d * (c[2] / 2.0 + d * (c[1] / 3.0 + d * c[0] / 4.0)))
        I = numpy.empty((Nr, B))
        if which == "inner":
            I[0] = 0.0
            numpy.cumsum(seg, axis=0, out=I[1:])
        else:
            I[-1] = 0.0
            I[:-1] = numpy.flip(numpy.cumsum(numpy.flip(seg, 0), axis=0), 0)
        return I.T.reshape(shape)
