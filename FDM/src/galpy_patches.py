"""
Workaround for a galpy 1.12.0.dev0 bug that breaks **vectorized** orbit integration in a
``MultipoleExpansionPotential`` (and ``interpSphericalPotential``).

When an ``Orbit`` holding N>1 orbits is integrated and *any* orbit leaves the potential's
radial interpolation grid, ``Orbit._integrate_impl`` emits an out-of-range warning formatted
as ``"... (min/max r = {:.3f},{:.3f}".format(self.rperi(), self.rap())`` (Orbits.py ~1920 &
~1959). For an N>1 ``Orbit``, ``rperi()``/``rap()`` return numpy arrays, and ``{:.3f}`` on an
array raises ``TypeError: unsupported format string passed to numpy.ndarray.__format__``.
Single orbits return scalars, so they're unaffected — which is why ``streamspraydf`` (it
integrates an N-star ``Orbit`` array) hits this but single-orbit integration does not.

The crash happens *after* the integration (the warning inspects the already-integrated
trajectory and `self.orbit` is populated), so we wrap ``_integrate_impl`` and swallow only
this specific formatting ``TypeError``, preserving the integrated result. Importing this
module applies the patch once; it is idempotent. Worth reporting upstream.
"""

from galpy.orbit import Orbit

_PATCH_FLAG = "_fdm_vectorized_multipole_warn_patched"
_orig_integrate_impl = Orbit.__dict__.get("_integrate_impl")


def apply():
    """Idempotently install the array-safe ``_integrate_impl`` wrapper."""
    if _orig_integrate_impl is None or getattr(Orbit, _PATCH_FLAG, False):
        return

    def _integrate_impl(self, *args, **kwargs):
        try:
            return _orig_integrate_impl(self, *args, **kwargs)
        except TypeError as exc:
            # The vectorized out-of-range warning formats an ndarray with {:.3f};
            # integration is already complete, so this is safe to ignore.
            if "numpy.ndarray.__format__" in str(exc):
                return None
            raise

    Orbit._integrate_impl = _integrate_impl
    setattr(Orbit, _PATCH_FLAG, True)


apply()
