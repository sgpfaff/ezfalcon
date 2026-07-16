"""
Tests for ``ExternalGalpyPotential``.
"""

import pickle

import numpy as np
import pytest

galpy = pytest.importorskip("galpy")
from galpy.potential import PlummerPotential, NFWPotential      # noqa: E402

from tambora.dynamics.forces import ExternalGalpyPotential      # noqa: E402


def _plummer(b=0.5):
    return PlummerPotential(amp=1e10, b=b, ro=8., vo=220.)


# --- input validation ---------------------------------------------------------

@pytest.mark.parametrize("bad", [
    pytest.param(object(), id='bare_object'),
    pytest.param('MWPotential2014', id='string'),
    pytest.param(42, id='int'),
    pytest.param(None, id='none'),
])
def test_a_non_potential_is_rejected(bad):
    with pytest.raises(TypeError, match="must be a galpy Potential object"):
        ExternalGalpyPotential(bad)


def test_a_potential_is_accepted():
    # The control: the guard above must not be rejecting everything.
    f = ExternalGalpyPotential(_plummer())
    assert f._pot is not None


def test_a_list_of_potentials_is_accepted_and_composed():
    # The supported list form: _ensure_pot folds it into a CompositePotential on
    # galpy >= 1.11 and leaves it a list on older galpy. Either way it validates.
    f = ExternalGalpyPotential([_plummer(b=0.5), _plummer(b=0.9)])
    pos = np.array([[8., 0., 0.]])
    assert np.isfinite(f.acc(pos, 0.0)).all()


def test_a_list_containing_a_non_potential_is_rejected_somehow():
    # A mixed list is rejected, but NOT by the per-component TypeError, and the
    # error depends on the galpy version:
    #
    #   galpy >= 1.11: _ensure_pot does reduce(operator.add, pot) BEFORE the
    #       validation loop, so `plummer + 'junk'` recurses inside galpy's
    #       __add__/__radd__ -> RecursionError. tambora's check never runs.
    #   galpy <  1.11: _ensure_pot returns the list untouched, _iter_components
    #       walks it, and the check raises a clean TypeError.
    #
    # So this asserts only that it *fails*. The RecursionError is a poor message
    # for a simple typo and is worth fixing by validating before composing.
    with pytest.raises((TypeError, RecursionError)):
        ExternalGalpyPotential([_plummer(), 'not a potential'])


# --- _dedup_key: the pickle path ----------------------------------------------

def test_equal_parameters_dedup_as_the_same_force():
    # The point of pickling rather than using identity: two separately
    # constructed but identical potentials ARE a duplicate registration.
    a, b = ExternalGalpyPotential(_plummer()), ExternalGalpyPotential(_plummer())
    assert a._pot is not b._pot                  # genuinely distinct objects
    assert a._dedup_key() == b._dedup_key()


def test_different_parameters_are_not_duplicates():
    a = ExternalGalpyPotential(_plummer(b=0.5))
    b = ExternalGalpyPotential(_plummer(b=0.9))
    assert a._dedup_key() != b._dedup_key()


def test_different_potential_types_are_not_duplicates():
    a = ExternalGalpyPotential(_plummer())
    b = ExternalGalpyPotential(NFWPotential(amp=1e12, a=16., ro=8., vo=220.))
    assert a._dedup_key() != b._dedup_key()


def test_dedup_key_is_hashable_and_names_the_type():
    # add_hook/add_external_force compare and store these, so they must be usable
    # as dict/set members, and must not collide across force classes.
    key = ExternalGalpyPotential(_plummer())._dedup_key()
    hash(key)
    assert key[0] is ExternalGalpyPotential


# --- _dedup_key: the unpicklable fallback -------------------------------------

def _unpicklable_plummer():
    """A real galpy potential carrying an attribute pickle cannot serialise."""
    p = _plummer()
    p._gotcha = lambda: None            # lambdas are not picklable
    return p


def test_the_unpicklable_potential_is_genuinely_unpicklable():
    # Guard for the two tests below: if galpy ever made this picklable, they
    # would silently start exercising the pickle path instead of the fallback.
    with pytest.raises(Exception):
        pickle.dumps(_unpicklable_plummer())


def test_an_unpicklable_potential_falls_back_to_object_identity():
    # `except Exception: return (type(self), id(self._pot))`. Without it,
    # constructing the force would be fine but *registering* it would raise
    # PicklingError from deep inside add_external_force.
    p = _unpicklable_plummer()
    f = ExternalGalpyPotential(p)
    assert f._dedup_key() == (ExternalGalpyPotential, id(p))


def test_the_fallback_still_dedups_the_same_object():
    # The fallback must remain useful for the case it can decide: the identical
    # potential object wrapped twice really is a duplicate.
    p = _unpicklable_plummer()
    assert ExternalGalpyPotential(p)._dedup_key() == ExternalGalpyPotential(p)._dedup_key()


def test_the_fallback_cannot_recognise_equal_but_distinct_potentials():
    # The documented cost of the fallback, pinned so it is a known trade rather
    # than a surprise: on the identity path two equal-but-distinct unpicklable
    # potentials do NOT compare equal, so a duplicate registration slips through.
    # Deliberate; dedup errs toward accepting, never toward wrongly rejecting a
    # force the user meant to add.
    a = ExternalGalpyPotential(_unpicklable_plummer())
    b = ExternalGalpyPotential(_unpicklable_plummer())
    assert a._dedup_key() != b._dedup_key()


def test_an_unpicklable_potential_is_still_a_working_force():
    # The fallback is about dedup identity only; the physics must be unaffected.
    f = ExternalGalpyPotential(_unpicklable_plummer())
    ref = ExternalGalpyPotential(_plummer())
    pos = np.array([[8., 0., 0.], [0., 8., 1.]])
    np.testing.assert_allclose(f.acc(pos, 0.0), ref.acc(pos, 0.0))
    np.testing.assert_allclose(f.potential(pos, 0.0), ref.potential(pos, 0.0))
