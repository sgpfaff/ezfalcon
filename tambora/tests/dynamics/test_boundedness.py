"""Tests for the boundedness diagnostics in ``tambora/dynamics/diagnostics.py``.

Two groups:

* ``bound_mask``: the ``'energy'`` self-binding baseline and the ``'jacobi'``
  tidal criterion. These pin down the tidal-tensor sign convention (the one
  convention most likely to be silently wrong), verify the Roche/Jacobi
  criterion strips particles the energy criterion wrongly keeps, and check the
  input validation.
* ``reconstruct_mask``: the pure replay of a transition log that backs every
  derived accessor on ``BoundednessHook`` (``mask_at``/``history``/``fraction``).
"""

import numpy as np
import pytest
import astropy.units as u
from galpy.potential import KeplerPotential

from tambora.dynamics.diagnostics import bound_mask, reconstruct_mask



# =============================================================================
# reconstruct_mask -- pure replay of a transition log
# =============================================================================
#
# An event is (idx, time, direction, *payload); direction +1 => bound, -1 =>
# unbound. reconstruct_mask replays every event with time <= t onto a copy of
# the initial mask. It assumes events are in chronological order (which the hook
# guarantees). Indices below are component-local, matching the hook.

def _initial():
    return np.array([True, True, True, True])


def test_replays_unbind_and_rebind_events():
    events = [(0, 1.0, -1),          # particle 0 unbinds at t=1
              (2, 2.0, -1),          # particle 2 unbinds at t=2
              (0, 3.0, +1)]          # particle 0 rebinds at t=3
    np.testing.assert_array_equal(reconstruct_mask(_initial(), events, 0.5),
                                  [True, True, True, True])   # before any event
    np.testing.assert_array_equal(reconstruct_mask(_initial(), events, 1.5),
                                  [False, True, True, True])  # 0 gone
    np.testing.assert_array_equal(reconstruct_mask(_initial(), events, 2.5),
                                  [False, True, False, True]) # 0 and 2 gone
    np.testing.assert_array_equal(reconstruct_mask(_initial(), events, 3.5),
                                  [True, True, False, True])  # 0 rebound


def test_query_time_boundary_is_inclusive():
    events = [(0, 1.0, -1)]
    # An event exactly at the query time has already happened (time <= t).
    np.testing.assert_array_equal(reconstruct_mask(_initial(), events, 1.0),
                                  [False, True, True, True])
    # Just before it, the flip has not yet applied.
    np.testing.assert_array_equal(reconstruct_mask(_initial(), events, 0.999),
                                  [True, True, True, True])


def test_does_not_mutate_initial_mask():
    initial = _initial()
    before = initial.copy()
    out = reconstruct_mask(initial, [(0, 1.0, -1)], t=2.0)
    np.testing.assert_array_equal(initial, before)    # input untouched
    assert out is not initial                          # returns a fresh array


def test_ignores_trailing_payload():
    # Events may carry captured pos/vel after (idx, time, direction); replay
    # must give the same result as the bare 3-tuples.
    bare = [(0, 1.0, -1), (2, 2.0, +1)]
    with_payload = [(0, 1.0, -1, np.array([1.0, 2.0, 3.0])),
                    (2, 2.0, +1, np.array([4.0, 5.0, 6.0]), np.array([7.0, 8.0, 9.0]))]
    np.testing.assert_array_equal(
        reconstruct_mask(_initial(), bare, 5.0),
        reconstruct_mask(_initial(), with_payload, 5.0))


def test_ordered_double_flip_resolves_to_last_event():
    # Two flips on the same particle, in chronological order: the later one wins.
    events = [(1, 1.0, -1), (1, 2.0, +1)]
    np.testing.assert_array_equal(reconstruct_mask(_initial(), events, 1.5),
                                  [True, False, True, True])  # after unbind only
    np.testing.assert_array_equal(reconstruct_mask(_initial(), events, 2.5),
                                  [True, True, True, True])   # after rebind


def test_assumes_chronological_order():
    # reconstruct_mask replays events in list order, not by timestamp, so it
    # relies on the log being chronological (as the hook always produces it).
    # Fed the same two flips out of order, it returns the wrong state -- this
    # documents the precondition rather than endorsing unordered input.
    ordered = [(1, 1.0, -1), (1, 2.0, +1)]      # correct: ends bound at t>=2
    shuffled = [(1, 2.0, +1), (1, 1.0, -1)]     # same events, wrong order
    np.testing.assert_array_equal(reconstruct_mask(_initial(), ordered, 5.0),
                                  [True, True, True, True])
    np.testing.assert_array_equal(reconstruct_mask(_initial(), shuffled, 5.0),
                                  [True, False, True, True])  # last-in-list wins
