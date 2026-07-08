"""Tests for hook firing cadences (``tambora/dynamics/hooks/cadence.py``).

Cadences determine when a hook fires during integration via its
``due(step, steps_per_output) -> bool`` method.

Structure
---------
Tests for each cadence are held in individual classes, with tests common
amongst all cadences held in the ``_CadenceContract`` mixin.

Adding a New Cadence
--------------------
To add a new cadence to tambora, create a new class that

* subclasses ``_CadenceContract``
* sets ``cadence`` to an instance the new cadence
* includes a truth table using a `pytest.mark.parametrize()` decorator 
"""

import pytest

from tambora.dynamics.hooks import (
    Cadence, EveryStep, EveryNSteps, EveryOutput, EveryNOutputs,
)


# --- shared contract ----------------------------------------------------------

class _CadenceContract:
    """Conditions every cadence must satisfy. Subclass and set ``cadence``."""

    cadence = None                       # subclass provides an instance

    def test_fires_at_step_zero(self):
        # _runner fires each hook once at step 0 to capture the initial state;
        # a cadence that skipped step 0 would silently drop that fire.
        assert self.cadence.due(0, steps_per_output=5) is True


# --- per-cadence truth tables -------------------------------------------------

class TestEveryStep(_CadenceContract):
    cadence = EveryStep()

    @pytest.mark.parametrize("step", [0, 1, 2, 7, 100])
    def test_always_due(self, step):
        assert self.cadence.due(step, steps_per_output=5) is True


class TestEveryNSteps(_CadenceContract):
    cadence = EveryNSteps(3)             # fires on multiples of 3

    # steps_per_output is irrelevant to EveryNSteps; pass an arbitrary value.
    @pytest.mark.parametrize("step, expected", [
        (0, True), (1, False), (2, False), (3, True), (4, False), (6, True), (9, True),
    ])
    def test_fires_on_multiples_of_n(self, step, expected):
        assert self.cadence.due(step, steps_per_output=7) is expected

    def test_n_one_fires_every_step(self):
        c = EveryNSteps(1)
        assert all(c.due(step, steps_per_output=7) for step in range(6))


class TestEveryOutput(_CadenceContract):
    cadence = EveryOutput()             # fires on multiples of steps_per_output

    @pytest.mark.parametrize("spo, step, expected", [
        (5, 0, True), (5, 4, False), (5, 5, True), (5, 10, True), (5, 12, False),
        (1, 3, True),                                  # spo=1 aligns with every step
        (10, 20, True), (10, 25, False),
    ])
    def test_fires_on_output_steps(self, spo, step, expected):
        assert self.cadence.due(step, steps_per_output=spo) is expected


class TestEveryNOutputs(_CadenceContract):
    cadence = EveryNOutputs(2)          # fires every 2nd output -> period 2*spo

    @pytest.mark.parametrize("spo, step, expected", [
        (5, 0, True), (5, 10, True), (5, 20, True),
        (5, 5, False), (5, 15, False),                 # outputs, but not every-2nd
        (4, 8, True), (4, 16, True), (4, 12, False),
    ])
    def test_fires_every_nth_output(self, spo, step, expected):
        assert self.cadence.due(step, steps_per_output=spo) is expected

    def test_n_one_matches_every_output(self):
        n1, out = EveryNOutputs(1), EveryOutput()
        for step in (0, 5, 7, 10, 12):
            assert n1.due(step, steps_per_output=5) is out.due(step, steps_per_output=5)


# --- abstract base ------------------------------------------------------------

class TestCadenceBase:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            Cadence()

    def test_subclass_without_due_is_abstract(self):
        class Incomplete(Cadence):     # does not implement due()
            pass
        with pytest.raises(TypeError):
            Incomplete()
