"""The phase collector: off by default, nested correctly, and installed where it says.

The collector exists so that "where does a longitudinal fit's time go" is answered by
phases that name themselves rather than by bucketing a ``cProfile`` on filename.  Three
things could make it worse than the instrument it replaces, and each has a test: it could
cost something when disabled, it could double-count a nested phase into its parent, and it
could be wired into the wrong place -- which a timing would never catch, because a wrong
share still looks like a share.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from cleverly import LTMLE, LongitudinalData
from cleverly.datasets import make_longitudinal
from cleverly.utils.phases import phase, profile_phases


def test_disabled_by_default_and_records_nothing() -> None:
    with phase("nothing"):
        pass
    with profile_phases() as profile:
        pass
    assert profile.inclusive == {}
    assert profile.counts == {}


def test_disabled_entry_allocates_no_span() -> None:
    """The same shared object every time, which is what makes the disabled path free."""
    assert phase("a") is phase("b")


def test_a_phase_records_its_time_and_its_count() -> None:
    with profile_phases() as profile:
        for _ in range(3):
            with phase("work"):
                time.sleep(0.01)
    assert profile.counts["work"] == 3
    assert profile.inclusive["work"] >= 0.03
    assert profile.total_seconds >= profile.inclusive["work"]


def test_a_nested_phase_is_subtracted_from_its_parent() -> None:
    """Inclusive overlaps, exclusive partitions -- the distinction the report rests on."""
    with profile_phases() as profile, phase("outer"):
        time.sleep(0.01)
        with phase("inner"):
            time.sleep(0.02)
    assert profile.inclusive["outer"] >= 0.03
    assert profile.inclusive["inner"] >= 0.02
    assert profile.exclusive["inner"] == pytest.approx(profile.inclusive["inner"])
    # The parent keeps only its own 10 ms, not the child's 20.
    assert profile.exclusive["outer"] < profile.inclusive["outer"] - 0.015
    accounted = sum(profile.exclusive.values())
    assert accounted <= profile.total_seconds + 1e-9


def test_the_stack_survives_an_exception() -> None:
    with profile_phases() as profile:
        with pytest.raises(RuntimeError, match="deliberate"), phase("outer"), phase("inner"):
            raise RuntimeError("deliberate")
        with phase("after"):
            pass
    assert profile.counts == {"inner": 1, "outer": 1, "after": 1}


def test_nesting_two_profiles_is_refused() -> None:
    with (
        profile_phases(),
        pytest.raises(RuntimeError, match="already being collected"),
        profile_phases(),
    ):
        pass


def test_the_collector_is_removed_after_the_block() -> None:
    with profile_phases():
        pass
    with profile_phases() as second, phase("work"):
        pass
    assert second.counts == {"work": 1}


def test_an_ltmle_fit_reports_every_phase_it_declares() -> None:
    """The wiring, checked by name.

    A phase that is never entered is a share silently missing from the denominator, and a
    phase entered in the wrong place is a share attributed to the wrong thing.  Neither
    shows up in a timing, so the set of names and the per-node counts are asserted
    directly.
    """
    frame, _ = make_longitudinal(n=600, seed=2)
    data = LongitudinalData.from_frame(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )
    estimator = LTMLE(
        {"always": 1, "never": 0},
        outcome_learner="glm",
        pseudo_learner="glm",
        treatment_learner="glm",
        n_folds=2,
        learner_folds=2,
        random_state=0,
    )
    with LTMLE.profile_phases() as profile:
        estimator.fit(data)

    expected = {
        "mechanism_fit",
        "mask_construction",
        "pseudo_outcome",
        "outcome_learner_fit",
        "clever_covariate",
        "fluctuation",
        "influence_curve",
        "inference",
    }
    assert expected <= set(profile.counts)
    # Two regimens x two nodes for the per-node phases; one scan per regimen plus the
    # mechanism's, which is what the change to a prefix scan is *for*.
    assert profile.counts["outcome_learner_fit"] == 4
    assert profile.counts["fluctuation"] == 4
    assert profile.counts["mask_construction"] == 5
    assert profile.total_seconds > 0.0
    assert sum(profile.exclusive.values()) <= profile.total_seconds


def test_profiling_does_not_change_the_fit() -> None:
    frame, _ = make_longitudinal(n=600, seed=2)
    columns = {
        "outcome": "Y",
        "treatment": ["A1", "A2"],
        "baseline": ["W1", "W2"],
        "time_varying": [[], ["L2"]],
        "censoring": ["C1", "C2"],
    }
    estimator = LTMLE(
        {"always": 1},
        outcome_learner="glm",
        pseudo_learner="glm",
        treatment_learner="glm",
        n_folds=2,
        learner_folds=2,
        random_state=0,
    )
    plain = estimator.fit(frame, **columns)
    with LTMLE.profile_phases():
        profiled = estimator.fit(frame, **columns)
    for name in list(plain.keys()):
        assert plain[name].psi == profiled[name].psi
        np.testing.assert_array_equal(plain[name].influence_curve, profiled[name].influence_curve)
