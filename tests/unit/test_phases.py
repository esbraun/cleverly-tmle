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
from typing import Any

import numpy as np
import pytest
import sklearn.linear_model

from cleverly.datasets import make_longitudinal
from cleverly.longitudinal import LTMLE, LongitudinalData
from cleverly.utils.phases import phase, profile_phases

#: Two is enough and four is not better: what is checked is that the parallel branch
#: is taken at all, which the same constant in ``test_parallel_invariance`` says too.
PARALLEL_JOBS = 2


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
        outcome_learner=sklearn.linear_model.LinearRegression(),
        pseudo_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
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
    # Two folds x two regimens x two nodes for each fold-specific recursion.  Masks are
    # scanned once per regimen, plus once for the shared mechanism -- the whole point of
    # the prefix scan, and a per-node count here would mean it had stopped happening.
    assert profile.counts["outcome_learner_fit"] == 8
    assert profile.counts["fluctuation"] == 8
    assert profile.counts["mask_construction"] == 3
    # One fan-out per regimen, which is the phase the workers' own phases hang under.
    assert profile.counts["outer_fold_recursion"] == 2
    assert profile.total_seconds > 0.0
    assert sum(profile.exclusive.values()) <= profile.total_seconds


def test_the_recursion_is_profiled_whether_or_not_it_ran_in_workers() -> None:
    """``n_jobs`` decides where the recursion runs, not whether it is measured.

    The four phases inside a fold recursion used to vanish entirely above one job: joblib's
    default backend is loky, the collector is thread-local, and a worker process starts
    with none.  A profile that silently drops the phases the fit spends its time in is
    worse than no profile, because the remaining shares still sum to something plausible.

    ``total_counts`` is the comparison rather than ``counts`` because *where* the work ran
    is a real difference the profile should keep: inline at ``n_jobs=1``, in ``workers``
    above it.  What must not differ is that it was measured at all.
    """
    frame, _ = make_longitudinal(n=300, seed=31)
    columns = {
        "outcome": "Y",
        "treatment": ["A1", "A2"],
        "baseline": ["W1", "W2"],
        "time_varying": [[], ["L2"]],
        "censoring": ["C1", "C2"],
    }

    def counts(n_jobs: int) -> tuple[dict[str, int], Any]:
        estimator = LTMLE(
            {"always": 1, "never": 0},
            outcome_learner=sklearn.linear_model.LinearRegression(),
            pseudo_learner=sklearn.linear_model.LinearRegression(),
            treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
            n_folds=2,
            learner_folds=2,
            random_state=0,
            n_jobs=n_jobs,
        )
        with LTMLE.profile_phases() as profile:
            estimator.fit(frame, **columns)
        return profile.total_counts, profile

    serial, serial_profile = counts(1)
    parallel, parallel_profile = counts(PARALLEL_JOBS)
    assert serial == parallel
    assert serial["fluctuation"] == 8

    # Where they ran, which is the difference the profile keeps rather than hides.
    assert serial_profile.workers is None
    assert parallel_profile.workers is not None
    assert parallel_profile.counts.get("fluctuation", 0) == 0
    assert parallel_profile.workers.counts["fluctuation"] == 8
    # Worker time is processor time across the folds and the parent's is wall time, so the
    # merge must never have put one into the other.
    assert sum(parallel_profile.exclusive.values()) <= parallel_profile.total_seconds


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
        outcome_learner=sklearn.linear_model.LinearRegression(),
        pseudo_learner=sklearn.linear_model.LinearRegression(),
        treatment_learner=sklearn.linear_model.LogisticRegression(max_iter=1000),
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
