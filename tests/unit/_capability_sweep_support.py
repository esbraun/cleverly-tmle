"""Builders for the kinds of fit whose capability rows RM23 checks against their calls.

A capability row reads ``available`` before any call. RM23 found rows that read
``available`` while their call then refused, and each such row was a kind of fit that
nobody had asked the row about. :data:`KINDS` names one builder per kind, so every test
that asks a row about a kind asks it about the same fit.

Each builder fits in sample with the explicit linear learners of
:func:`tests.conftest.linear_in_sample`, unless its docstring says otherwise. Each call
returns a fresh result, so no test reads a facade or a cached report that another test
filled.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.assessment import AssessmentStatus, replayability
from cleverly.datasets import (
    make_binary_outcome,
    make_instrument,
    make_linear_ate,
    make_missing_outcome,
    make_missing_outcome_binary,
)
from cleverly.estimators import DRTMLE, TMLE
from cleverly.exceptions import CapabilityError, DataError
from cleverly.interventions import Incremental, Shift
from cleverly.msm import MSM
from tests import discrete_law, discrete_law_mar, regimes
from tests.conftest import SELECTOR_CONFIGS, linear_ctmle, linear_in_sample


def _missing_frame() -> Any:
    """The missing-outcome law the RM23 probes fitted, ``make_missing_outcome(400, 4)``."""
    frame, _ = make_missing_outcome(n=400, seed=4)
    return frame


def fit_ordinary() -> Any:
    """A binary-treatment ATE fit with complete outcomes."""
    frame, _ = make_linear_ate(n=400, seed=2)
    return TMLE(**linear_in_sample()).fit(frame, outcome="Y", treatment="A").single()


def fit_missing() -> Any:
    """A binary-treatment fit with missing outcomes, which both tilt rows admit."""
    return (
        TMLE(**linear_in_sample())
        .fit(_missing_frame(), outcome="Y", treatment="A", delta="Delta")
        .single()
    )


def fit_incremental() -> Any:
    """An incremental fit with complete outcomes, whose estimand contains the propensity."""
    frame, _ = make_linear_ate(n=400, seed=2)
    return (
        TMLE(**linear_in_sample(incremental=[Incremental(2.0)]))
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


def fit_shift_missing() -> Any:
    """A shift fit with missing outcomes. The dose is the arm plus standard normal noise."""
    frame = _missing_frame()
    noise = np.random.default_rng(0).standard_normal(len(frame))
    dose = frame.assign(A=frame["A"] + noise)
    return (
        TMLE(**linear_in_sample(shifts=[Shift(0.5, cap=None)]))
        .fit(dose, outcome="Y", treatment="A", delta="Delta")
        .single()
    )


def fit_incremental_missing() -> Any:
    """An incremental fit with missing outcomes, which has both truncation axes."""
    return (
        TMLE(**linear_in_sample(incremental=[Incremental(2.0)]))
        .fit(_missing_frame(), outcome="Y", treatment="A", delta="Delta")
        .single()
    )


def fit_regime_missing() -> Any:
    """The three regimes of :mod:`tests.regimes` on the missing-outcome discrete law."""
    return (
        TMLE(**linear_in_sample(interventions=regimes.interventions()))
        .fit(
            discrete_law_mar.frame(),
            outcome="Y",
            treatment="A",
            covariates=("W",),
            delta="Delta",
        )
        .single()
    )


def fit_msm_missing() -> Any:
    """A linear working-model fit with missing outcomes, which reports coefficients only."""
    return (
        TMLE(**linear_in_sample(msm=MSM.linear()))
        .fit(_missing_frame(), outcome="Y", treatment="A", delta="Delta")
        .single()
    )


def fit_rr_missing() -> Any:
    """A risk-ratio-only fit with missing binary outcomes."""
    frame, _ = make_missing_outcome_binary(n=400, seed=4)
    return (
        TMLE(**linear_in_sample(estimands=("rr",)))
        .fit(frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )


def fit_natural_course() -> Any:
    """The missing-outcome natural-course mean, which fits no treatment mechanism.

    The outcome is binary, because this target refuses a continuous outcome without
    declared ``q_bounds``.
    """
    frame, _ = make_missing_outcome_binary(n=400, seed=4)
    return (
        TMLE(**linear_in_sample(estimands=("ey_obs",)))
        .fit(frame, outcome="Y", treatment="A", delta="Delta")
        .single()
    )


def fit_split_plan() -> Any:
    """A cross-fitted fit on the discrete law, given the split plan of a first fit.

    Two folds and a binary outcome, so no outcome support has to be declared. The plan
    labels the rows it was realised on, so ``refute`` refuses ``subset`` here.
    """
    frame = discrete_law.frame()
    folds = {"cross_fit": True, "n_folds": 2}
    first = TMLE(**linear_in_sample(**folds)).fit(frame, outcome="Y", treatment="A").single()
    supplied = TMLE(**linear_in_sample(**folds, split_plan=first.split_plan))
    return supplied.fit(frame, outcome="Y", treatment="A").single()


#: The covariate order :func:`fit_ctmle_ordered` declares, as ``make_instrument`` names it.
INSTRUMENT_ORDERING = ("W1", "W2", "W3")


def ctmle_ordered(ordering: tuple[str, ...] = INSTRUMENT_ORDERING) -> Any:
    """The ordered collaborative estimator of :func:`fit_ctmle_ordered`, at ``ordering``."""
    return linear_ctmle(
        "ordered",
        ordering=ordering,
        selection_folds=SELECTOR_CONFIGS["ordered"]["selection_folds"],
        estimands=("ate",),
    )


def fit_ctmle_ordered() -> Any:
    """An ordered collaborative fit with an explicit ordering, on ``make_instrument(500, 44)``."""
    frame, _ = make_instrument(n=500, seed=44)
    return ctmle_ordered().fit(frame, outcome="Y", treatment="A").single()


def drtmle_settings() -> dict[str, Any]:
    """Cross-fitted DR-TMLE with explicit linear learners, as the companion tests fit it."""
    return {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "reduced_outcome_learner": LinearRegression(),
        "reduced_treatment_learner": LogisticRegression(max_iter=1000),
        "n_folds": 3,
        "learner_folds": 2,
        "random_state": 0,
        "simultaneous": False,
        "estimands": ("ate",),
    }


def fit_drtmle(*, companion: bool = False) -> Any:
    """A DR-TMLE fit of a binary outcome, with an ``evaluation=`` companion when asked.

    The outcome is binary, so the cross-fitted fit needs no declared ``q_bounds``. The
    companion is an independent draw of the same law.
    """
    frame, _ = make_binary_outcome(n=240, seed=11)
    evaluation = make_binary_outcome(n=120, seed=12)[0] if companion else None
    estimator = DRTMLE(**drtmle_settings(), evaluation=evaluation)
    return estimator.fit(frame, outcome="Y", treatment="A").single()


def fit_drtmle_companion() -> Any:
    """:func:`fit_drtmle` with an ``evaluation=`` companion."""
    return fit_drtmle(companion=True)


def replay_disagreements(result: Any, estimands: tuple[str, ...]) -> list[str]:
    """Every point replay slot of ``result`` that disagrees with the call it stands for.

    ``retarget_cached_nuisances`` stands for ``estimator.retarget`` on the cached
    nuisances, and ``refit_nuisances`` for ``estimator.refit`` on the result's own data.
    A :class:`~cleverly.exceptions.CapabilityError` or
    :class:`~cleverly.exceptions.DataError` is a refusal. Any other exception is a defect,
    so it propagates. An empty list is agreement.
    """
    replay = replayability(result)
    estimator = result.estimator
    calls: dict[str, Callable[[], Any]] = {
        "retarget_cached_nuisances": lambda: estimator.retarget(
            result.data, result.nuisance, estimands=estimands
        ),
        "refit_nuisances": lambda: estimator.refit(
            result.data, intermediate_value=result.intermediate_value
        ),
    }
    problems = []
    for slot, call in calls.items():
        try:
            call()
        except (CapabilityError, DataError) as error:
            refused: str | None = f"{type(error).__name__}: {error}"
        else:
            refused = None
        if getattr(replay, slot) != (refused is None):
            problems.append(f"{slot} reads {getattr(replay, slot)}, and the call gave {refused}")
    return problems


def assert_replay_agrees(result: Any, estimands: tuple[str, ...]) -> None:
    """Each point replay slot of ``result`` reads true exactly when its call runs."""
    assert replay_disagreements(result, estimands) == []


def assert_replay_rows_refused(result: Any) -> None:
    """Every row that needs a replay slot reads unavailable, and a combined report runs.

    For a restored result whose replay slots read false. The list of rows is not empty,
    and the ``refute`` row reads unavailable too. A longitudinal ``refute`` row declares
    no replay slot, because it is unavailable for every longitudinal result.
    """
    rows = [
        row
        for facade in (result.diagnostics, result.sensitivity)
        for row in facade.capabilities
        if row.requires_replay is not None
    ]
    assert rows
    assert [row.operation for row in rows if row.available] == []
    assert not result.diagnostics.capability("refute").available
    report = result.assess(include_refits=True, include_retargets=True, random_state=0)
    assert report.diagnostics["refute"].status is AssessmentStatus.UNAVAILABLE


#: One builder per kind of fit, by the name the RM23 plan gives it.
KINDS: dict[str, Callable[[], Any]] = {
    "ordinary": fit_ordinary,
    "missing": fit_missing,
    "incremental": fit_incremental,
    "shift+missing": fit_shift_missing,
    "incremental+missing": fit_incremental_missing,
    "regime+missing": fit_regime_missing,
    "msm+missing": fit_msm_missing,
    "rr+missing": fit_rr_missing,
    "natural_course": fit_natural_course,
    "split_plan": fit_split_plan,
    "ctmle_ordered": fit_ctmle_ordered,
    "drtmle_companion": fit_drtmle_companion,
}
