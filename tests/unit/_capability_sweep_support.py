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

from cleverly.datasets import make_linear_ate, make_missing_outcome, make_missing_outcome_binary
from cleverly.estimators import TMLE
from cleverly.interventions import Incremental, Shift
from cleverly.msm import MSM
from tests import discrete_law_mar, regimes
from tests.conftest import linear_in_sample


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
}
