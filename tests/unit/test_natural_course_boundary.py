"""The deliberately narrow first implementation boundary for missing ``ey_obs``."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from cleverly import CapabilityError, PositivityWarning
from cleverly.estimators import CTMLE, DRTMLE
from tests.conftest import OracleMissingness, OracleOutcome, fast_tmle
from tests.unit._natural_course_support import (
    NeverFit,
    in_sample_tmle,
    never_fit_learners,
    stacked_tmle,
)


class _RealizedResponseSupport:
    """A law with safe realized response scores and impossible off-course cells."""

    @staticmethod
    def outcome_mean(w: Any, a: Any, z: Any = None) -> np.ndarray:
        levels = np.asarray(w, dtype=float).reshape(-1)
        arms = np.broadcast_to(np.asarray(a, dtype=float), levels.shape)
        return 0.25 + 0.15 * arms + 0.10 * levels

    @staticmethod
    def missingness(w: Any, a: Any) -> np.ndarray:
        levels = np.rint(np.asarray(w, dtype=float).reshape(-1))
        arms = np.broadcast_to(np.asarray(a, dtype=float), levels.shape)
        return np.where(arms == levels, 0.8, 0.001)


def _frame(*, continuous: bool = False, multi_arm: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(310)
    n = 150
    w = rng.normal(size=n)
    a = rng.integers(0, 3, size=n) if multi_arm else rng.binomial(1, 0.5, size=n)
    mean = 0.2 + 0.15 * a + 0.1 * w
    y = (
        mean + rng.normal(scale=0.1, size=n)
        if continuous
        else rng.binomial(1, 1 / (1 + np.exp(-mean)))
    )
    y = np.asarray(y, dtype=float)
    delta = rng.binomial(1, 0.8, size=n)
    y[delta == 0] = np.nan
    return pd.DataFrame(
        {
            "Y": y,
            "A": a,
            "W": w,
            "Delta": delta,
            "weight": np.linspace(0.5, 1.5, n),
            "id": np.arange(n) // 2,
            "stratum": (w > 0).astype(int),
            "Z": (w > -0.2).astype(int),
        }
    )


def _fit(frame: pd.DataFrame, **estimator_overrides: Any) -> Any:
    estimator = in_sample_tmle(**estimator_overrides)
    return estimator.fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=("W",),
        delta="Delta",
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cross_fit": True}, "stratify_folds='none'"),
        ({"fluctuation": "linear"}, "fluctuation='logistic'"),
        ({"targeting": "one_step"}, "targeting='iterative'"),
        ({"target_weights": True}, "target_weights=False"),
        ({"n_bootstrap": 2}, "n_bootstrap=0"),
        ({"estimands": ("ey_obs", "ate")}, "joint targeting"),
    ],
)
def test_unsupported_method_settings_refuse_before_fitting(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(CapabilityError, match=message):
        _fit(_frame(), **never_fit_learners(), **overrides)
    assert NeverFit.calls == 0


def test_repeated_splits_refuse_at_the_existing_method_boundary() -> None:
    with pytest.raises(ValueError, match="cross_fit=False makes no split"):
        _fit(_frame(), repeats=2)


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(
            lambda: CTMLE(
                strategy="oat", estimands=("ey_obs",), cross_fit=False, **never_fit_learners()
            ),
            id="ctmle",
        ),
        pytest.param(
            lambda: DRTMLE(estimands=("ey_obs",), cross_fit=False, **never_fit_learners()),
            id="drtmle-guarded",
        ),
        pytest.param(
            lambda: DRTMLE(
                estimands=("ey_obs",), guard=(), cross_fit=False, **never_fit_learners()
            ),
            id="drtmle-unguarded",
        ),
    ],
)
def test_estimator_variants_refuse_before_nuisance_fitting(build: Any) -> None:
    estimator = build()
    with pytest.raises(CapabilityError, match="use ordinary TMLE"):
        estimator.fit(
            _frame(),
            outcome="Y",
            treatment="A",
            covariates=("W",),
            delta="Delta",
        )
    assert NeverFit.calls == 0


#: The in-sample TMLE and the stacked CV-TMLE share every data-axis refusal. Each one
#: builds its learners from :func:`never_fit_learners`, so a refusal that let any learner
#: fit first fails on the assertion in the learner or on its call count.
ESTIMATORS = [
    pytest.param(in_sample_tmle, id="in-sample"),
    pytest.param(stacked_tmle, id="stacked"),
]


@pytest.mark.parametrize("build", ESTIMATORS)
@pytest.mark.parametrize(
    ("fit_roles", "message"),
    [
        ({"weights": "weight"}, "observation weights"),
        ({"id": "id"}, "clustered inference"),
        ({"strata": ("stratum",)}, "baseline strata"),
        ({"intermediate": "Z"}, "intermediate="),
    ],
    ids=("weighted", "clustered", "strata", "intermediate"),
)
def test_unsupported_data_compositions_refuse_before_fitting(
    build: Any, fit_roles: dict[str, Any], message: str
) -> None:
    estimator = build(**never_fit_learners())
    covariates = ("W", "stratum") if "strata" in fit_roles else ("W",)
    with pytest.raises(CapabilityError, match=message):
        estimator.fit(
            _frame(),
            outcome="Y",
            treatment="A",
            covariates=covariates,
            delta="Delta",
            **fit_roles,
        )
    assert NeverFit.calls == 0


@pytest.mark.parametrize("build", ESTIMATORS)
def test_multi_arm_treatment_refuses_before_fitting(build: Any) -> None:
    estimator = build(**never_fit_learners())
    with pytest.raises(CapabilityError, match="exactly two arms"):
        estimator.fit(
            _frame(multi_arm=True), outcome="Y", treatment="A", covariates=("W",), delta="Delta"
        )
    assert NeverFit.calls == 0


@pytest.mark.parametrize(
    ("build", "message"),
    [
        pytest.param(in_sample_tmle, "analyst-declared q_bounds", id="in-sample"),
        pytest.param(stacked_tmle, "requires a binary outcome", id="stacked"),
    ],
)
def test_an_unaudited_continuous_outcome_refuses_before_fitting(build: Any, message: str) -> None:
    estimator = build(**never_fit_learners())
    with pytest.raises(CapabilityError, match=message):
        estimator.fit(
            _frame(continuous=True), outcome="Y", treatment="A", covariates=("W",), delta="Delta"
        )
    assert NeverFit.calls == 0


def test_bounded_continuous_outcome_is_supported() -> None:
    result = _fit(_frame(continuous=True), q_bounds=(-1.0, 1.0)).single()
    assert np.isfinite(result.psi("ey_obs"))


def _response_frame(law: Any) -> pd.DataFrame:
    w = np.tile(np.array([0.0, 1.0]), 100)
    a = w.copy()
    delta = (np.arange(w.size) % 5 != 0).astype(float)
    mean = law.outcome_mean(w, a)
    y = (np.arange(w.size) % 3 < np.rint(3 * mean)).astype(float)
    return pd.DataFrame({"Y": np.where(delta == 1.0, y, np.nan), "A": a, "W": w, "Delta": delta})


def _fit_for_warnings(law: Any) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fast_tmle(
            estimands=("ey_obs",),
            outcome_learner=OracleOutcome(law),
            missingness_learner=OracleMissingness(law),
            cross_fit=False,
            nuisance_bound=0.01,
        ).fit(_response_frame(law), outcome="Y", treatment="A", covariates=("W",), delta="Delta")
    return list(caught)


def test_response_warning_reads_only_the_realized_natural_course() -> None:
    """The negative half: dangerous *off-course* cells must not raise the warning.

    Paired with the positive half below, which uses the same frame and the mirrored
    law. One alone would pass on a construction that always warns or never does.
    """
    caught = _fit_for_warnings(_RealizedResponseSupport())
    assert not any(issubclass(item.category, PositivityWarning) for item in caught)


class _LowRealizedResponse:
    """The mirror of :class:`_RealizedResponseSupport`: the realized cells are the bad ones."""

    @staticmethod
    def outcome_mean(w: Any, a: Any, z: Any = None) -> np.ndarray:
        levels = np.asarray(w, dtype=float).reshape(-1)
        arms = np.broadcast_to(np.asarray(a, dtype=float), levels.shape)
        return 0.25 + 0.15 * arms + 0.10 * levels

    @staticmethod
    def missingness(w: Any, a: Any) -> np.ndarray:
        levels = np.rint(np.asarray(w, dtype=float).reshape(-1))
        arms = np.broadcast_to(np.asarray(a, dtype=float), levels.shape)
        return np.where(arms == levels, 0.002, 0.9)


def test_a_low_realized_response_warns_with_the_natural_course_guidance() -> None:
    """The positive half of the realized-arm reduction.

    Its negative half above shows no warning when the *off-course* cells are the
    dangerous ones. Without this case the reduction would also pass if it silenced the
    warning outright, so the two together pin which column is read rather than only
    that some column is.
    """
    positivity = [
        item for item in _fit_for_warnings(_LowRealizedResponse()) if _is_positivity(item)
    ]
    assert positivity, "a response probability of 0.002 under a 0.01 bound must warn"
    message = str(positivity[0].message)
    assert "P(Delta = 1 | A, W)" in message
    # The natural-course sentence, not the arm-indexed one: this fit has no g(W) for
    # the clever covariate to divide by, so guidance naming g would be false.
    assert "natural-course response-residual covariate" in message
    assert "res.diagnostics.nuisance_models()" in message
    assert "just as g(W) does" not in message


def _is_positivity(item: warnings.WarningMessage) -> bool:
    return issubclass(item.category, PositivityWarning) and "P(Delta = 1" in str(item.message)


def test_ey_obs_beside_an_intermediate_refuses_on_a_complete_outcome() -> None:
    """The complete-outcome composition the natural-course boundary cannot see.

    ``_resolve_estimands_for_data`` returns at its first branch when no outcome is
    missing, so its ``intermediate=`` refusal never runs here. The guard that does run
    keyed on ``POPULATION_INTERVENTION_TARGETS``, which ``ey_obs`` left when its
    missing-outcome score equation landed. Without a replacement the fit succeeded and
    published the plain empirical mean of ``Y`` once per level of ``Z``, labelling each
    copy a controlled direct effect.
    """
    frame = _frame()
    frame = frame.assign(Y=frame["Y"].fillna(0.0), Z=(np.arange(len(frame)) % 2).astype(float))
    estimator = fast_tmle(estimands=("ey_obs",), cross_fit=False)
    with pytest.raises(CapabilityError, match=r"\['ey_obs'\] do not yet support intermediate="):
        estimator.fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=("W",),
            intermediate="Z",
        )


def test_all_estimands_beside_an_intermediate_drops_ey_obs_rather_than_refusing() -> None:
    """``estimands="all"`` asks for what the data supports, so it drops rather than fails.

    The refusal above must not reach this path. ``all`` named no target, so a
    composition it cannot express is a target to leave out, exactly as ``par`` and
    ``paf`` are left out. Refusing here would break a design that worked before
    ``ey_obs`` left ``POPULATION_INTERVENTION_TARGETS``, which is the shape of the
    regression the refusal above repairs.
    """
    frame = _frame()
    frame = frame.assign(Y=frame["Y"].fillna(0.0), Z=(np.arange(len(frame)) % 2).astype(float))
    result = (
        fast_tmle(estimands="all", cross_fit=False)
        .fit(frame, outcome="Y", treatment="A", covariates=("W",), intermediate="Z")
        .get(0.0)
    )
    assert result is not None
    reported = set(result.estimates)
    assert "ey_obs" not in reported
    assert not reported & {"par", "paf"}
    # It still reported the targets the composition does express, so the drop is a
    # narrowing and not a silent empty fit.
    assert {"ate", "ey1", "ey0"} <= reported


def test_the_summary_names_the_propensity_bound_only_when_a_propensity_was_fitted() -> None:
    """The witness for ``TMLEConfig.fits_treatment``.

    Deleting the guard leaves every other test green, because the line it suppresses
    is only wrong on the one fit that estimates no treatment mechanism. Both halves
    are asserted here: an ordinary fit must keep the line, and this one must not,
    so neither a missing guard nor an always-on one passes.
    """
    missing = _fit(_frame()).single()
    assert not missing.config.fits_treatment
    assert "propensity truncated" not in missing.summary()
    # It still names the mechanism it did bound, so the suppression is narrow.
    assert "P(Delta=1|A,W)" in missing.summary()

    complete = _frame()
    complete = complete.assign(Y=complete["Y"].fillna(0.0))
    ordinary = (
        fast_tmle(cross_fit=False)
        .fit(complete, outcome="Y", treatment="A", covariates=("W",))
        .single()
    )
    assert ordinary.config.fits_treatment
    assert "propensity truncated" in ordinary.summary()
