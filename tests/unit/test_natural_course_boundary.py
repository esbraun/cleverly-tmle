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
    settings: dict[str, Any] = {"estimands": ("ey_obs",), "cross_fit": False}
    settings.update(estimator_overrides)
    estimator = fast_tmle(**settings)
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
        ({"cross_fit": True}, "cross_fit=False"),
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
        _fit(_frame(), **overrides)


def test_repeated_splits_refuse_at_the_existing_method_boundary() -> None:
    with pytest.raises(ValueError, match="cross_fit=False makes no split"):
        _fit(_frame(), repeats=2)


@pytest.mark.parametrize(
    "estimator",
    [
        pytest.param(CTMLE(strategy="oat", estimands=("ey_obs",), cross_fit=False), id="ctmle"),
        pytest.param(DRTMLE(estimands=("ey_obs",), cross_fit=False), id="drtmle-guarded"),
        pytest.param(
            DRTMLE(estimands=("ey_obs",), guard=(), cross_fit=False), id="drtmle-unguarded"
        ),
    ],
)
def test_estimator_variants_refuse_before_nuisance_fitting(estimator: Any) -> None:
    with pytest.raises(CapabilityError, match="use ordinary TMLE"):
        estimator.fit(
            _frame(),
            outcome="Y",
            treatment="A",
            covariates=("W",),
            delta="Delta",
        )


@pytest.mark.parametrize(
    ("fit_roles", "message"),
    [
        ({"weights": "weight"}, "observation weights"),
        ({"id": "id"}, "clustered inference"),
        ({"strata": ("stratum",)}, "baseline strata"),
        ({"intermediate": "Z"}, "intermediate="),
    ],
)
def test_unsupported_data_compositions_refuse(fit_roles: dict[str, Any], message: str) -> None:
    estimator = fast_tmle(estimands=("ey_obs",), cross_fit=False)
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


def test_multi_arm_treatment_refuses() -> None:
    with pytest.raises(CapabilityError, match="exactly two arms"):
        _fit(_frame(multi_arm=True))


def test_continuous_outcome_requires_fixed_bounds() -> None:
    with pytest.raises(CapabilityError, match="analyst-declared q_bounds"):
        _fit(_frame(continuous=True))


def test_bounded_continuous_outcome_is_supported() -> None:
    result = _fit(_frame(continuous=True), q_bounds=(-1.0, 1.0)).single()
    assert np.isfinite(result.psi("ey_obs"))


def test_response_warning_reads_only_the_realized_natural_course() -> None:
    law = _RealizedResponseSupport()
    w = np.tile(np.array([0.0, 1.0]), 100)
    a = w.copy()
    delta = (np.arange(w.size) % 5 != 0).astype(float)
    mean = law.outcome_mean(w, a)
    y = (np.arange(w.size) % 3 < np.rint(3 * mean)).astype(float)
    frame = pd.DataFrame({"Y": np.where(delta == 1.0, y, np.nan), "A": a, "W": w, "Delta": delta})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fast_tmle(
            estimands=("ey_obs",),
            outcome_learner=OracleOutcome(law),
            missingness_learner=OracleMissingness(law),
            cross_fit=False,
            nuisance_bound=0.01,
        ).fit(frame, outcome="Y", treatment="A", covariates=("W",), delta="Delta")

    assert not any(issubclass(item.category, PositivityWarning) for item in caught)
