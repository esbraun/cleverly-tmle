"""Cached builders and independent witnesses for simulated-confounding tests."""

from __future__ import annotations

import importlib
from copy import copy
from dataclasses import dataclass, replace
from functools import cache
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from cleverly import CausalStudy, CollaborativeTMLEMethod, PointTreatment


def _collaborative_method(
    *,
    selection_estimand: str = "ate",
    overrides: dict[str, Any] | None = None,
) -> CollaborativeTMLEMethod:
    settings = dict(overrides or {})
    if settings.get("strategy", "greedy") != "oat":
        settings.setdefault("selection_folds", 2)
        settings.setdefault("selection_inner_folds", 2)
    return CollaborativeTMLEMethod(selection_estimand=selection_estimand, **settings)


#: The extra ``CollaborativeTMLEMethod`` settings each selection strategy needs, beyond the
#: folds ``_collaborative_method`` already defaults.  ``greedy`` and ``oat`` need none.
_STRATEGY_OVERRIDES: dict[str, dict[str, Any]] = {
    "greedy": {},
    "oat": {},
    "ordered": {"preorder": "logistic"},
    "discrete": {"candidates": ((), ("W",))},
}


def _strategy_method(strategy: str, *, selection_estimand: str) -> CollaborativeTMLEMethod:
    # The selector has to score the estimand this fit reports, so it follows the target.
    return _collaborative_method(
        selection_estimand=selection_estimand,
        overrides={"strategy": strategy, **_STRATEGY_OVERRIDES[strategy]},
    )


@cache
def confounding_study(
    *,
    backend: str = "pandas",
    labels: bool = False,
    weighted: bool = False,
    strata: bool = False,
    binary: bool = False,
    law: str = "policy",
    continuous: bool = False,
) -> CausalStudy:
    rng = np.random.default_rng(317 if law == "policy" else 912)
    n = 180
    w = rng.normal(size=n)
    v = np.where(np.arange(n) % 3 == 0, "small", "large")
    if law == "population":
        a = rng.binomial(1, 1 / (1 + np.exp(-0.8 * w + 0.9 * (v == "small"))))
        y = 0.4 + (1.2 + 1.8 * w) * a + 0.5 * w + rng.normal(scale=0.3, size=n)
        if continuous:
            a = 0.5 * w + rng.normal(size=n)
            y = 0.4 + (1.2 + 0.8 * w) * a + 0.5 * w + rng.normal(scale=0.3, size=n)
        elif binary:
            y = rng.binomial(1, 1 / (1 + np.exp(-0.3 - 0.5 * a - 0.4 * w)))
    else:
        a = rng.binomial(1, 1 / (1 + np.exp(-0.7 * w)))
        linear = 0.4 + 0.8 * w + a * (0.9 + 1.6 * w) + 0.3 * w**2
        y = (
            rng.binomial(1, 1 / (1 + np.exp(-linear)))
            if binary
            else linear + rng.normal(scale=0.35, size=n)
        )
    frame = pd.DataFrame(
        {"W": w, "V": v, "A": np.where(a, "treated", "control") if labels else a, "Y": y}
    )
    frame["weight"] = (
        np.where(v == "small", 3.1, 0.7) * np.where(w > 0, 1.8, 0.6)
        if law == "population"
        else np.where(w > 0, 2.8, 0.5)
    )
    if backend == "polars":
        import polars as pl

        frame = pl.from_pandas(frame)
    return CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W", "V"),
            strata=("V",) if strata else (),
            weights="weight" if weighted else None,
            treatment_kind="continuous" if continuous else "discrete",
        ),
    )


@cache
def confounding_estimate(
    study: CausalStudy,
    target: Any,
    *,
    repeats: int = 1,
    binary: bool = False,
    method: Any = "tmle",
) -> Any:
    return study.identify(target).estimate(
        method=method,
        outcome_learner=LogisticRegression(max_iter=1000)
        if binary
        else make_pipeline(PolynomialFeatures(2), LinearRegression()),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=12,
        simultaneous=False,
        repeats=repeats,
    )


def with_functional(result: Any, **changes: Any) -> Any:
    functional = replace(result.identified_effect.functional, **changes)
    identified = replace(result.identified_effect, functional=functional)
    return replace(result, identified_effect=identified)


def replacement(result: Any, surface: Any, treatment: float, outcome: float) -> Any:
    """Reconstruct the published law without the production perturbation helpers."""
    data = result.data
    latent = np.random.default_rng(surface.latent_seed).normal(size=data.n)
    a = data.treatment.copy()
    if data.is_continuous_treatment:
        a += treatment * latent
    elif treatment:
        changed = latent >= NormalDist().inv_cdf(1 - treatment)
        a[changed] = 1 - a[changed]
    y = data.outcome.copy()
    if data.family == "gaussian":
        y -= outcome * latent
    elif outcome:
        changed = latent >= NormalDist().inv_cdf(1 - outcome)
        y[changed] = 1 - y[changed]
    return data.with_treatment(a).with_outcome(
        y, family=data.family, name="simulated-confounding outcome"
    )


def forbid_draw_and_refit(monkeypatch: pytest.MonkeyPatch, estimator: Any) -> None:
    """Fail the test if the surface draws a latent vector or refits before it refuses."""
    module = importlib.import_module("cleverly.sensitivity.simulated_confounding")
    monkeypatch.setattr(module, "_latent_child_seed", lambda *_: pytest.fail("drew before refusal"))
    monkeypatch.setattr(
        estimator, "refit", lambda *_args, **_kwargs: pytest.fail("refit before refusal")
    )


def baseline_mask(result: Any, stratum: tuple[str, ...] | None) -> np.ndarray:
    if stratum is None:
        return np.ones(result.data.n, dtype=bool)
    code = result.data.strata_levels.index(stratum)
    return result.data.strata == code


def with_typed(result: Any, **changes: Any) -> Any:
    typed = replace(result.identified_effect.estimand, **changes)
    return replace(result, identified_effect=replace(result.identified_effect, estimand=typed))


def with_key(result: Any, alias: str, /, **changes: Any) -> Any:
    return replace(
        result,
        parameter_keys={
            **result.parameter_keys,
            alias: replace(result.parameter_keys[alias], **changes),
        },
    )


def with_estimator(result: Any, **changes: Any) -> Any:
    estimator = copy(result.estimator)
    for name, value in changes.items():
        setattr(estimator, name, value)
    return replace(result, estimator=estimator)


def with_last_nuisance(result: Any, **changes: Any) -> Any:
    draw = result.repeats[-1]
    return replace(
        result,
        repeats=(
            *result.repeats[:-1],
            replace(draw, nuisance=replace(draw.nuisance, **changes)),
        ),
    )


def alias_for(
    result: Any,
    target: str | None = None,
    stratum: tuple[str, ...] | None = None,
    *,
    value: Any = None,
    coefficient: str | None = None,
    estimate_prefix: str | None = None,
) -> str:
    if estimate_prefix is not None:
        # Corruption tests must still find the requested alias after they alter its key.
        matches = [alias for alias in result.estimates if alias.startswith(estimate_prefix)]
        assert len(matches) == 1
        return matches[0]
    matches = [
        alias
        for alias, key in result.parameter_keys.items()
        if (target is None or key.estimand == target)
        and key.stratum == stratum
        and (value is None or key.value == value)
        and (coefficient is None or key.term == coefficient)
    ]
    assert matches
    return sorted(matches)[0]


@dataclass(eq=False)
class Counter:
    function: Any
    calls: int = 0
    limit: int | None = None

    def __call__(self, *args: Any) -> Any:
        self.calls += 1
        if self.limit is not None and self.calls > self.limit:
            raise AssertionError("policy callback was replayed after validation")
        return self.function(*args)
