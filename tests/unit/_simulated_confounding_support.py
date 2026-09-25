"""Shared builders for simulated-confounding tests.

The fixed-policy and MSM builders come first. The shift-policy, support-constant and
attributable builders follow, because the capability-row tests fit the same fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly import (
    ATE,
    CausalStudy,
    CrossFitting,
    DRTMLEMethod,
    ModifiedTreatmentPolicy,
    ModifiedTreatmentPolicyEffect,
    MSMProjection,
    NaturalCourseMean,
    PointTreatment,
    PopulationAttributableFraction,
    PopulationAttributableRisk,
    RegimeContrast,
    RegimeMean,
    TMLEMethod,
)
from cleverly.datasets import make_linear_ate, make_shift_dose
from cleverly.interventions import Rule, Shift, Static, Stochastic
from cleverly.msm import MSM
from cleverly.sensitivity import ConfounderStrengthGrid
from cleverly.sensitivity.simulated_confounding import _latent_child_seed
from tests.conftest import IN_SAMPLE, mean_one_weights
from tests.unit._confounding_support import _STRATEGY_OVERRIDES, _strategy_method, alias_for
from tests.unit._confounding_support import confounding_estimate as _confounding_estimate
from tests.unit._confounding_support import confounding_study as _study

# Repeated fits require cross-fitting. Ordinary fixed-policy replay uses an in-sample fit.
_IN_SAMPLE_METHOD = TMLEMethod(cross_fitting=CrossFitting(enabled=False))


def _estimate(*args: Any, **kwargs: Any) -> Any:
    """Fit in sample unless the caller requests repeated cross-fitting."""
    if kwargs.get("repeats", 1) <= 1:
        kwargs.setdefault("method", _IN_SAMPLE_METHOD)
    return _confounding_estimate(*args, **kwargs)


_GRID = ConfounderStrengthGrid(treatment=(0.0, 0.22), outcome=(0.0, 0.17))


def _stochastic_density(w: Any) -> np.ndarray:
    p = 0.3 + 0.4 * (np.asarray(w["W"]) > 0)
    return np.column_stack((1 - p, p))


@dataclass(frozen=True)
class _MSMDesign:
    treated: Any = 1
    saturated: bool = False

    def __call__(self, a: Any, w: Any) -> np.ndarray:
        columns = [np.ones(len(w)), np.full(len(w), a == self.treated)]
        if not self.saturated:
            columns.append(np.asarray(w["W"]))
        return np.column_stack(columns)


@dataclass(frozen=True)
class _MSMWeight:
    treated: Any = 1

    def __call__(self, a: Any, w: Any) -> np.ndarray:
        return (0.5 + 2.0 * (np.asarray(w["W"]) > 0)) * (1.8 if a == self.treated else 0.6)


def _policy(kind: str, *, labels: bool = False) -> Any:
    """Build a fixed policy from baseline covariates and preserve its arm labels."""
    control, treated = ("control", "treated") if labels else (0, 1)
    if kind == "static":
        return Static(treated, name="policy")
    if kind == "rule":
        return Rule(
            lambda w: np.where(np.asarray(w["W"]) > 0, treated, control),
            name="policy",
            rule_kind="known",
        )
    if kind == "stochastic":
        return Stochastic(_stochastic_density, name="policy", density_kind="known")
    raise AssertionError(kind)


def _model(*, saturated: bool = False, labels: bool = False, link: str = "identity") -> MSM:
    treated = "treated" if labels else 1
    if saturated:
        return MSM(
            design=_MSMDesign(treated, True),
            terms=("intercept", "treatment"),
            link=link,
            design_kind="known",
        )
    return MSM(
        design=_MSMDesign(treated),
        terms=("intercept", "treatment", "baseline"),
        weights=_MSMWeight(treated),
        weights_kind="known",
        link=link,
        design_kind="known",
    )


@cache
def _fit_policy(
    kind: str = "stochastic",
    *,
    contrast: bool = True,
    backend: str = "pandas",
    labels: bool = False,
    weighted: bool = False,
    strata: bool = False,
    repeats: int = 1,
    binary: bool = False,
) -> Any:
    control = "control" if labels else 0
    interventions = (Static(control, name="reference"), _policy(kind, labels=labels))
    target = (
        RegimeContrast(interventions, reference="reference")
        if contrast
        else RegimeMean((interventions[1],))
    )
    return _estimate(
        _study(backend=backend, labels=labels, weighted=weighted, strata=strata, binary=binary),
        target,
        repeats=repeats,
        binary=binary,
    )


@cache
def _fit_msm(
    *,
    saturated: bool = False,
    weighted: bool = False,
    labels: bool = False,
    backend: str = "pandas",
    repeats: int = 1,
    binary: bool = False,
    link: str = "identity",
    strata: bool = False,
) -> Any:
    return _estimate(
        _study(weighted=weighted, labels=labels, backend=backend, binary=binary, strata=strata),
        MSMProjection(_model(saturated=saturated, labels=labels, link=link)),
        repeats=repeats,
        binary=binary,
    )


def _alias(
    result: Any, *, stratum: tuple[str, ...] | None = None, coefficient: str = "treatment"
) -> str:
    return alias_for(
        result,
        stratum=stratum,
        coefficient=coefficient if result.config.parameter_axis == "msm" else None,
        value=None if result.config.parameter_axis == "msm" else "policy",
    )


_TWO_POLICIES = (
    Shift(0.0, cap=3.0, name="natural course"),
    Shift(0.5, cap=3.0, name="up half"),
)


def _fit_shift_policies(
    *,
    family: str = "gaussian",
    seed: int = 7,
    policies: tuple[Shift, ...] = _TWO_POLICIES,
    means: bool = False,
    repeats: int = 1,
    weight_scale: float | None = None,
) -> Any:
    frame, _ = make_shift_dose(n=120, seed=seed)
    if family == "binomial":
        frame["Y"] = (frame["Y"] > frame["Y"].median()).astype(float)
        outcome_learner: Any = LogisticRegression(max_iter=1000)
    else:
        outcome_learner = LinearRegression()
    weight_name = None
    if weight_scale is not None:
        weight_name = "weight"
        frame[weight_name] = weight_scale * mean_one_weights(len(frame))
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3"),
            treatment_kind="continuous",
            weights=weight_name,
        ),
    )
    estimand: Any = (
        ModifiedTreatmentPolicy(shifts=policies)
        if means
        else ModifiedTreatmentPolicyEffect(shifts=policies)
    )
    return study.identify(estimand).estimate(
        method="tmle",
        outcome_learner=outcome_learner,
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=seed,
        repeats=repeats,
        simultaneous=False,
        **(IN_SAMPLE if family != "binomial" else {}),
    )


def _fit_with_a_support_constant_covariate(*, seed: int = 7) -> Any:
    """Fit weighted data whose ``W4`` varies on zero-weight rows alone.

    ``check_weights`` allows a zero weight, so such a row carries no mass. ``W4`` is
    therefore degenerate under the weighted law and calibrates nothing.
    """
    frame, _ = make_linear_ate(n=120, seed=seed)
    weights = mean_one_weights(len(frame))
    unsupported = np.zeros(len(frame), dtype=bool)
    unsupported[::20] = True
    weights[unsupported] = 0.0
    frame["weight"] = weights
    # The supported value is not a binary fraction, so the weighted mean of the column
    # carries a rounding residual and its weighted standard deviation is not exactly zero.
    degenerate = np.full(len(frame), 3.14)
    degenerate[unsupported] = 1.0 + np.arange(int(unsupported.sum()))
    frame["W4"] = degenerate
    study = CausalStudy(
        frame,
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=("W1", "W2", "W3", "W4"),
            weights="weight",
        ),
    )
    return study.identify(ATE()).estimate(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=2,
        learner_folds=2,
        random_state=seed,
        simultaneous=False,
        **IN_SAMPLE,
    )


#: Outcome-grid strength the zero-cell witness perturbs at.  ``_fit_attributable`` thresholds
#: its latent draw at the matching quantile, so the perturbation drives every outcome to zero.
#: One name, so the fixture and the grid cannot drift apart.
_ZERO_CELL_STRENGTH = 0.3


@cache
def _fit_attributable(
    target: str = "par",
    family: str = "binomial",
    *,
    method: str = "tmle",
    reference: Any = 0,
    strata: bool = True,
    repeats: int = 1,
    backend: str = "pandas",
    zero_cell: bool = False,
) -> Any:
    rng = np.random.default_rng(412)
    n = 180
    w = rng.normal(size=n)
    v = np.where(np.arange(n) % 3 == 0, "small", "large")
    a = rng.binomial(1, 1 / (1 + np.exp(-0.6 * w)))
    if zero_cell:
        # Set Y to one on exactly the units ``_flip_mask`` selects at ``_ZERO_CELL_STRENGTH``,
        # from the latent vector the surface draws at ``random_state=31``.  The perturbed
        # outcome is then identically zero.  Y is built here rather than after a discarded
        # ``rng.binomial`` draw; no statement below reads ``rng``, so the other paths keep
        # their draw sequence.
        latent = np.random.default_rng(_latent_child_seed(31)).normal(size=n)
        y = (latent >= NormalDist().inv_cdf(1 - _ZERO_CELL_STRENGTH)).astype(int)
    elif family == "binomial":
        y = rng.binomial(1, 1 / (1 + np.exp(0.6 - 1.7 * a - 0.4 * w)))
    else:
        y = 0.4 + (1.2 + 0.8 * w) * a + 0.6 * w + rng.normal(scale=0.3, size=n)
    if isinstance(reference, str):
        a = np.where(a == 1, "active", "control")
    frame = pd.DataFrame({"W": w, "V": v, "A": a, "Y": y})
    frame["weight"] = np.where(v == "small", 3.1, 0.7) * np.where(w > 0, 1.8, 0.6)
    if backend == "polars":
        import polars as pl

        frame = pl.from_pandas(frame)
    targets = {
        "par": PopulationAttributableRisk(reference=reference),
        "paf": PopulationAttributableFraction(reference=reference),
        "ey_obs": NaturalCourseMean(),
    }
    configured: Any = method
    if method == "drtmle":
        configured = DRTMLEMethod(
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=LogisticRegression(max_iter=1000),
        )
    if method in _STRATEGY_OVERRIDES:
        configured = _strategy_method(method, selection_estimand=target)
    return (
        CausalStudy(
            frame,
            design=PointTreatment(
                outcome="Y",
                treatment="A",
                adjustment=("W", "V"),
                strata=("V",) if strata else (),
                weights="weight",
                treatment_kind="discrete",
            ),
        )
        .identify(targets[target])
        .estimate(
            method=configured,
            outcome_learner=DummyRegressor()
            if zero_cell
            else LogisticRegression(max_iter=1000)
            if family == "binomial"
            else LinearRegression(),
            treatment_learner=LogisticRegression(max_iter=1000),
            n_folds=2,
            learner_folds=2,
            random_state=12,
            repeats=repeats,
            simultaneous=False,
            # This suite is about simulated confounding, not cross-fitting. A
            # cross-fitted continuous outcome now needs a declared q_bounds, which is
            # beside the point here, so the gaussian branch fits in sample. The
            # zero-cell fixture is fit in sample for the same reason as the surface's
            # own degenerate refit below: a fold whose training complement gets an
            # all-zero perturbed outcome has no observed-outcome-1 row to fit on, and
            # that is the *later*, cross-fit-specific refusal this test is not about --
            # the one it is about is PAF's own "observed risk is zero" refusal, which
            # an in-sample fit still reaches. Binary, non-zero-cell fits keep
            # cross-fitting: this file's other pinned numbers are measured under it.
            cross_fit=family != "gaussian" and not zero_cell,
        )
    )
