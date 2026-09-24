"""Shared fixed-policy and MSM builders for simulated-confounding tests."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

from cleverly import CrossFitting, MSMProjection, RegimeContrast, RegimeMean, TMLEMethod
from cleverly.interventions import Rule, Static, Stochastic
from cleverly.msm import MSM
from cleverly.sensitivity import ConfounderStrengthGrid
from tests.unit._confounding_support import alias_for
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
