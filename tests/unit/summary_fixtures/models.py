"""Outcome and treatment model checks whose summaries the contract test checks (X89 T3).

One fixture per table, reading and note the model check can print, so each of its note
codes fires on at least one fixture. Fits are compact (400 rows) with logistic and linear
learners. Findings a correct compact fit does not give (separation, a mean-only library, a
planted calibration slope) are built from real checks with ``dataclasses.replace``.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import replace
from typing import Any

from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_binary_outcome, make_linear_ate
from cleverly.estimators import CTMLE, TMLE
from cleverly.validation.nuisance import NuisanceDiagnostics

from . import Case
from ._check_fidelity import read_models

__all__ = ["FIXTURES"]


def _logistic() -> LogisticRegression:
    return LogisticRegression(max_iter=1000)


@functools.cache
def _binary() -> Any:
    return make_binary_outcome(n=400, seed=1)[0]


def _fit(estimator: Any, frame: Any) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator.fit(frame, outcome="Y", treatment="A").single()


@functools.cache
def cross_fitted() -> NuisanceDiagnostics:
    """A cross-fitted fit of a binary outcome: two probability models, tested slopes."""
    fit = _fit(
        TMLE(outcome_learner=_logistic(), treatment_learner=_logistic(), n_folds=3, random_state=0),
        _binary(),
    )
    return fit.diagnostics.nuisance_models()


@functools.cache
def in_sample() -> NuisanceDiagnostics:
    """An in-sample fit of a continuous outcome: a probability model and a mean model."""
    fit = _fit(
        TMLE(
            outcome_learner=LinearRegression(),
            treatment_learner=_logistic(),
            cross_fit=False,
            random_state=0,
        ),
        make_linear_ate(n=400, seed=1)[0],
    )
    return fit.diagnostics.nuisance_models()


@functools.cache
def repeated() -> NuisanceDiagnostics:
    """A fit repeated over three splits, which prints the spread between them."""
    fit = _fit(
        TMLE(
            outcome_learner=_logistic(),
            treatment_learner=_logistic(),
            n_folds=3,
            repeats=3,
            simultaneous=False,
            estimands=("ate", "rr"),
            random_state=0,
        ),
        _binary(),
    )
    return fit.diagnostics.nuisance_models()


@functools.cache
def collaborative() -> NuisanceDiagnostics:
    """A collaborative fit: a working treatment model, its selection and no interval."""
    fit = _fit(
        CTMLE(
            outcome_learner=_logistic(),
            treatment_learner=_logistic(),
            n_folds=3,
            random_state=0,
            strategy="greedy",
            estimands=("ate",),
        ),
        _binary(),
    )
    return fit.diagnostics.nuisance_models()


def _with(check: NuisanceDiagnostics, name: str, **changes: Any) -> NuisanceDiagnostics:
    models = tuple(
        replace(model, **changes) if model.name == name else model for model in check.models
    )
    return replace(check, models=models)


def findings() -> NuisanceDiagnostics:
    """A check with separation, a mean-only library and a planted calibration slope."""
    check = cross_fitted()
    propensity = check["propensity"]
    outcome = check["outcome"]
    check = _with(
        check,
        "propensity",
        metrics={**propensity.metrics, "auc": 0.95},
        learner_weights={"mean": 0.9, "glm": 0.1},
    )
    return _with(
        check,
        "outcome",
        metrics={**outcome.metrics, "calibration_slope": 0.4, "calibration_slope_se": 0.05},
    )


def weak_outcome() -> NuisanceDiagnostics:
    """A mean model that explains little of the outcome."""
    check = in_sample()
    outcome = check["outcome"]
    return _with(check, "outcome", metrics={**outcome.metrics, "r2": 0.01})


def weak_treatment() -> NuisanceDiagnostics:
    """A treatment model barely better than chance, read as a property of overlap."""
    check = in_sample()
    propensity = check["propensity"]
    return _with(check, "propensity", metrics={**propensity.metrics, "auc": 0.52})


def recording() -> NuisanceDiagnostics:
    """A recording model that predicts almost perfectly, so some weights are extreme."""
    check = cross_fitted()
    recorded = replace(
        check["propensity"],
        name="missingness",
        metrics={**check["propensity"].metrics, "auc": 0.97},
    )
    return replace(check, models=(*check.models, recorded))


def declared() -> NuisanceDiagnostics:
    """A declared treatment law, which the treatment rows compare with the data."""
    return replace(in_sample(), treatment_role="declared_treatment_law")


def missing_parts() -> NuisanceDiagnostics:
    """A collaborative check whose selection is missing and whose outcome was not fitted."""
    check = collaborative()
    return replace(
        check,
        selection=None,
        selection_omission="the fitted result retains no ctmle method artifact",
        outcome_omission="this estimator fits no outcome regression",
        models=tuple(model for model in check.models if model.name != "outcome"),
    )


def none() -> NuisanceDiagnostics:
    """A check with no model, which a check built by hand can be."""
    return replace(in_sample(), models=())


def _models(build: Any) -> Case:
    return Case(build, fidelity=read_models)


FIXTURES = {
    "models_cross_fitted": _models(cross_fitted),
    "models_in_sample": _models(in_sample),
    "models_repeated": _models(repeated),
    "models_collaborative": _models(collaborative),
    "models_findings": _models(findings),
    "models_weak_outcome": _models(weak_outcome),
    "models_weak_treatment": _models(weak_treatment),
    "models_recording": _models(recording),
    "models_declared": _models(declared),
    "models_missing_parts": _models(missing_parts),
    "models_none": _models(none),
}
