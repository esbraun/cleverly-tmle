"""Longitudinal fits whose summaries the contract test checks (X89 T2).

One fixture per kind of row, table and note a ``LongitudinalResult`` summary can print: each
estimator, plan kind and outcome kind, the bands and the bootstrap, clusters and weights, and
the witnesses that fire the conditional notes. Every fit has 300 rows or fewer and uses
explicit linear or logistic learners, except the one that needs a flexible learner.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import (
    make_longitudinal,
    make_longitudinal_competing,
    make_longitudinal_repeated,
    make_longitudinal_survival,
)
from cleverly.learners import GLM
from cleverly.longitudinal import (
    IPSI,
    LTMLE,
    SDR,
    DynamicRegimen,
    ParametricGFormula,
    SequentialGComputation,
    SequentialIPW,
)

__all__ = ["COLUMNS", "FIXTURES", "learners"]

#: The column roles of :func:`cleverly.datasets.make_longitudinal`.
COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W1", "W2"],
    "time_varying": [[], ["L2"]],
    "censoring": ["C1", "C2"],
}
PLANS = {"always": 1, "never": 0}


def learners(**overrides: Any) -> dict[str, Any]:
    """Linear and logistic learners for every model, in-sample, without bands."""
    settings: dict[str, Any] = {
        "outcome_learner": LinearRegression(),
        "pseudo_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "censoring_learner": LogisticRegression(max_iter=1000),
        "n_folds": 1,
        "simultaneous": False,
        "random_state": 0,
    }
    settings.update(overrides)
    return settings


@functools.cache
def _frame(n: int = 300, seed: int = 0) -> Any:
    return make_longitudinal(n=n, seed=seed)[0]


@functools.cache
def means() -> Any:
    """Two plan means, cross-fitted, with simultaneous bands."""
    return LTMLE(PLANS, **learners(n_folds=2, simultaneous=True)).fit(_frame(), **COLUMNS)


def _study() -> Any:
    from cleverly import CausalStudy, LongitudinalTreatment

    design = LongitudinalTreatment(
        outcome="Y",
        treatment=("A1", "A2"),
        baseline=("W1", "W2"),
        time_varying=((), ("L2",)),
        censoring=("C1", "C2"),
    )
    return CausalStudy(_frame(), design=design)


@functools.cache
def contrast() -> Any:
    """One contrast, so the header names the question and no key table prints."""
    from cleverly import RegimeContrast

    return _study().estimate(
        RegimeContrast(PLANS, reference="never"), cross_fit=False, **_study_learners()
    )


def _study_learners(**overrides: Any) -> dict[str, Any]:
    settings = learners(**overrides)
    for key in ("n_folds", "simultaneous"):
        settings.pop(key)
    return settings


@functools.cache
def rule() -> Any:
    """A plan that follows a declared rule at the second time point."""
    adaptive = DynamicRegimen(
        "adaptive", (1, lambda h: (h["L2"] > 0).astype(int)), rule_kind="known"
    )
    estimator = LTMLE({"never": 0, "adaptive": adaptive}, reference="never", **learners())
    return estimator.fit(_frame(), **COLUMNS)


@functools.cache
def sdr() -> Any:
    """The sequentially doubly robust estimator, cross-fitted."""
    return SDR(PLANS, reference="never", **learners(n_folds=2)).fit(_frame(), **COLUMNS)


@functools.cache
def ipsi() -> Any:
    """An odds tilt against the natural course."""
    from cleverly.interventions import Incremental

    settings = learners(n_folds=2)
    del settings["pseudo_learner"]
    estimator = IPSI([Incremental(1.0), Incremental(2.0)], **settings)
    return estimator.fit(_frame(), **COLUMNS)


@functools.cache
def gcomp() -> Any:
    """Sequential g-computation: a model-based interval and no weights."""
    return SequentialGComputation(PLANS, reference="never", outcome_learner=GLM()).fit(
        _frame(), **COLUMNS
    )


@functools.cache
def ipw() -> Any:
    """Sequential inverse probability weighting."""
    return SequentialIPW(PLANS, reference="never", treatment_learner=GLM()).fit(_frame(), **COLUMNS)


@functools.cache
def gformula() -> Any:
    """The parametric g-formula, with one modelled covariate."""
    estimator = ParametricGFormula(
        PLANS, reference="never", covariate_types={"L2": "normal"}, n_draws=20, random_state=0
    )
    return estimator.fit(_frame(), **COLUMNS)


@functools.cache
def repeats() -> Any:
    """Three random splits into folds, combined by their median."""
    estimator = LTMLE(PLANS, reference="never", repeats=3, **learners(n_folds=2))
    return estimator.fit(_frame(), **COLUMNS)


@functools.cache
def bootstrap() -> Any:
    """A full-refit bootstrap with its interval and its sup-t bands."""
    estimator = LTMLE(
        PLANS,
        n_bootstrap=20,
        bootstrap_interval="bca",
        **learners(simultaneous=True),
    )
    return estimator.fit(_frame(), **COLUMNS)


@functools.cache
def survival() -> Any:
    """The event risk at two time points under each plan."""
    frame = make_longitudinal_survival(n=300, seed=12)[0]
    columns = {**COLUMNS, "outcome": ["Y1", "Y2"]}
    return LTMLE(PLANS, **learners()).fit(frame, **columns)


@functools.cache
def repeated_outcome() -> Any:
    """An outcome measured at every time point."""
    from cleverly.longitudinal import Repeated

    frame = make_longitudinal_repeated(n=300, seed=7)[0]
    estimator = LTMLE({"never": 0, "always": 1}, reference="never", **learners())
    return estimator.fit(
        frame,
        outcome=Repeated(["Y1", "Y2", "Y3"]),
        treatment=["A1", "A2", "A3"],
        baseline=["W"],
        time_varying=[[], ["L2"], ["L3"]],
        censoring=["C1", "C2", "C3"],
    )


@functools.cache
def competing() -> Any:
    """Two competing causes."""
    frame = make_longitudinal_competing(n=300, seed=1)[0]
    estimator = LTMLE(
        PLANS,
        reference="never",
        **learners(outcome_learner=LogisticRegression(max_iter=1000)),
    )
    return estimator.fit(
        frame,
        outcome={"relapse": ("R1", "R2"), "death": ("D1", "D2")},
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


@functools.cache
def working_model() -> Any:
    """A marginal structural working model over the plan means."""
    from tests.unit.test_longitudinal_msm import DOSE, fitted_regimens

    return fitted_regimens(msm=DOSE)


@functools.cache
def clustered_weighted() -> Any:
    """Eight clusters and observation weights: t intervals and a weighted population."""
    frame = _frame().copy()
    frame["site"] = np.arange(len(frame)) % 8
    frame["w"] = np.random.default_rng(0).uniform(0.5, 2.0, len(frame))
    estimator = LTMLE(PLANS, reference="never", **learners(n_folds=2))
    return estimator.fit(frame, id="site", weights="w", **COLUMNS)


@functools.cache
def capped() -> Any:
    """Witness: a lower limit high enough to move some probabilities."""
    estimator = LTMLE(PLANS, reference="never", g_bounds=(0.2, 1.0), **learners())
    return estimator.fit(_frame(), **COLUMNS)


@functools.cache
def flexible_in_sample() -> Any:
    """Witness: a flexible outcome learner fitted and evaluated on the same rows."""
    from cleverly import RegimeContrast

    forest = RandomForestRegressor(n_estimators=20, max_depth=3, random_state=0)
    return _study().estimate(
        RegimeContrast(PLANS, reference="never"),
        cross_fit=False,
        **_study_learners(outcome_learner=forest, pseudo_learner=forest),
    )


@functools.cache
def unsolved() -> Any:
    """Witness: one update step with a tight tolerance leaves an equation unsolved."""
    estimator = LTMLE(PLANS, reference="never", max_iter=1, tol=1e-12, **learners())
    return estimator.fit(_frame(), **COLUMNS)


FIXTURES = {
    "means": means,
    "contrast": contrast,
    "rule": rule,
    "sdr": sdr,
    "ipsi": ipsi,
    "gcomp": gcomp,
    "ipw": ipw,
    "gformula": gformula,
    "repeats": repeats,
    "bootstrap": bootstrap,
    "survival": survival,
    "repeated_outcome": repeated_outcome,
    "competing": competing,
    "working_model": working_model,
    "clustered_weighted": clustered_weighted,
    "capped": capped,
    "flexible_in_sample": flexible_in_sample,
    "unsolved": unsolved,
}
