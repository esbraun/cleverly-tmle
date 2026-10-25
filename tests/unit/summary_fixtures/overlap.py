"""Overlap checks whose summaries the contract test checks (X89 T3).

One fixture per table, reading, result and note the overlap check can print, so each of its
note codes and each branch of its grade fires on at least one fixture. Fits are compact
(400 rows) and in sample with linear and logistic learners. A tier a compact fit does not
reach (a group's cap above the fit's, a capped or concentrated recording probability, a
known treatment law, an omitted equation row) is built from a real report with
``dataclasses.replace``.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import replace
from typing import Any

from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.data.weighting import SCORE_LOAD_SHAPE_MISMATCH
from cleverly.datasets import (
    make_binary_outcome,
    make_linear_ate,
    make_missing_outcome,
    make_multi_arm,
    make_weak_overlap,
)
from cleverly.estimators import TMLE
from cleverly.sensitivity.positivity import PositivityReport

from . import Case
from ._check_fidelity import read_overlap

__all__ = ["FIXTURES"]


def _estimator(**settings: Any) -> TMLE:
    return TMLE(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        cross_fit=False,
        random_state=0,
        **settings,
    )


def _support(estimator: TMLE, frame: Any, **roles: Any) -> PositivityReport:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = estimator.fit(frame, outcome="Y", treatment="A", **roles).single()
        return result.diagnostics.support()


@functools.cache
def plain() -> PositivityReport:
    """A two-arm fit with good overlap: nothing capped, two equation rows."""
    return _support(_estimator(estimands=("ate", "att")), make_linear_ate(n=400, seed=71)[0])


@functools.cache
def weak() -> PositivityReport:
    """Weak overlap: the cap moves more than the serious share of units."""
    return _support(
        _estimator(estimands=("ate",), g_bounds=(0.1, 0.9)), make_weak_overlap(n=400, seed=72)[0]
    )


def strained() -> PositivityReport:
    """A cap that moves between the two shares, read as strain."""
    report = plain()
    return replace(report, truncated={**report.truncated, "count": 12.0, "fraction": 0.03})


def group_limits() -> PositivityReport:
    """A group whose own limits moved more units than the fit's: the tier reads the group."""
    report = plain()
    load = report.group_score_load["att"]
    return replace(
        report,
        group_score_load={
            **report.group_score_load,
            "att": {**load, "clipped_count": 30.0, "clipped_fraction": 0.075},
        },
    )


@functools.cache
def recording() -> PositivityReport:
    """Missing outcomes with an ``att``: the other probabilities and the excluded estimand."""
    frame, _ = make_missing_outcome(n=400, seed=13, strength=1.5)
    return _support(
        _estimator(estimands=("ate", "att"), missingness_learner=LogisticRegression(max_iter=1000)),
        frame,
        covariates=["W1", "W2", "W3"],
        delta="Delta",
    )


def recording_capped() -> PositivityReport:
    """The recording probability is capped on more than the strain share of its values."""
    report = recording()
    name = "P(Delta=1|A,W)"
    return replace(
        report,
        mechanisms={
            **report.mechanisms,
            name: {**report.mechanisms[name], "clipped": 20.0, "clipped_fraction": 0.05},
        },
        truncated={**report.truncated, "count": 30.0, "fraction": 0.075},
    )


def recording_concentrated() -> PositivityReport:
    """The joint probability leaves a small effective share: reported, not graded."""
    report = recording()
    name = "P(A=a,Delta=1|W)"
    return replace(
        report,
        mechanisms={**report.mechanisms, name: {**report.mechanisms[name], "ess_ratio": 0.4}},
    )


@functools.cache
def three_arms() -> PositivityReport:
    """Three arms with tight limits: the capped probabilities no longer sum to one."""
    frame, _ = make_multi_arm(n=400, seed=5)
    return _support(_estimator(estimands=("ey",), g_bounds=(0.2, 0.8)), frame)


@functools.cache
def repeated() -> PositivityReport:
    """A fit repeated over three splits, which shows the first."""
    estimator = TMLE(
        outcome_learner=LinearRegression(),
        treatment_learner=LogisticRegression(max_iter=1000),
        n_folds=3,
        repeats=3,
        simultaneous=False,
        estimands=("ate",),
        random_state=0,
    )
    return _support(estimator, make_binary_outcome(n=400, seed=1)[0])


def known() -> PositivityReport:
    """A treatment law the data declares, and an equation row the fit could not report."""
    report = plain()
    return replace(
        report,
        treatment_mechanism="known",
        group_score_load={"att": report.group_score_load["att"]},
        clever_covariate_max={"att": report.clever_covariate_max["att"]},
        group_score_load_omissions={
            "mean": SCORE_LOAD_SHAPE_MISMATCH,
            "att.clipping": "the fitted clipping mask was not retained",
        },
    )


def _overlap(build: Any) -> Case:
    return Case(build, fidelity=read_overlap)


FIXTURES = {
    "overlap_plain": _overlap(plain),
    "overlap_weak": _overlap(weak),
    "overlap_strained": _overlap(strained),
    "overlap_group_limits": _overlap(group_limits),
    "overlap_recording": _overlap(recording),
    "overlap_recording_capped": _overlap(recording_capped),
    "overlap_recording_concentrated": _overlap(recording_concentrated),
    "overlap_three_arms": _overlap(three_arms),
    "overlap_repeated": _overlap(repeated),
    "overlap_known": _overlap(known),
}
