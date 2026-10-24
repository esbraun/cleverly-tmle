"""Equation and DR-TMLE correction checks whose summaries the contract test checks (X89 T3).

One fixture per table, note and result an equation check or a correction check can print,
so each of their note codes fires on at least one fixture. Fits are compact (300 rows) with
linear and logistic learners. The failed checks, the check of a fit that reports no
interval, and the check whose cap binds are built from real checks with
``dataclasses.replace``: a correct compact fit gives none of them.
"""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import Any

from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_linear_ate
from cleverly.estimators import DRTMLE, TMLE
from cleverly.validation.drtmle import CorrectionCheck
from cleverly.validation.score import ScoreCheck

from . import Case
from ._check_fidelity import read_corrections, read_equations

__all__ = ["FIXTURES"]


def _learners(**changes: Any) -> dict[str, Any]:
    return {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "random_state": 0,
        **changes,
    }


@functools.cache
def _frame() -> Any:
    return make_linear_ate(n=300, seed=1)[0]


@functools.cache
def tmle() -> ScoreCheck:
    """A plain fit: the outcome means, the effects among the treated and the untreated."""
    return (
        TMLE(**_learners())
        .fit(_frame(), outcome="Y", treatment="A")
        .single()
        .diagnostics.score_equations()
    )


@functools.cache
def _drtmle() -> Any:
    fit = DRTMLE(**_learners(estimands=["ate"])).fit(_frame(), outcome="Y", treatment="A")
    return fit.single()


def drtmle() -> ScoreCheck:
    """A doubly robust fit, which adds its corrections table and its curve's note."""
    return _drtmle().diagnostics.score_equations()


@functools.cache
def declared() -> ScoreCheck:
    """A fit under a declared stopping rule, whose estimate rows are held to nothing."""
    learners = _learners(stopping="scaled_var", targeting="one_step", step_size=0.01)
    return (
        TMLE(**learners)
        .fit(_frame(), outcome="Y", treatment="A")
        .single()
        .diagnostics.score_equations()
    )


@functools.cache
def _partial() -> Any:
    learners = _learners(estimands=["ate"], guard=("g",))
    return DRTMLE(**learners).fit(_frame(), outcome="Y", treatment="A").single()


def partial_guard() -> ScoreCheck:
    """A fit that solves one of the two corrections, so the other is described only."""
    return _partial().diagnostics.score_equations()


def failed() -> ScoreCheck:
    """A check with a step that stopped short, folds that did not converge, an ill-posed
    step and a broken identity, each of which fails or warns."""
    check = drtmle()
    rows = list(check.rows)
    step = next(index for index, row in enumerate(rows) if row.kind == "fluctuation")
    rows[step] = replace(
        rows[step],
        score=3 * rows[step].threshold,
        passed=False,
        converged=False,
        failure="max_iter_reached",
        folds_converged=(2, 3),
        hessian_condition=4.2e9,
    )
    identity = next(index for index, row in enumerate(rows) if row.kind == "identity")
    rows[identity] = replace(rows[identity], score=5 * rows[identity].threshold, passed=False)
    return replace(check, rows=tuple(rows))


def rule_unmet() -> ScoreCheck:
    """A fit whose correction step missed its declared stopping rule on one equation."""
    check = declared()
    rows = list(check.rows)
    rows[0] = replace(rows[0], score=2 * rows[0].threshold, passed=False)
    return replace(check, rows=tuple(rows))


def labelled() -> ScoreCheck:
    """A fit that reports no interval, so its check verifies the equations only."""
    return replace(tmle(), inference="estimated_mechanism_plugin")


def corrections() -> CorrectionCheck:
    """The corrections of an in-sample doubly robust fit, which the theorem covers."""
    return _drtmle().diagnostics.corrections()


def corrections_cross_fitted() -> CorrectionCheck:
    """The corrections of a cross-fitted fit, which the theorem does not cover."""
    return replace(corrections(), cross_fitted=True)


def corrections_splits() -> CorrectionCheck:
    """The corrections of two splits, which print a split column."""
    check = corrections()
    second = tuple(replace(row, draw=1, reported=2 * row.reported) for row in check.rows)
    return replace(check, rows=(*check.rows, *second))


def splits() -> ScoreCheck:
    """An equation check of two splits, whose rows name their split."""
    check = drtmle()
    rows = []
    for draw in (0, 1):
        for row in check.rows:
            if row.kind != "influence curve":
                rows.append(replace(row, name=f"{row.name}[draw {draw}]"))
    estimates = [row for row in check.rows if row.kind == "influence curve"]
    return replace(check, rows=(*rows, *estimates))


def corrections_partial() -> CorrectionCheck:
    """The corrections of a one-guard fit: the unsolved one is described only."""
    return _partial().diagnostics.corrections()


def corrections_capped() -> CorrectionCheck:
    """A fit whose treatment model sits at its cap at the end, outside the theorem."""
    check = corrections()
    return replace(check, rows=tuple(replace(row, margin=0.0) for row in check.rows))


def corrections_failed() -> CorrectionCheck:
    """A correction that is not negligible and an identity that does not hold."""
    check = corrections()
    rows = list(check.rows)
    rows[0] = replace(rows[0], reported=5 * check.threshold, stored=5 * check.threshold)
    rows[1] = replace(rows[1], stored=rows[1].reported + 10 * check.identity_threshold)
    return replace(check, rows=tuple(rows))


def corrections_none() -> CorrectionCheck:
    """A plain fit reports no corrections."""
    return replace(corrections(), rows=())


def _equations(build: Any) -> Case:
    return Case(build, fidelity=read_equations)


def _corrections(build: Any) -> Case:
    return Case(build, fidelity=read_corrections)


FIXTURES = {
    "score_tmle": _equations(tmle),
    "score_drtmle": _equations(drtmle),
    "score_declared": _equations(declared),
    "score_partial_guard": _equations(partial_guard),
    "score_failed": _equations(failed),
    "score_rule_unmet": _equations(rule_unmet),
    "score_labelled": _equations(labelled),
    "score_splits": _equations(splits),
    "corrections_in_sample": _corrections(corrections),
    "corrections_cross_fitted": _corrections(corrections_cross_fitted),
    "corrections_splits": _corrections(corrections_splits),
    "corrections_partial": _corrections(corrections_partial),
    "corrections_capped": _corrections(corrections_capped),
    "corrections_failed": _corrections(corrections_failed),
    "corrections_none": _corrections(corrections_none),
}
