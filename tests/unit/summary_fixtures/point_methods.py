"""Point-estimate objects whose summaries the contract test checks (X89, thread T1).

One fixture per object a point-treatment fit returns besides its result: the method records
of C-TMLE, DR-TMLE and adaptive IPTW, the cross-validated report, a result set, the
inference objects, a variable-importance result and the super learner's diagnostics. Each
note code T1 adds fires on at least one fixture (``tests/unit/test_point_method_summaries.py``
checks that). Every fit is compact and uses explicit learners.

None of these objects carries an estimates frame with its data, so each fixture gives a
reader from ``_point_readers`` that checks every printed number against the object.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from cleverly.datasets import make_binary_outcome, make_cde, make_linear_ate, make_missing_outcome
from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.estimators.iptw import AdaptiveIPTW
from cleverly.inference.bootstrap import BootstrapResult
from cleverly.inference.results import RepeatSpread
from cleverly.learners import ParametricMechanism, ParametricOutcome, SuperLearner
from cleverly.learners.super_learner import SuperLearnerDiagnostics
from cleverly.variable_importance import variable_importance
from tests.unit.summary_fixtures import Case
from tests.unit.summary_fixtures import _point_readers as read

__all__ = ["FIXTURES"]

COVARIATES = ("W1", "W2", "W3", "W4")


def _learners(**overrides: Any) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "outcome_learner": LinearRegression(),
        "treatment_learner": LogisticRegression(max_iter=1000),
        "cross_fit": False,
        "simultaneous": False,
        "random_state": 0,
    }
    settings.update(overrides)
    return settings


def _frame(n: int = 300, seed: int = 1) -> Any:
    return make_linear_ate(n=n, seed=seed)[0]


def _fit(estimator: Any) -> Any:
    return estimator.fit(_frame(), outcome="Y", treatment="A").single()


def _record(result: Any, key: str) -> Any:
    return result.extra[key]


# ----------------------------------------------------------------- C-TMLE records


@functools.cache
def greedy() -> Any:
    """The greedy search: every covariate on the path, one of them left out of the pick."""
    return _fit(CTMLE(**_learners(strategy="greedy", estimands=("ate",))))


@functools.cache
def greedy_selection() -> Any:
    return _record(greedy(), "ctmle")


@functools.cache
def ordered_patience() -> Any:
    """The ordered search with patience 1, which stops before the last candidate."""
    return _record(
        _fit(CTMLE(**_learners(strategy="ordered", patience=1, estimands=("ate",)))), "ctmle"
    )


@functools.cache
def lasso_selection() -> Any:
    """The lasso path, longer than the table shows."""
    return _record(_fit(CTMLE(**_learners(strategy="lasso", estimands=("ate",)))), "ctmle")


@functools.cache
def learners_unordered() -> Any:
    """Candidate learners listed out of order of complexity."""
    candidates = [LogisticRegression(max_iter=1000), LogisticRegression(C=0.01, max_iter=1000)]
    return _fit(
        CTMLE(**_learners(strategy="learners", treatment_candidates=candidates, estimands=("ate",)))
    )


@functools.cache
def learners_selection() -> Any:
    return _record(learners_unordered(), "ctmle")


@functools.cache
def outcome_adaptive() -> Any:
    """The outcome-adaptive record whose stacked interval is refused."""
    return _record(_fit(CTMLE(**_learners(strategy="oat", estimands=("ate",)))), "ctmle")


@functools.cache
def stacked() -> Any:
    """The parametric outcome-adaptive fit, which reports the stacked interval."""
    return _fit(
        CTMLE(
            **_learners(
                strategy="oat",
                outcome_learner=ParametricOutcome(),
                treatment_learner=ParametricMechanism(),
                cross_fit=True,
                n_folds=3,
                q_bounds=(-20, 20),
                estimands=("ate",),
            )
        )
    )


@functools.cache
def stacked_record() -> Any:
    return _record(stacked(), "ctmle")


@functools.cache
def derivative() -> Any:
    return _fit(CTMLE(**_learners(strategy="derivative", estimands=("ate",))))


@functools.cache
def derivative_record() -> Any:
    return _record(derivative(), "ctmle")


# ----------------------------------------------------------------- DR-TMLE and IPTW


def _reduced_learner() -> Any:
    return SuperLearner(
        library=[LinearRegression(), DecisionTreeRegressor(max_depth=2, random_state=0)],
        n_folds=3,
        random_state=0,
    )


@functools.cache
def drtmle() -> Any:
    return _fit(
        DRTMLE(
            **_learners(
                estimands=("ate",),
                reduced_outcome_learner=_reduced_learner(),
                reduced_treatment_learner=_reduced_learner(),
            )
        )
    )


@functools.cache
def reduced() -> Any:
    return _record(drtmle(), "drtmle")


@functools.cache
def reduced_bivariate() -> Any:
    """The bivariate reduction, whose treatment regression also takes the probability."""
    return _record(_fit(DRTMLE(**_learners(estimands=("ate",), reduction="bivariate"))), "drtmle")


@functools.cache
def reduced_missing() -> Any:
    """A randomized trial with outcomes missing at random: five extra regressions per arm."""
    frame = make_missing_outcome(n=300, seed=4)[0]
    return _record(
        DRTMLE(**_learners(estimands=("ate",)), randomized=True)
        .fit(frame, outcome="Y", treatment="A", delta="Delta")
        .single(),
        "drtmle",
    )


@functools.cache
def reduced_unguarded() -> Any:
    """``guard=()``: the reduced regressions are not fitted, so the fit is an ordinary TMLE."""
    return _record(_fit(DRTMLE(**_learners(estimands=("ate",), guard=()))), "drtmle")


@functools.cache
def adaptive_iptw() -> Any:
    return _fit(AdaptiveIPTW(**_learners()))


@functools.cache
def iptw_record() -> Any:
    return _record(adaptive_iptw(), "adaptive_iptw")


@functools.cache
def iptw_capped() -> Any:
    """One round only, so the iteration stops before the tolerance."""
    return _record(_fit(AdaptiveIPTW(**_learners(max_iter=1))), "adaptive_iptw")


# ----------------------------------------------------------------- reports and sets


@functools.cache
def cv_report() -> Any:
    """Cross-fitted, fold-evaluated, two repeats: both reports print."""
    return _fit(
        TMLE(
            **_learners(
                cross_fit=True,
                n_folds=3,
                cv_evaluation=True,
                repeats=2,
                q_bounds=(-20, 20),
                estimands=("ate", "ey1", "ey0"),
            )
        )
    ).cv_targeting


@functools.cache
def result_set_one() -> Any:
    """A fit without an intermediate variable: a set of one result."""
    return AdaptiveIPTW(**_learners()).fit(_frame(), outcome="Y", treatment="A")


@functools.cache
def result_set() -> Any:
    """A controlled direct effect at two levels of the intermediate variable."""
    frame = make_cde(n=300, seed=3)[0]
    return TMLE(**_learners(estimands=("ate",))).fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=["W1", "W2", "W3"],
        intermediate="Z",
    )


# ----------------------------------------------------------------- inference objects


@functools.cache
def ratio() -> Any:
    """The risk ratio, whose null value is 1 on its own scale."""
    frame = make_binary_outcome(n=300, seed=2)[0]
    return (
        TMLE(**_learners(outcome_learner=LogisticRegression(max_iter=1000), estimands=("rr",)))
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


@functools.cache
def estimate() -> Any:
    return ratio().estimates["rr"]


@functools.cache
def wald() -> Any:
    return ratio().estimates["rr"].wald_test()


@functools.cache
def bootstrap() -> Any:
    """Literal draws with two failed replicates."""
    rng = np.random.default_rng(4)
    return BootstrapResult(
        draws={"ate": rng.normal(1.2, 0.2, size=48), "ey1": rng.normal(3.1, 0.15, size=48)},
        n_requested=50,
        n_failed=2,
        resampling="iid",
    )


@functools.cache
def bootstrap_interval() -> Any:
    return bootstrap().interval("ate")


@functools.cache
def bands() -> Any:
    return (
        TMLE(**_learners(simultaneous=True, n_multiplier=200, estimands=("ate", "ey1", "ey0")))
        .fit(_frame(), outcome="Y", treatment="A")
        .single()
        .simultaneous
    )


@functools.cache
def repeated() -> Any:
    return (
        TMLE(
            **_learners(
                cross_fit=True,
                n_folds=3,
                repeats=3,
                q_bounds=(-20, 20),
                estimands=("ate", "ey1", "ey0"),
            )
        )
        .fit(_frame(), outcome="Y", treatment="A")
        .single()
    )


@functools.cache
def spread() -> Any:
    return repeated().repeat_spread()


@functools.cache
def spread_missing() -> Any:
    """A spread with one split the log scale could not take."""
    return RepeatSpread(
        {"rr": float("nan"), "ate": 0.012},
        n_repeats=3,
        std_errors={"rr": 0.08, "ate": 0.11},
        inferential={"rr": True, "ate": True},
        scales={"rr": "log", "ate": "original"},
    )


@functools.cache
def spread_withheld() -> Any:
    """Every estimate withholds inference, so the spread is read against the plug-in."""
    return RepeatSpread(
        {"ate": 0.02, "ey1": 0.015},
        n_repeats=4,
        std_errors={"ate": 0.1, "ey1": 0.09},
        inferential={"ate": False, "ey1": False},
        scales={"ate": "original", "ey1": "original"},
    )


@functools.cache
def importance() -> Any:
    """Two candidate exposures fitted in sample by a flexible learner."""
    rng = np.random.default_rng(5)
    n = 180
    w = rng.normal(size=n)
    x1 = rng.binomial(1, 1.0 / (1.0 + np.exp(-w)))
    x2 = rng.binomial(1, 0.5, size=n)
    y = 1.2 * x1 + 0.2 * w + rng.normal(scale=0.5, size=n)
    frame = pd.DataFrame({"Y": y, "X1": x1, "X2": x2, "W": w})
    return variable_importance(
        frame,
        outcome="Y",
        candidates=("X1", "X2"),
        covariates=("W",),
        estimator=TMLE(
            **_learners(outcome_learner=DecisionTreeRegressor(max_depth=3, random_state=0))
        ),
    )


@functools.cache
def ensemble() -> Any:
    """A fitted classification super learner, scored by the negative log-likelihood."""
    frame = _frame()
    learner = SuperLearner(
        library=[
            LogisticRegression(max_iter=1000),
            LogisticRegression(C=0.01, max_iter=1000),
        ],
        n_folds=3,
        random_state=0,
    )
    learner.fit(frame[list(COVARIATES)].to_numpy(), frame["A"].to_numpy())
    return learner.diagnostics_


@functools.cache
def ensemble_failed() -> Any:
    """Literal regression diagnostics with one candidate that failed to fit."""
    return SuperLearnerDiagnostics(
        names=("lm", "tree"),
        cv_risk=np.array([0.91, 1.07]),
        weights=np.array([0.8, 0.2]),
        best="lm",
        ensemble_cv_risk=0.89,
        loss="mse",
        failed=("forest",),
    )


FIXTURES: dict[str, Case] = {
    "greedy_selection": Case(greedy_selection, names=COVARIATES, fidelity=read.read_selection),
    "ordered_patience": Case(ordered_patience, names=COVARIATES, fidelity=read.read_selection),
    "lasso_selection": Case(lasso_selection, names=COVARIATES, fidelity=read.read_selection),
    "learners_selection": Case(learners_selection, names=COVARIATES, fidelity=read.read_selection),
    "outcome_adaptive": Case(outcome_adaptive, fidelity=read.read_oat),
    "stacked": Case(stacked_record, fidelity=read.read_oat),
    "derivative": Case(derivative_record, fidelity=read.read_derivative),
    "reduced": Case(reduced, fidelity=read.read_reduced),
    "reduced_bivariate": Case(reduced_bivariate, fidelity=read.read_reduced),
    "reduced_missing": Case(reduced_missing, fidelity=read.read_reduced),
    "reduced_unguarded": Case(reduced_unguarded, fidelity=read.read_reduced),
    "iptw": Case(iptw_record, fidelity=read.read_iptw),
    "iptw_capped": Case(iptw_capped, fidelity=read.read_iptw),
    "cv_report": Case(cv_report, fidelity=read.read_cv),
    "result_set": Case(result_set, names=("Z",), fidelity=read.read_result_set),
    "result_set_one": Case(result_set_one, fidelity=read.read_result_set),
    "estimate": Case(estimate, fidelity=read.read_estimate),
    "wald": Case(wald, fidelity=read.read_wald),
    "bootstrap": Case(bootstrap, fidelity=read.read_bootstrap),
    "bootstrap_interval": Case(bootstrap_interval, fidelity=read.read_bootstrap_interval),
    "bands": Case(bands, fidelity=read.read_bands),
    "spread": Case(spread, fidelity=read.read_spread),
    "spread_missing": Case(spread_missing, fidelity=read.read_spread),
    "spread_withheld": Case(spread_withheld, fidelity=read.read_spread),
    "importance": Case(importance, names=("X1", "X2", "W"), fidelity=read.read_importance),
    "ensemble": Case(ensemble, fidelity=read.read_ensemble),
    "ensemble_failed": Case(
        ensemble_failed, names=("lm", "tree", "forest"), fidelity=read.read_ensemble
    ),
}

#: Fitted results whose summaries print a method's record, for the note-witness test.
RESULTS: dict[str, Callable[[], Any]] = {
    "greedy": greedy,
    "learners_unordered": learners_unordered,
    "stacked": stacked,
    "derivative": derivative,
    "drtmle": drtmle,
    "adaptive_iptw": adaptive_iptw,
}
