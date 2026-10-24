"""Point-treatment fits whose summaries the contract test checks (X89 T0).

One fixture per kind of row, column and note a ``TMLEResult`` summary can print, so each
note code of the pilot fires on at least one fixture. Every fit is compact (300 rows or
fewer, or a small cluster design) and uses explicit linear or logistic learners, except the
one that needs a flexible learner to fire its note.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.datasets import make_binary_outcome, make_clustered, make_linear_ate
from cleverly.estimators import CTMLE, TMLE

__all__ = ["FIXTURES"]


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


@functools.cache
def canonical() -> Any:
    """The five default estimands, with simultaneous bands."""
    return (
        TMLE(**_learners(simultaneous=True, n_multiplier=200))
        .fit(_frame(), outcome="Y", treatment="A")
        .single()
    )


@functools.cache
def single_ate() -> Any:
    """One estimand, so the header names the question and no key table prints."""
    return TMLE(**_learners(estimands=("ate",))).fit(_frame(), outcome="Y", treatment="A").single()


@functools.cache
def clustered_t() -> Any:
    """Twelve clusters: a t reference with a df column, and experimental rows."""
    frame = make_clustered(n=48, cluster_size=4, seed=5)[0]
    return (
        TMLE(**_learners(estimands=("ate", "ey1", "ey0")))
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cluster")
        .single()
    )


@functools.cache
def mixed_strata() -> Any:
    """Fifty clusters, two of them in one stratum: that stratum's row gives no interval."""
    frame = make_clustered(n=500, cluster_size=10, seed=7)[0]
    frame = frame.assign(S=np.where(frame["cluster"] < 2, "small", "big"))
    return (
        TMLE(**_learners(estimands=("ate",)))
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "S"],
            id="cluster",
            strata=["S"],
        )
        .single()
    )


@functools.cache
def collaborative() -> Any:
    """A greedy C-TMLE fit, whose status withholds every interval."""
    return (
        CTMLE(**_learners(estimands=("ate", "ey1", "ey0"), selection_folds=3))
        .fit(_frame(), outcome="Y", treatment="A")
        .single()
    )


@functools.cache
def bootstrap() -> Any:
    """A full-refit bootstrap table beside the influence-curve table."""
    return (
        TMLE(**_learners(estimands=("ate",), n_bootstrap=20))
        .fit(_frame(), outcome="Y", treatment="A")
        .single()
    )


@functools.cache
def collaborative_bootstrap() -> Any:
    """A bootstrap that prints as a diagnostic, because the fit withholds its interval."""
    return (
        CTMLE(**_learners(estimands=("ate",), selection_folds=3, n_bootstrap=8))
        .fit(_frame(), outcome="Y", treatment="A")
        .single()
    )


@functools.cache
def markup_names() -> Any:
    """Column names and arm labels that hold markup, which the HTML must escape."""
    frame = _frame().rename(columns={"Y": "<script>y", "W1": "<b>w1</b>"})
    frame["A"] = np.where(frame["A"] == 1, "<i>yes</i>", "no & none")
    return (
        TMLE(**_learners(reference="no & none"))
        .fit(frame, outcome="<script>y", treatment="A")
        .single()
    )


@functools.cache
def id_like_covariate() -> Any:
    """A covariate named like a roadmap id, which a summary must print as written."""
    frame = _frame().rename(columns={"W1": "X1", "W2": "F18"})
    return TMLE(**_learners(estimands=("ate",))).fit(frame, outcome="Y", treatment="A").single()


@functools.cache
def flexible_in_sample() -> Any:
    """Random forests fitted and evaluated on the same rows."""
    return (
        TMLE(
            **_learners(
                estimands=("ate",),
                outcome_learner=RandomForestRegressor(n_estimators=10, random_state=0),
                treatment_learner=RandomForestClassifier(n_estimators=10, random_state=0),
            )
        )
        .fit(_frame(), outcome="Y", treatment="A")
        .single()
    )


@functools.cache
def weighted() -> Any:
    """Observation weights, so the estimate describes the weighted population."""
    frame = _frame()
    frame["w"] = np.random.default_rng(3).uniform(0.5, 2.0, len(frame))
    return (
        TMLE(**_learners(estimands=("ate",)))
        .fit(frame, outcome="Y", treatment="A", weights="w")
        .single()
    )


@functools.cache
def repeated() -> Any:
    """Cross-fitting over three random splits, each row the median."""
    frame = make_binary_outcome(n=300, seed=2)[0]
    return (
        TMLE(
            **_learners(
                estimands=("ate", "ey1", "ey0"),
                outcome_learner=LogisticRegression(),
                cross_fit=True,
                n_folds=3,
                repeats=3,
            )
        )
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


@functools.cache
def ratio() -> Any:
    """Risk and odds ratios, whose std. error is on the log scale."""
    frame = make_binary_outcome(n=300, seed=2)[0]
    return (
        TMLE(**_learners(estimands=("rr", "or"), outcome_learner=LogisticRegression()))
        .fit(frame, outcome="Y", treatment="A")
        .single()
    )


@functools.cache
def capped() -> Any:
    """Tight probability limits that move many rows, and the warning the fit raises."""
    return (
        TMLE(**_learners(estimands=("ate",), g_bounds=(0.4, 0.6)))
        .fit(_frame(), outcome="Y", treatment="A")
        .single()
    )


@functools.cache
def folds_reduced() -> Any:
    """Twenty folds asked for on twelve clusters, which allow only twelve."""
    frame = make_clustered(n=48, cluster_size=4, seed=5)[0]
    return (
        TMLE(
            **_learners(
                estimands=("ate",),
                cross_fit=True,
                n_folds=20,
                q_bounds=(-20.0, 20.0),
            )
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"], id="cluster")
        .single()
    )


@functools.cache
def declared_rule() -> Any:
    """A declared stopping rule, so the note states the remaining bias it allows."""
    return (
        TMLE(**_learners(estimands=("ate",), stopping="scaled_var"))
        .fit(_frame(), outcome="Y", treatment="A")
        .single()
    )


@functools.cache
def data_warning() -> Any:
    """A constant covariate the data check drops, and the warning it raises."""
    frame = _frame().assign(W5=1.0)
    return TMLE(**_learners(estimands=("ate",))).fit(frame, outcome="Y", treatment="A").single()


FIXTURES: dict[str, Callable[[], Any]] = {
    "canonical": canonical,
    "single_ate": single_ate,
    "clustered_t": clustered_t,
    "mixed_strata": mixed_strata,
    "collaborative": collaborative,
    "bootstrap": bootstrap,
    "collaborative_bootstrap": collaborative_bootstrap,
    "markup_names": markup_names,
    "id_like_covariate": id_like_covariate,
    "flexible_in_sample": flexible_in_sample,
    "weighted": weighted,
    "repeated": repeated,
    "ratio": ratio,
    "capped": capped,
    "folds_reduced": folds_reduced,
    "declared_rule": declared_rule,
    "data_warning": data_warning,
}
