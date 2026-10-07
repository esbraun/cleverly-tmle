"""Repeated-sampling accuracy of LTMLE on declared known node mechanisms.

One family, fixed before any registered run.  On one draw of the SMART law of
:mod:`tests.studies.known_mechanism_law` per replication, n = 2,000 and 1,000 replications, the
contrast of ``always`` against ``never`` is fitted with the wrong outcome regressions twice:
with every factor declared, and with the treatment declared beside a censoring factor
estimated by an unpenalized logistic regression on ``(L0, A1)``.  The second is the usual
SMART analysis, where the double-robust conditions apply to the censoring factor only.  Each
cell answers to the accuracy rule of :mod:`tests.studies.known_mechanism_properties`.  A
replication that raises is never redrawn.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import known_mechanism_law as law
from tests.studies.canonical_known_node_mechanisms import STUDY, fit_cleverly, truth
from tests.studies.evidence.properties import REPLICATE_COLUMNS, replicate_row
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.evidence.seeds import stream_seed

ACCURACY = "known_mechanism_accuracy"
ACCURACY_N = 2_000
ACCURACY_REPLICATES = 1_000
ESTIMAND = "ate_regimen[always vs never]"
TRUTH = truth()[ESTIMAND]


class NoMechanismFit(BaseEstimator, ClassifierMixin):
    """A treatment learner that must never run: every treatment node is declared."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> NoMechanismFit:
        raise AssertionError("a declared treatment node fitted a treatment learner")

    def predict_proba(self, design: Any) -> Any:
        raise AssertionError("a declared treatment node fitted a treatment learner")


def _replication(replicate: int) -> list[dict[str, Any]]:
    seed = stream_seed(STUDY, "property_sample", ACCURACY, "paired", replicate)
    frame = law.smart_sample(ACCURACY_N, seed)
    fits = {
        "ate_regimen__ltmle_known__q_wrong": fit_cleverly(
            frame, treatment_learner=NoMechanismFit(), censoring_learner=NoMechanismFit()
        ),
        "ate_regimen__ltmle_known_treatment__censoring_estimated": fit_cleverly(
            frame,
            censoring="estimated",
            treatment_learner=NoMechanismFit(),
            censoring_learner=LogisticRegression(penalty=None, max_iter=10_000),
        ),
    }
    return [
        replicate_row(
            property_name=ACCURACY,
            cell=cell,
            role="positive",
            replicate=replicate,
            n=ACCURACY_N,
            requested=ACCURACY_REPLICATES,
            truth=TRUTH,
            estimate=result[ESTIMAND],
            alpha=STUDY.margins.alpha,
        )
        for cell, result in fits.items()
    ]


def declared_cells() -> tuple[tuple[str, str, str, float], ...]:
    """Every ``(family, cell, estimand, truth)`` the replication file carries."""
    return tuple((ACCURACY, cell, ESTIMAND, TRUTH) for cell in STUDY.property_cells[ACCURACY])


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    replicates = ACCURACY_REPLICATES if budget is None else min(budget, ACCURACY_REPLICATES)
    outcomes = map_parallel(_replication, [(r,) for r in range(replicates)], n_jobs=n_jobs)
    rows = pd.DataFrame([row for outcome in outcomes for row in outcome])
    if budget is not None:
        # A capped smoke run requests what it runs, so the summary reads it as complete.
        rows["requested_replicates"] = rows["requested_replicates"].clip(upper=budget)
    return rows.loc[:, list(REPLICATE_COLUMNS)]


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    margins = STUDY.margins
    family = summary["property"] == ACCURACY
    summary.loc[family, "passed"] = (
        summary.loc[family, "bias_equivalent"].astype(bool)
        & (summary.loc[family, "coverage_ci_lower"] >= margins.coverage_floor)
        & summary.loc[family, "se_ratio"].between(*margins.se_ratio_sanity)
    )
    return finish(summary, rates)
