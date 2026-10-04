"""Repeated-sampling calibration of the identity-link MSM with baseline strata.

One family, against the exact truths of :mod:`tests.studies.stratified_alternating_law`.
Declared before any run:

==========================  ========================================================  =====  =====
family                      cells                                                     n      R
==========================  ========================================================  =====  =====
``interval_calibration``    ``identity_<scope>_<term>``: the three marginal and nine  2,000  2,000
                            stratum coefficients of the primary fit, with
                            shrunken-SE and noise controls
==========================  ========================================================  =====  =====

The fits are the primary scenario's own, at fresh seeds.  The efficiency bounds are exact under
the bounded-outcome law (:func:`~tests.studies.stratified_alternating_law.beta_efficiency_sd`),
and the outcome and treatment learners are correctly specified for the mean, so the cells
publish efficiency ratios inside :data:`EFFICIENCY_RATIO_BAND`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import norm

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import stratified_alternating_law as law
from tests.studies.canonical_stratified_msm_identity import (
    CALIBRATION_KINDS,
    CALIBRATION_LABELS,
    STUDY,
    fit_cleverly,
    label_name,
    logistic,
)
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell, replicate_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.seeds import stream_seed

CALIBRATION_N = 2_000
CALIBRATION_REPLICATES = 2_000
SHRUNKEN_SE_FACTOR = 0.70
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

EFFICIENCY_SD: dict[str, float] = {
    label: law.beta_efficiency_sd(label_name(label)) for label in CALIBRATION_LABELS
}


class BoundedSampler:
    """The bounded-outcome L1 as a coverage-study sampler with its own exact truths."""

    name = "stratified_l1_bounded"

    def truth(self) -> dict[str, float]:
        return dict(law.truths("identity"))

    def __call__(self, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
        return law.beta_sample(n, seed), self.truth()


def _seed(replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", "interval_calibration", "identity", replicate)


def _fit_replication(payload: tuple[int, int, int, int]) -> list[dict[str, Any]]:
    replicate, n, requested, seed = payload
    result = fit_cleverly(law.beta_sample(n, seed))
    truth = law.truths("identity")
    return [
        replicate_row(
            property_name="interval_calibration",
            cell=f"{label}__correctly_specified",
            role="positive",
            replicate=replicate,
            n=n,
            requested=requested,
            truth=float(truth[label_name(label)]),
            estimate=result[label_name(label)],
            alpha=STUDY.margins.alpha,
        )
        for label in CALIBRATION_LABELS
    ]


def _payloads(budget: int | None = None) -> list[tuple[tuple[int, int, int, int]]]:
    count = CALIBRATION_REPLICATES if budget is None else budget
    return [
        ((replicate, CALIBRATION_N, CALIBRATION_REPLICATES, _seed(replicate)),)
        for replicate in range(count)
    ]


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every declared replication, or the first ``budget``, then derive the controls."""
    outcomes = map_parallel(_fit_replication, _payloads(budget), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=CALIBRATION_LABELS,
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def generate_smoke_property_rows(*, n_jobs: int = 1, replicates: int = 1) -> pd.DataFrame:
    """Fit the first ``replicates`` declared replications, unsummarized."""
    outcomes = map_parallel(_fit_replication, _payloads(replicates), n_jobs=n_jobs)
    return pd.DataFrame([row for result in outcomes for row in result])


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared rules and the calibration rule with the efficiency band."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, rate_labels=(), efficiency_bounds=EFFICIENCY_SD
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    return finish(summary, rates)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published cell, with the law, the estimand and the stream root it reads."""
    roles = dict(zip(CALIBRATION_KINDS, ("positive", "control", "control"), strict=True))
    return tuple(
        PropertyCell(
            property="interval_calibration",
            cell=f"{label}__{kind}",
            dgp=BoundedSampler(),
            outcome_learner=logistic,
            treatment_learner=logistic,
            n=CALIBRATION_N,
            replicates=CALIBRATION_REPLICATES,
            seed=_seed(0),
            role=role,
            estimand=label_name(label),
        )
        for label in CALIBRATION_LABELS
        for kind, role in roles.items()
    )
