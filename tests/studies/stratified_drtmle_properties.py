"""Repeated-sampling properties of DR-TMLE with baseline strata.

Two families, against the exact quadrature truths of the stratified paper law.  Declared before
any run:

==========================  ===========================================================  =====  =====
family                      cells                                                        n      R
==========================  ===========================================================  =====  =====
``interval_calibration``    ``marginal_ate`` and ``v<s>_ate``: the marginal and each     2,000  2,000
                            stratum ATE of the both-correct fit, with shrunken-SE and
                            noise controls
``double_robustness``       ``<label>__{both_correct, outcome_correct,                   2,000  2,000
                            treatment_correct, both_wrong}``; a wrong GLM omits ``W12``  2,000  800
==========================  ===========================================================  =====  =====

The marginal cells read the same fits as the stratum cells, so they add no fit.  The marginal
estimate of a stratified fit reduces on ``(g_n, V)``.  It is a different estimator from the
unstratified fit, and the primary scenario does not pair it.

The calibration cells read 2,000 both-correct fits, seeded by
``stream_seed(STUDY, "property_sample", "both_correct", r)``.  The ``both_correct`` robustness
cells read 2,000 more on their own stream, ``"robustness_both_correct"``, and each other
configuration runs 800, the shipped DR-TMLE study's budget, on the stream of its name.  Every fit
is the primary scenario's cross-fitted fit.

The declared run at ``70005d5`` read the calibration cells and the ``both_correct`` robustness
cells off one set of fits, so two claims shared one sample stream, which
``tests/unit/test_method_evidence.py`` forbids.  The robustness cells moved to their own stream
before the re-run.  The calibration stream, its draws and every other stream are unchanged, so no
red cell moved.

The shipped DR-TMLE study's contraction ladder is not repeated per stratum: its cells are
written for one marginal parameter, and a stratum's rate statement is the same theorem on the
law given the stratum.  ``tests/unit/test_stratified_drtmle_exact.py`` carries the
per-stratum reduction, the step a pooled reduction would get wrong.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.stats import norm

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import stratified_alternating_law as law
from tests.studies.canonical_stratified_drtmle import (
    ATE_LABELS,
    CALIBRATION_KINDS,
    DOUBLE_ROBUST_CONFIGURATIONS,
    EFFICIENCY_SD,
    SCENARIO,
    STUDY,
    ate_name,
    draw_from_seed,
    fit_cleverly,
)
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell, replicate_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.seeds import stream_seed

PROPERTY_N = 2_000
CALIBRATION_REPLICATES = 2_000
DOUBLE_ROBUST_REPLICATES = 800
SHRUNKEN_SE_FACTOR = 0.70
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: The calibration batch: its configuration, stream and budget.
CALIBRATION_STREAM = "both_correct"
#: The stream of each robustness configuration.  ``both_correct`` has its own, apart from the
#: calibration stream.
ROBUSTNESS_STREAMS = {
    "both_correct": "robustness_both_correct",
    "outcome_correct": "outcome_correct",
    "treatment_correct": "treatment_correct",
    "both_wrong": "both_wrong",
}
#: Each robustness configuration's budget.
BUDGETS = {
    "both_correct": CALIBRATION_REPLICATES,
    "outcome_correct": DOUBLE_ROBUST_REPLICATES,
    "treatment_correct": DOUBLE_ROBUST_REPLICATES,
    "both_wrong": DOUBLE_ROBUST_REPLICATES,
}


class PaperSampler:
    """The stratified paper law as a coverage-study sampler, with its own exact truths."""

    name = "stratified_paper_law"

    def truth(self) -> dict[str, float]:
        return law.paper_truths()

    def __call__(self, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
        return draw_from_seed(SCENARIO, n, seed)


def _seed(configuration: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", configuration, replicate)


def _fit_replication(payload: tuple[str, str, int, int, int, int]) -> list[dict[str, Any]]:
    family, configuration, replicate, n, requested, seed = payload
    frame, truth = draw_from_seed(SCENARIO, n, seed)
    result = fit_cleverly(frame, configuration)
    rows: list[dict[str, Any]] = []
    for label in ATE_LABELS:
        name = ate_name(label)
        cell = (
            f"{label}__correctly_specified"
            if family == "interval_calibration"
            else f"{label}__{configuration}"
        )
        rows.append(
            replicate_row(
                property_name=family,
                cell=cell,
                role="control" if configuration == "both_wrong" else "positive",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=float(truth[name]),
                estimate=result[name],
                alpha=STUDY.margins.alpha,
            )
        )
    return rows


def _payloads(budget: int | None = None) -> list[tuple[tuple[str, str, int, int, int, int]]]:
    batches = [("interval_calibration", "both_correct", CALIBRATION_STREAM, CALIBRATION_REPLICATES)]
    batches += [
        (
            "double_robustness",
            configuration,
            ROBUSTNESS_STREAMS[configuration],
            BUDGETS[configuration],
        )
        for configuration in DOUBLE_ROBUST_CONFIGURATIONS
    ]
    out: list[tuple[tuple[str, str, int, int, int, int]]] = []
    for family, configuration, stream, declared in batches:
        count = declared if budget is None else budget
        out += [
            ((family, configuration, replicate, PROPERTY_N, declared, _seed(stream, replicate)),)
            for replicate in range(count)
        ]
    return out


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every declared replication, or the first ``budget`` of each, then the controls."""
    outcomes = map_parallel(_fit_replication, _payloads(budget), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=ATE_LABELS,
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=PROPERTY_N,
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
    """Fit the first ``replicates`` declared replications of each configuration, unsummarized."""
    outcomes = map_parallel(_fit_replication, _payloads(replicates), n_jobs=n_jobs)
    return pd.DataFrame([row for result in outcomes for row in result])


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared rules, and the calibration rule with the efficiency band."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, rate_labels=(), efficiency_bounds=EFFICIENCY_SD
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    return finish(summary, rates)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every published cell, with the law, the estimand and the stream root it reads."""
    sampler = PaperSampler()
    cells: list[PropertyCell] = []
    roles = dict(zip(CALIBRATION_KINDS, ("positive", "control", "control"), strict=True))
    for label in ATE_LABELS:
        name = ate_name(label)
        for kind, role in roles.items():
            cells.append(
                PropertyCell(
                    property="interval_calibration",
                    cell=f"{label}__{kind}",
                    dgp=sampler,
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=PROPERTY_N,
                    replicates=CALIBRATION_REPLICATES,
                    seed=_seed(CALIBRATION_STREAM, 0),
                    role=role,
                    estimand=name,
                )
            )
        for configuration in DOUBLE_ROBUST_CONFIGURATIONS:
            cells.append(
                PropertyCell(
                    property="double_robustness",
                    cell=f"{label}__{configuration}",
                    dgp=sampler,
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=PROPERTY_N,
                    replicates=BUDGETS[configuration],
                    seed=_seed(ROBUSTNESS_STREAMS[configuration], 0),
                    role="control" if configuration == "both_wrong" else "positive",
                    estimand=name,
                )
            )
    return tuple(cells)
