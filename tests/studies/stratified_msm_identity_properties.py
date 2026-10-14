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
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell, replicate_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_derived,
    calibration_verdicts,
    finish,
)
from tests.studies.evidence.replication import fit_properties
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


#: The calibration positive arm of each label to the controls derived from it.
DERIVED_CELLS = calibration_derived(CALIBRATION_LABELS)

_Key = tuple[tuple[int, int, int, int]]


def _draw(key: _Key) -> pd.DataFrame:
    """The sample a key fits, a function of the key alone."""
    _, n, _, seed = key[0]
    return law.beta_sample(n, seed)


def _cells_of(key: _Key) -> list[str]:
    """The ``property/cell`` names the key's rows and derived controls fill."""
    names = []
    for label in CALIBRATION_LABELS:
        name = f"interval_calibration/{label}__correctly_specified"
        names += [name, *DERIVED_CELLS[name]]
    return names


def _cell_of(key: _Key) -> str:
    return "interval_calibration/identity"


def _replicate_of(key: _Key) -> int:
    return key[0][0]


def _fit_replication(key: _Key, frame: pd.DataFrame) -> list[dict[str, Any]]:
    replicate, n, requested, _seed = key[0]
    result = fit_cleverly(frame)
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


def _payloads(budget: int | None = None) -> list[_Key]:
    count = CALIBRATION_REPLICATES if budget is None else budget
    return [
        ((replicate, CALIBRATION_N, CALIBRATION_REPLICATES, _seed(replicate)),)
        for replicate in range(count)
    ]


#: The share of a cell's requested replications that the accounting lets a run lose.  Zero: a
#: single lost replication turns the cell red (RM39 part b2, OQ-3).
FAILURE_BOUND = 0.0


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every declared replication, or the first ``budget``, then derive the controls."""
    outcomes = fit_properties(
        _payloads(budget),
        _fit_replication,
        cells_of=_cells_of,
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
        draw=_draw,
        derived=DERIVED_CELLS,
    )
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


#: The failure accounting's declared rules (RM39b2 rev 3, D3'a).  The calibration rule takes
#: this study's efficiency band as an explicit argument (D3'd).
PROPERTY_RULES = (property_verdicts.calibration_rule(efficiency_band=EFFICIENCY_RATIO_BAND),)
