"""Truth-based repeated-sampling properties for the DR-TMLE protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_drtmle import STUDY, draw_from_seed, fit_cleverly
from tests.studies.evidence.properties import REPLICATE_COLUMNS, replicate_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    contraction_rates,
    contraction_verdicts,
    finish,
)
from tests.studies.evidence.seeds import stream_seed

DOUBLE_ROBUST_REPLICATES = 800
RATE_REPLICATES = 800
CALIBRATION_REPLICATES = 2_400
RATE_SIZES = (500, 1500, 4500)
CALIBRATION_N = 3000

#: The sizes the contraction ladder is fitted over, and how many replications each rung gets.
#:
#: **Why this family exists.**  ``double_robustness`` judges the bias at one size against an
#: equivalence margin of a quarter of an empirical standard deviation, and on this law the two
#: one-correct cells exceed it at ``n = 1,500``.  A single red cell cannot say which of two
#: very different things happened: a second-order remainder that has not yet decayed, which is
#: what Theorem 1 predicts and leaves the interval eventually valid, or an estimator that is
#: not consistent at all.  Those have the same appearance at one size and opposite meanings.
#:
#: Fitting log |bias| on log ``n`` separates them.  A second-order remainder gives a slope near
#: ``-1``, a first-order one near ``-1/2``, and an inconsistent estimator near ``0``.  The
#: ``both_wrong`` arm rides along as the control that must fail to contract.
#:
#: Raising the level margin instead was considered and rejected: measured over ``n`` in
#: (1500, 3000, 6000), the standardized bias under a correct mechanism runs 0.357, 0.171 and
#: 0.135, so no size on any affordable ladder brings the 99% interval inside 0.25.  The level
#: cell is left red and this family says what kind of red it is.
#:
#: The rungs are judged on *coverage* rather than on that bias -- see
#: :func:`~tests.studies.evidence.property_verdicts.contraction_verdicts`. One rung is red:
#: at ``n = 1,500`` with the
#: outcome regression misspecified the exact coverage interval dips below the declared floor.
#: That is a small-sample statement in one regime, and it is a result the single-size study
#: had no way to reach.
CONTRACTION_SIZES = (1500, 3000, 6000)
CONTRACTION_REPLICATES = 800
CONTRACTION_SCENARIOS = ("outcome_correct", "treatment_correct", "both_wrong")


@dataclass(frozen=True)
class Cell:
    property: str
    cell: str
    scenario: str
    n: int
    replicates: int
    seed_offset: int
    role: str = "positive"


def cells() -> tuple[Cell, ...]:
    out = [
        Cell(
            "double_robustness",
            scenario,
            scenario,
            1500,
            DOUBLE_ROBUST_REPLICATES,
            10_000 + index * 1000,
            "control" if scenario == "both_wrong" else "positive",
        )
        for index, scenario in enumerate(
            ("both_correct", "outcome_correct", "treatment_correct", "both_wrong")
        )
    ]
    out.extend(
        Cell(
            "root_n_and_efficiency",
            f"n_{size}",
            "both_correct",
            size,
            RATE_REPLICATES,
            20_000 + index * 1000,
            "control" if index == 0 else "positive",
        )
        for index, size in enumerate(RATE_SIZES)
    )
    out.append(
        Cell(
            "interval_calibration",
            "correctly_specified",
            "both_correct",
            CALIBRATION_N,
            CALIBRATION_REPLICATES,
            30_000,
        )
    )
    out.extend(
        Cell(
            "double_robust_contraction",
            f"{scenario}_n{size}",
            scenario,
            size,
            CONTRACTION_REPLICATES,
            # One offset per rung, so no two rungs share a replication stream. The ladder is
            # fitted across sizes, and reusing a stream would correlate the rungs and narrow
            # the slope interval for a reason that has nothing to do with the estimator.
            40_000 + scenario_index * 3_000 + size_index * 1_000,
            "control" if scenario == "both_wrong" else "positive",
        )
        for scenario_index, scenario in enumerate(CONTRACTION_SCENARIOS)
        for size_index, size in enumerate(CONTRACTION_SIZES)
    )
    return tuple(out)


def _property_replicate(payload: tuple[Cell, int]) -> dict[str, Any]:
    cell, replicate = payload
    frame, reference_truth = draw_from_seed(
        cell.scenario,
        cell.n,
        stream_seed(
            STUDY,
            "property",
            cell.property,
            cell.cell,
            str(replicate + cell.seed_offset),
        ),
    )
    estimate = fit_cleverly(frame, cell.scenario).estimates["ate"]
    return replicate_row(
        property_name=cell.property,
        cell=cell.cell,
        role=cell.role,
        replicate=replicate,
        n=cell.n,
        requested=cell.replicates,
        truth=float(reference_truth["ate"]),
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    payloads = [((cell, replicate),) for cell in cells() for replicate in range(cell.replicates)]
    rows = map_parallel(_property_replicate, payloads, n_jobs=n_jobs)
    return pd.DataFrame(rows).loc[:, list(REPLICATE_COLUMNS)]


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(rows, STUDY)
    contraction_verdicts(summary, STUDY)
    rates.extend(contraction_rates(rows, STUDY, summary.columns, scenarios=CONTRACTION_SCENARIOS))
    return finish(summary, rates)
