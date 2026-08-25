"""Truth-based repeated-sampling properties for the DR-TMLE protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_drtmle import STUDY, draw_from_seed, fit_cleverly
from tests.studies.evidence.properties import REPLICATE_COLUMNS, replicate_row
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.evidence.seeds import stream_seed

DOUBLE_ROBUST_REPLICATES = 800
RATE_REPLICATES = 800
CALIBRATION_REPLICATES = 2_400
RATE_SIZES = (500, 1500, 4500)
CALIBRATION_N = 3000


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
    return finish(summary, rates)
