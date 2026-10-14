"""False-warning and detection rates of the calibration-slope rule.

Two families.  ``warning_rate`` holds the laws on which a calibration warning is false: the
known propensity at a weak and a strong signal, a correct unpenalized logistic model at a weak
signal with one covariate and with four, and a fitted model on a randomized law.  Each positive
cell must bound the rule's rate at or below the type-I ceiling.  Four of those laws carry a
``fixed_band`` control on the same fits, which reads the pooled slope against the band the
package applied before RM15 and must rise above the ceiling.

``power`` holds two tempered learners whose limit slopes are 1/2 and 2, and the rule must
detect each.

Each law draws its own samples from ``stream_seed(STUDY, "property_sample", family, law,
replicate)``.  A control reads the fits of its positive cell, so the two differ only in the rule.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from tests.parallel import STUDY_JOBS
from tests.studies.calibration_slope_warning import (
    BAND_CONTROLLED,
    LAWS,
    POWER_LAWS,
    PRIMARY_N,
    PROPERTY_CELLS,
    STUDY,
    WARNING_LAWS,
    draw_law,
    fit,
    read,
    rule_covers,
)
from tests.studies.evidence import property_verdicts
from tests.studies.evidence.properties import REPLICATE_COLUMNS
from tests.studies.evidence.property_verdicts import apply_shared_verdicts, finish
from tests.studies.evidence.replication import fit_properties, splat_draw, splat_drawn
from tests.studies.evidence.seeds import stream_seed

#: Replications per property law.  A control reads its positive cell's fits.
PROPERTY_REPLICATES = 10_000


def _row(
    family: str,
    cell: str,
    role: str,
    replicate: int,
    replicates: int,
    truth: float,
    estimate: float,
    std_error: float,
    covered: bool,
    rejected: bool,
) -> dict[str, Any]:
    return {
        "property": family,
        "cell": cell,
        "role": role,
        "replicate": replicate,
        "n": PRIMARY_N,
        "requested_replicates": replicates,
        "failed_replicates": 0,
        "truth": truth,
        "estimate": estimate,
        "std_error": std_error,
        "covered": int(covered),
        "rejected": int(rejected),
    }


def _draw(family: str, law_name: str, replicate: int, replicates: int) -> Any:
    seed = stream_seed(STUDY, "property_sample", family, law_name, replicate)
    return draw_law(law_name, PRIMARY_N, seed)


def _replicate(
    family: str, law_name: str, replicate: int, replicates: int, drawn: Any
) -> list[dict[str, Any]]:
    seed = stream_seed(STUDY, "property_sample", family, law_name, replicate)
    result = fit(drawn, law_name, seed)
    reading = read(result)
    truth = LAWS[law_name].limit_slope
    covered = rule_covers(reading, truth)
    rows = [
        _row(
            family,
            f"{law_name}__rule",
            "positive",
            replicate,
            replicates,
            truth,
            reading.slope,
            reading.std_error,
            covered,
            reading.finding,
        )
    ]
    if family == "warning_rate" and law_name in BAND_CONTROLLED:
        rows.append(
            _row(
                family,
                f"{law_name}__fixed_band",
                "control",
                replicate,
                replicates,
                truth,
                reading.slope,
                reading.std_error,
                covered,
                reading.band,
            )
        )
    return rows


def _cells_of(key: tuple[str, str, int, int]) -> list[str]:
    """The ``property/cell`` names the key's rows fill."""
    family, law_name = key[:2]
    cells = [f"{family}/{law_name}__rule"]
    if family == "warning_rate" and law_name in BAND_CONTROLLED:
        cells.append(f"{family}/{law_name}__fixed_band")
    return cells


def _cell_of(key: tuple[str, str, int, int]) -> str:
    return f"{key[0]}/{key[1]}__rule"


def _replicate_of(key: tuple[str, str, int, int]) -> int:
    return key[2]


#: The share of a cell's requested replications that the accounting lets a run lose, declared
#: for the study's next property run (RM39b2 rev 3 section 6, confirmed 2026-10-14).  Up to
#: ``floor(0.01 R)`` lost replicates are judged by the worst case of each clause; one more is
#: ``bound exceeded``.
FAILURE_BOUND = 0.01

#: The draw-level statistic the accounting reports for each lost replicate (rev 3 D3'f).
PROPERTY_WITNESS = property_verdicts.SAMPLE_WITNESS


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, replicates: int = PROPERTY_REPLICATES
) -> pd.DataFrame:
    """Fit every property replication and return one row per cell and replication."""
    laws = [("warning_rate", law) for law in WARNING_LAWS] + [("power", law) for law in POWER_LAWS]
    payloads = [
        (family, law, index, replicates) for family, law in laws for index in range(replicates)
    ]
    outcomes = fit_properties(
        payloads,
        splat_drawn(_replicate),
        cells_of=_cells_of,
        draw=splat_draw(_draw),
        cell_of=_cell_of,
        replicate_of=_replicate_of,
        n_jobs=n_jobs,
    )
    rows = pd.DataFrame([row for records in outcomes for row in records])
    for family, declared in PROPERTY_CELLS.items():
        cells = set(rows.loc[rows["property"] == family, "cell"])
        if cells != set(declared):
            raise RuntimeError(f"{family} cells {sorted(cells ^ set(declared))} differ")
    return rows.loc[:, list(REPLICATE_COLUMNS)]


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The verdict of each cell, each family against its own rule."""
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=())
    return finish(summary, rates)
