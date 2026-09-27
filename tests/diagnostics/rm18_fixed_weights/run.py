"""Design FW-A: the efficiency ladder of the weighted cross-fitted calibration cell.

RM18 of ``docs/roadmap.md`` declares the design in "Design FW".  Part FW-A fits the registered
cell's law and fit (arm W) and its unweighted twin (arm U) at n = 2,000, 8,000 and 32,000, with
8,300 fresh replicates for each arm and size, and reads the empirical efficiency ratio of W at
32,000.  Part FW-B shares its draws with BD-P, so ``tests/diagnostics/rm18_boundary`` runs it.

    python -m tests.diagnostics.rm18_fixed_weights.run --part FW-A --output <scratch>

``--read-only`` rebuilds ``fw-a-reading.csv`` from ``fw-a-rows.csv.gz`` and
``fw-a-validation.csv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from tests.diagnostics import rm18_seeds
from tests.diagnostics import rm18_shared as shared
from tests.studies import weighted_longitudinal_properties_common as weighted
from tests.studies.canonical_weighted_ltmle_crossfit import STUDY as CROSSFIT
from tests.studies.evidence.inference import Interval
from tests.studies.ltmle_crossfit_properties import EFFICIENCY_SD as UNWEIGHTED_EFFICIENCY_SD

HERE = Path(__file__).resolve().parent
DESIGN = "fixed-weights"
PART = "FW-A"
ARMS = ("W", "U")
SIZES = (2_000, 8_000, 32_000)
REPLICATES = 8_300
#: The exact efficiency bound of each arm's own law.
BOUNDS = {
    "W": weighted.EFFICIENCY_SD["static"],
    "U": UNWEIGHTED_EFFICIENCY_SD["static"],
}
#: The registered cell this part reads.
CELL = ("interval_calibration", "static__correctly_specified")
#: The reading reads the W arm at this size.
READ_SIZE = 32_000
#: The contraction window: 1.5 times the R3 half-width of 0.020 either side of 1.
WINDOW = (0.97, 1.03)

PERSISTENT = "persistent excess"
CONTRACTING = "finite-sample, contracting"
UNRESOLVED = "unresolved"
WEIGHTS_ADD = "weights add excess"
REVERSE = "reverse"
NO_WEIGHT_EXCESS = "no weight-specific excess"


def labels() -> list[tuple[Any, ...]]:
    """The declared seed labels ``("rm18", DESIGN, arm, n, replicate)``, in assignment order."""
    return shared.ladder_labels(("rm18", DESIGN), ARMS, SIZES, REPLICATES)


def payloads(cap: int | None = None) -> list[tuple[Any, ...]]:
    """The declared fresh draws, the first ``cap`` of each rung under a smoke cap."""
    return shared.ladder_payloads(
        CROSSFIT,
        cross_fit=True,
        part=PART,
        null=False,
        labels=labels(),
        seeds=rm18_seeds.weighted_seeds()[PART],
        replicates=REPLICATES,
        cap=cap,
    )


def reading_label(interval: Interval) -> str:
    """The declared reading of the W empirical efficiency interval at 32,000."""
    if interval.low > WINDOW[1]:
        return PERSISTENT
    if interval.within(*WINDOW):
        return CONTRACTING
    return UNRESOLVED


def attribution(interval: Interval) -> str:
    """The labelled attribution of one ``Delta(n)`` interval."""
    if interval.low > 0.0:
        return WEIGHTS_ADD
    if interval.high < 0.0:
        return REVERSE
    return NO_WEIGHT_EXCESS


def reading_table(rows: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Every declared statistic, the attributions and the reading, from the fresh rows."""
    out = shared.validation_readings(validation, PART)
    scope = f"W, n = {READ_SIZE}"
    if not shared.validated(validation, PART):
        out.append(shared.reading(PART, scope, "reading", result=shared.NOT_VALIDATED))
        return shared.reading_frame(out)
    statistics, kept, smoke = shared.ladder_table(
        rows,
        CROSSFIT,
        design=DESIGN,
        part=PART,
        arms=ARMS,
        sizes=SIZES,
        bounds=BOUNDS,
        declared=REPLICATES,
    )
    out.extend(statistics)
    for n in SIZES:
        delta = shared.difference_interval(
            kept["W", n]["draws"]["efficiency_empirical"],
            kept["U", n]["draws"]["efficiency_empirical"],
        )
        out.append(
            shared.reading(
                PART,
                f"n = {n}",
                "Delta(n)",
                value=kept["W", n]["points"]["efficiency_empirical"]
                - kept["U", n]["points"]["efficiency_empirical"],
                interval=delta,
                result=shared.label(attribution(delta), smoke),
            )
        )
    interval = shared.interval_of(kept["W", READ_SIZE]["draws"]["efficiency_empirical"])
    out.append(
        shared.reading(
            PART,
            scope,
            "reading",
            interval=interval,
            result=shared.label(reading_label(interval), smoke),
        )
    )
    return shared.reading_frame(out)


def run_part(part: str, output: Path, cap: int | None, jobs: int) -> None:
    """R4 on the 2,400 committed replicates of the calibration cell, then the fresh ladder."""
    rows_path, validation_path, _ = shared.part_paths(output, part)
    validation = shared.validation_frame(
        shared.validate_weighted(
            CROSSFIT,
            cross_fit=True,
            part=part,
            payload=("interval_calibration", "correctly_specified"),
            cells=[CELL],
            cap=cap,
            jobs=jobs,
        )
    )
    shared.write_table(validation, validation_path)
    if shared.validated(validation, part):
        rows = pd.DataFrame(shared.pool(shared.ladder_replicate, payloads(cap), jobs))
        shared.write_table(shared.require_finite(rows), rows_path)


def table(part: str, output: Path, jobs: int) -> pd.DataFrame:
    """The reading table from the rows and the validation record in ``output``."""
    del jobs
    rows_path, validation_path, _ = shared.part_paths(output, part)
    return reading_table(shared.optional_rows(rows_path), shared.read_rows(validation_path))


if __name__ == "__main__":
    shared.main(__doc__.splitlines()[0], (PART,), HERE, run_part, table)
