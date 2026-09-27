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

import numpy as np
import pandas as pd

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
#: The registered cell this part reads, and its committed empirical efficiency interval.
CELL = ("interval_calibration", "static__correctly_specified")
REGISTERED_EFFICIENCY = (1.060724, 1.019495, 1.100601)
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


def payloads(replicates: int) -> list[tuple[Any, ...]]:
    """The declared fresh draws, with seeds ``stream_seed(CROSSFIT, "rm18", DESIGN, arm, n, k)``."""
    return shared.ladder_payloads(
        CROSSFIT,
        cross_fit=True,
        part=PART,
        law_name="selected",
        prefix=("rm18", DESIGN),
        arms=ARMS,
        sizes=SIZES,
        replicates=replicates,
    )


def validate(cap: int | None, jobs: int) -> pd.DataFrame:
    """R4 on the 2,400 committed replicates of the calibration cell."""
    return shared.validate_weighted(
        CROSSFIT,
        cross_fit=True,
        part=PART,
        payload=("interval_calibration", "correctly_specified"),
        cells=[CELL],
        cap=cap,
        jobs=jobs,
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
    points: dict[tuple[str, int], float] = {}
    draws: dict[tuple[str, int], np.ndarray] = {}
    for arm in ARMS:
        for n in SIZES:
            group = rows.loc[(rows["arm"] == arm) & (rows["n"] == n)]
            statistics, point, draw = shared.ladder_statistics(
                group, CROSSFIT, design=DESIGN, part=PART, bound=BOUNDS[arm]
            )
            out.extend(statistics)
            points[arm, n] = point["efficiency_empirical"]
            draws[arm, n] = draw["efficiency_empirical"]
    for n in SIZES:
        delta = shared.difference_interval(draws["W", n], draws["U", n])
        out.append(
            shared.reading(
                PART,
                f"n = {n}",
                "Delta(n)",
                value=points["W", n] - points["U", n],
                interval=delta,
                result=attribution(delta),
            )
        )
    interval = shared.interval_of(draws["W", READ_SIZE])
    out.append(
        shared.reading(PART, scope, "reading", interval=interval, result=reading_label(interval))
    )
    return shared.reading_frame(out)


def main() -> None:
    arguments = shared.arguments(__doc__.splitlines()[0], (PART,), HERE)
    output = arguments.output
    rows_path, validation_path, reading_path = shared.part_paths(output, PART)
    if not arguments.read_only:
        with shared.run_log(output, f"{PART}, cap {arguments.replicates}"):
            validation = validate(arguments.replicates, arguments.jobs)
            shared.write_table(validation, validation_path)
            if shared.validated(validation, PART):
                replicates = shared.budget(REPLICATES, arguments.replicates)
                rows = pd.DataFrame(
                    shared.pool(shared.ladder_replicate, payloads(replicates), arguments.jobs)
                )
                shared.write_table(rows, rows_path)
            shared.write_table(table(output), reading_path)
    else:
        shared.write_table(table(output), reading_path)
    shared.show(reading_path)


def table(output: Path) -> pd.DataFrame:
    """The reading table from the rows and the validation record in ``output``."""
    rows_path, validation_path, _ = shared.part_paths(output, PART)
    return reading_table(shared.optional_rows(rows_path), shared.read_rows(validation_path))


if __name__ == "__main__":
    main()
