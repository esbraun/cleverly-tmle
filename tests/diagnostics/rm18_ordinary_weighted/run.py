"""Design OW: the calibration ladder, the sharp null and the targeting family of `weighted-ltmle`.

RM18 of ``docs/roadmap.md`` declares the design in "Design OW".  Each part runs on its own:

* OW-A fits the ordinary weighted fit (arm W) and its unweighted twin (arm U) at n = 2,000,
  8,000 and 32,000, 17,000 fresh replicates each, and reads the SE ratio of W at 32,000;
* OW-B fits both arms on the null law at 4,000 and 16,000, 3,200 fresh replicates each, and
  reads the rejection rate and the coverage of W at 4,000;
* OW-C refits the registered targeting family on 2,655 fresh draws, and computes ``b_inf``,
  the exact population limit of the static untargeted plug-in.

    python -m tests.diagnostics.rm18_ordinary_weighted.run --part OW-A --output <scratch>

``--read-only`` rebuilds ``reading.csv`` from the rows and ``validation.csv`` in ``--output``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import norm, t

from tests import discrete_law_longitudinal as law
from tests.diagnostics import rm18_shared as shared
from tests.diagnostics.rm18_fixed_weights.run import BOUNDS
from tests.studies import weighted_longitudinal_properties_common as weighted
from tests.studies.canonical_weighted_ltmle import STUDY as ORDINARY
from tests.studies.evidence.inference import Interval
from tests.studies.evidence.properties import summarize_cells
from tests.studies.evidence.property_verdicts import necessity_verdicts

HERE = Path(__file__).resolve().parent
DESIGN = "ordinary-weighted"
PARTS = ("OW-A", "OW-B", "OW-C")
ARMS = ("W", "U")

CALIBRATION_SIZES = (2_000, 8_000, 32_000)
CALIBRATION_REPLICATES = 17_000
NULL_SIZES = (4_000, 16_000)
NULL_REPLICATES = 3_200
FAMILY_N = 2_000
FAMILY_REPLICATES = 2_655

#: OW-A reads W at this size, against this window around 1.
CALIBRATION_READ = 32_000
WINDOW = (0.97, 1.03)
#: OW-B reads W at the registered size, against the registered gate.
NULL_READ = 4_000
REJECTION_CEILING = ORDINARY.margins.alpha + ORDINARY.margins.type_i_margin
COVERAGE_FLOOR = ORDINARY.margins.coverage_floor

#: OW-C: the registered family, its margin and displacement threshold, and its budget.
FAMILY = "targeting_necessity"
FAMILY_CELLS = tuple(
    (FAMILY, f"{label}__{arm}")
    for label in weighted.CONTRASTS
    for arm in ("targeted", "untargeted")
)
CONTROL = "static__untargeted"
MARGIN = ORDINARY.margins.standardized_bias
REGISTERED_FAMILY_REPLICATES = weighted.NECESSITY_REPLICATES
MINIMUM_POWER = 0.80
#: The exact-law precondition: an untargeted value equals its longhand follower mean.
EXACT_TOLERANCE = 1e-12

PERSISTENT = "persistent deficit"
CONTRACTING = "finite-sample, contracting"
UNRESOLVED = "unresolved"
OUTSIDE = "resolved outside gate"
WITHIN = "resolved within gate"
INERT = "control inert"
UNDERPOWERED = "control underpowered by design"
DISCRIMINABLE = "control discriminable by design"
RESOLVED = "family resolved"
NOT_RESOLVED = "family not resolved"


def calibration_payloads(replicates: int) -> list[tuple[Any, ...]]:
    return shared.ladder_payloads(
        ORDINARY,
        cross_fit=False,
        part="OW-A",
        law_name="selected",
        prefix=("rm18", DESIGN, "interval_calibration"),
        arms=ARMS,
        sizes=CALIBRATION_SIZES,
        replicates=replicates,
    )


def null_payloads(replicates: int) -> list[tuple[Any, ...]]:
    return shared.ladder_payloads(
        ORDINARY,
        cross_fit=False,
        part="OW-B",
        law_name="null",
        prefix=("rm18", DESIGN, "type_i_error"),
        arms=ARMS,
        sizes=NULL_SIZES,
        replicates=replicates,
    )


def family_payloads(replicates: int) -> list[tuple[Any, ...]]:
    """The registered targeting payload on fresh seeds, one draw for all four cells."""
    labels = [("rm18", DESIGN, FAMILY, index) for index in range(replicates)]
    seeds = shared.fresh_seeds(ORDINARY, labels, shared.weighted_registered_seeds(ORDINARY))
    return [
        (FAMILY, "targeted", index, FAMILY_N, replicates, seed, "mechanism_correct")
        for index, seed in enumerate(seeds)
    ]


def _family_replicate(payload: tuple[Any, ...]) -> list[dict[str, Any]]:
    rows = weighted._fit_replication(ORDINARY, False, payload)
    return [{**row, "seed": payload[5]} for row in rows]


VALIDATION = {
    "OW-A": (
        ("interval_calibration", "correctly_specified"),
        [("interval_calibration", "static__correctly_specified")],
    ),
    "OW-B": (("type_i_error", "sharp_null"), [("type_i_error", "static__sharp_null")]),
    "OW-C": ((FAMILY, "targeted"), list(FAMILY_CELLS)),
}


def validate(part: str, cap: int | None, jobs: int) -> pd.DataFrame:
    """R4 on the committed replicates of the part's registered cells."""
    payload, cells = VALIDATION[part]
    frame = shared.validate_weighted(
        ORDINARY, cross_fit=False, part=part, payload=payload, cells=cells, cap=cap, jobs=jobs
    )
    if part == "OW-C":
        held, largest = exact_precondition()
        frame.loc[len(frame)] = {
            "part": part,
            "check": "exact-law precondition",
            "compared": 2,
            "largest_difference": largest,
            "result": shared.HOLDS if held else shared.FAILS,
        }
    return frame


def exact_precondition() -> tuple[bool, float]:
    """The exact-law untargeted value of each static regimen equals its longhand mean."""
    values = shared.exact_untargeted("mechanism_correct")
    largest = max(abs(values[label] - shared.follower_mean(label)) for label in values)
    return bool(largest <= EXACT_TOLERANCE), float(largest)


def b_inf() -> float:
    """The exact population limit of the static untargeted plug-in, minus its truth."""
    values = shared.exact_untargeted("mechanism_correct")
    return float(values["always"] - values["never"] - law.TRUTH[shared.STATIC])


def discrimination_probability(standardized: float, replicates: int) -> float:
    """The declared normal approximation of the chance that a family control discriminates."""
    quantile = float(t.ppf(0.995, replicates - 1))
    root = math.sqrt(replicates)
    return float(
        norm.cdf(root * (standardized - MARGIN) - quantile)
        + norm.cdf(-root * (standardized + MARGIN) - quantile)
    )


def calibration_label(interval: Interval) -> str:
    """OW-A, on the SE-ratio interval of W at 32,000."""
    if interval.high < WINDOW[0]:
        return PERSISTENT
    if interval.within(*WINDOW):
        return CONTRACTING
    return UNRESOLVED


def null_label(rejection: Interval, coverage: Interval) -> str:
    """OW-B, on the rejection and coverage intervals of W at 4,000."""
    if rejection.low > REJECTION_CEILING or coverage.high < COVERAGE_FLOOR:
        return OUTSIDE
    if rejection.high <= REJECTION_CEILING and coverage.low >= COVERAGE_FLOOR:
        return WITHIN
    return UNRESOLVED


def population_label(standardized: float, power: float) -> str:
    """OW-C, on the population standardized bias and the discrimination probability."""
    if standardized <= MARGIN:
        return INERT
    if power < MINIMUM_POWER:
        return UNDERPOWERED
    return DISCRIMINABLE


def ladder_readings(
    rows: pd.DataFrame, part: str, sizes: tuple[int, ...], bounds: dict[str, float] | None
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    """Every arm and size of one ladder part, and the statistics its reading needs."""
    out: list[dict[str, Any]] = []
    kept: dict[tuple[str, int], dict[str, Any]] = {}
    for arm in ARMS:
        for n in sizes:
            group = rows.loc[(rows["arm"] == arm) & (rows["n"] == n)]
            statistics, points, draws = shared.ladder_statistics(
                group,
                ORDINARY,
                design=DESIGN,
                part=part,
                bound=None if bounds is None else bounds[arm],
            )
            out.extend(statistics)
            kept[arm, n] = {"points": points, "draws": draws, "rows": statistics}
    return out, kept


def _interval(statistics: list[dict[str, Any]], name: str) -> Interval:
    row = next(row for row in statistics if row["statistic"] == name)
    return Interval(float(row["ci_lower"]), float(row["ci_upper"]))


def calibration_table(rows: pd.DataFrame) -> list[dict[str, Any]]:
    out, kept = ladder_readings(rows, "OW-A", CALIBRATION_SIZES, BOUNDS)
    for n in CALIBRATION_SIZES:
        delta = shared.difference_interval(
            kept["W", n]["draws"]["se_ratio"], kept["U", n]["draws"]["se_ratio"]
        )
        out.append(
            shared.reading(
                "OW-A",
                f"n = {n}",
                "Delta_SE(n)",
                value=kept["W", n]["points"]["se_ratio"] - kept["U", n]["points"]["se_ratio"],
                interval=delta,
                result="descriptive",
            )
        )
    interval = _interval(kept["W", CALIBRATION_READ]["rows"], "SE ratio")
    out.append(
        shared.reading(
            "OW-A",
            f"W, n = {CALIBRATION_READ}",
            "reading",
            interval=interval,
            result=calibration_label(interval),
        )
    )
    return out


def null_table(rows: pd.DataFrame) -> list[dict[str, Any]]:
    out, kept = ladder_readings(rows, "OW-B", NULL_SIZES, None)
    statistics = kept["W", NULL_READ]["rows"]
    label = null_label(_interval(statistics, "rejection rate"), _interval(statistics, "coverage"))
    out.append(shared.reading("OW-B", f"W, n = {NULL_READ}", "reading", result=label))
    return out


def family_table(rows: pd.DataFrame, limit: float) -> list[dict[str, Any]]:
    """OW-C: the fresh family under its registered rule, and the population reading."""
    margins = ORDINARY.margins
    summary = summarize_cells(
        rows,
        margin=margins.standardized_bias,
        confidence_level=margins.confidence_level,
        alpha=margins.alpha,
    )
    necessity_verdicts(
        summary,
        rows,
        family=FAMILY,
        labels=tuple(weighted.CONTRASTS),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=weighted.NECESSITY_DISPLACEMENT,
    )
    out = []
    for row in summary.itertuples(index=False):
        out.append(
            shared.reading(
                "OW-C",
                str(row.cell),
                "bias",
                value=float(row.bias),
                interval=Interval(float(row.bias_ci_lower), float(row.bias_ci_upper)),
                result=f"margin {row.bias_margin:.6f}; {'passes' if row.passed else 'fails'}",
            )
        )
    displacement = float(summary["targeting_displacement"].iloc[0])
    family_holds = bool(summary["property_passed"].iloc[0])
    out.append(shared.reading("OW-C", "family", "displacement", value=displacement))
    spread = float(rows.loc[rows["cell"] == CONTROL, "estimate"].std(ddof=1))
    standardized = abs(limit) / spread
    power = discrimination_probability(standardized, REGISTERED_FAMILY_REPLICATES)
    out += [
        shared.reading("OW-C", CONTROL, "b_inf", value=limit),
        shared.reading("OW-C", CONTROL, "fresh SD", value=spread),
        shared.reading("OW-C", CONTROL, "abs(b_inf) / SD", value=standardized),
        shared.reading("OW-C", CONTROL, "p_1200", value=power),
        shared.reading(
            "OW-C", CONTROL, "population reading", result=population_label(standardized, power)
        ),
        shared.reading(
            "OW-C", "family", "fresh reading", result=RESOLVED if family_holds else NOT_RESOLVED
        ),
    ]
    return out


def reading_table(
    part: str, rows: pd.DataFrame, validation: pd.DataFrame, limit: float | None = None
) -> pd.DataFrame:
    out = shared.validation_readings(validation, part)
    if not shared.validated(validation, part):
        out.append(shared.reading(part, "", "reading", result=shared.NOT_VALIDATED))
    elif part == "OW-A":
        out += calibration_table(rows)
    elif part == "OW-B":
        out += null_table(rows)
    else:
        out += family_table(rows, b_inf() if limit is None else limit)
    return shared.reading_frame(out)


def draw(part: str, cap: int | None, jobs: int) -> pd.DataFrame:
    if part == "OW-A":
        calls = calibration_payloads(shared.budget(CALIBRATION_REPLICATES, cap))
        return pd.DataFrame(shared.pool(shared.ladder_replicate, calls, jobs))
    if part == "OW-B":
        calls = null_payloads(shared.budget(NULL_REPLICATES, cap))
        return pd.DataFrame(shared.pool(shared.ladder_replicate, calls, jobs))
    calls = family_payloads(shared.budget(FAMILY_REPLICATES, cap))
    return pd.DataFrame(
        [row for rows in shared.pool(_family_replicate, calls, jobs) for row in rows]
    )


def table(output: Path, part: str) -> pd.DataFrame:
    rows_path, validation_path, _ = shared.part_paths(output, part)
    return reading_table(part, shared.optional_rows(rows_path), shared.read_rows(validation_path))


def main() -> None:
    arguments = shared.arguments(__doc__.splitlines()[0], PARTS, HERE)
    output, part = arguments.output, arguments.part
    rows_path, validation_path, reading_path = shared.part_paths(output, part)
    if not arguments.read_only:
        with shared.run_log(output, f"{part}, cap {arguments.replicates}"):
            validation = validate(part, arguments.replicates, arguments.jobs)
            shared.write_table(validation, validation_path)
            if shared.validated(validation, part):
                shared.write_table(draw(part, arguments.replicates, arguments.jobs), rows_path)
            shared.write_table(table(output, part), reading_path)
    else:
        shared.write_table(table(output, part), reading_path)
    shared.show(reading_path)


if __name__ == "__main__":
    main()
