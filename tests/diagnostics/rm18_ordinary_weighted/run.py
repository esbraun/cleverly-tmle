"""Design OW: the calibration ladder, the sharp null and the targeting family of `weighted-ltmle`.

RM18 of ``docs/roadmap.md`` declares the design in "Design OW".  Each part runs on its own:

* OW-A fits the ordinary weighted fit (arm W) and its unweighted twin (arm U) at n = 2,000,
  8,000 and 32,000, 17,000 fresh replicates each, and reads the SE ratio of W at 32,000;
* OW-B fits both arms on the null law at 4,000 and 16,000, 3,200 fresh replicates each, and
  reads the rejection rate and the coverage of W at 4,000;
* OW-C refits the registered targeting family on 2,655 fresh draws, and computes ``b_inf``,
  the exact population limit of the static untargeted plug-in.

    python -m tests.diagnostics.rm18_ordinary_weighted.run --part OW-A --output <scratch>

``--read-only`` rebuilds the part's reading from its rows and validation record in ``--output``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from tests import discrete_law_longitudinal as law
from tests.diagnostics import rm18_seeds
from tests.diagnostics import rm18_shared as shared
from tests.diagnostics.rm18_fixed_weights.run import BOUNDS
from tests.studies import weighted_longitudinal_properties_common as weighted
from tests.studies.canonical_weighted_ltmle import STUDY as ORDINARY
from tests.studies.evidence.inference import Interval, bootstrap
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

#: Each part's registered payload family and summary cells, which R4 refits and recomputes.
VALIDATION = {
    "OW-A": (
        ("interval_calibration", "correctly_specified"),
        [("interval_calibration", "static__correctly_specified")],
    ),
    "OW-B": (("type_i_error", "sharp_null"), [("type_i_error", "static__sharp_null")]),
    "OW-C": ((FAMILY, "targeted"), list(FAMILY_CELLS)),
}


# ---------------------------------------------------------------------------------- seeds


def calibration_labels() -> list[tuple[Any, ...]]:
    prefix = ("rm18", DESIGN, "interval_calibration")
    return shared.ladder_labels(prefix, ARMS, CALIBRATION_SIZES, CALIBRATION_REPLICATES)


def null_labels() -> list[tuple[Any, ...]]:
    return shared.ladder_labels(("rm18", DESIGN, "type_i_error"), ARMS, NULL_SIZES, NULL_REPLICATES)


def family_labels() -> list[tuple[Any, ...]]:
    return [("rm18", DESIGN, FAMILY, index) for index in range(FAMILY_REPLICATES)]


def calibration_payloads(cap: int | None = None) -> list[tuple[Any, ...]]:
    return shared.ladder_payloads(
        ORDINARY,
        cross_fit=False,
        part="OW-A",
        null=False,
        labels=calibration_labels(),
        seeds=rm18_seeds.weighted_seeds()["OW-A"],
        replicates=CALIBRATION_REPLICATES,
        cap=cap,
    )


def null_payloads(cap: int | None = None) -> list[tuple[Any, ...]]:
    return shared.ladder_payloads(
        ORDINARY,
        cross_fit=False,
        part="OW-B",
        null=True,
        labels=null_labels(),
        seeds=rm18_seeds.weighted_seeds()["OW-B"],
        replicates=NULL_REPLICATES,
        cap=cap,
    )


def family_payloads(cap: int | None = None) -> list[tuple[Any, ...]]:
    """The registered targeting payload on fresh seeds; one draw gives all four cells."""
    replicates = shared.budget(FAMILY_REPLICATES, cap)
    seeds = rm18_seeds.weighted_seeds()["OW-C"][:replicates]
    return [
        (
            ORDINARY,
            False,
            (FAMILY, "targeted", index, FAMILY_N, replicates, seed, "mechanism_correct"),
        )
        for index, seed in enumerate(seeds)
    ]


# ------------------------------------------------------------------------ the exact limit


def follower_mean(label: str, *, use_weights: bool = True) -> float:
    """The weighted mean outcome of the units that follow static regimen ``label``, longhand.

    Read off the support and ``SELECTED_PROBS`` alone: no learner, no fit and no frame.
    """
    arm = {"always": 1.0, "never": 0.0}[label]
    followed = np.array(
        [
            point[2] == 1 and point[1] == arm and point[5] == 1 and point[4] == arm
            for point in law.SUPPORT
        ]
    )
    outcome = np.array([0.0 if point[6] is None else float(point[6]) for point in law.SUPPORT])
    weights = weighted.OBS_WEIGHTS if use_weights else np.ones_like(weighted.OBS_WEIGHTS)
    mass = weighted.SELECTED_PROBS * weights * followed
    return float(np.sum(mass * outcome) / np.sum(mass))


def exact_untargeted(configuration: str, *, use_weights: bool = True) -> dict[str, float]:
    """The ordinary fit and the untargeted plug-in of each static regimen on the exact frame."""
    frame = shared.exact_selected_frame(use_weights=use_weights)
    result = weighted.fit(frame, configuration, cross_fit=False)
    return {
        label: weighted.untargeted(frame, label, configuration, cross_fit=False, folds=result.folds)
        for label in ("always", "never")
    }


def exact_precondition() -> tuple[bool, float]:
    """The exact-law untargeted value of each static regimen equals its longhand mean."""
    values = exact_untargeted("mechanism_correct")
    largest = max(abs(values[label] - follower_mean(label)) for label in values)
    return bool(largest <= EXACT_TOLERANCE), float(largest)


def b_inf() -> float:
    """The exact population limit of the static untargeted plug-in, minus its truth."""
    values = exact_untargeted("mechanism_correct")
    return float(values["always"] - values["never"] - law.TRUTH[shared.STATIC])


def discrimination_probability(standardized: float, replicates: int) -> float:
    """The declared normal approximation of the chance that a family control discriminates."""
    quantile = float(t.ppf(0.995, replicates - 1))
    root = math.sqrt(replicates)
    return float(
        norm.cdf(root * (standardized - MARGIN) - quantile)
        + norm.cdf(-root * (standardized + MARGIN) - quantile)
    )


# ------------------------------------------------------------------------------- readings


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


def calibration_table(rows: pd.DataFrame) -> list[dict[str, Any]]:
    out, kept, smoke = shared.ladder_table(
        rows,
        ORDINARY,
        design=DESIGN,
        part="OW-A",
        arms=ARMS,
        sizes=CALIBRATION_SIZES,
        bounds=BOUNDS,
        declared=CALIBRATION_REPLICATES,
    )
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
    interval = shared.row_interval(kept["W", CALIBRATION_READ]["rows"], "SE ratio")
    result = shared.label(calibration_label(interval), smoke)
    out.append(
        shared.reading(
            "OW-A", f"W, n = {CALIBRATION_READ}", "reading", interval=interval, result=result
        )
    )
    return out


def null_table(rows: pd.DataFrame) -> list[dict[str, Any]]:
    out, kept, smoke = shared.ladder_table(
        rows,
        ORDINARY,
        design=DESIGN,
        part="OW-B",
        arms=ARMS,
        sizes=NULL_SIZES,
        bounds=None,
        declared=NULL_REPLICATES,
    )
    statistics = kept["W", NULL_READ]["rows"]
    label = null_label(
        shared.row_interval(statistics, "rejection rate"),
        shared.row_interval(statistics, "coverage"),
    )
    out.append(
        shared.reading("OW-B", f"W, n = {NULL_READ}", "reading", result=shared.label(label, smoke))
    )
    return out


def control_sd_interval(estimates: np.ndarray) -> Interval:
    """The 99% percentile bootstrap interval of the control's SD, on its declared stream."""
    draws = bootstrap(
        {"estimate": estimates},
        {"sd": lambda draw: draw["estimate"].std(axis=1, ddof=1)},
        replicates=ORDINARY.margins.bootstrap_replicates,
        seed=shared.bootstrap_seed(ORDINARY, DESIGN, "control SD"),
    )["sd"]
    return shared.interval_of(draws)


def family_table(rows: pd.DataFrame, limit: float) -> list[dict[str, Any]]:
    """OW-C: the fresh family under its registered rule, and the population readings."""
    margins = ORDINARY.margins
    smoke = bool((rows.groupby("cell").size() != FAMILY_REPLICATES).any())
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
    out = [
        shared.reading(
            "OW-C",
            str(row.cell),
            "bias",
            value=float(row.bias),
            interval=Interval(float(row.bias_ci_lower), float(row.bias_ci_upper)),
            result=f"margin {row.bias_margin:.6f}, {'passes' if row.passed else 'fails'}",
        )
        for row in summary.itertuples(index=False)
    ]
    family_holds = bool(summary["property_passed"].iloc[0])
    out.append(
        shared.reading(
            "OW-C", "family", "displacement", value=float(summary["targeting_displacement"].iloc[0])
        )
    )
    estimates = rows.loc[rows["cell"] == CONTROL, "estimate"].to_numpy(dtype=float)
    spread = float(estimates.std(ddof=1))
    standardized = abs(limit) / spread
    power = discrimination_probability(standardized, REGISTERED_FAMILY_REPLICATES)
    sd = control_sd_interval(estimates)
    out += [
        shared.reading("OW-C", CONTROL, "b_inf", value=limit),
        shared.reading("OW-C", CONTROL, "fresh SD", value=spread, interval=sd),
        shared.reading("OW-C", CONTROL, "abs(b_inf) / SD", value=standardized),
        shared.reading("OW-C", CONTROL, "p_1200", value=power),
        shared.reading(
            "OW-C",
            CONTROL,
            "p_1200 at the SD interval's ends",
            interval=Interval(
                discrimination_probability(abs(limit) / sd.high, REGISTERED_FAMILY_REPLICATES),
                discrimination_probability(
                    abs(limit) / sd.low if sd.low > 0 else math.inf, REGISTERED_FAMILY_REPLICATES
                ),
            ),
            result="supplementary",
        ),
        shared.reading(
            "OW-C",
            CONTROL,
            "population reading",
            result=shared.label(population_label(standardized, power), smoke),
        ),
        shared.reading(
            "OW-C",
            "family",
            "fresh reading",
            result=shared.label(RESOLVED if family_holds else NOT_RESOLVED, smoke),
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


# ---------------------------------------------------------------------------------- the run


def validate(part: str, cap: int | None, jobs: int) -> pd.DataFrame:
    """R4 on the part's registered cells; OW-C adds the exact-law precondition."""
    payload, cells = VALIDATION[part]
    rows = shared.validate_weighted(
        ORDINARY, cross_fit=False, part=part, payload=payload, cells=cells, cap=cap, jobs=jobs
    )
    if part == "OW-C":
        held, largest = exact_precondition()
        rows.append(shared.validation_row(part, "exact-law precondition", (held, largest, 2)))
    return shared.validation_frame(rows)


def draw(part: str, cap: int | None, jobs: int) -> pd.DataFrame:
    if part == "OW-A":
        return pd.DataFrame(shared.pool(shared.ladder_replicate, calibration_payloads(cap), jobs))
    if part == "OW-B":
        return pd.DataFrame(shared.pool(shared.ladder_replicate, null_payloads(cap), jobs))
    fitted = shared.pool(shared.weighted_draw, family_payloads(cap), jobs)
    return pd.DataFrame([row for rows in fitted for row in rows])


def run_part(part: str, output: Path, cap: int | None, jobs: int) -> None:
    rows_path, validation_path, _ = shared.part_paths(output, part)
    validation = validate(part, cap, jobs)
    shared.write_table(validation, validation_path)
    if shared.validated(validation, part):
        shared.write_table(shared.require_finite(draw(part, cap, jobs)), rows_path)


def table(part: str, output: Path, jobs: int) -> pd.DataFrame:
    del jobs
    rows_path, validation_path, _ = shared.part_paths(output, part)
    return reading_table(part, shared.optional_rows(rows_path), shared.read_rows(validation_path))


if __name__ == "__main__":
    shared.main(__doc__.splitlines()[0], PARTS, HERE, run_part, table)
