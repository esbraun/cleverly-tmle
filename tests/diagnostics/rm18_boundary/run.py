"""Design BD: re-read the boundary cells on fresh draws, and the paired rows of FW-B and BD-P.

RM18 of ``docs/roadmap.md`` at commit ``985849c6`` declares the design in "Design BD",
and FW-B in "Design FW".  Each
part runs on its own, in this order:

* BD-0 computes the pass probability of each registered budget at the nominal truth;
* BD-1 re-reads four coverage cells at 6,000 fresh replicates each;
* BD-2 re-reads the multi-arm calibration cell at 17,000;
* BD-3 re-reads the weighted both-wrong control at 2,655, and computes its ``b_inf``;
* BD-P1, BD-P-pilot and BD-P2 are the paired parts in :mod:`.paired`.

    python -m tests.diagnostics.rm18_boundary.run --part BD-1 --output <scratch>

``--read-only`` rebuilds the part's reading from what ``--output`` holds; a paired part's
reading comes from its committed comparisons.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom, norm

from tests import discrete_law_longitudinal as law
from tests.diagnostics import rm18_seeds
from tests.diagnostics import rm18_shared as shared
from tests.diagnostics.rm18_boundary import paired
from tests.studies import canonical_drtmle
from tests.studies import drtmle_properties as binary
from tests.studies import weighted_longitudinal_properties_common as weighted
from tests.studies.canonical_drtmle import STUDY as BINARY
from tests.studies.canonical_multi_arm_drtmle import STUDY as MULTI
from tests.studies.canonical_weighted_ltmle_crossfit import STUDY as CROSSFIT
from tests.studies.evidence.inference import Interval, clopper_pearson
from tests.studies.evidence.properties import PropertyCell, replicate_row, run_cells
from tests.studies.evidence.property_verdicts import (
    CONTRACTION_FAMILY,
    UNION_MODEL_SE_BAND,
    apply_shared_verdicts,
    contraction_verdicts,
)
from tests.studies.evidence.registry import StudyRecord

HERE = Path(__file__).resolve().parent
DESIGN = "boundary"
PARTS = ("BD-0", "BD-1", "BD-2", "BD-3", *paired.PARTS)

CONTROL_REPLICATES = 2_655
CONTROL = ("double_robustness", "static__both_wrong")

#: Each cell a part re-reads: its record, its property module, its key and its budget.
CELLS = {part: rm18_seeds.cells(part) for part in ("BD-1", "BD-2")}

#: BD-0: each registered budget, the gate it is read against, and the nominal truth.
NOMINAL_COVERAGE = 0.95
BD0_CELLS = (
    ("SELECTOR root_n_and_efficiency/n_500", 400, "coverage"),
    ("BINARY double_robust_contraction/treatment_correct_n1500", 800, "coverage"),
    ("MULTI double_robust_contraction/outcome_correct_n4000", 600, "coverage"),
    ("MULTI root_n_and_efficiency/n_500", 400, "coverage"),
    ("MULTI interval_calibration/correctly_specified, SE-ratio leg", 1_600, "SE ratio"),
)

SATISFIES = "resolved: truth satisfies the gate"
FAILS_GATE = "resolved: truth fails the gate"
UNRESOLVED = "unresolved"
INERT = "control inert"
NOT_INERT = "control not inert"


# --------------------------------------------------------------------------------- BD-0


def coverage_pass_probability(replicates: int, margins: Any) -> float:
    """P(the exact interval's lower end clears the floor) when the true coverage is 0.95."""
    needed = next(
        count
        for count in range(math.floor(margins.coverage_floor * replicates), replicates + 1)
        if clopper_pearson(count, replicates, confidence_level=margins.confidence_level).low
        >= margins.coverage_floor
    )
    return float(binom.sf(needed - 1, replicates, NOMINAL_COVERAGE))


def se_band_pass_probability(replicates: int, margins: Any) -> float:
    """P(the SE-ratio interval lies inside the band) at a true ratio of 1, by the normal
    approximation with a relative SD of ``1 / sqrt(2R)``."""
    spread = 1.0 / math.sqrt(2.0 * replicates)
    quantile = float(norm.ppf(0.5 + margins.confidence_level / 2.0))
    low, high = margins.calibration_se_ratio
    room = min(1.0 - low, high - 1.0) - quantile * spread
    return float(max(0.0, 2.0 * norm.cdf(room / spread) - 1.0))


def bd0_table() -> list[dict[str, Any]]:
    margins = MULTI.margins
    out = []
    for name, replicates, kind in BD0_CELLS:
        probability = (
            coverage_pass_probability(replicates, margins)
            if kind == "coverage"
            else se_band_pass_probability(replicates, margins)
        )
        out.append(
            shared.reading("BD-0", name, "pass probability at the nominal truth", value=probability)
        )
    return out


# ----------------------------------------------------------------------- BD-1 and BD-2


def registered_cell(module: ModuleType, property_name: str, cell: str) -> Any:
    return next(
        item for item in module.cells() if (item.property, item.cell) == (property_name, cell)
    )


def _binary_replicate(payload: tuple[Any, int, int, int]) -> dict[str, Any]:
    """One fresh binary rung draw: ``_property_replicate`` with the fresh seed."""
    cell, index, requested, seed = payload
    frame, truth = canonical_drtmle.draw_from_seed(cell.scenario, cell.n, seed)
    estimate = canonical_drtmle.fit_cleverly(frame, cell.scenario).estimates["ate"]
    row = replicate_row(
        property_name=cell.property,
        cell=cell.cell,
        role=cell.role,
        replicate=index,
        n=cell.n,
        requested=requested,
        truth=float(truth["ate"]),
        estimate=estimate,
        alpha=BINARY.margins.alpha,
    )
    return {**row, "seed": seed}


def _coverage_rows(module: ModuleType, cell: PropertyCell, jobs: int) -> pd.DataFrame:
    rows = run_cells([cell], module._estimator, n_jobs=jobs)
    if len(rows) != cell.replicates or int(rows["failed_replicates"].max()) != 0:
        raise RuntimeError(
            f"{cell.property}/{cell.cell}: {len(rows)} of {cell.replicates} replicates returned; "
            f"a fit that raises stops the part"
        )
    return rows


def fresh_rows(
    record: StudyRecord,
    module: ModuleType,
    property_name: str,
    cell_name: str,
    declared: int,
    cap: int | None,
    jobs: int,
) -> pd.DataFrame:
    """The fresh re-read of one cell, with each replicate's sample seed."""
    cell = registered_cell(module, property_name, cell_name)
    replicates = shared.budget(declared, cap)
    if module is binary:
        seeds = rm18_seeds.binary_seeds()[:replicates]
        calls = [(cell, index, replicates, seed) for index, seed in enumerate(seeds)]
        rows = pd.DataFrame(shared.pool(_binary_replicate, calls, jobs))
    else:
        seed = rm18_seeds.coverage_seeds()[record.slug, property_name, cell_name]
        rows = _coverage_rows(module, replace(cell, seed=seed, replicates=replicates), jobs)
        rows["seed"] = np.random.SeedSequence(seed).generate_state(replicates)[rows["replicate"]]
    return shared.require_finite(rows).assign(study=record.slug)


def validate_cells(part: str, cap: int | None, jobs: int) -> pd.DataFrame:
    """R4 on every committed replicate of each cell, with one summary per registered study."""
    out = []
    by_record: dict[str, list[tuple[Any, ...]]] = {}
    for entry in CELLS[part]:
        by_record.setdefault(entry[0].slug, []).append(entry)
    keys = ["property", "cell", "replicate"]
    for entries in by_record.values():
        record = entries[0][0]
        committed = shared.read_rows(record.artifact("property-replicates.csv.gz"))
        refits = []
        for _, module, property_name, cell_name, _ in entries:
            selection = committed.loc[
                (committed["property"] == property_name) & (committed["cell"] == cell_name)
            ]
            count = shared.budget(len(selection), cap)
            selection = selection.loc[selection["replicate"] < count]
            cell = registered_cell(module, property_name, cell_name)
            if module is binary:
                calls = [(cell, index) for index in range(count)]
                refit = pd.DataFrame(shared.pool(binary._property_replicate, calls, jobs))
            else:
                refit = _coverage_rows(module, replace(cell, replicates=count), jobs)
            scope = f"{record.slug} {property_name}/{cell_name}"
            out.append(
                shared.validation_row(
                    part, f"{scope} refit rows", shared.compare_rows(refit, selection, keys)
                )
            )
            refits.append(refit)
        cells = [(entry[2], entry[3]) for entry in entries]
        outcome = shared.summary_check(
            record, committed, pd.concat(refits, ignore_index=True), cells
        )
        out.append(shared.validation_row(part, f"{record.slug} summary rows", outcome))
    return shared.validation_frame(out)


def registered_summary(rows: pd.DataFrame, record: StudyRecord, label: str | None) -> pd.DataFrame:
    """The registered rule of each fresh cell, through the framework, on the diagnostic stream.

    The framework derives its bootstrap stream from the record, so the record carries the
    diagnostic stream as its ``resampling_seed``.  ``label=None`` keeps the registered stream,
    which reproduces the published summary of the registered rows.
    """
    stream = (
        record
        if label is None
        else replace(record, resampling_seed=shared.bootstrap_seed(record, DESIGN, label))
    )
    summary, _ = apply_shared_verdicts(
        rows.drop(columns=["seed", "study"], errors="ignore"), stream, rate_labels=()
    )
    contraction_verdicts(summary, stream)
    return summary


def fails_gate(row: Any, margins: Any) -> bool:
    """The declared failing side of each registered rule."""
    coverage = Interval(float(row.coverage_ci_lower), float(row.coverage_ci_upper))
    if row.property == "root_n_and_efficiency":
        return bool(
            coverage.high < margins.coverage_floor
            or bool(row.bias_discriminated)
            or not margins.se_ratio_sanity[0] <= float(row.se_ratio) <= margins.se_ratio_sanity[1]
        )
    if row.property == CONTRACTION_FAMILY:
        return bool(coverage.high < margins.coverage_floor)
    if row.property == "interval_calibration":
        ratio = Interval(float(row.se_ratio_ci_lower), float(row.se_ratio_ci_upper))
        return bool(
            ratio.outside(*margins.calibration_se_ratio)
            or coverage.outside(*margins.calibration_coverage)
        )
    if row.property == "double_robustness":
        return bool(
            bool(row.bias_equivalent)
            or not UNION_MODEL_SE_BAND[0] <= float(row.se_ratio) <= UNION_MODEL_SE_BAND[1]
        )
    raise ValueError(f"no declared failing side for {row.property!r}")


def gate_label(row: Any, margins: Any) -> str:
    if bool(row.passed):
        return SATISFIES
    if fails_gate(row, margins):
        return FAILS_GATE
    return UNRESOLVED


def cell_readings(
    part: str, summary: pd.DataFrame, record: StudyRecord, declared: int
) -> list[dict[str, Any]]:
    out = []
    for row in summary.itertuples(index=False):
        scope = f"{record.slug} {row.property}/{row.cell}"
        smoke = int(row.replicates) != declared
        out += [
            shared.reading(part, scope, "replicates", value=float(row.replicates)),
            shared.reading(
                part,
                scope,
                "coverage",
                value=float(row.coverage),
                interval=Interval(float(row.coverage_ci_lower), float(row.coverage_ci_upper)),
            ),
            shared.reading(
                part,
                scope,
                "bias",
                value=float(row.bias),
                interval=Interval(float(row.bias_ci_lower), float(row.bias_ci_upper)),
                result=f"margin {float(row.bias_margin):.6g}",
            ),
            shared.reading(part, scope, "SE ratio", value=float(row.se_ratio)),
        ]
        if row.property == "interval_calibration":
            out.append(
                shared.reading(
                    part,
                    scope,
                    "SE-ratio interval",
                    interval=Interval(float(row.se_ratio_ci_lower), float(row.se_ratio_ci_upper)),
                )
            )
        out.append(
            shared.reading(
                part, scope, "reading", result=shared.label(gate_label(row, record.margins), smoke)
            )
        )
    return out


def coverage_table(part: str, rows: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record, _, property_name, cell_name, declared in CELLS[part]:
        group = rows.loc[
            (rows["study"] == record.slug)
            & (rows["property"] == property_name)
            & (rows["cell"] == cell_name)
        ]
        out += cell_readings(part, registered_summary(group, record, cell_name), record, declared)
        if property_name == "interval_calibration":
            out.append(
                shared.reading(
                    part,
                    f"{record.slug} {property_name}/{cell_name}",
                    "RMS-SE ratio",
                    value=shared.rms_se_ratio(group),
                    result="supplementary",
                )
            )
    return out


# --------------------------------------------------------------------------------- BD-3


def control_labels() -> list[tuple[Any, ...]]:
    return [
        ("rm18", DESIGN, "double_robustness", "both_wrong", index)
        for index in range(CONTROL_REPLICATES)
    ]


def control_payloads(cap: int | None = None) -> list[tuple[Any, ...]]:
    replicates = shared.budget(CONTROL_REPLICATES, cap)
    seeds = rm18_seeds.weighted_seeds()["BD-3"][:replicates]
    return [
        (
            CROSSFIT,
            True,
            (
                "double_robustness",
                "both_wrong",
                index,
                weighted.DOUBLE_ROBUST_N,
                replicates,
                seed,
                "both_wrong",
            ),
        )
        for index, seed in enumerate(seeds)
    ]


def control_limit() -> float:
    """``b_inf``: the both-wrong fit on the exact-law frame with one outer fold, minus truth."""
    result = weighted.fit(shared.exact_selected_frame(), "both_wrong", cross_fit=False)
    return float(result[shared.STATIC].psi) - float(law.TRUTH[shared.STATIC])


def control_table(rows: pd.DataFrame) -> list[dict[str, Any]]:
    static = rows.loc[rows["cell"] == CONTROL[1]]
    smoke = len(static) != CONTROL_REPLICATES
    out = cell_readings(
        "BD-3", registered_summary(static, CROSSFIT, CONTROL[1]), CROSSFIT, CONTROL_REPLICATES
    )
    limit = control_limit()
    spread = float(static["estimate"].std(ddof=1))
    ratio = abs(limit) / spread
    result = INERT if ratio <= CROSSFIT.margins.standardized_bias else NOT_INERT
    out += [
        shared.reading("BD-3", CONTROL[1], "b_inf", value=limit),
        shared.reading("BD-3", CONTROL[1], "fresh SD", value=spread),
        shared.reading("BD-3", CONTROL[1], "abs(b_inf) / SD", value=ratio),
        shared.reading(
            "BD-3", CONTROL[1], "population reading", result=shared.label(result, smoke)
        ),
    ]
    return out


# ----------------------------------------------------------------------------- the parts


def run_part(part: str, output: Path, cap: int | None, jobs: int) -> None:
    rows_path, validation_path, _ = shared.part_paths(output, part)
    if part == "BD-0":
        return
    if part in paired.PARTS:
        paired.run_part(part, output, cap, jobs)
        return
    if part in CELLS:
        validation = validate_cells(part, cap, jobs)
    else:
        validation = shared.validation_frame(
            shared.validate_weighted(
                CROSSFIT,
                cross_fit=True,
                part=part,
                payload=("double_robustness", "both_wrong"),
                cells=[CONTROL, ("double_robustness", "dynamic__both_wrong")],
                cap=cap,
                jobs=jobs,
            )
        )
    shared.write_table(validation, validation_path)
    if not shared.validated(validation, part):
        return
    if part in CELLS:
        rows = pd.concat(
            [fresh_rows(*entry, cap, jobs) for entry in CELLS[part]], ignore_index=True
        )
    else:
        fitted = shared.pool(shared.weighted_draw, control_payloads(cap), jobs)
        rows = shared.require_finite(pd.DataFrame([row for rows in fitted for row in rows]))
    shared.write_table(rows, rows_path)


def table(part: str, output: Path, jobs: int) -> pd.DataFrame:
    del jobs
    rows_path, validation_path, _ = shared.part_paths(output, part)
    if part == "BD-0":
        return shared.reading_frame(bd0_table())
    validation = shared.read_rows(validation_path)
    out = shared.validation_readings(validation, part)
    if not shared.validated(validation, part):
        out.append(shared.reading(part, "", "reading", result=shared.NOT_VALIDATED))
    elif part in paired.PARTS:
        out += paired.table(part, output)
    elif part in CELLS:
        out += coverage_table(part, shared.read_rows(rows_path))
    else:
        out += control_table(shared.read_rows(rows_path))
    return shared.reading_frame(out)


def pushed(part: str) -> tuple[str, ...]:
    """BD-P2 reads the pilot, which the upstream must hold before a declared step 2."""
    return (paired.PILOT,) if part == "BD-P2" else ()


if __name__ == "__main__":
    shared.main(__doc__.splitlines()[0], PARTS, HERE, run_part, table, pushed)
