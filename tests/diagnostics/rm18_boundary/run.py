"""Design BD: re-read the boundary cells on fresh draws, and the paired rows of FW-B and BD-P.

RM18 of ``docs/roadmap.md`` declares the design in "Design BD", and FW-B in "Design FW".  Each
part runs on its own, in this order:

* BD-0 computes the pass probability of each registered budget at the nominal truth;
* BD-1 re-reads four coverage cells at 6,000 fresh replicates each;
* BD-2 re-reads the multi-arm calibration cell at 17,000;
* BD-3 re-reads the weighted both-wrong control at 2,655, and computes its ``b_inf``;
* BD-P1 compares the committed 800 paired draws again, ``native`` and ``hajek``;
* BD-P-pilot validates the paired harness on the registered draws, fits the 800-draw pilot and
  writes ``pilot.csv`` with the step 2 budget ``R_p``;
* BD-P2 fits ``R_p`` fresh paired draws.  It reads ``pilot.csv`` from ``--output``, and refuses
  one that differs from the committed ``pilot.csv``.  Its comparisons go to
  ``bd-p2-comparisons.csv``, and ``--read-only`` computes them again from the rows.

    python -m tests.diagnostics.rm18_boundary.run --part BD-1 --output <scratch>

``--read-only`` rebuilds the part's reading from the rows and the validation record.
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
from tests.diagnostics import rm18_shared as shared
from tests.diagnostics.rm18_boundary import paired
from tests.studies import canonical_drtmle
from tests.studies import drtmle_properties as binary
from tests.studies import multi_arm_ctmle_selector_properties as selector
from tests.studies import multi_arm_drtmle_properties as multi
from tests.studies import weighted_longitudinal_properties_common as weighted
from tests.studies.canonical_drtmle import STUDY as BINARY
from tests.studies.canonical_multi_arm_ctmle_selector import STUDY as SELECTOR
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
from tests.studies.evidence.seeds import replicate_seed, stream_seed

HERE = Path(__file__).resolve().parent
DESIGN = "boundary"
PARTS = ("BD-0", "BD-1", "BD-2", "BD-3", "BD-P1", "BD-P-pilot", "BD-P2")

COVERAGE_REPLICATES = 6_000
CALIBRATION_REPLICATES = 17_000
CONTROL_REPLICATES = 2_655
#: The outer-rung budget Design SL declares; the multi-arm registered seed set holds it.
SL_OUTER_REPLICATES = 73_000
SL_OUTER_SIZES = (2_000, 8_000)

#: Each coverage-study cell a part re-reads: its record, its property module, its key, budget.
CELLS: dict[str, tuple[tuple[StudyRecord, ModuleType, str, str, int], ...]] = {
    "BD-1": (
        (SELECTOR, selector, "root_n_and_efficiency", "n_500", COVERAGE_REPLICATES),
        (
            BINARY,
            binary,
            "double_robust_contraction",
            "treatment_correct_n1500",
            COVERAGE_REPLICATES,
        ),
        (MULTI, multi, "double_robust_contraction", "outcome_correct_n4000", COVERAGE_REPLICATES),
        (MULTI, multi, "root_n_and_efficiency", "n_500", COVERAGE_REPLICATES),
    ),
    "BD-2": (
        (MULTI, multi, "interval_calibration", "correctly_specified", CALIBRATION_REPLICATES),
    ),
}
CONTROL = ("double_robustness", "static__both_wrong")

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


# ------------------------------------------------------------------------- seeds (R2)


def study_primary_seeds(record: StudyRecord) -> set[int]:
    return {
        replicate_seed(record, scenario, index)
        for scenario in record.scenarios
        for index in range(record.replicates)
    }


def coverage_registered_states(record: StudyRecord, module: ModuleType) -> set[int]:
    """Every replicate seed a registered coverage-study module draws, SL's budget included."""
    states = study_primary_seeds(record)
    for cell in module.cells():
        count = cell.replicates
        if module is multi and cell.property == CONTRACTION_FAMILY and cell.n in SL_OUTER_SIZES:
            count = max(count, SL_OUTER_REPLICATES)
        states.update(np.random.SeedSequence(cell.seed).generate_state(count).tolist())
    return states


def coverage_seed(
    record: StudyRecord, labels: tuple[str, ...], replicates: int, registered: set[int]
) -> int:
    """The ``CoverageStudy`` seed under the collision rule, fixed at the declared budget."""
    seed = stream_seed(record, *labels)
    counter = 0
    while True:
        states = np.random.SeedSequence(seed).generate_state(replicates).tolist()
        if len(set(states)) == replicates and registered.isdisjoint(states):
            return seed
        counter += 1
        seed = stream_seed(record, *labels, "retry", counter)


def binary_registered_seeds() -> set[int]:
    seeds = study_primary_seeds(BINARY)
    seeds.update(
        stream_seed(BINARY, "property", cell.property, cell.cell, str(index + cell.seed_offset))
        for cell in binary.cells()
        for index in range(cell.replicates)
    )
    return seeds


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


def _binary_registered(payload: tuple[Any, int]) -> dict[str, Any]:
    return binary._property_replicate(payload)


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
    labels = ("rm18", DESIGN, property_name, cell_name)
    if module is binary:
        order = [(*labels, index) for index in range(replicates)]
        seeds = shared.fresh_seeds(record, order, binary_registered_seeds())
        calls = [(cell, index, replicates, seed) for index, seed in enumerate(seeds)]
        rows = pd.DataFrame(shared.pool(_binary_replicate, calls, jobs))
    else:
        seed = coverage_seed(record, labels, declared, coverage_registered_states(record, module))
        rows = _coverage_rows(module, replace(cell, seed=seed, replicates=replicates), jobs)
        rows["seed"] = np.random.SeedSequence(seed).generate_state(replicates)[rows["replicate"]]
    return rows.assign(study=record.slug)


def validate_cell(
    record: StudyRecord,
    module: ModuleType,
    property_name: str,
    cell_name: str,
    part: str,
    cap: int | None,
    jobs: int,
) -> list[dict[str, Any]]:
    """R4 on every committed replicate of one cell."""
    committed = shared.read_rows(record.artifact("property-replicates.csv.gz"))
    selection = committed.loc[
        (committed["property"] == property_name) & (committed["cell"] == cell_name)
    ]
    count = shared.budget(len(selection), cap)
    selection = selection.loc[selection["replicate"] < count]
    cell = registered_cell(module, property_name, cell_name)
    if module is binary:
        refit = pd.DataFrame(
            shared.pool(_binary_registered, [(cell, index) for index in range(count)], jobs)
        )
    else:
        refit = _coverage_rows(module, replace(cell, replicates=count), jobs)
    keys = ["property", "cell", "replicate"]
    scope = f"{record.slug} {property_name}/{cell_name}"
    return [
        shared.validation_row(
            part, f"{scope} refit rows", shared.compare_rows(refit, selection, keys)
        ),
        shared.validation_row(
            part,
            f"{scope} summary row",
            shared.summary_check(record, committed, refit, [(property_name, cell_name)]),
        ),
    ]


def registered_summary(rows: pd.DataFrame, record: StudyRecord, label: str) -> pd.DataFrame:
    """The registered rule of each fresh cell, through the framework, on the diagnostic stream."""
    stream = shared.bootstrap_record(record, DESIGN, label)
    summary, _ = apply_shared_verdicts(rows, stream, rate_labels=())
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


def cell_readings(part: str, summary: pd.DataFrame, record: StudyRecord) -> list[dict[str, Any]]:
    out = []
    for row in summary.itertuples(index=False):
        scope = f"{record.slug} {row.property}/{row.cell}"
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
        out.append(shared.reading(part, scope, "reading", result=gate_label(row, record.margins)))
    return out


def coverage_table(part: str, rows: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record, _, property_name, cell_name, _ in CELLS[part]:
        group = rows.loc[
            (rows["study"] == record.slug)
            & (rows["property"] == property_name)
            & (rows["cell"] == cell_name)
        ]
        summary = registered_summary(group, record, cell_name)
        out += cell_readings(part, summary, record)
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


def control_payloads(replicates: int) -> list[tuple[Any, ...]]:
    labels = [
        ("rm18", DESIGN, "double_robustness", "both_wrong", index) for index in range(replicates)
    ]
    seeds = shared.fresh_seeds(CROSSFIT, labels, shared.weighted_registered_seeds(CROSSFIT))
    return [
        (
            "double_robustness",
            "both_wrong",
            index,
            weighted.DOUBLE_ROBUST_N,
            replicates,
            seed,
            "both_wrong",
        )
        for index, seed in enumerate(seeds)
    ]


def _control_replicate(payload: tuple[Any, ...]) -> list[dict[str, Any]]:
    rows = weighted._fit_replication(CROSSFIT, True, payload)
    return [{**row, "seed": payload[5]} for row in rows]


def control_limit() -> float:
    """``b_inf``: the both-wrong fit on the exact-law frame with one outer fold, minus truth."""
    result = weighted.fit(shared.exact_selected_frame(), "both_wrong", cross_fit=False)
    return float(result[shared.STATIC].psi) - float(law.TRUTH[shared.STATIC])


def control_table(rows: pd.DataFrame) -> list[dict[str, Any]]:
    static = rows.loc[rows["cell"] == CONTROL[1]]
    summary = registered_summary(static, CROSSFIT, CONTROL[1])
    out = cell_readings("BD-3", summary, CROSSFIT)
    limit = control_limit()
    spread = float(static["estimate"].std(ddof=1))
    ratio = abs(limit) / spread
    out += [
        shared.reading("BD-3", CONTROL[1], "b_inf", value=limit),
        shared.reading("BD-3", CONTROL[1], "fresh SD", value=spread),
        shared.reading("BD-3", CONTROL[1], "abs(b_inf) / SD", value=ratio),
        shared.reading(
            "BD-3",
            CONTROL[1],
            "population reading",
            result=INERT if ratio <= CROSSFIT.margins.standardized_bias else NOT_INERT,
        ),
    ]
    return out


# ----------------------------------------------------------------------------- the parts


def run_part(part: str, output: Path, cap: int | None, jobs: int) -> None:
    rows_path, validation_path, _ = shared.part_paths(output, part)
    if part == "BD-0":
        return
    if part in CELLS:
        validation = pd.DataFrame(
            [
                row
                for record, module, property_name, cell_name, _ in CELLS[part]
                for row in validate_cell(record, module, property_name, cell_name, part, cap, jobs)
            ],
            columns=list(shared.VALIDATION_COLUMNS),
        )
        shared.write_table(validation, validation_path)
        if shared.validated(validation, part):
            frames = [
                fresh_rows(record, module, property_name, cell_name, declared, cap, jobs)
                for record, module, property_name, cell_name, declared in CELLS[part]
            ]
            shared.write_table(pd.concat(frames, ignore_index=True), rows_path)
    elif part == "BD-3":
        validation = shared.validate_weighted(
            CROSSFIT,
            cross_fit=True,
            part=part,
            payload=("double_robustness", "both_wrong"),
            cells=[
                ("double_robustness", "static__both_wrong"),
                ("double_robustness", "dynamic__both_wrong"),
            ],
            cap=cap,
            jobs=jobs,
        )
        shared.write_table(validation, validation_path)
        if shared.validated(validation, part):
            calls = control_payloads(shared.budget(CONTROL_REPLICATES, cap))
            fitted = shared.pool(_control_replicate, calls, jobs)
            shared.write_table(pd.DataFrame([row for rows in fitted for row in rows]), rows_path)
    elif part == "BD-P1":
        rows = paired.committed_paired()
        rows = rows.loc[rows["replicate"] < shared.budget(CROSSFIT.replicates, cap)]
        checks = []
        if cap is None:
            compared = paired.comparisons(rows, CROSSFIT, jobs)
            checks.append(
                shared.validation_row(
                    part, "native reproduces equivalence.csv", paired.reproduces_committed(compared)
                )
            )
        else:
            checks.append(
                shared.validation_row(
                    part, "smoke run, no comparison with equivalence.csv", (True, 0.0, 0)
                )
            )
        shared.write_table(
            pd.DataFrame(checks, columns=list(shared.VALIDATION_COLUMNS)), validation_path
        )
        shared.write_table(rows, rows_path)
    elif part == "BD-P-pilot":
        validation = pd.DataFrame(
            paired.registered_draws_reproduce(jobs, cap), columns=list(shared.VALIDATION_COLUMNS)
        )
        shared.write_table(validation, validation_path)
        if shared.validated(validation, part):
            rows, inference = paired.draw_both(
                paired.PILOT_RECORD, shared.budget(paired.PILOT_REPLICATES, cap), jobs
            )
            pilot_rows = paired.paired_rows(rows, inference)
            shared.write_table(pilot_rows, rows_path)
            compared = paired.comparisons(pilot_rows, paired.PILOT_RECORD, jobs)
            shared.write_table(paired.pilot_table(compared), output / "pilot.csv")
    elif part == "BD-P2":
        pilot = shared.read_rows(output / "pilot.csv")
        committed = HERE / "pilot.csv"
        held = not committed.exists() or shared.read_rows(committed).equals(pilot)
        validation = pd.DataFrame(
            [
                shared.validation_row(
                    part, "pilot.csv is the committed pilot", (held, 0.0, int(committed.exists()))
                )
            ],
            columns=list(shared.VALIDATION_COLUMNS),
        )
        shared.write_table(validation, validation_path)
        if held:
            _, budget = paired.paired_budget(pilot)
            rows, inference = paired.draw_both(
                paired.PAIRED_RECORD, shared.budget(budget, cap), jobs
            )
            shared.write_table(paired.paired_rows(rows, inference), rows_path)
            compare_paired(output, jobs)
    else:  # pragma: no cover - argparse restricts the choices
        raise ValueError(part)


def reading_table(part: str, output: Path, jobs: int) -> pd.DataFrame:
    rows_path, validation_path, _ = shared.part_paths(output, part)
    if part == "BD-0":
        return shared.reading_frame(bd0_table())
    validation = shared.read_rows(validation_path)
    out = shared.validation_readings(validation, part)
    if not shared.validated(validation, part):
        out.append(shared.reading(part, "", "reading", result=shared.NOT_VALIDATED))
        return shared.reading_frame(out)
    rows = shared.optional_rows(rows_path)
    if part in CELLS:
        out += coverage_table(part, rows)
    elif part == "BD-3":
        out += control_table(rows)
    elif part == "BD-P1":
        out += paired.comparison_readings(
            part, paired.comparisons(rows, CROSSFIT, jobs), labelled=False
        )
    elif part == "BD-P-pilot":
        pilot = shared.read_rows(output / "pilot.csv")
        resolution, budget = paired.paired_budget(pilot)
        out += [
            shared.reading(part, "pilot", "r_pilot", value=resolution),
            shared.reading(part, "pilot", "R_p", value=float(budget)),
        ]
    else:
        out += paired.comparison_readings(
            part, shared.read_rows(comparisons_path(output)), labelled=True
        )
    return shared.reading_frame(out)


def comparisons_path(output: Path) -> Path:
    """Step 2's comparisons, kept beside its rows: their bootstraps gather ``1,000 x R_p``."""
    return output / "bd-p2-comparisons.csv"


def compare_paired(output: Path, jobs: int) -> None:
    """Compare the step 2 rows under both conventions, and write the comparisons."""
    rows = shared.read_rows(shared.part_paths(output, "BD-P2")[0])
    shared.write_table(
        paired.comparisons(rows, paired.PAIRED_RECORD, jobs), comparisons_path(output)
    )


def main() -> None:
    arguments = shared.arguments(__doc__.splitlines()[0], PARTS, HERE)
    output, part = arguments.output, arguments.part
    _, _, reading_path = shared.part_paths(output, part)
    if not arguments.read_only:
        with shared.run_log(output, f"{part}, cap {arguments.replicates}"):
            run_part(part, output, arguments.replicates, arguments.jobs)
            shared.write_table(reading_table(part, output, arguments.jobs), reading_path)
    else:
        if part == "BD-P2":
            compare_paired(output, arguments.jobs)
        shared.write_table(reading_table(part, output, arguments.jobs), reading_path)
    shared.show(reading_path)


if __name__ == "__main__":
    main()
