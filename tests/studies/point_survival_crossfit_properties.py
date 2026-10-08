"""Property families of ``point-treatment-survival-crossfit``.

The laws, estimands, learners and exact efficiency bounds are those of
``point-treatment-survival`` (:mod:`tests.studies.point_survival_properties`); every fit here
uses five outer folds.

==============================  ===========================================================
family                          cells, replications and size
==============================  ===========================================================
``double_robustness``           ``survival_t5`` in four nuisance configurations; 1,200 at
                                n = 2,000
``interval_calibration``        ``survival_t5`` (9,600) and ``three_arm_t3`` (1,600) at
                                n = 2,000, with the two derived controls of each
``simultaneous_coverage``       ``all_reported``: the band over the fifteen parameters of the
                                ``survival_t5`` calibration fits, and its pointwise control
==============================  ===========================================================
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_point_survival_crossfit as study
from tests.studies import point_survival_properties as base
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell, replicate_row
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import joint_coverage_rows, joint_property_cells

STUDY = study.STUDY
DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
CALIBRATION_N = 2_000
CALIBRATION_REPLICATES = {"survival_t5": 9_600, "three_arm_t3": 1_600}
CALIBRATION_LABELS = tuple(CALIBRATION_REPLICATES)
EFFICIENCY_SD = {label: base.EFFICIENCY_SD[label] for label in CALIBRATION_LABELS}
EFFICIENCY_RATIO_BAND = base.EFFICIENCY_RATIO_BAND
SHRUNKEN_SE_FACTOR = base.SHRUNKEN_SE_FACTOR
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
JOINT_LABEL = "all_reported"

FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = (
    *(
        ("double_robustness", c, c, DOUBLE_ROBUST_N, DOUBLE_ROBUST_REPLICATES, "survival_t5")
        for c in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")
    ),
    *(
        (
            "interval_calibration",
            label,
            "both_correct",
            CALIBRATION_N,
            CALIBRATION_REPLICATES[label],
            label,
        )
        for label in CALIBRATION_LABELS
    ),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream."""
    cells: list[PropertyCell] = []
    for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong"):
        cells.append(
            PropertyCell(
                property="double_robustness",
                cell=f"survival_t5__{configuration}",
                dgp=base.DeclaredLaw("survival_t5"),
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=DOUBLE_ROBUST_N,
                replicates=DOUBLE_ROBUST_REPLICATES,
                seed=_seed("double_robustness", configuration, 0),
                role="control" if configuration == "both_wrong" else "positive",
                estimand=base.ESTIMAND["survival_t5"],
            )
        )
    for label in CALIBRATION_LABELS:
        for cell, role in (
            ("correctly_specified", "positive"),
            ("shrunken_se_control", "control"),
            ("noise_control", "control"),
        ):
            cells.append(
                PropertyCell(
                    property="interval_calibration",
                    cell=f"{label}__{cell}",
                    dgp=base.DeclaredLaw(label),
                    outcome_learner=lambda: None,
                    treatment_learner=lambda: None,
                    n=CALIBRATION_N,
                    replicates=CALIBRATION_REPLICATES[label],
                    seed=_seed("interval_calibration", label, 0),
                    role=role,
                    estimand=base.ESTIMAND[label],
                )
            )
    cells.extend(
        joint_property_cells(
            JOINT_LABEL,
            n=CALIBRATION_N,
            replicates=CALIBRATION_REPLICATES["survival_t5"],
            seed=_seed("interval_calibration", "survival_t5", 0),
        )
    )
    return tuple(cells)


def _fit_set_rows(payload: tuple[str, str, str, int, int, int, str]) -> list[dict[str, Any]]:
    family, stream, configuration, replicate, n, requested, label = payload
    frame = base.draw(label, n, _seed(family, stream, replicate))
    joint = family == "interval_calibration" and label == "survival_t5"
    result = base.fit_label(
        label, frame, configuration=configuration, simultaneous=joint, n_folds=study.N_FOLDS
    )
    common_row = {"replicate": replicate, "n": n, "requested": requested}
    cell = (
        f"{label}__correctly_specified"
        if family == "interval_calibration"
        else f"survival_t5__{configuration}"
    )
    role = "control" if configuration == "both_wrong" else "positive"
    rows = [
        replicate_row(
            property_name=family,
            cell=cell,
            role=role,
            truth=base.TRUTHS[label][base.ESTIMAND[label]],
            estimate=base.estimate_of(result, label),
            alpha=STUDY.margins.alpha,
            **common_row,
        )
    ]
    if joint:
        rows.extend(
            joint_coverage_rows(
                result,
                base.TRUTHS["survival_t5"],
                tuple(result.estimates),
                label=JOINT_LABEL,
                replicate=replicate,
                n=n,
                requested=requested,
                pointwise_critical=CRITICAL,
            )
        )
    return rows


def _payloads(budget: int | None) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    for family, stream, configuration, n, replicates, label in FIT_SETS:
        requested = replicates if budget is None else budget
        out.extend(
            ((family, stream, configuration, r, n, requested, label),) for r in range(requested)
        )
    return out


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check."""
    outcomes = map_parallel(_fit_set_rows, _payloads(budget), n_jobs=n_jobs)
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


def failure_probe(draws: int, *, n_jobs: int = 1, start: int = 0) -> dict[str, int]:
    """Failed fits per fit set over streams ``start`` to ``start + draws - 1``."""
    payloads = [
        ((family, stream, configuration, r, n, draws, label),)
        for family, stream, configuration, n, _, label in FIT_SETS
        for r in range(start, start + draws)
    ]
    outcomes = map_parallel(_probe_one, payloads, n_jobs=n_jobs)
    counts = {f"{family}/{stream}": 0 for family, stream, *_ in FIT_SETS}
    for (payload,), failed in zip(payloads, outcomes, strict=True):
        counts[f"{payload[0]}/{payload[1]}"] += int(failed)
    return counts


def _probe_one(payload: tuple[str, str, str, int, int, int, str]) -> bool:
    try:
        rows = _fit_set_rows(payload)
    except Exception:
        return True
    return not all(np.isfinite(row["estimate"]) and np.isfinite(row["std_error"]) for row in rows)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows, STUDY, rate_labels=(), efficiency_bounds=EFFICIENCY_SD
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    return finish(summary, rates)
