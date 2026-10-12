"""Property families of ``point-survival-curve-crossfit``.

Every fit is ``LTMLE`` with ``survival_model="hazard"`` and ``n_folds=5``, and every fit set reuses
the replicate rows of :func:`tests.studies.point_survival_curve_properties.fit_set_rows` on this
study's own streams.  The learners of every positive cell are the saturated cell fits, which are
correct on these finite laws.  Every efficiency bound is exact.

================================  =========================================================
family                            cells, replications and size
================================  =========================================================
``interval_calibration``          ``survival_t5`` one step (9,600, with the shrunken-SE and
                                  noise controls); ``survival_t5_per_horizon`` (1,600);
                                  ``competing_t4`` one step (1,600); n = 2,000
``double_robustness``             ``survival_t5`` one step in four nuisance configurations;
                                  1,200 at n = 2,000
``simultaneous_coverage``         from the ``survival_t5`` calibration fits: the band over
                                  each arm's five-visit risk curve and over all fifteen
                                  reported parameters, each with its pointwise control
================================  =========================================================

The failure rule is the registry's: a replication that raises is never redrawn, and the shared
harness refuses a cell that lost one.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_point_survival_curve_crossfit as study
from tests.studies import point_survival_curve_properties as base
from tests.studies.evidence.properties import REPLICATE_COLUMNS, PropertyCell
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import joint_property_cells

STUDY = study.STUDY
N_FOLDS = study.N_FOLDS
DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
CALIBRATION_N = 2_000
CALIBRATION_REPLICATES = {
    "survival_t5": 9_600,
    "survival_t5_per_horizon": 1_600,
    "competing_t4": 1_600,
}
CONTROLLED_LABELS = ("survival_t5",)
CONFIGURATIONS = base.CONFIGURATIONS
EFFICIENCY_SD = {
    "survival_t5": base.EFFICIENCY_SD["survival_t5"],
    "survival_t5_per_horizon": base.EFFICIENCY_SD["survival_t5"],
    "competing_t4": base.EFFICIENCY_SD["competing_t4"],
}
EFFICIENCY_RATIO_BAND = base.EFFICIENCY_RATIO_BAND
SHRUNKEN_SE_FACTOR = base.SHRUNKEN_SE_FACTOR
CRITICAL = base.CRITICAL

#: Every fit set: ``(family, stream label, configuration, n, replicates, label)``.
FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = (
    *(
        ("interval_calibration", label, "both_correct", CALIBRATION_N, replicates, label)
        for label, replicates in CALIBRATION_REPLICATES.items()
    ),
    *(
        ("double_robustness", c, c, DOUBLE_ROBUST_N, DOUBLE_ROBUST_REPLICATES, "survival_t5")
        for c in CONFIGURATIONS
    ),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream."""
    cells: list[PropertyCell] = []

    def add(family: str, cell: str, role: str, label: str, reps: int, stream: str) -> None:
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=base.DeclaredLaw(label),
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=CALIBRATION_N,
                replicates=reps,
                seed=_seed(family, stream, 0),
                role=role,
                estimand=base.ESTIMAND[label],
            )
        )

    for label, replicates in CALIBRATION_REPLICATES.items():
        arms = [("correctly_specified", "positive")]
        if label in CONTROLLED_LABELS:
            arms += [("shrunken_se_control", "control"), ("noise_control", "control")]
        for cell, role in arms:
            add("interval_calibration", f"{label}__{cell}", role, label, replicates, label)
    for configuration in CONFIGURATIONS:
        add(
            "double_robustness",
            f"survival_t5__{configuration}",
            "control" if configuration == "both_wrong" else "positive",
            "survival_t5",
            DOUBLE_ROBUST_REPLICATES,
            configuration,
        )
    for label in (*(f"curve_{arm}" for arm in base.CURVE_ARMS), base.JOINT_LABEL):
        cells.extend(
            joint_property_cells(
                label,
                n=CALIBRATION_N,
                replicates=CALIBRATION_REPLICATES["survival_t5"],
                seed=_seed("interval_calibration", "survival_t5", 0),
            )
        )
    return tuple(cells)


fit_set_rows = partial(base.fit_set_rows, n_folds=N_FOLDS, seed=_seed)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check."""
    outcomes = map_parallel(fit_set_rows, base.payloads(FIT_SETS, budget), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=CONTROLLED_LABELS,
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
    """Failed or non-finite fits per fit set over streams ``start`` to ``start + draws - 1``."""
    probes = [
        ((family, stream, configuration, r, n, draws, label),)
        for family, stream, configuration, n, _, label in FIT_SETS
        for r in range(start, start + draws)
    ]
    outcomes = map_parallel(_probe_one, probes, n_jobs=n_jobs)
    counts = {f"{family}/{stream}": 0 for family, stream, *_ in FIT_SETS}
    for (payload,), failed in zip(probes, outcomes, strict=True):
        counts[f"{payload[0]}/{payload[1]}"] += int(failed)
    return counts


def _probe_one(payload: tuple[str, str, str, int, int, int, str]) -> bool:
    try:
        rows = fit_set_rows(payload)
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
