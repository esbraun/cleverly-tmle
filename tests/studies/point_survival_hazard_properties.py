"""Property families of ``point-survival-hazard``.

Each label names one estimand on one law of :mod:`tests.studies.point_survival_common`, as in
:mod:`tests.studies.point_survival_properties`.  Every fit is ``LTMLE`` with
``survival_model="hazard"`` and ``targeting="iterative"``.  The learners of every positive cell
are the saturated cell fits, which are correct on these finite laws.  Every efficiency bound is
exact (:func:`~tests.studies.point_survival_common.efficiency_sd`).

==============================  ===========================================================
family                          cells, replications and size
==============================  ===========================================================
``interval_calibration``        ``survival_t5`` (the visit-5 difference; 9,600, with the
                                shrunken-SE and noise controls) and ``competing_t4`` (the
                                death incidence difference at visit 4; 1,600); n = 2,000
``double_robustness``           ``survival_t5`` in four nuisance configurations; 1,200 at
                                n = 2,000.  ``outcome_correct`` is the correct hazard with a
                                wrong mechanism
``targeting_necessity``         ``survival_t5``: the targeted estimate and the plug-in of the
                                same fit, with a wrong hazard and the correct mechanism; 1,200
                                at n = 2,000
``root_n_and_efficiency``       ``survival_t5`` at n = 1,000, 2,000 and 8,000; 800 each.  The
                                1,000 rung is the ladder's control
``power``                       ``survival_t5`` at n = 2,000; 800
``simultaneous_coverage``       ``all_reported``: the band over the fifteen parameters of the
                                ``survival_t5`` calibration fits, and its pointwise control
==============================  ===========================================================

The wrong hazard learner is :class:`~tests.studies.point_survival_properties.InterceptOnly`,
which predicts one hazard per node for both arms, so its plug-in contrast is zero in every
sample.  The failure rule is the registry's: a replication that raises is never redrawn, and the
shared harness refuses a cell that lost one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_point_survival_hazard as study
from tests.studies import point_survival_common as common
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    necessity_verdicts,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import joint_coverage_rows, joint_property_cells
from tests.studies.point_survival_properties import (
    EFFICIENCY_SD as POINT_EFFICIENCY_SD,
)
from tests.studies.point_survival_properties import (
    ESTIMAND,
    LABELS,
    TRUTHS,
    InterceptOnly,
    MarginalProbabilities,
)

STUDY = study.STUDY
TARGETING = study.ALGORITHM
DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
NECESSITY_REPLICATES = 1_200
NECESSITY_N = 2_000
RATE_REPLICATES = 800
RATE_SIZES = (1_000, 2_000, 8_000)
CALIBRATION_N = 2_000
CALIBRATION_REPLICATES = {"survival_t5": 9_600, "competing_t4": 1_600}
CALIBRATION_LABELS = tuple(CALIBRATION_REPLICATES)
#: The labels that carry the two derived calibration controls.
CONTROLLED_LABELS = ("survival_t5",)
POWER_REPLICATES = 800
POWER_N = 2_000

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
JOINT_LABEL = "all_reported"
#: The declared floor on the targeted arm's displacement from the plug-in, in the targeted
#: arm's empirical standard deviations.  The plug-in contrast is zero in every sample.
TARGETING_DISPLACEMENT = 0.25
CONFIGURATIONS = ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")

#: The exact efficiency bounds of the labels, from the X13 module that pins them.
EFFICIENCY_SD = {label: POINT_EFFICIENCY_SD[label] for label in CALIBRATION_LABELS}


def learners(configuration: str = "both_correct") -> dict[str, Any]:
    """The cell learners of one nuisance configuration.  No pseudo-outcome is fitted."""
    hazard_correct = configuration in {"both_correct", "outcome_correct"}
    mechanism_correct = configuration in {"both_correct", "mechanism_correct"}
    return {
        "outcome_learner": common.CellMeans() if hazard_correct else InterceptOnly(),
        "treatment_learner": (
            common.CellProbabilities() if mechanism_correct else MarginalProbabilities()
        ),
        "censoring_learner": common.CellMeans() if mechanism_correct else MarginalProbabilities(),
    }


def fit_label(
    label: str,
    frame: pd.DataFrame,
    *,
    configuration: str = "both_correct",
    simultaneous: bool = False,
    targeting: str = TARGETING,
    n_folds: int = 1,
) -> Any:
    """The property fit of one label's sample."""
    scenario, regimens, *_ = LABELS[label]
    return LTMLE(
        regimens,
        reference="arm0",
        n_folds=n_folds,
        simultaneous=simultaneous,
        survival_model="hazard",
        targeting=targeting,
        max_iter=study.MAX_ITER,
        tol=study.TOL,
        random_state=0,
        **learners(configuration),
    ).fit(scenario.container(frame))


def draw(label: str, n: int, seed: int) -> pd.DataFrame:
    return LABELS[label][0].draw(n, seed)


def plug_in(result: Any, name: str) -> float:
    """The initial-hazard plug-in of one reported parameter of a hazard fit."""
    return study.initial_estimates(result, (name,))[name]


# ------------------------------------------------------------------ the grid

#: Every fit set: ``(family, stream label, configuration, n, replicates, label)``.
FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = (
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
    *(
        ("double_robustness", c, c, DOUBLE_ROBUST_N, DOUBLE_ROBUST_REPLICATES, "survival_t5")
        for c in CONFIGURATIONS
    ),
    (
        "targeting_necessity",
        "wrong_hazard",
        "mechanism_correct",
        NECESSITY_N,
        NECESSITY_REPLICATES,
        "survival_t5",
    ),
    *(
        ("root_n_and_efficiency", f"n_{size}", "both_correct", size, RATE_REPLICATES, "survival_t5")
        for size in RATE_SIZES
    ),
    ("power", "alternative", "both_correct", POWER_N, POWER_REPLICATES, "survival_t5"),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


class DeclaredLaw:
    """The law a declared cell reads, with its exact truths.

    Parameters
    ----------
    label : str
        The label of :data:`~tests.studies.point_survival_properties.LABELS`.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.name = f"point_survival__{label}"

    def truth(self) -> dict[str, float]:
        return dict(TRUTHS[self.label])


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream."""
    cells: list[PropertyCell] = []

    def add(family: str, cell: str, role: str, label: str, n: int, reps: int, stream: str) -> None:
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=DeclaredLaw(label),
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=n,
                replicates=reps,
                seed=_seed(family, stream, 0),
                role=role,
                estimand=ESTIMAND[label],
            )
        )

    for label in CALIBRATION_LABELS:
        arms = [("correctly_specified", "positive")]
        if label in CONTROLLED_LABELS:
            arms += [("shrunken_se_control", "control"), ("noise_control", "control")]
        for cell, role in arms:
            add(
                "interval_calibration",
                f"{label}__{cell}",
                role,
                label,
                CALIBRATION_N,
                CALIBRATION_REPLICATES[label],
                label,
            )
    for configuration in CONFIGURATIONS:
        add(
            "double_robustness",
            f"survival_t5__{configuration}",
            "control" if configuration == "both_wrong" else "positive",
            "survival_t5",
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            configuration,
        )
    for arm, role in (("targeted", "positive"), ("untargeted", "control")):
        add(
            "targeting_necessity",
            f"survival_t5__{arm}",
            role,
            "survival_t5",
            NECESSITY_N,
            NECESSITY_REPLICATES,
            "wrong_hazard",
        )
    for size in RATE_SIZES:
        add(
            "root_n_and_efficiency",
            f"survival_t5__n_{size}",
            "control" if size == min(RATE_SIZES) else "positive",
            "survival_t5",
            size,
            RATE_REPLICATES,
            f"n_{size}",
        )
    add(
        "power",
        "survival_t5__alternative",
        "positive",
        "survival_t5",
        POWER_N,
        POWER_REPLICATES,
        "alternative",
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


# ------------------------------------------------------------------ rows


def fit_set_rows(
    payload: tuple[str, str, str, int, int, int, str],
    *,
    targeting: str = TARGETING,
    n_folds: int = 1,
    seed: Any = None,
) -> list[dict[str, Any]]:
    """The replicate rows of one fit of one fit set."""
    family, stream, configuration, replicate, n, requested, label = payload
    frame = draw(label, n, (seed or _seed)(family, stream, replicate))
    common_row = {"replicate": replicate, "n": n, "requested": requested}

    def row(cell: str, role: str, estimate: Any) -> dict[str, Any]:
        return replicate_row(
            property_name=family,
            cell=cell,
            role=role,
            truth=TRUTHS[label][ESTIMAND[label]],
            estimate=estimate,
            alpha=STUDY.margins.alpha,
            **common_row,
        )

    if family == "interval_calibration":
        joint = label == "survival_t5"
        result = fit_label(label, frame, simultaneous=joint, targeting=targeting, n_folds=n_folds)
        rows = [row(f"{label}__correctly_specified", "positive", result[ESTIMAND[label]])]
        if joint:
            rows.extend(
                joint_coverage_rows(
                    result,
                    TRUTHS["survival_t5"],
                    tuple(result.estimates),
                    label=JOINT_LABEL,
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    pointwise_critical=CRITICAL,
                )
            )
        return rows
    result = fit_label(
        label, frame, configuration=configuration, targeting=targeting, n_folds=n_folds
    )
    estimate = result[ESTIMAND[label]]
    if family == "double_robustness":
        role = "control" if configuration == "both_wrong" else "positive"
        return [row(f"survival_t5__{configuration}", role, estimate)]
    if family == "targeting_necessity":
        return [
            row("survival_t5__targeted", "positive", estimate),
            control_row(
                property_name=family,
                cell="survival_t5__untargeted",
                truth=TRUTHS[label][ESTIMAND[label]],
                estimate=plug_in(result, ESTIMAND[label]),
                standard_error=float(estimate.std_error),
                critical=CRITICAL,
                **common_row,
            ),
        ]
    if family == "root_n_and_efficiency":
        role = "control" if n == min(RATE_SIZES) else "positive"
        return [row(f"survival_t5__n_{n}", role, estimate)]
    return [row("survival_t5__alternative", "positive", estimate)]


def _payloads(
    fit_sets: Sequence[tuple[str, str, str, int, int, str]], budget: int | None
) -> list[tuple[Any, ...]]:
    out: list[tuple[Any, ...]] = []
    for family, stream, configuration, n, replicates, label in fit_sets:
        requested = replicates if budget is None else budget
        out.extend(
            ((family, stream, configuration, r, n, requested, label),) for r in range(requested)
        )
    return out


def generate_property_rows(*, n_jobs: int = STUDY_JOBS, budget: int | None = None) -> pd.DataFrame:
    """Fit every property replication; ``budget`` caps each fit set for a pre-run check."""
    outcomes = map_parallel(fit_set_rows, _payloads(FIT_SETS, budget), n_jobs=n_jobs)
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
    """Failed or non-converged fits per fit set over streams ``start`` to ``start + draws - 1``."""
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
        rows = fit_set_rows(payload)
    except Exception:
        return True
    return not all(np.isfinite(row["estimate"]) and np.isfinite(row["std_error"]) for row in rows)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement",),
        rate_labels=("survival_t5",),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=("survival_t5",),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    return finish(summary, rates)
