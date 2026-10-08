"""Property families of ``point-treatment-survival``.

Each label names one estimand on one law of :mod:`tests.studies.point_survival_common`.  The
learners of every positive cell are the saturated cell means, which are correct on these
finite laws.  Every efficiency bound is exact (:func:`~tests.studies.point_survival_common.efficiency_sd`).

==============================  ===========================================================
family                          cells, replications and size
==============================  ===========================================================
``double_robustness``           ``survival_t5`` (the visit-5 difference) in four nuisance
                                configurations; 1,200 at n = 2,000
``root_n_and_efficiency``       ``survival_t5`` at n = 1,000, 2,000 and 8,000; 800 each.  The
                                1,000 rung is the ladder's control
``interval_calibration``        ``survival_t5`` (9,600), ``rmst_5`` (the RMST difference at
                                visit 5 on the same fits), ``competing_t4`` (the death
                                incidence difference at visit 4), ``three_arm_t3`` (arm 2
                                against arm 0 at visit 3, with ``L_t``, wide), ``continuous_t4``
                                (the difference at grid time 3 with an exponential event time),
                                ``end_of_study`` (the held end-of-study difference, D5) and
                                ``weighted_t5`` (the tilted law of the weight ``1 + W1 / 2``); 1,600
                                each at n = 2,000, with the two derived controls of each
``power``                       ``survival_t5`` at n = 4,000; 800
``clustered_inference``         ``clustered_t5`` (the arm-0 risk at visit 5) on 25 clusters of
                                80 that share ``W``; the cluster-robust interval and the same
                                fit's IID interval; 1,600
``simultaneous_coverage``       ``all_reported``: the band over the fifteen parameters of the
                                ``survival_t5`` calibration fits, and its pointwise control
==============================  ===========================================================

The failure rule is the registry's: a replication that raises is never redrawn, and the shared
harness refuses a cell that lost one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator

from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_point_survival as study
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
    clustered_inference_verdicts,
    finish,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import (
    joint_coverage_rows,
    joint_property_cells,
)

STUDY = study.STUDY
DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
RATE_REPLICATES = 800
RATE_SIZES = (1_000, 2_000, 8_000)
CALIBRATION_N = 2_000
#: The calibration size of each label.  ``rmst_5`` reads the ``survival_t5`` fits.
CALIBRATION_REPLICATES = {
    "survival_t5": 9_600,
    "rmst_5": 9_600,
    "competing_t4": 1_600,
    "three_arm_t3": 1_600,
    "continuous_t4": 1_600,
    "end_of_study": 1_600,
    "weighted_t5": 1_600,
}
CALIBRATION_LABELS = tuple(CALIBRATION_REPLICATES)
POWER_REPLICATES = 800
POWER_N = 4_000
CLUSTER_REPLICATES = 1_600
CLUSTER_SIZE = 80
CLUSTER_N = 2_000

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
JOINT_LABEL = "all_reported"

THREE_ARMS = {"arm0": 0, "arm1": 1, "arm2": 2}


def weight(w1: Any, w2: Any) -> Any:
    """The observation weight of ``weighted_t5``: ``1 + W1 / 2``, not a whole number."""
    return 1.0 + 0.5 * np.asarray(w1, dtype=float)


def _contrast(arm: int, horizon: int, cause: int = 1) -> list[common.Target]:
    return [
        common.Target(1.0, "static", arm, horizon, cause),
        common.Target(-1.0, "static", 0, horizon, cause),
    ]


#: Each label: ``(scenario, regimens, estimand, efficiency targets, weighted)``.
LABELS: dict[str, tuple[common.Scenario, dict[str, int], str, list[common.Target], bool]] = {
    "survival_t5": (
        common.SURVIVAL,
        study.REGIMENS,
        common.contrast_name("arm1", "arm0", 5),
        _contrast(1, 5),
        False,
    ),
    "rmst_5": (
        common.SURVIVAL,
        study.REGIMENS,
        "rmst_regimen[arm1 vs arm0 @ t=5]",
        [
            common.Target(sign * -1.0, "static", arm, horizon)
            for horizon in range(1, 5)
            for arm, sign in ((1, 1.0), (0, -1.0))
        ],
        False,
    ),
    "competing_t4": (
        common.COMPETING,
        study.REGIMENS,
        common.contrast_name("arm1", "arm0", 4, "death"),
        _contrast(1, 4, cause=2),
        False,
    ),
    "three_arm_t3": (
        common.THREE_ARM,
        THREE_ARMS,
        common.contrast_name("arm2", "arm0", 3),
        _contrast(2, 3),
        False,
    ),
    "continuous_t4": (
        common.CONTINUOUS,
        study.REGIMENS,
        common.contrast_name("arm1", "arm0", 4),
        _contrast(1, 4),
        False,
    ),
    "end_of_study": (
        common.END_OF_STUDY,
        study.REGIMENS,
        "ate_regimen[arm1 vs arm0]",
        _contrast(1, 2),
        False,
    ),
    "weighted_t5": (
        common.SURVIVAL,
        study.REGIMENS,
        common.contrast_name("arm1", "arm0", 5),
        _contrast(1, 5),
        True,
    ),
    "clustered_t5": (
        common.SURVIVAL,
        study.REGIMENS,
        common.risk_name("arm0", 5),
        [common.Target(1.0, "static", 0, 5)],
        False,
    ),
}


def _truth_of(label: str) -> dict[str, float]:
    scenario, regimens, _, _, weighted = LABELS[label]
    plans = {name: common.static(arm, scenario.levels) for name, arm in regimens.items()}
    values = common.truths(scenario, plans, "arm0", weight=weight if weighted else None)
    if label == "rmst_5":
        values["rmst_regimen[arm1 vs arm0 @ t=5]"] = -sum(
            values[common.contrast_name("arm1", "arm0", horizon)] for horizon in range(1, 5)
        )
    return values


TRUTHS = {label: _truth_of(label) for label in LABELS}
ESTIMAND = {label: entry[2] for label, entry in LABELS.items()}


def efficiency_bound(label: str) -> float:
    """The exact efficiency bound of a label's estimand, as a standard deviation."""
    scenario, _, _, targets, weighted = LABELS[label]
    return common.efficiency_sd(scenario, targets, weight=weight if weighted else None)


#: The bounds, pinned from :func:`efficiency_bound`; the design test recomputes them.
EFFICIENCY_SD: dict[str, float] = {
    "survival_t5": 1.0884501904200183,
    "rmst_5": 2.999011460445784,
    "competing_t4": 0.8067434575609278,
    "three_arm_t3": 1.228538336985197,
    "continuous_t4": 1.0670608901200032,
    "end_of_study": 1.0811546949337272,
    "weighted_t5": 1.1150237573293247,
}


# ------------------------------------------------------------------ learners


class InterceptOnly(BaseEstimator):
    """A regression that ignores its design: the weighted mean of the target."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> InterceptOnly:
        target = np.asarray(y, dtype=float).reshape(-1)
        weights = np.ones_like(target) if sample_weight is None else np.asarray(sample_weight)
        self.mean_ = float(np.average(target, weights=weights))
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(np.asarray(X)), self.mean_)

    def predict_proba(self, X: Any) -> np.ndarray:
        p = self.predict(X)
        return np.column_stack([1.0 - p, p])


class MarginalProbabilities(BaseEstimator):
    """A mechanism that ignores its design: the weighted class shares."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> MarginalProbabilities:
        target = np.asarray(y, dtype=float).reshape(-1)
        weights = np.ones_like(target) if sample_weight is None else np.asarray(sample_weight)
        self.classes_ = np.unique(target)
        shares = np.array([np.sum(weights[target == level]) for level in self.classes_])
        self.shares_ = shares / shares.sum()
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.tile(self.shares_, (len(np.asarray(X)), 1))


def learners(configuration: str = "both_correct") -> dict[str, Any]:
    """The cell learners of one nuisance configuration."""
    q_correct = configuration in {"both_correct", "outcome_correct"}
    g_correct = configuration in {"both_correct", "mechanism_correct"}
    return {
        "outcome_learner": common.CellMeans() if q_correct else InterceptOnly(),
        "pseudo_learner": common.CellMeans() if q_correct else InterceptOnly(),
        "treatment_learner": common.CellProbabilities() if g_correct else MarginalProbabilities(),
        "censoring_learner": common.CellMeans() if g_correct else MarginalProbabilities(),
    }


def fit_label(
    label: str,
    frame: pd.DataFrame,
    *,
    configuration: str = "both_correct",
    simultaneous: bool = False,
    n_folds: int = 1,
) -> Any:
    """The property fit of one label's sample, in sample unless ``n_folds`` says otherwise."""
    scenario, regimens, _, _, weighted = LABELS[label]
    if weighted:
        frame = frame.assign(w=weight(frame["W1"], frame["W2"]))
    data = scenario.container(
        frame,
        weights="w" if weighted else None,
        id="id" if label == "clustered_t5" else None,
    )
    return LTMLE(
        regimens,
        reference="arm0",
        n_folds=n_folds,
        simultaneous=simultaneous,
        max_iter=100,
        tol=1e-10,
        random_state=0,
        **learners(configuration),
    ).fit(data)


def draw(label: str, n: int, seed: int) -> pd.DataFrame:
    scenario = LABELS[label][0]
    return scenario.draw(n, seed, cluster_size=CLUSTER_SIZE if label == "clustered_t5" else None)


def estimate_of(result: Any, label: str) -> Any:
    if label == "rmst_5":
        return result.rmst("arm1", 5, versus="arm0")
    return result[ESTIMAND[label]]


# ------------------------------------------------------------------ the grid

#: Every fit set: ``(family, stream label, configuration, n, replicates, label)``.
FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = (
    *(
        ("double_robustness", c, c, DOUBLE_ROBUST_N, DOUBLE_ROBUST_REPLICATES, "survival_t5")
        for c in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")
    ),
    *(
        ("root_n_and_efficiency", f"n_{size}", "both_correct", size, RATE_REPLICATES, "survival_t5")
        for size in RATE_SIZES
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
        if label != "rmst_5"
    ),
    ("power", "alternative", "both_correct", POWER_N, POWER_REPLICATES, "survival_t5"),
    (
        "clustered_inference",
        "clustered_t5",
        "both_correct",
        CLUSTER_N,
        CLUSTER_REPLICATES,
        "clustered_t5",
    ),
)


def _seed(family: str, label: str, replicate: int) -> int:
    return stream_seed(STUDY, "property_sample", family, label, replicate)


class DeclaredLaw:
    """The law a declared cell reads, with its exact truths.

    Parameters
    ----------
    label : str
        The label of :data:`LABELS`.
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

    for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong"):
        add(
            "double_robustness",
            f"survival_t5__{configuration}",
            "control" if configuration == "both_wrong" else "positive",
            "survival_t5",
            DOUBLE_ROBUST_N,
            DOUBLE_ROBUST_REPLICATES,
            configuration,
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
    for label in CALIBRATION_LABELS:
        stream = "survival_t5" if label == "rmst_5" else label
        for cell, role in (
            ("correctly_specified", "positive"),
            ("shrunken_se_control", "control"),
            ("noise_control", "control"),
        ):
            add(
                "interval_calibration",
                f"{label}__{cell}",
                role,
                label,
                CALIBRATION_N,
                CALIBRATION_REPLICATES[label],
                stream,
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
    for cell, role in (("cluster_robust", "positive"), ("iid_control", "control")):
        add(
            "clustered_inference",
            f"clustered_t5__{cell}",
            role,
            "clustered_t5",
            CLUSTER_N,
            CLUSTER_REPLICATES,
            "clustered_t5",
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


def _fit_set_rows(payload: tuple[str, str, str, int, int, int, str]) -> list[dict[str, Any]]:
    family, stream, configuration, replicate, n, requested, label = payload
    frame = draw(label, n, _seed(family, stream, replicate))
    common_row = {"replicate": replicate, "n": n, "requested": requested}

    def row(cell: str, role: str, name_label: str, estimate: Any) -> dict[str, Any]:
        return replicate_row(
            property_name=family,
            cell=cell,
            role=role,
            truth=TRUTHS[name_label][ESTIMAND[name_label]],
            estimate=estimate,
            alpha=STUDY.margins.alpha,
            **common_row,
        )

    if family == "interval_calibration":
        joint = label == "survival_t5"
        result = fit_label(label, frame, simultaneous=joint)
        rows = [row(f"{label}__correctly_specified", "positive", label, estimate_of(result, label))]
        if joint:
            rmst = estimate_of(result, "rmst_5")
            rows.append(row("rmst_5__correctly_specified", "positive", "rmst_5", rmst))
            names = tuple(result.estimates)
            for joint_row in joint_coverage_rows(
                result,
                TRUTHS["survival_t5"],
                names,
                label=JOINT_LABEL,
                replicate=replicate,
                n=n,
                requested=requested,
                pointwise_critical=CRITICAL,
            ):
                rows.append(joint_row)
        return rows
    if family == "clustered_inference":
        result = fit_label(label, frame)
        estimate = estimate_of(result, label)
        truth = TRUTHS[label][ESTIMAND[label]]
        curve = np.asarray(estimate.influence_curve, dtype=float)
        iid_se = float(np.sqrt(np.mean(curve**2) / len(curve)))
        return [
            row("clustered_t5__cluster_robust", "positive", label, estimate),
            control_row(
                property_name=family,
                cell="clustered_t5__iid_control",
                truth=truth,
                estimate=float(estimate.psi),
                standard_error=iid_se,
                critical=CRITICAL,
                **common_row,
            ),
        ]
    result = fit_label(label, frame, configuration=configuration)
    estimate = estimate_of(result, label)
    if family == "double_robustness":
        role = "control" if configuration == "both_wrong" else "positive"
        return [row(f"survival_t5__{configuration}", role, label, estimate)]
    if family == "root_n_and_efficiency":
        role = "control" if n == min(RATE_SIZES) else "positive"
        return [row(f"survival_t5__n_{n}", role, label, estimate)]
    return [row("survival_t5__alternative", "positive", label, estimate)]


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
        rows,
        STUDY,
        extra_columns=("coverage_gain_ci_lower", "coverage_gain_ci_upper"),
        rate_labels=("survival_t5",),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    clustered_inference_verdicts(
        summary,
        rows,
        STUDY,
        positive_cell="clustered_t5__cluster_robust",
        control_cell="clustered_t5__iid_control",
    )
    simultaneous_coverage_verdicts(summary, margins=STUDY.margins)
    return finish(summary, rates)
