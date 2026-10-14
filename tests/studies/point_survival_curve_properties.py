"""Property families of ``point-survival-curve``.

Each label names one estimand on one law.  Every fit is ``LTMLE`` with
``survival_model="hazard"`` and ``targeting="one_step"``, except where the table says otherwise.
The learners of every positive cell are the saturated cell fits, which are correct on these
finite laws.  Every efficiency bound is exact.

================================  =========================================================
family                            cells, replications and size
================================  =========================================================
``interval_calibration``          ``survival_t5`` (the visit-5 difference; 9,600, with the
                                  shrunken-SE and noise controls); ``competing_t4`` (the
                                  death incidence difference at visit 4); ``rule_t3`` and
                                  ``policy_t3`` (one fit of the three-arm law without
                                  :math:`L_t`: the rule :math:`d(W) = 1\\{W_1 = 1\\}` and a
                                  known policy, each against arm 0 at visit 3);
                                  ``weighted_t5`` (weight :math:`1 + W_1/2`), one step and
                                  per horizon; ``isotonic_t2`` (``risk_regimen[always @
                                  t=2]`` of the law with a treatment at two nodes, fitted
                                  with ``survival_model="iterated_mean"`` and
                                  ``isotonic=True``).  1,600 each except the first; n = 2,000
``isotonic_projection``           ``isotonic_t2__moved``: per replication, how far the
                                  projection moved the level.  A diagnostic: its rejection
                                  rate is the binding share, and it states no verdict
``clustered_inference``           ``clustered_t5`` (the arm-0 risk at visit 5, 25 clusters of
                                  80), one step and per horizon; 1,600 each
``double_robustness``             ``survival_t5`` in four nuisance configurations; 1,200
``targeting_necessity``           ``survival_t5``: the targeted estimate and the plug-in of
                                  the same fit, with a wrong hazard; 1,200
``root_n_and_efficiency``         ``survival_t5`` at n = 1,000, 2,000 and 8,000; 800 each
``power``                         ``survival_t5`` at n = 2,000; 800
``simultaneous_coverage``         from the ``survival_t5`` calibration fits: the band over
                                  each arm's five-visit risk curve (``curve_arm1``,
                                  ``curve_arm0``) and over all fifteen reported parameters
                                  (``all_reported``), each with its pointwise control
================================  =========================================================

**The isotonic cell does not exercise a binding projection.**  Its curve rises from 0.375 at
node 1 to 0.758 at node 2, a step of about nine standard errors at n = 2,000, so the declared
binding share is zero (``X14-impl-review.md`` F4).  The cell checks that the projection
leaves a regular curve's inference unchanged.  It is not evidence for an interval after a
projection that moves a level.  The S-mono law that would have bound has no feasible design
(``X14-progress.md``, deviation 1).

A curve band is the shipped multiplier band (:func:`cleverly.inference.multiplier.
simultaneous_bands`, 2,000 rademacher draws, ``random_state=0``) over the five risks of one
arm, which is the band a fit of that one regimen reports.  The wrong hazard learner is
:class:`~tests.studies.point_survival_properties.InterceptOnly`.  The failure rule is the
registry's: a replication that raises is never redrawn, and the shared harness refuses a cell
that lost one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.inference.multiplier import simultaneous_bands
from cleverly.interventions import Stochastic
from cleverly.longitudinal import LTMLE
from cleverly.longitudinal.regimen import DynamicRegimen
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_survival as mnode
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_point_survival_curve as study
from tests.studies import point_survival_common as common
from tests.studies import point_survival_properties as x13
from tests.studies.canonical_ltmle import G_BOUNDS, declared_regimens
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    clustered_inference_verdicts,
    finish,
    necessity_verdicts,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import joint_coverage_rows, joint_property_cells
from tests.studies.ltmle_survival_properties import sample as mnode_sample
from tests.studies.point_survival_hazard_properties import learners, plug_in

STUDY = study.STUDY
DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
NECESSITY_REPLICATES = 1_200
NECESSITY_N = 2_000
RATE_REPLICATES = 800
RATE_SIZES = (1_000, 2_000, 8_000)
CALIBRATION_N = 2_000
CALIBRATION_REPLICATES = {
    "survival_t5": 9_600,
    "competing_t4": 1_600,
    "three_arm": 1_600,
    "weighted_t5": 1_600,
    "weighted_t5_per_horizon": 1_600,
    "isotonic_t2": 1_600,
}
CLUSTER_N = 2_000
CLUSTER_SIZE = x13.CLUSTER_SIZE
CLUSTER_REPLICATES = 1_600
CLUSTER_LABELS = ("clustered_t5", "clustered_t5_per_horizon")
#: The labels that carry the two derived calibration controls.
CONTROLLED_LABELS = ("survival_t5",)
POWER_REPLICATES = 800
POWER_N = 2_000

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
#: The declared floor on the targeted arm's displacement from the plug-in, in the targeted
#: arm's empirical standard deviations.  The plug-in contrast is zero in every sample.
TARGETING_DISPLACEMENT = 0.25
CONFIGURATIONS = ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong")
#: The shipped band's settings, which the curve bands repeat.
MULTIPLIER = {"n_replicates": 2000, "kind": "rademacher", "random_state": 0}
CURVE_ARMS = ("arm1", "arm0")
JOINT_LABEL = "all_reported"
ISOTONIC_DIAGNOSTIC = "isotonic_t2__moved"

# ------------------------------------------------------------------ the laws

#: S3: the three-arm law of X13 without :math:`L_t`.
THREE_ARM = common.Scenario("three_arm_s3", 3, arms=3, wide=True)
#: The isotonic cell's level, on the law with a treatment at two nodes.
ISOTONIC_ESTIMAND = "risk_regimen[always @ t=2]"


def rule_arm(w1: Any) -> Any:
    """The rule :math:`d(W) = 1\\{W_1 = 1\\}`: arm 1 where ``W1`` is one, else arm 0."""
    return np.where(np.asarray(w1) == 1, 1, 0)


def policy_probabilities(w1: Any) -> np.ndarray:
    """The known policy over three arms: ``(1/4, 1/4 + W1/4, 1/2 - W1/4)``."""
    w1 = np.asarray(w1, dtype=float)
    return np.column_stack([np.full_like(w1, 0.25), 0.25 + 0.25 * w1, 0.5 - 0.25 * w1])


def _rule_assign(w1: int, w2: int) -> tuple[float, ...]:
    arm = int(rule_arm(w1))
    return tuple(1.0 if level == arm else 0.0 for level in THREE_ARM.levels)


def _policy_assign(w1: int, w2: int) -> tuple[float, ...]:
    return tuple(float(value) for value in policy_probabilities(np.array([w1]))[0])


THREE_ARM_REGIMENS: dict[str, Any] = {
    "arm0": 0,
    "rule": DynamicRegimen("rule", (lambda frame: rule_arm(frame["W1"]),), rule_kind="known"),
    "policy": Stochastic(
        lambda frame: policy_probabilities(frame["W1"]), "q", density_kind="known"
    ),
}
THREE_ARM_PLANS = {
    "arm0": common.static(0, THREE_ARM.levels),
    "rule": _rule_assign,
    "policy": _policy_assign,
}
THREE_ARM_TRUTH = common.truths(THREE_ARM, THREE_ARM_PLANS, "arm0")

#: Each label: ``(estimand, truths)``.
TRUTHS: dict[str, dict[str, float]] = {
    "survival_t5": x13.TRUTHS["survival_t5"],
    "competing_t4": x13.TRUTHS["competing_t4"],
    "rule_t3": THREE_ARM_TRUTH,
    "policy_t3": THREE_ARM_TRUTH,
    "weighted_t5": x13.TRUTHS["weighted_t5"],
    "weighted_t5_per_horizon": x13.TRUTHS["weighted_t5"],
    # Read by ``point-survival-curve-crossfit``, which reuses :func:`fit_set_rows`.
    "survival_t5_per_horizon": x13.TRUTHS["survival_t5"],
    "isotonic_t2": dict(mnode.TRUTH),
    "clustered_t5": x13.TRUTHS["clustered_t5"],
    "clustered_t5_per_horizon": x13.TRUTHS["clustered_t5"],
}
ESTIMAND = {
    "survival_t5": x13.ESTIMAND["survival_t5"],
    "competing_t4": x13.ESTIMAND["competing_t4"],
    "rule_t3": common.contrast_name("rule", "arm0", 3),
    "policy_t3": common.contrast_name("policy", "arm0", 3),
    "weighted_t5": x13.ESTIMAND["weighted_t5"],
    "weighted_t5_per_horizon": x13.ESTIMAND["weighted_t5"],
    "survival_t5_per_horizon": x13.ESTIMAND["survival_t5"],
    "isotonic_t2": ISOTONIC_ESTIMAND,
    "clustered_t5": x13.ESTIMAND["clustered_t5"],
    "clustered_t5_per_horizon": x13.ESTIMAND["clustered_t5"],
}


def efficiency_bound(label: str) -> float:
    """The exact efficiency bound of a label's estimand, as a standard deviation."""
    if label in ("rule_t3", "policy_t3"):
        assign = _rule_assign if label == "rule_t3" else _policy_assign
        targets = [
            common.Target(1.0, "policy", assign, 3),
            common.Target(-1.0, "static", 0, 3),
        ]
        return common.efficiency_sd(THREE_ARM, targets)
    if label == "isotonic_t2":
        return float(np.sqrt(np.sum(mnode.PROBS * mnode.eif(ISOTONIC_ESTIMAND) ** 2)))
    return x13.efficiency_bound(label.removesuffix("_per_horizon"))


#: The bounds, pinned from :func:`efficiency_bound`; the design test recomputes them.
EFFICIENCY_SD: dict[str, float] = {
    "survival_t5": x13.EFFICIENCY_SD["survival_t5"],
    "competing_t4": x13.EFFICIENCY_SD["competing_t4"],
    "rule_t3": 0.8508366425845435,
    "policy_t3": 0.7662820005333323,
    "weighted_t5": x13.EFFICIENCY_SD["weighted_t5"],
    "weighted_t5_per_horizon": x13.EFFICIENCY_SD["weighted_t5"],
    "isotonic_t2": 1.7351666267699624,
}


# ------------------------------------------------------------------ fits


def _settings(targeting: str) -> dict[str, Any]:
    if targeting == "one_step":
        return {"targeting": "one_step", "step_size": study.STEP_SIZE, "tol": study.TOL}
    return {"targeting": "iterative", "max_iter": study.PER_HORIZON_MAX_ITER, "tol": study.TOL}


def fit_label(
    label: str,
    frame: pd.DataFrame,
    *,
    configuration: str = "both_correct",
    simultaneous: bool = False,
    n_folds: int = 1,
) -> Any:
    """The property fit of one label's sample."""
    if label == "isotonic_t2":
        return LTMLE(
            declared_regimens(mnode.REGIMEN_SPEC),
            reference=mnode.REGIMEN_REFERENCE,
            outcome_learner=mnode.CellMeans(),
            pseudo_learner=mnode.CellMeans(),
            treatment_learner=mnode.CellMeans(),
            censoring_learner=mnode.CellMeans(),
            n_folds=n_folds,
            g_bounds=G_BOUNDS,
            simultaneous=False,
            isotonic=True,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        ).fit(
            frame,
            outcome=["Y1", "Y2"],
            treatment=["A1", "A2"],
            baseline=["W"],
            time_varying=[[], ["L2"]],
            censoring=["C1", "C2"],
        )
    targeting = "iterative" if label.endswith("_per_horizon") else "one_step"
    base = label.removesuffix("_per_horizon")
    if base in ("rule_t3", "policy_t3"):
        scenario, regimens = THREE_ARM, THREE_ARM_REGIMENS
    else:
        scenario, regimens = x13.LABELS[base][0], x13.LABELS[base][1]
    weighted = base == "weighted_t5"
    if weighted:
        frame = frame.assign(w=x13.weight(frame["W1"], frame["W2"]))
    data = scenario.container(
        frame,
        weights="w" if weighted else None,
        id="id" if base == "clustered_t5" else None,
    )
    return LTMLE(
        regimens,
        reference="arm0",
        n_folds=n_folds,
        simultaneous=simultaneous,
        survival_model="hazard",
        random_state=0,
        **_settings(targeting),
        **learners(configuration),
    ).fit(data)


def draw(label: str, n: int, seed: int) -> pd.DataFrame:
    if label == "isotonic_t2":
        return mnode_sample(mnode.PROBS, n, seed)
    base = label.removesuffix("_per_horizon")
    if base in ("rule_t3", "policy_t3", "three_arm"):
        return THREE_ARM.draw(n, seed)
    return x13.LABELS[base][0].draw(
        n, seed, cluster_size=CLUSTER_SIZE if base == "clustered_t5" else None
    )


# ------------------------------------------------------------------ the grid

#: Every fit set: ``(family, stream label, configuration, n, replicates, label)``.  The
#: ``three_arm`` calibration set fits once and publishes the ``rule_t3`` and ``policy_t3``
#: cells.
FIT_SETS: tuple[tuple[str, str, str, int, int, str], ...] = (
    *(
        (
            "interval_calibration",
            label,
            "both_correct",
            CALIBRATION_N,
            replicates,
            label,
        )
        for label, replicates in CALIBRATION_REPLICATES.items()
    ),
    *(
        ("clustered_inference", label, "both_correct", CLUSTER_N, CLUSTER_REPLICATES, label)
        for label in CLUSTER_LABELS
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
        A key of :data:`TRUTHS`.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.name = f"point_survival_curve__{label}"

    def truth(self) -> dict[str, float]:
        return dict(TRUTHS[self.label])


class MovedLaw:
    """The zero the isotonic diagnostic records each replication's movement against."""

    name = "point_survival_curve__isotonic_t2_moved"

    def truth(self) -> dict[str, float]:
        return {"moved": 0.0}


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with the law it reads, its estimand, its size and its stream."""
    cells: list[PropertyCell] = []

    def add(
        family: str,
        cell: str,
        role: str,
        label: str,
        n: int,
        reps: int,
        stream: str,
        *,
        law: Any = None,
        estimand: str | None = None,
    ) -> None:
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=law or DeclaredLaw(label),
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=n,
                replicates=reps,
                seed=_seed(family, stream, 0),
                role=role,
                estimand=estimand or ESTIMAND[label],
            )
        )

    for stream, replicates in CALIBRATION_REPLICATES.items():
        labels = ("rule_t3", "policy_t3") if stream == "three_arm" else (stream,)
        for label in labels:
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
                    replicates,
                    stream,
                )
    add(
        "isotonic_projection",
        ISOTONIC_DIAGNOSTIC,
        DIAGNOSTIC_ROLE,
        "isotonic_t2",
        CALIBRATION_N,
        CALIBRATION_REPLICATES["isotonic_t2"],
        "isotonic_t2",
        law=MovedLaw(),
        estimand="moved",
    )
    for label in CLUSTER_LABELS:
        for arm, role in (("cluster_robust", "positive"), ("iid_control", "control")):
            add(
                "clustered_inference",
                f"{label}__{arm}",
                role,
                label,
                CLUSTER_N,
                CLUSTER_REPLICATES,
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
    for label in (*(f"curve_{arm}" for arm in CURVE_ARMS), JOINT_LABEL):
        cells.extend(
            joint_property_cells(
                label,
                n=CALIBRATION_N,
                replicates=CALIBRATION_REPLICATES["survival_t5"],
                seed=_seed("interval_calibration", "survival_t5", 0),
            )
        )
    return tuple(cells)


# ------------------------------------------------------------------ rows


class _Family:
    """One fit's estimates with a band over a subfamily, as :func:`joint_coverage_rows` reads."""

    def __init__(self, result: Any, names: Sequence[str], cluster: Any = None) -> None:
        self._result = result
        self.simultaneous = simultaneous_bands(
            {name: result[name] for name in names},
            alpha=STUDY.margins.alpha,
            cluster=cluster,
            **MULTIPLIER,
        )

    def __getitem__(self, name: str) -> Any:
        return self._result[name]


def _curve_names(arm: str) -> tuple[str, ...]:
    return tuple(common.risk_name(arm, t) for t in study.HORIZONS)


def _moved(result: Any) -> float:
    """How far the projection moved the isotonic cell's level; zero if it did not bind."""
    for regimen, _cause, horizon, before, after in result.isotonic_moves:
        if regimen == "always" and horizon == 2:
            return float(after - before)
    return 0.0


def fit_set_rows(
    payload: tuple[str, str, str, int, int, int, str],
    *,
    n_folds: int = 1,
    seed: Callable[[str, str, int], int] | None = None,
) -> list[dict[str, Any]]:
    """The replicate rows of one fit of one fit set."""
    family, stream, configuration, replicate, n, requested, label = payload
    frame = draw(label, n, (seed or _seed)(family, stream, replicate))
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
        fitted = "rule_t3" if label == "three_arm" else label
        result = fit_label(fitted, frame, simultaneous=joint, n_folds=n_folds)
        if label == "three_arm":
            return [
                row(f"{name}__correctly_specified", "positive", name, result[ESTIMAND[name]])
                for name in ("rule_t3", "policy_t3")
            ]
        rows = [row(f"{label}__correctly_specified", "positive", label, result[ESTIMAND[label]])]
        if label == "isotonic_t2":
            moved = _moved(result)
            rows.append(
                {
                    "property": "isotonic_projection",
                    "cell": ISOTONIC_DIAGNOSTIC,
                    "role": DIAGNOSTIC_ROLE,
                    "replicate": replicate,
                    "n": n,
                    "requested_replicates": requested,
                    "failed_replicates": 0,
                    "truth": 0.0,
                    "estimate": moved,
                    "std_error": np.nan,
                    "covered": int(moved == 0.0),
                    "rejected": int(moved != 0.0),
                }
            )
        if joint:
            truth = TRUTHS["survival_t5"]
            for arm in CURVE_ARMS:
                rows.extend(
                    joint_coverage_rows(
                        _Family(result, _curve_names(arm)),
                        truth,
                        _curve_names(arm),
                        label=f"curve_{arm}",
                        replicate=replicate,
                        n=n,
                        requested=requested,
                        pointwise_critical=CRITICAL,
                    )
                )
            rows.extend(
                joint_coverage_rows(
                    result,
                    truth,
                    tuple(result.estimates),
                    label=JOINT_LABEL,
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    pointwise_critical=CRITICAL,
                )
            )
        return rows
    result = fit_label(label, frame, configuration=configuration, n_folds=n_folds)
    estimate = result[ESTIMAND[label]]
    if family == "clustered_inference":
        curve = np.asarray(estimate.influence_curve, dtype=float)
        iid_se = float(np.sqrt(np.mean(curve**2) / len(curve)))
        return [
            row(f"{label}__cluster_robust", "positive", label, estimate),
            control_row(
                property_name=family,
                cell=f"{label}__iid_control",
                truth=TRUTHS[label][ESTIMAND[label]],
                estimate=float(estimate.psi),
                standard_error=iid_se,
                critical=CRITICAL,
                **common_row,
            ),
        ]
    if family == "double_robustness":
        role = "control" if configuration == "both_wrong" else "positive"
        return [row(f"survival_t5__{configuration}", role, label, estimate)]
    if family == "targeting_necessity":
        return [
            row("survival_t5__targeted", "positive", label, estimate),
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
        return [row(f"survival_t5__n_{n}", role, label, estimate)]
    return [row("survival_t5__alternative", "positive", label, estimate)]


def payloads(
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
    outcomes = map_parallel(fit_set_rows, payloads(FIT_SETS, budget), n_jobs=n_jobs)
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
    return not all(
        np.isfinite(row["estimate"])
        and (row["role"] == DIAGNOSTIC_ROLE or np.isfinite(row["std_error"]))
        for row in rows
    )


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(
            "targeting_displacement",
            "coverage_gain_ci_lower",
            "coverage_gain_ci_upper",
        ),
        rate_labels=("survival_t5",),
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=STUDY.margins, efficiency_band=EFFICIENCY_RATIO_BAND)
    for label in CLUSTER_LABELS:
        clustered_inference_verdicts(
            summary,
            rows,
            STUDY,
            positive_cell=f"{label}__cluster_robust",
            control_cell=f"{label}__iid_control",
        )
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
    # The binding share is reported, not gated: its rejection rate is the share of
    # replications the projection moved, and the row states no verdict.
    diagnostic = summary["property"] == "isotonic_projection"
    summary.loc[diagnostic, "passed"] = True
    summary.loc[diagnostic, "property_passed"] = True
    return finish(summary, rates)
