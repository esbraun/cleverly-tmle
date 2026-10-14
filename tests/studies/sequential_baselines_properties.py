r"""Property families of ``sequential-baselines`` (X41).

Every fit is :func:`tests.studies.canonical_sequential_baselines.fit`: the declared ``GLM``
nuisances of each law, ``g_bounds = (0.01, 1)`` and ``q_bounds`` at the outcome's support.
Each draw is shared by every estimator and every cell of its fit set, so the estimators are
paired within a law.

=============================  ===================================================================
family                         cells, replications and size
=============================  ===================================================================
``interval_calibration``       the 43 labels of :data:`~tests.studies.canonical_sequential_
                               baselines.CALIBRATION_LABELS` at n = 2,000 (``D2C`` at 8,000), with
                               :data:`CALIBRATION_REPLICATES` each, and the two derived controls of
                               each label (a shrunken standard error and added noise)
``root_n_and_efficiency``      the six ``*_contrast`` labels of ``D2`` and ``G`` at n = 2,000,
                               4,000 and 8,000, :data:`LADDER_REPLICATES` each; every rung is
                               a positive cell (a parametric fit is root-n at each size)
``root_n_rate``                the rate rows of each ladder label
``misspecification_control``   K1: ``treat_all vs treat_none`` on ``X`` for each estimator, n = 2,000,
                               :data:`CALIBRATION_REPLICATES`; it publishes the positive
                               calibration rule, and red is the declared outcome
``fixed_mechanism_se``         K2: on the ``G`` and ``D2`` calibration draws, the IPW standard
                               error with a fixed mechanism (no mechanism term), for Hajek and HT;
                               on ``D2`` the targets ``static_12``, ``policy`` and ``rule``
``first_node_se``              K3: on the ``G`` draws, the ICE standard error of
                               :math:`w (\bar Q_1 - \psi)` alone
``adaptive_learners``          A: ``treat_all vs treat_none`` on ``X`` with the default learner library,
                               ICE and IPW-Hajek on the same draws, at n = 1,000,
                               :data:`ADAPTIVE_REPLICATES`; points only
=============================  ===================================================================

**Rules.**  ``interval_calibration`` reads the shared calibration rule: the 99% bootstrap
interval of the SE ratio inside (0.93, 1.07) and the 99% coverage interval inside (0.92, 0.98).
A positive cell also needs its 99% standardized-bias interval inside 0.25 SD
(``bias_equivalent``).  The shared rule reads no bias, and a 0.3 SD bias keeps coverage inside
its band at this R.  There is no efficiency band: a parametric plug-in is not the efficient
estimator of the nonparametric model, so :attr:`StudyRecord.calibration_efficiency_ratio` is
``False``, and the noise control's scale is :data:`NOISE_SD`, one pilot sampling SD at the
cell's size.  ``root_n_and_efficiency`` reads the shared ladder rule.
``misspecification_control`` publishes the positive rule: it is in ``EXPECTED_RED``, and the
design test shows the failure is powered.

``fixed_mechanism_se`` and ``first_node_se`` carry a predicted direction: the fixed-mechanism
ratio above 1 (Lunceford and Davidian 2004, Section 3.1) and the first-node ratio below 1.  A
cell in :data:`GATED_DIRECTIONS` passes when its whole SE-ratio interval leaves the
calibration band on the predicted side.  That reading shows the calibration instrument would
catch a dropped mechanism term (K2) or a dropped chain (K3).  The design test gates a cell
only where its predictive power is at least 0.95, and it lists every such cell; the others
report.  K3 is a property of the law, not of the estimator, so it cannot fail for a broken
estimator.  On ``D2`` Hajek's mechanism term moves its SE by about 1%, so only HT's ``D2`` cells
carry the multinomial mechanism evidence.  ``adaptive_learners`` publishes no verdict.

Every fit's inference status is checked before its row is written (``fit_set_rows``).

**Failures.**  A replicate whose fit raises a recordable error is recorded by
:func:`~tests.studies.evidence.replication.fit_replicates` and never redrawn.  The shared
property summary refuses a cell that lost a replicate, so the declared 0.01 bound
(:data:`FAILURE_BOUND`) is not yet operative here.  The declared route: a recordable
``DataError`` in the property phase stops it with its records.  The phase then reruns unchanged
(laws, n, R and seeds) once RM39b2 makes the 0.01 bound operative, with the failures accounted
under its rule.  No size, law or seed moves.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import norm

from cleverly.learners import SuperLearner
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_sequential_baselines as study
from tests.studies import sequential_baselines_laws as laws
from tests.studies.evidence import replication
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    PropertyCell,
    control_row,
    ratio_intervals,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    se_ratio_leaves_band,
)
from tests.studies.evidence.seeds import stream_seed

STUDY = study.STUDY
FAILURE_BOUND = study.FAILURE_BOUND
CALIBRATION_N = study.CALIBRATION_N
LADDER = study.LADDER
#: The replicates of every calibration cell and of K1, sized by the arm-pooled predictive rule
#: in ``tests/unit/test_sequential_baselines_design.py``.
CALIBRATION_REPLICATES = 8_800
LADDER_REPLICATES = 1_000
ADAPTIVE_REPLICATES = 100
ADAPTIVE_SIZES = study.ADAPTIVE_SIZES
SHRUNKEN_SE_FACTOR = 0.70
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))
ESTIMATORS = study.ESTIMATORS

#: The empirical SD of each calibration label in the design pilot times
#: :math:`\sqrt{2000}` (``CALIBRATION_N``), so the noise control adds one sampling SD at the
#: cell's own size (``D2C`` too, at 8,000).  It scales the noise control only.  The design test
#: pins these values.
NOISE_SD: dict[str, float] = {
    "hajek_d2_contrast": 1.9551,
    "hajek_d2_policy": 0.8512,
    "hajek_d2_rule": 1.3488,
    "hajek_d2_static_00": 1.4718,
    "hajek_d2_static_12": 1.2719,
    "hajek_d2c_relapse_h2": 0.5873,
    "hajek_d2s_contrast_h2": 1.7783,
    "hajek_d2s_risk_h1": 0.5524,
    "hajek_d2s_risk_h2": 1.1889,
    "hajek_d2w_static_12": 1.2754,
    "hajek_g_contrast": 3.4405,
    "hajek_g_rule": 3.7606,
    "hajek_g_treat_all": 2.6414,
    "hajek_g_treat_none": 2.6979,
    "ht_d2_contrast": 2.0162,
    "ht_d2_policy": 0.8448,
    "ht_d2_rule": 1.6249,
    "ht_d2_static_00": 1.5426,
    "ht_d2_static_12": 1.3095,
    "ht_d2c_relapse_h2": 0.5993,
    "ht_d2s_contrast_h2": 1.8326,
    "ht_d2s_risk_h1": 0.5527,
    "ht_d2s_risk_h2": 1.2279,
    "ht_g_contrast": 4.1558,
    "ht_g_rule": 3.9228,
    "ht_g_treat_all": 2.6486,
    "ht_g_treat_none": 3.5863,
    "ice_d2_contrast": 1.9432,
    "ice_d2_policy": 0.8479,
    "ice_d2_rule": 1.3441,
    "ice_d2_static_00": 1.4568,
    "ice_d2_static_12": 1.2677,
    "ice_d2c_relapse_h2": 0.5887,
    "ice_d2s_contrast_h2": 1.803,
    "ice_d2s_risk_h1": 0.5532,
    "ice_d2s_risk_h2": 1.1879,
    "ice_d2w_static_12": 1.2724,
    "ice_g_contrast": 2.7709,
    "ice_g_rule": 2.1091,
    "ice_g_treat_all": 2.063,
    "ice_g_treat_none": 2.0801,
    "ice_gm_contrast": 0.7108,
    "ice_gm_down": 1.268,
}
#: Every K2 and K3 cell whose predicted side of the calibration band the design test shows
#: powered (predictive power >= 0.95 at :data:`CALIBRATION_REPLICATES`).  The other K2 cells
#: report: Hajek on ``D2`` and on ``G``'s levels, and HT's ``treat_all`` on ``G``.
GATED_DIRECTIONS: dict[str, Literal["below", "above"]] = {
    "hajek_g_contrast__fixed_mechanism": "above",
    "ht_g_treat_none__fixed_mechanism": "above",
    "ht_g_contrast__fixed_mechanism": "above",
    "ht_d2_static_12__fixed_mechanism": "above",
    "ht_d2_policy__fixed_mechanism": "above",
    "ht_d2_rule__fixed_mechanism": "above",
    **{f"ice_g_{key}__first_node": "below" for key in study.G_REPORTED},
}

# ------------------------------------------------------------------------------ the truths


def _d2_names() -> dict[str, tuple[str, float]]:
    truth = study.d2_truths()
    names = {
        "static_12": "ey_regimen[static_12]",
        "static_00": "ey_regimen[static_00]",
        "rule": "ey_regimen[rule]",
        "policy": "ey_regimen[policy]",
        "contrast": "ate_regimen[static_12 vs static_00]",
    }
    return {key: (name, truth[name]) for key, name in names.items()}


def _g_names(law: laws.ContinuousLaw) -> dict[str, tuple[str, float]]:
    truth = study.g_estimand_truths(law)
    names = {
        "treat_all": "ey_regimen[treat_all]",
        "treat_none": "ey_regimen[treat_none]",
        "rule": "ey_regimen[rule]",
        "contrast": "ate_regimen[treat_all vs treat_none]",
    }
    return {key: (name, truth[name]) for key, name in names.items()}


def _d2s_names() -> dict[str, tuple[str, float]]:
    plans = laws.D2_PLANS
    h1 = laws.D2S.truth(plans["static_12"], horizon=1)
    h2 = laws.D2S.truth(plans["static_12"], horizon=2)
    h2_00 = laws.D2S.truth(plans["static_00"], horizon=2)
    return {
        "risk_h1": ("risk_regimen[static_12 @ t=1]", h1),
        "risk_h2": ("risk_regimen[static_12 @ t=2]", h2),
        "contrast_h2": ("ate_regimen[static_12 vs static_00 @ t=2]", h2 - h2_00),
    }


def _d2c_names() -> dict[str, tuple[str, float]]:
    value = laws.D2C.truth(laws.D2_PLANS["static_12"], horizon=2, cause=1)
    return {"relapse_h2": ("cif_regimen[static_12, relapse @ t=2]", value)}


def _gm_names() -> dict[str, tuple[str, float]]:
    truth = laws.gm_truths()
    return {
        "down": ("ey_regimen[down]", truth["down"]),
        "contrast": ("ate_regimen[down vs natural]", truth["contrast"]),
    }


def _d2w_names() -> dict[str, tuple[str, float]]:
    value = laws.D2.truth(laws.D2_PLANS["static_12"], weight=laws.observation_weight)
    return {"static_12": ("ey_regimen[static_12]", value)}


#: ``label -> (estimand, truth)`` for every calibration label.
TARGETS: dict[str, tuple[str, float]] = {}
for _estimator in ESTIMATORS:
    for _prefix, _names in (
        ("d2", _d2_names()),
        ("g", _g_names(laws.G)),
        ("d2s", _d2s_names()),
        ("d2c", _d2c_names()),
        ("gm", _gm_names()),
        ("d2w", _d2w_names()),
        ("x", _g_names(laws.X)),
    ):
        for _key, _target in _names.items():
            TARGETS[f"{_estimator}_{_prefix}_{_key}"] = _target

# ------------------------------------------------------------------------------- fit sets

#: Every fit set: ``name -> (law, n, replicates)``.  One draw of the law per replicate.
FIT_SETS: dict[str, tuple[str, int, int]] = {
    "calibration_d2": ("D2", CALIBRATION_N, CALIBRATION_REPLICATES),
    "calibration_g": ("G", CALIBRATION_N, CALIBRATION_REPLICATES),
    "calibration_d2s": ("D2S", CALIBRATION_N, CALIBRATION_REPLICATES),
    "calibration_d2c": ("D2C", study.D2C_N, CALIBRATION_REPLICATES),
    "calibration_gm": ("GM", CALIBRATION_N, CALIBRATION_REPLICATES),
    "misspecification_x": ("X", CALIBRATION_N, CALIBRATION_REPLICATES),
    **{f"ladder_d2_n_{size}": ("D2", size, LADDER_REPLICATES) for size in LADDER},
    **{f"ladder_g_n_{size}": ("G", size, LADDER_REPLICATES) for size in LADDER},
    **{f"adaptive_x_n_{size}": ("X", size, ADAPTIVE_REPLICATES) for size in ADAPTIVE_SIZES},
}


def sample_seed(fit_set: str, replicate: int) -> int:
    """The seed of one draw: this study's own ``property_sample`` stream."""
    return stream_seed(STUDY, "property_sample", fit_set, replicate)


def draw(law: str, n: int, seed: int) -> pd.DataFrame:
    """One draw of ``law``; the ``D2`` draws carry the known weight column ``w``."""
    if law in laws.DISCRETE_LAWS:
        frame = laws.DISCRETE_LAWS[law].sample(n, seed)
        frame["w"] = laws.observation_weight(frame["W"])
        return frame
    if law == "GM":
        return laws.gm_sample(n, seed)
    return (laws.G if law == "G" else laws.X).sample(n, seed)


def _scaled_first_node(result: Any, label: str) -> np.ndarray:
    """The first node's carried prediction of one plan, on the outcome's scale."""
    first = result.fits[label].steps[0]
    values = first.initial if first.marginal is None else first.marginal
    return np.asarray(result.scaler.unscale_levels(np.asarray(values, dtype=float)), dtype=float)


def first_node_standard_error(result: Any, key: str) -> float:
    r"""K3: the standard error of :math:`w (\bar Q_1 - \psi)` alone, for one G target."""
    plans = {
        "treat_all": ("treat_all",),
        "treat_none": ("treat_none",),
        "contrast": ("treat_all", "treat_none"),
    }[key]
    values = _scaled_first_node(result, plans[0])
    if len(plans) == 2:
        values = values - _scaled_first_node(result, plans[1])
    weights = np.asarray(result.data.weights, dtype=float)
    curve = weights * (values - np.average(values, weights=weights))
    return float(np.sqrt(np.mean(curve**2) / len(curve)))


def _row(
    family: str,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    truth: float,
    est: Any,
) -> dict[str, Any]:
    return replicate_row(
        property_name=family,
        cell=cell,
        role=role,
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=est,
        alpha=STUDY.margins.alpha,
    )


def _target_names(estimator: str, prefix: str) -> list[str]:
    return [
        TARGETS[label][0]
        for label in study.CALIBRATION_LABELS
        if label.startswith(f"{estimator}_{prefix}_")
    ]


def fit_set_rows(fit_set: str, replicate: int, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Every row one draw of one fit set publishes.

    Each fit's inference status is checked before a row is written
    (:func:`~tests.studies.canonical_sequential_baselines.require_status`): ``influence_curve``
    on every calibration, ladder and K1 fit, and ``adaptive_baseline_plugin`` on the K2 and A
    fits.  A mismatch stops the run rather than calibrating the wrong curve.
    """
    law, n, requested = FIT_SETS[fit_set]
    rows: list[dict[str, Any]] = []
    calibration = "interval_calibration"

    def add(family: str, cell: str, role: str, label: str, result: Any) -> None:
        name, truth = TARGETS[label]
        rows.append(_row(family, cell, role, replicate, n, requested, truth, result[name]))

    if fit_set.startswith("ladder_"):
        prefix = fit_set.split("_")[1]
        for estimator in ESTIMATORS:
            result = study.fit(estimator, law, frame)
            label = f"{estimator}_{prefix}_contrast"
            study.require_status(result, [TARGETS[label][0]], study.INFERENTIAL)
            add("root_n_and_efficiency", f"{label}__n_{n}", "positive", label, result)
        return rows
    if fit_set.startswith("adaptive_"):
        name, truth = TARGETS["ice_x_contrast"]
        fits = {
            "ice": study.fit("ice", "X", frame, outcome=SuperLearner(random_state=0)),
            "hajek": study.fit("hajek", "X", frame, mechanism=SuperLearner(random_state=0)),
        }
        for estimator, result in fits.items():
            study.require_status(result, [name], study.DIAGNOSTIC)
            rows.append(
                _row(
                    "adaptive_learners",
                    f"{estimator}_x_contrast__n_{n}",
                    DIAGNOSTIC_ROLE,
                    replicate,
                    n,
                    requested,
                    truth,
                    result[name],
                )
            )
        return rows
    if fit_set == "misspecification_x":
        for estimator in ESTIMATORS:
            result = study.fit(estimator, "X", frame)
            label = f"{estimator}_x_contrast"
            study.require_status(result, [TARGETS[label][0]], study.INFERENTIAL)
            add("misspecification_control", f"{label}__main_terms", "control", label, result)
        return rows
    prefix = {"D2": "d2", "G": "g", "D2S": "d2s", "D2C": "d2c", "GM": "gm"}[law]
    estimators = ("ice",) if law == "GM" else ESTIMATORS
    for estimator in estimators:
        result = study.fit(estimator, law, frame)
        study.require_status(result, _target_names(estimator, prefix), study.INFERENTIAL)
        for label in study.CALIBRATION_LABELS:
            if label.startswith(f"{estimator}_{prefix}_"):
                add(calibration, f"{label}__correctly_specified", "positive", label, result)
        if law == "G" and estimator == "ice":
            for key in study.G_REPORTED:
                name, truth = TARGETS[f"ice_g_{key}"]
                rows.append(
                    control_row(
                        property_name="first_node_se",
                        cell=f"ice_g_{key}__first_node",
                        replicate=replicate,
                        n=n,
                        requested=requested,
                        truth=truth,
                        estimate=float(result[name].psi),
                        standard_error=first_node_standard_error(result, key),
                        critical=CRITICAL,
                        role=DIAGNOSTIC_ROLE,
                    )
                )
        if law in study.K2_TARGETS and estimator != "ice":
            fixed = study.fit(estimator, law, frame, mechanism=study.FixedMechanismGLM())
            keys = study.K2_TARGETS[law]
            names = [TARGETS[f"{estimator}_{prefix}_{key}"][0] for key in keys]
            study.require_status(fixed, names, study.DIAGNOSTIC)
            for key in keys:
                name, truth = TARGETS[f"{estimator}_{prefix}_{key}"]
                rows.append(
                    _row(
                        "fixed_mechanism_se",
                        f"{estimator}_{prefix}_{key}__fixed_mechanism",
                        DIAGNOSTIC_ROLE,
                        replicate,
                        n,
                        requested,
                        truth,
                        fixed[name],
                    )
                )
        if law == "D2" and estimator in ("ice", "hajek"):
            weighted = study.fit(estimator, law, frame, weights="w")
            label = f"{estimator}_d2w_static_12"
            study.require_status(weighted, [TARGETS[label][0]], study.INFERENTIAL)
            add(calibration, f"{label}__correctly_specified", "positive", label, weighted)
    return rows


# ------------------------------------------------------------------------------ the grid


def declared_cells() -> tuple[PropertyCell, ...]:
    """Every sampled cell, with its law, estimand, size and stream root."""
    cells: list[PropertyCell] = []

    def add(family: str, cell: str, role: str, fit_set: str, label: str) -> None:
        law, n, replicates = FIT_SETS[fit_set]
        cells.append(
            PropertyCell(
                property=family,
                cell=cell,
                dgp=law,
                outcome_learner=lambda: None,
                treatment_learner=lambda: None,
                n=n,
                replicates=replicates,
                seed=sample_seed(fit_set, 0),
                role=role,
                estimand=TARGETS[label][0],
            )
        )

    sets = {"d2": "calibration_d2", "g": "calibration_g", "d2s": "calibration_d2s"}
    sets |= {"d2c": "calibration_d2c", "gm": "calibration_gm", "d2w": "calibration_d2"}
    for label in study.CALIBRATION_LABELS:
        prefix = label.split("_")[1]
        add(
            "interval_calibration", f"{label}__correctly_specified", "positive", sets[prefix], label
        )
    for label in study.LADDER_LABELS:
        prefix = label.split("_")[1]
        for size in LADDER:
            role = "positive"
            add(
                "root_n_and_efficiency",
                f"{label}__n_{size}",
                role,
                f"ladder_{prefix}_n_{size}",
                label,
            )
    for estimator in ESTIMATORS:
        label = f"{estimator}_x_contrast"
        add(
            "misspecification_control",
            f"{label}__main_terms",
            "control",
            "misspecification_x",
            label,
        )
    for k2_law, keys in study.K2_TARGETS.items():
        k2_prefix = k2_law.lower()
        for estimator in ("hajek", "ht"):
            for key in keys:
                label = f"{estimator}_{k2_prefix}_{key}"
                add(
                    "fixed_mechanism_se",
                    f"{label}__fixed_mechanism",
                    DIAGNOSTIC_ROLE,
                    f"calibration_{k2_prefix}",
                    label,
                )
    for key in study.G_REPORTED:
        add(
            "first_node_se",
            f"ice_g_{key}__first_node",
            DIAGNOSTIC_ROLE,
            "calibration_g",
            f"ice_g_{key}",
        )
    for estimator in ("ice", "hajek"):
        for size in ADAPTIVE_SIZES:
            add(
                "adaptive_learners",
                f"{estimator}_x_contrast__n_{size}",
                DIAGNOSTIC_ROLE,
                f"adaptive_x_n_{size}",
                "ice_x_contrast",
            )
    return tuple(cells)


# ------------------------------------------------------------------------------- the loop


def _draw_key(key: tuple[str, int]) -> pd.DataFrame:
    fit_set, replicate = key
    law, n, _ = FIT_SETS[fit_set]
    return draw(law, n, sample_seed(fit_set, replicate))


def _fit_key(key: tuple[str, int], frame: pd.DataFrame) -> list[dict[str, Any]]:
    return fit_set_rows(key[0], key[1], frame)


def run_fit_set(fit_set: str, *, n_jobs: int, replicates: int | None = None) -> pd.DataFrame:
    """Every row of one fit set; ``replicates`` caps the budget for a smoke or a pilot.

    A recordable failure is recorded and never redrawn.  The shared summary refuses a cell
    that lost a replicate, so a failure stops the phase here with its records.
    """
    _, _, declared = FIT_SETS[fit_set]
    count = declared if replicates is None else replicates
    outcomes = replication.fit_replicates(
        [(fit_set, replicate) for replicate in range(count)],
        _fit_key,
        draw=_draw_key,
        n_jobs=n_jobs,
        label=f"properties_{fit_set}",
        implementation=study.IMPLEMENTATION,
        cell_of=lambda key: key[0],
        replicate_of=lambda key: key[1],
        seed_of=lambda key: sample_seed(key[0], key[1]),
    )
    if outcomes.failed:
        raise RuntimeError(
            f"{fit_set}: {len(outcomes.failed)} of {count} replicates failed (declared bound "
            f"{FAILURE_BOUND} per cell); records:\n{outcomes.failures.to_string(index=False)}"
        )
    rows = pd.DataFrame([row for rows in outcomes.values for row in rows])
    if replicates is not None:
        rows["requested_replicates"] = count
    return rows


def generate_property_rows(
    *, n_jobs: int = STUDY_JOBS, budget: int | None = None, fit_sets: Any = None
) -> pd.DataFrame:
    """Fit every property replicate; ``budget`` caps each fit set for a smoke or a pilot."""
    names = list(FIT_SETS) if fit_sets is None else list(fit_sets)
    rows = pd.concat(
        [run_fit_set(name, n_jobs=n_jobs, replicates=budget) for name in names],
        ignore_index=True,
    )
    labels = [
        label
        for label in study.CALIBRATION_LABELS
        if f"{label}__correctly_specified" in set(rows["cell"])
    ]
    if labels:
        rows = pd.concat(
            [
                rows,
                calibration_controls(
                    rows,
                    STUDY,
                    labels=labels,
                    efficiency_bounds=NOISE_SD,
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


# ---------------------------------------------------------------------------- verdicts

#: The families with a study-specific rule.
OWN_FAMILIES = ("misspecification_control", "fixed_mechanism_se", "first_node_se")


def _own_verdicts(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    margins = STUDY.margins
    for index in summary.index[summary["property"].isin(OWN_FAMILIES)]:
        family = str(summary.loc[index, "property"])
        cell = str(summary.loc[index, "cell"])
        group = rows.loc[(rows["property"] == family) & (rows["cell"] == cell)]
        ratio = ratio_intervals(
            group,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=stream_seed(STUDY, family, cell),
        )["se_ratio"]
        summary.loc[index, "se_ratio_ci_lower"] = ratio.low
        summary.loc[index, "se_ratio_ci_upper"] = ratio.high
        if family == "misspecification_control":
            coverage = (
                float(summary.loc[index, "coverage_ci_lower"]),
                float(summary.loc[index, "coverage_ci_upper"]),
            )
            summary.loc[index, "passed"] = bool(
                ratio.within(*margins.calibration_se_ratio)
                and margins.calibration_coverage[0] <= coverage[0]
                and coverage[1] <= margins.calibration_coverage[1]
            )
            continue
        side = GATED_DIRECTIONS.get(cell)
        # A gated cell must leave the calibration band on its predicted side, so it shows that
        # the calibration instrument would catch the dropped term.  Other cells report only.
        summary.loc[index, "passed"] = (
            True
            if side is None
            else se_ratio_leaves_band(ratio, margins.calibration_se_ratio, side)
        )
    reported = summary["property"] == "adaptive_learners"
    summary.loc[reported, "passed"] = True


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    """The shared verdicts and the study's own declared rules."""
    labels = tuple(
        label for label in study.LADDER_LABELS if (rows["cell"] == f"{label}__n_{LADDER[0]}").any()
    )
    summary, rates = apply_shared_verdicts(rows, STUDY, rate_labels=labels)
    calibration_verdicts(summary, margins=STUDY.margins)
    # The bias leg (design review B2): the shared calibration rule reads the SE ratio and
    # coverage only, and a bias of 0.3 SD keeps coverage inside its band at this R.  Every
    # positive calibration cell also needs its 99% bias interval inside 0.25 SD.
    positive = (summary["property"] == "interval_calibration") & summary["cell"].str.endswith(
        "__correctly_specified"
    )
    summary.loc[positive, "passed"] = summary.loc[positive, "passed"].astype(bool) & summary.loc[
        positive, "bias_equivalent"
    ].astype(bool)
    _own_verdicts(summary, rows)
    return finish(summary, rates)


def cell_label(cell: str) -> str:
    """The label of a cell name, before ``__``."""
    return cell.split("__", 1)[0]


__all__ = [
    "FIT_SETS",
    "declared_cells",
    "generate_property_rows",
    "summarize_properties",
]
