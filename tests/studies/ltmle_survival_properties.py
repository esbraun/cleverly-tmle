"""Independent repeated-sampling properties for ordinary survival LTMLE."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor

from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_survival as law
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_ltmle import G_BOUNDS
from tests.studies.canonical_ltmle_survival import PROPERTY_LABELS, STUDY
from tests.studies.canonical_properties import apply_shared_verdicts, finish
from tests.studies.evidence.inference import Interval
from tests.studies.evidence.properties import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import stream_seed

DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
RATE_REPLICATES = 800
RATE_SIZES = (1_000, 2_000, 8_000)
#: The horizon-two SE-ratio interval is the binding calibration endpoint.  A 2,400-draw
#: pilot put its lower endpoint near 0.92 around a point ratio near 0.95.  Four times that
#: budget reduces the interval width by half and resolves the predeclared 0.93 boundary
#: without changing the boundary after seeing the study.
CALIBRATION_REPLICATES = 9_600
CALIBRATION_N = 2_000
NULL_REPLICATES = 800
NULL_N = 4_000
TARGETING_REPLICATES = DOUBLE_ROBUST_REPLICATES
TARGETING_N = DOUBLE_ROBUST_N
RECURSION_REPLICATES = DOUBLE_ROBUST_REPLICATES
RECURSION_N = DOUBLE_ROBUST_N

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
TARGETING_DISPLACEMENT = 0.25
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

REGIMENS = law.REGIMEN_SPEC
REFERENCE = law.REGIMEN_REFERENCE
CONTRASTS = {
    "static_t1": "ate_regimen[always vs never @ t=1]",
    "static_t2": "ate_regimen[always vs never @ t=2]",
    "dynamic_t2": "ate_regimen[continue_if_l2 vs never @ t=2]",
}
EFFICIENCY_SD = {
    label: float(np.sqrt(np.sum(law.PROBS * law.eif(name) ** 2)))
    for label, name in CONTRASTS.items()
}

# A null that still needs longitudinal adjustment.  The first hazard has opposing
# conditional treatment effects whose marginal effects cancel.  The second hazard varies
# with L2, while every regimen-specific L2 average is one half.  All three contrasts are
# therefore exactly zero without making either hazard a baseline-only constant.
NULL_H1 = np.array([[0.25, 0.50], [0.75, 0.50]])
NULL_H2 = np.array(law.H2, copy=True)
NULL_H2[0, 0, 0, 0], NULL_H2[0, 0, 1, 0] = 0.40, 0.80
NULL_H2[1, 0, 0, 0], NULL_H2[1, 0, 1, 0] = 0.25, 0.75
for w in (0, 1):
    NULL_H2[w, 1, 0, 0] = 0.20
    NULL_H2[w, 1, 0, 1] = 0.20
    NULL_H2[w, 1, 1, 1] = 0.60

# The original horizon-two static effect is intentionally small.  A power control must
# reject reliably, so use a predeclared alternative with low treated and high untreated
# second-node hazards.  The horizon-one alternative remains the original law.
POWER_H2 = np.array(law.H2, copy=True)
POWER_H2[:, 0, :, 0] = 0.80
POWER_H2[:, 1, :, 1] = 0.15
POWER_H2[:, 1, 0, 0] = 0.15


def probabilities(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    """Observable support probabilities under replacement survival hazards."""
    masses: list[float] = []
    for point in law.SUPPORT:
        w, a1, c1, y1, l2, a2, c2, y2 = point
        mass = law.P_W[w] * (law.G1[w] if a1 == 1 else 1.0 - law.G1[w])
        if c1 == 0:
            masses.append(float(mass * (1.0 - law.C1[w, a1])))
            continue
        mass *= law.C1[w, a1]
        mass *= h1[w, a1] if y1 == 1 else 1.0 - h1[w, a1]
        if y1 == 1:
            masses.append(float(mass))
            continue
        mass *= law.P_L2[w, a1] if l2 == 1 else 1.0 - law.P_L2[w, a1]
        mass *= law.G2[w, a1, l2] if a2 == 1 else 1.0 - law.G2[w, a1, l2]
        if c2 == 0:
            masses.append(float(mass * (1.0 - law.C2[w, a1, l2, a2])))
            continue
        mass *= law.C2[w, a1, l2, a2]
        mass *= h2[w, a1, l2, a2] if y2 == 1 else 1.0 - h2[w, a1, l2, a2]
        masses.append(float(mass))
    out = np.asarray(masses, dtype=float)
    return out / out.sum()


NULL_PROBS = probabilities(NULL_H1, NULL_H2)
POWER_PROBS = probabilities(law.H1, POWER_H2)
NULL_TRUTH = {label: float(law.functional(NULL_PROBS, name)) for label, name in CONTRASTS.items()}
POWER_TRUTH = {label: float(law.functional(POWER_PROBS, name)) for label, name in CONTRASTS.items()}


def sample(probs: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = rng.choice(len(law.SUPPORT), size=n, p=probs)
    names = ("W", "A1", "C1", "Y1", "L2", "A2", "C2", "Y2")
    return pd.DataFrame(
        {
            name: np.array(
                [
                    np.nan if point[position] is None else float(point[position])
                    for point in law.SUPPORT
                ]
            )[cells]
            for position, name in enumerate(names)
        }
    )


def _learners(configuration: str) -> tuple[Any, Any, Any, Any]:
    q_correct = configuration in {"both_correct", "outcome_correct"}
    g_correct = configuration in {"both_correct", "mechanism_correct"}
    return (
        law.CellMeans() if q_correct else DummyClassifier(strategy="prior"),
        law.CellMeans() if q_correct else DummyRegressor(strategy="mean"),
        law.CellMeans() if g_correct else DummyClassifier(strategy="prior"),
        law.CellMeans() if g_correct else DummyClassifier(strategy="prior"),
    )


def fit(frame: pd.DataFrame, configuration: str = "both_correct") -> Any:
    outcome, pseudo, treatment, censoring = _learners(configuration)
    return LTMLE(
        REGIMENS,
        reference=REFERENCE,
        outcome_learner=outcome,
        pseudo_learner=pseudo,
        treatment_learner=treatment,
        censoring_learner=censoring,
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
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


def _plan_arms(frame: pd.DataFrame, label: str) -> tuple[np.ndarray, np.ndarray]:
    node1, node2 = law.REGIMEN_ARMS[label]
    w = frame["W"].to_numpy().astype(int)
    l2 = np.nan_to_num(frame["L2"].to_numpy()).astype(int)
    first = np.full(len(frame), float(node1)) if np.ndim(node1) == 0 else np.asarray(node1)[w]
    second = np.full(len(frame), float(node2)) if np.ndim(node2) == 0 else np.asarray(node2)[w, l2]
    return np.asarray(first), np.asarray(second)


def untargeted(frame: pd.DataFrame, label: str, horizon: int, configuration: str) -> float:
    """Survival recursion using the same nuisance learners and no fluctuation."""
    outcome, pseudo, _, _ = _learners(configuration)
    first, second = _plan_arms(frame, label)
    y1 = np.nan_to_num(frame["Y1"].to_numpy())
    followed_one = (frame["C1"].to_numpy() == 1.0) & (frame["A1"].to_numpy() == first)
    baseline = frame[["W"]].to_numpy(dtype=float)
    if horizon == 1:
        model = clone(outcome).fit(baseline[followed_one], y1[followed_one])
        return float(np.mean(model.predict_proba(baseline)[:, 1]))

    history = np.column_stack([baseline[:, 0], np.nan_to_num(frame["L2"].to_numpy())])
    followed_two = (
        followed_one
        & (y1 == 0.0)
        & (frame["C2"].to_numpy() == 1.0)
        & (frame["A2"].to_numpy() == second)
    )
    later = clone(outcome).fit(history[followed_two], frame["Y2"].to_numpy()[followed_two])
    carried = np.asarray(later.predict_proba(history))[:, 1]
    pseudo_outcome = y1 + (1.0 - y1) * carried
    earlier = clone(pseudo).fit(baseline[followed_one], pseudo_outcome[followed_one])
    return float(np.mean(earlier.predict(baseline)))


def survivor_only(frame: pd.DataFrame) -> float:
    """End-of-study analysis among those who did not fail at the first node."""
    kept = frame[(frame["C1"] != 1.0) | (frame["Y1"] == 0.0)].reset_index(drop=True)
    result = LTMLE(
        {"always": 1},
        reference="always",
        outcome_learner=law.CellMeans(),
        pseudo_learner=law.CellMeans(),
        treatment_learner=law.CellMeans(),
        censoring_learner=law.CellMeans(),
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
    ).fit(
        kept,
        outcome="Y2",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )
    return float(result.psi("ey_regimen[always]"))


def _row(
    *,
    property_name: str,
    cell: str,
    role: str,
    replicate: int,
    n: int,
    requested: int,
    truth: float,
    estimate: Any,
) -> dict[str, Any]:
    low, high = estimate.ci
    return {
        "property": property_name,
        "cell": cell,
        "role": role,
        "replicate": replicate,
        "n": n,
        "requested_replicates": requested,
        "failed_replicates": 0,
        "truth": truth,
        "estimate": float(estimate.psi),
        "std_error": float(estimate.std_error),
        "covered": int(low <= truth <= high),
        "rejected": int(estimate.pvalue < STUDY.margins.alpha),
    }


def _control_row(
    *,
    property_name: str,
    cell: str,
    replicate: int,
    n: int,
    requested: int,
    truth: float,
    estimate: float,
    standard_error: float,
) -> dict[str, Any]:
    half = CRITICAL * standard_error
    return {
        "property": property_name,
        "cell": cell,
        "role": "control",
        "replicate": replicate,
        "n": n,
        "requested_replicates": requested,
        "failed_replicates": 0,
        "truth": truth,
        "estimate": estimate,
        "std_error": standard_error,
        "covered": int(estimate - half <= truth <= estimate + half),
        "rejected": int(abs(estimate / standard_error) > CRITICAL),
    }


def _fit_replication(payload: tuple[str, str, int, int, int, int, str]) -> list[dict[str, Any]]:
    property_name, suffix, replicate, n, requested, seed, configuration = payload
    probs = (
        NULL_PROBS
        if property_name == "type_i_error"
        else POWER_PROBS
        if property_name == "power"
        else law.PROBS
    )
    frame = sample(probs, n, seed)
    result = fit(frame, configuration)
    rows: list[dict[str, Any]] = []
    for label, name in CONTRASTS.items():
        truth = (
            NULL_TRUTH[label]
            if property_name == "type_i_error"
            else POWER_TRUTH[label]
            if property_name == "power"
            else float(law.TRUTH[name])
        )
        role = (
            "control"
            if suffix == "both_wrong"
            or (property_name == "root_n_and_efficiency" and n == min(RATE_SIZES))
            else "positive"
        )
        rows.append(
            _row(
                property_name=property_name,
                cell=f"{label}__{suffix}",
                role=role,
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=result[name],
            )
        )
        if property_name == "targeting_necessity":
            left, right = name[len("ate_regimen[") : -1].rsplit(" @ t=", 1)[0].split(" vs ")
            horizon = int(name.rsplit(" @ t=", 1)[1][:-1])
            unfluctuated = untargeted(frame, left, horizon, configuration) - untargeted(
                frame, right, horizon, configuration
            )
            rows.append(
                _control_row(
                    property_name=property_name,
                    cell=f"{label}__untargeted",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=truth,
                    estimate=unfluctuated,
                    standard_error=float(result[name].std_error),
                )
            )
    return rows


def _recursion_replication(payload: tuple[int, int, int, int]) -> list[dict[str, Any]]:
    replicate, n, requested, seed = payload
    frame = sample(law.PROBS, n, seed)
    result = fit(frame, "both_correct")
    name = "risk_regimen[always @ t=2]"
    estimate = result[name]
    truth = float(law.TRUTH[name])
    positive = _row(
        property_name="survival_recursion_necessity",
        cell="always_t2__survival",
        role="positive",
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=estimate,
    )
    control = _control_row(
        property_name="survival_recursion_necessity",
        cell="always_t2__survivor_only",
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=survivor_only(frame),
        standard_error=float(estimate.std_error),
    )
    return [positive, control]


def _payloads() -> list[tuple[tuple[str, str, int, int, int, int, str]]]:
    specs: list[tuple[str, str, int, int, str]] = []
    for configuration in ("both_correct", "outcome_correct", "mechanism_correct", "both_wrong"):
        specs.append(
            (
                "double_robustness",
                configuration,
                DOUBLE_ROBUST_N,
                DOUBLE_ROBUST_REPLICATES,
                configuration,
            )
        )
    for size in RATE_SIZES:
        specs.append(("root_n_and_efficiency", f"n_{size}", size, RATE_REPLICATES, "both_correct"))
    specs.extend(
        [
            (
                "interval_calibration",
                "correctly_specified",
                CALIBRATION_N,
                CALIBRATION_REPLICATES,
                "both_correct",
            ),
            ("type_i_error", "sharp_null", NULL_N, NULL_REPLICATES, "both_correct"),
            ("power", "alternative", NULL_N, NULL_REPLICATES, "both_correct"),
            (
                "targeting_necessity",
                "targeted",
                TARGETING_N,
                TARGETING_REPLICATES,
                "mechanism_correct",
            ),
        ]
    )
    payloads: list[tuple[tuple[str, str, int, int, int, int, str]]] = []
    for property_name, cell, n, replicates, configuration in specs:
        for replicate in range(replicates):
            seed = stream_seed(STUDY, "property_sample", property_name, cell, replicate)
            payloads.append(((property_name, cell, replicate, n, replicates, seed, configuration),))
    return payloads


def _calibration_controls(rows: pd.DataFrame) -> pd.DataFrame:
    source = rows.loc[
        (rows["property"] == "interval_calibration")
        & rows["cell"].str.endswith("__correctly_specified")
    ]
    controls: list[pd.DataFrame] = []
    for label in CONTRASTS:
        base = source.loc[source["cell"] == f"{label}__correctly_specified"].copy()
        shrunken = base.copy()
        shrunken["cell"] = f"{label}__shrunken_se_control"
        shrunken["role"] = "control"
        shrunken["std_error"] *= SHRUNKEN_SE_FACTOR
        shrunken["covered"] = (
            (shrunken["estimate"] - CRITICAL * shrunken["std_error"] <= shrunken["truth"])
            & (shrunken["truth"] <= shrunken["estimate"] + CRITICAL * shrunken["std_error"])
        ).astype(int)
        shrunken["rejected"] = (
            np.abs(shrunken["estimate"] / shrunken["std_error"]) > CRITICAL
        ).astype(int)
        controls.append(shrunken)

        noisy = base.copy()
        rng = np.random.default_rng(stream_seed(STUDY, "efficiency_noise", label))
        noisy["cell"] = f"{label}__noise_control"
        noisy["role"] = "control"
        noisy["estimate"] += rng.normal(
            scale=EFFICIENCY_SD[label] / np.sqrt(CALIBRATION_N), size=len(noisy)
        )
        noisy["covered"] = (
            (noisy["estimate"] - CRITICAL * noisy["std_error"] <= noisy["truth"])
            & (noisy["truth"] <= noisy["estimate"] + CRITICAL * noisy["std_error"])
        ).astype(int)
        noisy["rejected"] = (np.abs(noisy["estimate"] / noisy["std_error"]) > CRITICAL).astype(int)
        controls.append(noisy)
    return pd.concat(controls, ignore_index=True)


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    outcomes = map_parallel(_fit_replication, _payloads(), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    recursion_payloads = [
        (
            (
                replicate,
                RECURSION_N,
                RECURSION_REPLICATES,
                stream_seed(STUDY, "recursion", replicate),
            ),
        )
        for replicate in range(RECURSION_REPLICATES)
    ]
    recursion = map_parallel(_recursion_replication, recursion_payloads, n_jobs=n_jobs)
    rows = pd.concat(
        [
            rows,
            pd.DataFrame([row for result in recursion for row in result]),
            _calibration_controls(rows),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def _interval(summary: pd.DataFrame, index: Any, prefix: str) -> Interval:
    return Interval(
        float(summary.loc[index, f"{prefix}_ci_lower"]),
        float(summary.loc[index, f"{prefix}_ci_upper"]),
    )


def _paired_displacement(rows: pd.DataFrame, family: str, left: str, right: str) -> float:
    arms = {
        name: rows.loc[(rows["property"] == family) & (rows["cell"] == cell)].sort_values(
            "replicate"
        )
        for name, cell in (("left", left), ("right", right))
    }
    if not np.array_equal(arms["left"]["replicate"], arms["right"]["replicate"]):
        raise ValueError(f"{family} controls are not paired on replication")
    spread = float(arms["left"]["estimate"].std(ddof=1))
    moved = float(arms["right"]["estimate"].mean() - arms["left"]["estimate"].mean())
    return abs(moved) / spread


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement", "recursion_displacement"),
        rate_labels=PROPERTY_LABELS,
        efficiency_bounds=EFFICIENCY_SD,
    )
    margins = STUDY.margins

    calibration = summary["property"] == "interval_calibration"
    for index in summary.index[calibration]:
        kind = str(summary.loc[index, "cell"]).split("__", 1)[1]
        ratio = _interval(summary, index, "se_ratio")
        empirical = _interval(summary, index, "efficiency_empirical")
        reported = _interval(summary, index, "efficiency_reported")
        coverage = _interval(summary, index, "coverage")
        if kind == "correctly_specified":
            passed = (
                ratio.within(*margins.calibration_se_ratio)
                and coverage.within(*margins.calibration_coverage)
                and empirical.within(*EFFICIENCY_RATIO_BAND)
                and reported.within(*EFFICIENCY_RATIO_BAND)
            )
        elif kind == "shrunken_se_control":
            passed = ratio.high < margins.calibration_se_ratio[0]
        else:
            passed = empirical.low > EFFICIENCY_RATIO_BAND[1]
        summary.loc[index, "passed"] = bool(passed)

    targeting = summary["property"] == "targeting_necessity"
    summary.loc[targeting & (summary["role"] == "positive"), "passed"] = summary.loc[
        targeting & (summary["role"] == "positive"), "bias_equivalent"
    ]
    summary.loc[targeting & (summary["role"] == "control"), "passed"] = summary.loc[
        targeting & (summary["role"] == "control"), "bias_discriminated"
    ]
    displacements = [
        _paired_displacement(
            rows,
            "targeting_necessity",
            f"{label}__targeted",
            f"{label}__untargeted",
        )
        for label in PROPERTY_LABELS
    ]
    targeting_displacement = min(displacements)
    summary.loc[targeting, "targeting_displacement"] = targeting_displacement
    summary.loc[targeting, "property_passed"] = bool(
        summary.loc[targeting, "passed"].all() and targeting_displacement >= TARGETING_DISPLACEMENT
    )

    recursion = summary["property"] == "survival_recursion_necessity"
    summary.loc[recursion & (summary["role"] == "positive"), "passed"] = summary.loc[
        recursion & (summary["role"] == "positive"), "bias_equivalent"
    ]
    summary.loc[recursion & (summary["role"] == "control"), "passed"] = summary.loc[
        recursion & (summary["role"] == "control"), "bias_discriminated"
    ]
    recursion_displacement = _paired_displacement(
        rows,
        "survival_recursion_necessity",
        "always_t2__survival",
        "always_t2__survivor_only",
    )
    summary.loc[recursion, "recursion_displacement"] = recursion_displacement
    summary.loc[recursion, "property_passed"] = bool(
        summary.loc[recursion, "passed"].all() and recursion_displacement >= TARGETING_DISPLACEMENT
    )
    return finish(summary, rates)
