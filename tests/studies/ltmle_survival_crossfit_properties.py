"""Independent repeated-sampling properties for cross-fitted survival LTMLE."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from cleverly.datasets import make_longitudinal_survival
from cleverly.learners.crossfit import Folds
from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_survival as law
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_ltmle import KnownLongitudinalMechanism
from tests.studies.canonical_ltmle_crossfit import G_BOUNDS
from tests.studies.canonical_ltmle_survival_crossfit import PROPERTY_LABELS, STUDY
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    control_row,
    paired_displacement,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    crossfit_overfitting_verdicts,
    finish,
)
from tests.studies.evidence.seeds import stream_seed

DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
RATE_REPLICATES = 800
#: The ladder starts at 1,000 rather than at the 500 the end-of-study study uses, and the
#: reason is the absorbing event rather than a preference.  At ``n = 500`` this law does not
#: always *have* a horizon-two parameter: sweeping this study's own
#: ``stream_seed(STUDY, "property_sample", "root_n_and_efficiency", "n_500", replicate)``
#: stream, replicate 80 leaves five followers of ``always`` through the second node and every
#: one of them has ``Y2 = 0``, so the second regression has nothing to separate and the
#: estimator refuses with ``LongitudinalError``.  That refusal is correct, and
#: ``docs/development/method-benchmarking.md`` forbids replacing a failed replication, so 500
#: is unavailable here.  The end-of-study law keeps it because it has no risk set to thin: its
#: outcome sits at the end, so no unit leaves before the node the plan is followed through.
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
#: Use the same precision-sized paired budget as the end-of-study cross-fit instrument.
OVERFIT_REPLICATES = 8_000
OVERFIT_N = 1_000

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
TARGETING_DISPLACEMENT = 0.25

#: How far the survivor-only control must sit from the survival recursion, in empirical
#: standard deviations of the recursion's own estimate, before the family will call the
#: recursion load bearing.
#:
#: The same number as :data:`TARGETING_DISPLACEMENT`, and declared separately rather than
#: reused, because the two families answer different questions and a reader has to be able to
#: see which threshold each verdict was read against.  The *value* is the same for the reason
#: that one is: it is :attr:`Margins.standardized_bias`, the bias a positive cell is allowed
#: to carry, so a control has to move the estimate by at least as much as the study is
#: prepared to overlook.  Publishing one margin under one description for two families is how
#: a page comes to state a rule nothing checks.
RECURSION_DISPLACEMENT = TARGETING_DISPLACEMENT

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


class KnownDiscreteMechanism(BaseEstimator):
    """The exact treatment or censoring mechanism of the survival property law."""

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> KnownDiscreteMechanism:
        del X, y, sample_weight
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        w = matrix[:, 0].astype(int)
        if self.kind == "treatment" and matrix.shape[1] == 1:
            probability = law.G1[w]
        elif self.kind == "treatment" and matrix.shape[1] == 3:
            l2 = matrix[:, 1].astype(int)
            a1 = matrix[:, 2].astype(int)
            probability = law.G2[w, a1, l2]
        elif self.kind == "censoring" and matrix.shape[1] == 2:
            a1 = matrix[:, 1].astype(int)
            probability = law.C1[w, a1]
        elif self.kind == "censoring" and matrix.shape[1] == 4:
            l2 = matrix[:, 1].astype(int)
            a1 = matrix[:, 2].astype(int)
            a2 = matrix[:, 3].astype(int)
            probability = law.C2[w, a1, l2, a2]
        else:  # pragma: no cover - a changed design is a study-contract failure
            raise ValueError(f"unexpected {self.kind} mechanism design {matrix.shape}")
        return np.column_stack([1.0 - probability, probability])


# A sharp null that still needs the whole recursion.  The first hazard has opposing
# conditional treatment effects whose marginal effects cancel.  The second hazard varies with
# L2 on every followed path, and the three plans' weighted second-node averages coincide.  So
# all three contrasts are exactly zero without making either hazard a baseline-only constant,
# and a baseline-only standardisation comes out biased by 0.0349 rather than by nothing.
# ``test_ltmle_survival_method_study`` asserts that bias.  It is the only witness that says
# this null is one a longitudinal fit has to work for.
#
# Every value is a multiple of 1/4, which is not cosmetic: ``law.probabilities`` rounds to
# whole rows and refuses anything else, so the derived law is realised *exactly* by an N-row
# sample just as ``law.PROBS`` is.  A hazard off that grid leaves the null usable for sampling
# and unusable for any exact-law control built on ``law.frame()``.
NULL_H1 = np.array([[0.25, 0.50], [0.75, 0.50]])
NULL_H2 = np.array(law.H2, copy=True)
NULL_H2[0, 0, 0, 0], NULL_H2[0, 0, 1, 0] = 0.25, 0.75
NULL_H2[1, 0, 0, 0], NULL_H2[1, 0, 1, 0] = 0.75, 0.25
NULL_H2[0, 1, 0, 1], NULL_H2[0, 1, 1, 1] = 0.25, 0.50
NULL_H2[1, 1, 0, 1], NULL_H2[1, 1, 1, 1] = 0.75, 0.25
NULL_H2[0, 1, 0, 0] = 0.50
NULL_H2[1, 1, 0, 0] = 0.50

# The original horizon-two static effect is intentionally small, so the power cell needs a
# predeclared alternative.  This one keeps the law's own first hazard and replaces the second
# on the quarter grid, for the reason the null does.
#
# It also makes the two horizon-two contrasts *different parameters*: always comes to -0.2578
# and the dynamic plan to -0.1953.  The previous alternative left the second hazard constant
# in L2 on every followed path, which made those two contrasts numerically identical and
# published two power cells for one claim -- in a section whose own opening says it reports
# each unique parameter once.
POWER_H2 = np.array(law.H2, copy=True)
POWER_H2[0, 0, 0, 0], POWER_H2[0, 0, 1, 0] = 0.75, 0.50
POWER_H2[1, 0, 0, 0], POWER_H2[1, 0, 1, 0] = 0.50, 0.75
POWER_H2[0, 1, 0, 1], POWER_H2[0, 1, 1, 1] = 0.25, 0.50
POWER_H2[1, 1, 0, 1], POWER_H2[1, 1, 1, 1] = 0.50, 0.25
POWER_H2[0, 1, 0, 0] = 0.75
POWER_H2[1, 1, 0, 0] = 0.75

NULL_PROBS = law.probabilities(NULL_H1, NULL_H2)
POWER_PROBS = law.probabilities(h2=POWER_H2)
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
    """The four nuisance learners one configuration hands the estimator.

    The correctly specified mechanism is the law's own conditional probabilities rather than
    the saturated ``law.CellMeans`` fit the ordinary study uses, because a training complement
    under a five-fold split can meet an empty cell.  See
    :func:`tests.studies.ltmle_crossfit_properties._learners`, which carries the argument.
    """
    if configuration in {"overfit_crossfit", "overfit_control"}:
        return (
            DecisionTreeClassifier(min_samples_leaf=1, random_state=0),
            DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
            law.CellMeans(),
            law.CellMeans(),
        )
    q_correct = configuration in {"both_correct", "outcome_correct"}
    g_correct = configuration in {"both_correct", "mechanism_correct"}
    return (
        law.CellMeans() if q_correct else DummyClassifier(strategy="prior"),
        law.CellMeans() if q_correct else DummyRegressor(strategy="mean"),
        KnownDiscreteMechanism("treatment") if g_correct else DummyClassifier(strategy="prior"),
        KnownDiscreteMechanism("censoring") if g_correct else DummyClassifier(strategy="prior"),
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
        n_folds=1 if configuration == "overfit_control" else 5,
        learner_folds=5,
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


def _overfit(frame: pd.DataFrame, *, cross_fit: bool) -> Any:
    """The continuous-history survival law where a tree can memorize event outcomes."""
    return LTMLE(
        {"never": 0, "always": 1},
        reference="never",
        outcome_learner=DecisionTreeClassifier(min_samples_leaf=1, random_state=0),
        pseudo_learner=DecisionTreeRegressor(min_samples_leaf=1, random_state=0),
        treatment_learner=KnownLongitudinalMechanism("treatment"),
        censoring_learner=KnownLongitudinalMechanism("censoring"),
        n_folds=5 if cross_fit else 1,
        learner_folds=5,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome=["Y1", "Y2"],
        treatment=["A1", "A2"],
        baseline=["W1", "W2"],
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


def untargeted(
    frame: pd.DataFrame, label: str, horizon: int, configuration: str, folds: Folds
) -> float:
    """Survival recursion using the same nuisance learners and no fluctuation.

    Fold-specific, like :func:`fit`: each outer training complement fits its own regressions
    and each fold's held-out rows are read off its own, so the difference between this and the
    estimator is the targeting step and nothing else.  See
    :func:`tests.studies.ltmle_crossfit_properties.untargeted`, which carries the argument for
    why the folds are passed in rather than rebuilt.

    The first horizon is a one-node recursion and the second composes a pseudo-outcome from
    the first node's event indicator, which is the structure the survivor-only control exists
    to separate this from.
    """
    outcome, pseudo, _, _ = _learners(configuration)
    first, second = _plan_arms(frame, label)
    y1 = np.nan_to_num(frame["Y1"].to_numpy())
    followed_one = (frame["C1"].to_numpy() == 1.0) & (frame["A1"].to_numpy() == first)
    baseline = frame[["W"]].to_numpy(dtype=float)
    stitched = np.empty(len(frame), dtype=float)

    if horizon == 1:
        for held_out in range(folds.n_folds):
            training = folds.assignment != held_out
            model = clone(outcome).fit(
                baseline[training & followed_one], y1[training & followed_one]
            )
            evaluated = folds.assignment == held_out
            stitched[evaluated] = model.predict_proba(baseline[evaluated])[:, 1]
        return float(np.mean(stitched))

    history = np.column_stack([baseline[:, 0], np.nan_to_num(frame["L2"].to_numpy())])
    followed_two = (
        followed_one
        & (y1 == 0.0)
        & (frame["C2"].to_numpy() == 1.0)
        & (frame["A2"].to_numpy() == second)
    )
    events = frame["Y2"].to_numpy()
    for held_out in range(folds.n_folds):
        training = folds.assignment != held_out
        later = clone(outcome).fit(
            history[training & followed_two], events[training & followed_two]
        )
        carried = np.asarray(later.predict_proba(history))[:, 1]
        pseudo_outcome = y1 + (1.0 - y1) * carried
        earlier = clone(pseudo).fit(
            baseline[training & followed_one], pseudo_outcome[training & followed_one]
        )
        evaluated = folds.assignment == held_out
        stitched[evaluated] = earlier.predict(baseline[evaluated])
    return float(np.mean(stitched))


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


def _fit_replication(payload: tuple[str, str, int, int, int, int, str]) -> list[dict[str, Any]]:
    property_name, suffix, replicate, n, requested, seed, configuration = payload
    if property_name == "crossfit_overfitting":
        frame, truth = make_longitudinal_survival(n=n, seed=seed, censoring=True, backend="pandas")
        name = "ate_regimen[always vs never @ t=2]"
        result = _overfit(frame, cross_fit=configuration == "overfit_crossfit")
        return [
            replicate_row(
                property_name=property_name,
                cell=suffix,
                role="control" if configuration == "overfit_control" else "positive",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=float(truth[name]),
                estimate=result[name],
                alpha=STUDY.margins.alpha,
            )
        ]
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
            replicate_row(
                property_name=property_name,
                cell=f"{label}__{suffix}",
                role=role,
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=result[name],
                alpha=STUDY.margins.alpha,
            )
        )
        if property_name == "targeting_necessity":
            left, right = name[len("ate_regimen[") : -1].rsplit(" @ t=", 1)[0].split(" vs ")
            horizon = int(name.rsplit(" @ t=", 1)[1][:-1])
            unfluctuated = untargeted(
                frame, left, horizon, configuration, result.folds
            ) - untargeted(frame, right, horizon, configuration, result.folds)
            rows.append(
                control_row(
                    property_name=property_name,
                    cell=f"{label}__untargeted",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=truth,
                    estimate=unfluctuated,
                    standard_error=float(result[name].std_error),
                    critical=CRITICAL,
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
    positive = replicate_row(
        property_name="survival_recursion_necessity",
        cell="always_t2__survival",
        role="positive",
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )
    control = control_row(
        property_name="survival_recursion_necessity",
        cell="always_t2__survivor_only",
        replicate=replicate,
        n=n,
        requested=requested,
        truth=truth,
        estimate=survivor_only(frame),
        standard_error=float(estimate.std_error),
        critical=CRITICAL,
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
            (
                "crossfit_overfitting",
                "cross_fitted_survival_ltmle",
                OVERFIT_N,
                OVERFIT_REPLICATES,
                "overfit_crossfit",
            ),
            (
                "crossfit_overfitting",
                "in_sample_control",
                OVERFIT_N,
                OVERFIT_REPLICATES,
                "overfit_control",
            ),
        ]
    )
    payloads: list[tuple[tuple[str, str, int, int, int, int, str]]] = []
    for property_name, cell, n, replicates, configuration in specs:
        for replicate in range(replicates):
            seed_cell = "paired" if property_name == "crossfit_overfitting" else cell
            seed = stream_seed(STUDY, "property_sample", property_name, seed_cell, replicate)
            payloads.append(((property_name, cell, replicate, n, replicates, seed, configuration),))
    return payloads


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
            calibration_controls(
                rows,
                STUDY,
                labels=PROPERTY_LABELS,
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


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=(
            "targeting_displacement",
            "recursion_displacement",
            "coverage_gain_ci_lower",
            "coverage_gain_ci_upper",
        ),
        rate_labels=PROPERTY_LABELS,
        efficiency_bounds=EFFICIENCY_SD,
    )
    margins = STUDY.margins

    calibration_verdicts(summary, margins=margins, efficiency_band=EFFICIENCY_RATIO_BAND)

    targeting = summary["property"] == "targeting_necessity"
    summary.loc[targeting & (summary["role"] == "positive"), "passed"] = summary.loc[
        targeting & (summary["role"] == "positive"), "bias_equivalent"
    ]
    summary.loc[targeting & (summary["role"] == "control"), "passed"] = summary.loc[
        targeting & (summary["role"] == "control"), "bias_discriminated"
    ]
    displacements = [
        paired_displacement(
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
    recursion_displacement = paired_displacement(
        rows,
        "survival_recursion_necessity",
        "always_t2__survival",
        "always_t2__survivor_only",
    )
    summary.loc[recursion, "recursion_displacement"] = recursion_displacement
    summary.loc[recursion, "property_passed"] = bool(
        summary.loc[recursion, "passed"].all() and recursion_displacement >= RECURSION_DISPLACEMENT
    )
    crossfit_overfitting_verdicts(summary, rows, STUDY, positive_cell="cross_fitted_survival_ltmle")
    return finish(summary, rates)
