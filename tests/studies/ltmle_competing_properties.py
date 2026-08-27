"""Independent repeated-sampling properties for competing-risk LTMLE."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor

from cleverly.learners.crossfit import Folds
from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_competing as law
from tests.parallel import STUDY_JOBS
from tests.studies import canonical_ltmle_competing as canonical
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    crossfit_overfitting_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.seeds import stream_seed

DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 4_000
RATE_REPLICATES = 1_600
RATE_SIZES = (4_000, 8_000, 32_000)
CALIBRATION_REPLICATES = 9_600
CALIBRATION_N = 32_000
NULL_REPLICATES = 1_600
NULL_N = 4_000
TARGETING_REPLICATES = DOUBLE_ROBUST_REPLICATES
TARGETING_N = DOUBLE_ROBUST_N
RECURSION_REPLICATES = DOUBLE_ROBUST_REPLICATES
RECURSION_N = DOUBLE_ROBUST_N

EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70
TARGETING_DISPLACEMENT = 0.25

#: How far the cause-specific-survival control must sit from the all-cause recursion, in
#: empirical standard deviations of the recursion's own estimate, before the family will call
#: the all-cause risk set load bearing.
#:
#: The same number as :data:`TARGETING_DISPLACEMENT`, and declared separately rather than
#: reused, because the two families answer different questions and a reader has to be able to
#: see which threshold each verdict was read against.  The *value* is the same for the reason
#: that one is: it is :attr:`Margins.standardized_bias`, the bias a positive cell is allowed
#: to carry, so a control has to move the estimate by at least as much as the study is
#: prepared to overlook.
RECURSION_DISPLACEMENT = TARGETING_DISPLACEMENT

STUDY = canonical.STUDY
REGIMENS = law.REGIMEN_SPEC
REFERENCE = law.REGIMEN_REFERENCE
PROPERTY_LABELS = canonical.PROPERTY_LABELS
CONTRASTS = {
    "relapse_dynamic_t2": "ate_regimen[continue_if_l2 vs never, relapse @ t=2]",
    "death_static_t2": "ate_regimen[always vs never, death @ t=2]",
}
EFFICIENCY_SD = {
    label: float(np.sqrt(np.sum(law.PROBS * law.eif(name) ** 2)))
    for label, name in CONTRASTS.items()
}

# A sharp null that still needs the whole longitudinal recursion.
#
# Every contrast the fit reports is exactly zero -- both causes, both horizons, both plans --
# and none of it is bought by making a hazard constant.  ``NULL_H1`` does not depend on ``A1``,
# which zeroes the first horizon and, because the all-cause survival ``S_1`` is then the same
# in every arm, leaves the second horizon a statement about ``NULL_H2`` alone.  ``NULL_H2``
# then *does* vary with ``L2``, with ``A1`` and with ``A2``: the plans' second-node averages
# coincide only after averaging, and they are averaged under two different ``L2`` laws, because
# ``P(L2 = 1 | w, A1 = 0)`` is ``(0.25, 0.50)`` while ``P(L2 = 1 | w, A1 = 1)`` is
# ``(0.75, 0.75)``.  That mismatch is the freedom ``ltmle_properties._null_outcome`` documents.
#
# The cancellation runs *across* ``W``, not within it.  At ``w = 1`` the never path weights
# ``L2`` as ``(0.50, 0.50)`` and either treated path as ``(0.25, 0.75)``; on the quarter grid
# the only per-stratum matches are the flat ones.  So a null that cancels stratum by stratum is
# forced to hold the second hazard constant in ``L2`` -- which is what the first version of
# these constants did, and why it needed no longitudinal estimator at all.
#
# ``test_ltmle_competing_method_study`` is the witness.  A standardisation that adjusts for
# ``W`` and never for ``L2`` misses the horizon-two null by -0.0077 (death, always against
# never) and +0.0091 (relapse, the rule against never).  The previous constants left the second
# hazard constant in ``L2``, ``A1`` and ``A2``, and that same analysis recovered the null
# exactly, so the type-I cell could not tell a sequential-regression fit from two cross
# sections.  A crude comparison of arms misses it at both horizons, so the cell still requires
# baseline adjustment as well.
#
# Every value is a multiple of ``1/4`` and strictly positive, and every all-cause sum is
# ``0.75``, so a quarter survives each node and ``law.counts`` realises the derived law exactly
# in ``N`` rows just as it does :data:`law.PROBS`.
NULL_H1 = np.array(
    [
        [[0.25, 0.25], [0.50, 0.50]],  # relapse: 0.25 at W = 0, 0.50 at W = 1
        [[0.50, 0.50], [0.25, 0.25]],  # death:   0.50 at W = 0, 0.25 at W = 1
    ]
)

# Only the eight cells the three plans traverse and the null needs moved are overwritten.  The
# rest of ``law.H2`` stands, including the six cells holding ``A2`` opposite the arm the plan
# assigns at that ``L2``: no plan reaches them, so they enter the observed law and the nuisance
# fits but neither the truth nor the bias, and there is nothing to choose there.
# Indexed ``[cause - 1, w, a1, l2, a2]``.
NULL_H2 = np.array(law.H2, copy=True)
NULL_H2[0, 0, 1, 0, 0] = 0.25  # relapse, W=0: the rule's L2 = 0 arm
NULL_H2[0, 0, 1, 1, 1] = 0.50  # relapse, W=0: the L2 = 1 arm both treated plans share
NULL_H2[0, 1, 0, 0, 0] = 0.50  # relapse, W=1: the never path, now moving with L2
NULL_H2[0, 1, 1, 0, 1] = 0.50  # relapse, W=1: the always path at L2 = 0
NULL_H2[1, 0, 0, 0, 0] = 0.50  # death,   W=0: the never path, now moving with L2
NULL_H2[1, 0, 1, 0, 1] = 0.25  # death,   W=0: the always path at L2 = 0
NULL_H2[1, 1, 1, 0, 1] = 0.25  # death,   W=1: the always path at L2 = 0
NULL_H2[1, 1, 1, 1, 1] = 0.50  # death,   W=1: the always path at L2 = 1

NULL_PROBS = law.probabilities(NULL_H1, NULL_H2)
NULL_TRUTH = {label: float(law.functional(NULL_PROBS, name)) for label, name in CONTRASTS.items()}

# No separate alternative, unlike both sibling studies.  ``ltmle_survival_properties`` needs one
# because its horizon-two static effect "is intentionally small, so the power cell needs a
# predeclared alternative".  This law's two registered contrasts are 0.1211 and -0.1523 under
# its own hazards, which is a material effect already, so the primary law *is* the declared
# alternative and the power cell samples it directly.
# ``test_each_power_control_has_a_material_cause_specific_effect`` is what pins that.
POWER_TRUTH = {label: float(law.TRUTH[name]) for label, name in CONTRASTS.items()}

_SUPPORT_FRAME = law.frame().iloc[law.first_row_of()].reset_index(drop=True)


def sample(probs: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    cells = np.random.default_rng(seed).choice(len(law.SUPPORT), size=n, p=probs)
    return _SUPPORT_FRAME.iloc[cells].reset_index(drop=True)


def _learners(configuration: str) -> tuple[Any, Any, Any, Any]:
    q_correct = configuration in {"both_correct", "outcome_correct"}
    g_correct = configuration in {"both_correct", "mechanism_correct"}
    return (
        law.CellMeans() if q_correct else DummyClassifier(strategy="prior"),
        law.CellMeans() if q_correct else DummyRegressor(strategy="mean"),
        canonical.KnownCompetingMechanism("treatment")
        if g_correct
        else DummyClassifier(strategy="prior"),
        canonical.KnownCompetingMechanism("censoring")
        if g_correct
        else DummyClassifier(strategy="prior"),
    )


def fit(frame: pd.DataFrame, configuration: str = "both_correct", *, n_folds: int = 1) -> Any:
    outcome, pseudo, treatment, censoring = _learners(configuration)
    return LTMLE(
        REGIMENS,
        reference=REFERENCE,
        horizons=(2,),
        outcome_learner=outcome,
        pseudo_learner=pseudo,
        treatment_learner=treatment,
        censoring_learner=censoring,
        n_folds=n_folds,
        learner_folds=5,
        g_bounds=canonical.G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome=law.outcome_columns(),
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


def _plan_arms(frame: pd.DataFrame, label: str) -> tuple[np.ndarray, np.ndarray]:
    first, second = law.REGIMEN_ARMS[label]
    w = frame["W"].to_numpy().astype(int)
    l2 = np.nan_to_num(frame["L2"].to_numpy()).astype(int)
    node1 = np.full(len(frame), float(first)) if np.ndim(first) == 0 else np.asarray(first)[w]
    node2 = (
        np.full(len(frame), float(second)) if np.ndim(second) == 0 else np.asarray(second)[w, l2]
    )
    return np.asarray(node1), np.asarray(node2)


def _fold_masks(folds: Folds, held_out: int) -> tuple[np.ndarray, np.ndarray]:
    evaluated = folds.assignment == held_out
    training = np.ones_like(evaluated) if folds.n_folds == 1 else ~evaluated
    return training, evaluated


def untargeted(
    frame: pd.DataFrame,
    label: str,
    cause: str,
    horizon: int,
    configuration: str,
    folds: Folds,
    *,
    cause_specific_survival: bool = False,
) -> float:
    """Fold-specific unfluctuated recursion for one cause and plan."""
    outcome, pseudo, _, _ = _learners(configuration)
    first, second = _plan_arms(frame, label)
    target1, target2 = law.outcome_columns()[cause]
    other = next(value for value in law.CAUSES if value != cause)
    other1 = law.outcome_columns()[other][0]
    y1 = np.nan_to_num(frame[target1].to_numpy())
    d1 = np.nan_to_num(frame[other1].to_numpy())
    followed_one = (frame["C1"].to_numpy() == 1.0) & (frame["A1"].to_numpy() == first)
    baseline = frame[["W"]].to_numpy(dtype=float)
    stitched = np.empty(len(frame), dtype=float)

    if horizon == 1:
        for held_out in range(folds.n_folds):
            training, evaluated = _fold_masks(folds, held_out)
            model = clone(outcome).fit(
                baseline[training & followed_one], y1[training & followed_one]
            )
            stitched[evaluated] = model.predict_proba(baseline[evaluated])[:, 1]
        return float(np.mean(stitched))

    history = np.column_stack([baseline[:, 0], np.nan_to_num(frame["L2"].to_numpy())])
    event_free = (y1 == 0.0) & (d1 == 0.0)
    followed_two = (
        followed_one
        & event_free
        & (frame["C2"].to_numpy() == 1.0)
        & (frame["A2"].to_numpy() == second)
    )
    events = frame[target2].to_numpy()
    for held_out in range(folds.n_folds):
        training, evaluated = _fold_masks(folds, held_out)
        later = clone(outcome).fit(
            history[training & followed_two], events[training & followed_two]
        )
        carried = np.asarray(later.predict_proba(history))[:, 1]
        survival = 1.0 - y1 if cause_specific_survival else event_free.astype(float)
        pseudo_outcome = y1 + survival * carried
        earlier = clone(pseudo).fit(
            baseline[training & followed_one], pseudo_outcome[training & followed_one]
        )
        stitched[evaluated] = earlier.predict(baseline[evaluated])
    return float(np.mean(stitched))


def _fit_replication(
    payload: tuple[str, str, int, int, int, int, str],
    *,
    study: Any,
    fit_fn: Callable[[pd.DataFrame, str], Any],
) -> list[dict[str, Any]]:
    property_name, suffix, replicate, n, requested, seed, configuration = payload
    # The power cell samples the primary law, so it needs no branch of its own here; see
    # :data:`POWER_TRUTH` for why this study declares no separate alternative.
    probs = NULL_PROBS if property_name == "type_i_error" else law.PROBS
    frame = sample(probs, n, seed)
    result = fit_fn(frame, configuration)
    rows: list[dict[str, Any]] = []
    for label, name in CONTRASTS.items():
        truth = NULL_TRUTH[label] if property_name == "type_i_error" else float(law.TRUTH[name])
        role = "control" if suffix == "both_wrong" else "positive"
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
                alpha=study.margins.alpha,
            )
        )
        if property_name == "targeting_necessity":
            inside = name[len("ate_regimen[") : -1]
            comparison_cause, horizon_text = inside.rsplit(" @ t=", 1)
            comparison, cause = comparison_cause.rsplit(", ", 1)
            left, right = comparison.split(" vs ")
            horizon = int(horizon_text)
            unfluctuated = untargeted(
                frame, left, cause, horizon, configuration, result.folds
            ) - untargeted(frame, right, cause, horizon, configuration, result.folds)
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
                    critical=float(norm.ppf(1.0 - study.margins.alpha / 2.0)),
                )
            )
    return rows


def _recursion_replication(
    payload: tuple[int, int, int, int],
    *,
    study: Any,
    fit_fn: Callable[[pd.DataFrame, str], Any],
) -> list[dict[str, Any]]:
    replicate, n, requested, seed = payload
    frame = sample(law.PROBS, n, seed)
    result = fit_fn(frame, "both_correct")
    rows: list[dict[str, Any]] = []
    for cause in law.CAUSES:
        name = f"cif_regimen[always, {cause} @ t=2]"
        estimate = result[name]
        truth = float(law.TRUTH[name])
        rows.append(
            replicate_row(
                property_name="competing_risk_recursion_necessity",
                cell=f"{cause}_always_t2__all_cause",
                role="positive",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=estimate,
                alpha=study.margins.alpha,
            )
        )
        rows.append(
            control_row(
                property_name="competing_risk_recursion_necessity",
                cell=f"{cause}_always_t2__cause_specific_control",
                replicate=replicate,
                n=n,
                requested=requested,
                truth=truth,
                estimate=untargeted(
                    frame,
                    "always",
                    cause,
                    2,
                    "both_correct",
                    result.folds,
                    cause_specific_survival=True,
                ),
                standard_error=float(estimate.std_error),
                critical=float(norm.ppf(1.0 - study.margins.alpha / 2.0)),
            )
        )
    return rows


def _payloads(study: Any) -> list[tuple[tuple[str, str, int, int, int, int, str]]]:
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
    return [
        (
            (
                property_name,
                cell,
                replicate,
                n,
                replicates,
                stream_seed(study, "property_sample", property_name, cell, replicate),
                configuration,
            ),
        )
        for property_name, cell, n, replicates, configuration in specs
        for replicate in range(replicates)
    ]


def generate_for(
    study: Any,
    fit_fn: Callable[[pd.DataFrame, str], Any],
    *,
    n_jobs: int,
) -> pd.DataFrame:
    def runner(payload: tuple[str, str, int, int, int, int, str]) -> list[dict[str, Any]]:
        return _fit_replication(payload, study=study, fit_fn=fit_fn)

    outcomes = map_parallel(runner, _payloads(study), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    recursion_payloads = [
        (
            (
                replicate,
                RECURSION_N,
                RECURSION_REPLICATES,
                stream_seed(study, "competing_recursion", replicate),
            ),
        )
        for replicate in range(RECURSION_REPLICATES)
    ]

    def recursion_runner(payload: tuple[int, int, int, int]) -> list[dict[str, Any]]:
        return _recursion_replication(payload, study=study, fit_fn=fit_fn)

    recursion = map_parallel(recursion_runner, recursion_payloads, n_jobs=n_jobs)
    rows = pd.concat(
        [
            rows,
            pd.DataFrame([row for result in recursion for row in result]),
            calibration_controls(
                rows,
                study,
                labels=PROPERTY_LABELS,
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=CALIBRATION_N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=float(norm.ppf(1.0 - study.margins.alpha / 2.0)),
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    return generate_for(STUDY, lambda frame, config: fit(frame, config), n_jobs=n_jobs)


def summarize_for(
    rows: pd.DataFrame, study: Any, *, crossfit_positive_cell: str | None = None
) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        study,
        extra_columns=("targeting_displacement", "recursion_displacement"),
        rate_labels=PROPERTY_LABELS,
        efficiency_bounds=EFFICIENCY_SD,
    )
    calibration_verdicts(summary, margins=study.margins, efficiency_band=EFFICIENCY_RATIO_BAND)

    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=PROPERTY_LABELS,
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )
    necessity_verdicts(
        summary,
        rows,
        family="competing_risk_recursion_necessity",
        labels=tuple(f"{cause}_always_t2" for cause in law.CAUSES),
        arms=("all_cause", "cause_specific_control"),
        column="recursion_displacement",
        threshold=RECURSION_DISPLACEMENT,
    )
    if crossfit_positive_cell is not None:
        crossfit_overfitting_verdicts(summary, rows, study, positive_cell=crossfit_positive_cell)
    return finish(summary, rates)


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    return summarize_for(rows, STUDY)
