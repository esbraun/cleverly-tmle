"""Independent properties for ordinary end-of-study longitudinal TMLE."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor

from cleverly.longitudinal import LTMLE
from cleverly.utils.parallel import map_parallel
from tests import discrete_law_longitudinal as law
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_ltmle import G_BOUNDS, STUDY
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    control_row,
    replicate_row,
)
from tests.studies.evidence.property_verdicts import (
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    necessity_verdicts,
)
from tests.studies.evidence.seeds import stream_seed

DOUBLE_ROBUST_REPLICATES = 1_200
DOUBLE_ROBUST_N = 2_000
RATE_REPLICATES = 800
RATE_SIZES = (500, 2_000, 8_000)
CALIBRATION_REPLICATES = 2_400
CALIBRATION_N = 2_000
#: Twice the shared budget, for the reason ``ctmle_oat_properties.OAT_NULL_REPLICATES`` is:
#: the binding endpoint here is the *upper* bound on the rejection rate, and at 400
#: replications a one-sided 99% bound sits within a few thousandths of the published ceiling
#: of ``alpha + type_i_margin``.  A cell that clears its gate by less than the width of one
#: extra rejection is reporting the budget rather than the estimator.  Sized against that
#: bound before the run, not after seeing the point estimate.
NULL_REPLICATES = 800
NULL_N = 4_000

#: How far from the exact efficiency bound the sampling spread and the reported standard error
#: may sit.  Two-sided: the bound is a floor, so a ratio *below* it is an understated interval
#: rather than an unusually good estimator.
EFFICIENCY_RATIO_BAND = (0.90, 1.10)
SHRUNKEN_SE_FACTOR = 0.70

#: How far removing the fluctuation must move the estimate, in empirical standard deviations
#: of the targeted arm, before ``targeting_necessity`` will call the targeting step load
#: bearing.  Predeclared as :attr:`Margins.standardized_bias`, and not by coincidence: that is
#: exactly the bias a positive cell is allowed to carry, so the control has to move the
#: estimate by at least as much as the study is prepared to overlook.
TARGETING_DISPLACEMENT = 0.25
TARGETING_REPLICATES = DOUBLE_ROBUST_REPLICATES
TARGETING_N = DOUBLE_ROBUST_N

#: Two-sided Wald critical value at the study's declared size, used by every row this
#: module builds outside the estimator -- the deliberately shrunken and noised calibration
#: controls, and the untargeted arm, which has no interval of its own to copy.
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

REGIMENS = {key: law.REGIMEN_SPEC[key] for key in ("never", "always", "treat_if_l2")}
REFERENCE = "never"
CONTRASTS = {
    "static": "ate_regimen[always vs never]",
    "dynamic": "ate_regimen[treat_if_l2 vs never]",
}
EFFICIENCY_SD = {
    label: float(np.sqrt(np.sum(law.PROBS * law.eif(name) ** 2)))
    for label, name in CONTRASTS.items()
}


def _null_outcome() -> np.ndarray:
    r"""``P(Y = 1 | W, A1, L2, A2, C2 = 1)`` under a sharp null that stays longitudinal.

    The cells the two contrasted plans traverse are overwritten and nothing else is, so the
    null shares this law's treatment, censoring and :math:`L_2` mechanisms exactly and differs
    only in what :math:`Y` answers.  Off-path cells keep their original values, which is what
    leaves :math:`A_2` moving :math:`Y` in the *observed* data while the two regimen means
    coincide.

    **Why not simply make** :math:`Y` **a function of** :math:`W`.  That is what this cell used
    to do, and it made the null degenerate: with
    :math:`P(Y = 1 \mid W, L_2, A_2, C_2 = 1) = 0.25 + 0.5 W` the outcome is independent of
    :math:`(A_1, L_2, A_2, C_1, C_2)` given :math:`W`, so a baseline-only standardisation
    returns *exactly* zero and the type-I cell could not tell a longitudinal fit from one that
    ignored :math:`L_2` altogether.  Here :math:`L_2` moves :math:`Y` from ``0.25`` to ``0.75``,
    censoring is informative through it, and the same baseline-only analysis is biased by
    ``-0.0088``.

    The construction works because :math:`P(L_2 = 1 \mid w, A_1 = 1)` differs from
    :math:`P(L_2 = 1 \mid w, A_1 = 0)`: that is the freedom that lets two :math:`l_2`-varying
    outcome rows average to one number under two different :math:`L_2` laws.  Every value stays
    a multiple of ``1/4``, so the null law is realised exactly by an ``N``-row sample just as
    :data:`law.PROBS` is.

    Both contrasts come out exactly zero, the dynamic one as well as the static.  Only the
    static one is registered, because ``docs/development/method-benchmarking.md`` requires a
    type-I cell to carry a nonzero-effect power control and the dynamic contrast's power at
    ``NULL_N`` is about ``0.43`` against :data:`MINIMUM_POWER` -- it would need ``n`` near
    ``10,000``.  The dynamic null is recorded here as a property of the law rather than
    claimed as a cell.
    """
    outcome = np.array(law.Q, copy=True)
    # [w, a1, l2, a2].  Reading down each column: the "always" path (a1 = a2 = 1), the "never"
    # path (a1 = a2 = 0), and the dynamic path (a1 = 0, a2 = 1{l2 = 1}), whose l2 = 1 arm is
    # the only cell the third plan reaches that the second does not.
    outcome[0, 1, 0, 1], outcome[0, 1, 1, 1] = 0.25, 0.75  # E[Y | W=0, always] = 0.625
    outcome[0, 0, 0, 0], outcome[0, 0, 1, 0] = 0.75, 0.25  # E[Y | W=0, never ] = 0.625
    outcome[0, 0, 1, 1] = 0.25
    outcome[1, 1, 0, 1], outcome[1, 1, 1, 1] = 0.75, 0.25  # E[Y | W=1, always] = 0.375
    outcome[1, 0, 0, 0], outcome[1, 0, 1, 0] = 0.25, 0.50  # E[Y | W=1, never ] = 0.375
    outcome[1, 0, 1, 1] = 0.50
    return outcome


NULL_OUTCOME = _null_outcome()
NULL_PROBS = law.probabilities(NULL_OUTCOME)
NULL_TRUTH = float(law.functional(NULL_PROBS, CONTRASTS["static"]))


def sample(probs: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cells = rng.choice(len(law.SUPPORT), size=n, p=probs)
    return pd.DataFrame(
        {
            name: np.array(
                [
                    np.nan if point[position] is None else float(point[position])
                    for point in law.SUPPORT
                ]
            )[cells]
            for position, name in enumerate(("W", "A1", "C1", "L2", "A2", "C2", "Y"))
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
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
    )


def _plan_arms(frame: pd.DataFrame, label: str) -> tuple[Any, Any]:
    """The two arms ``label`` assigns each row, read off :data:`law.REGIMEN_ARMS` longhand.

    Written from the oracle's table rather than from the callable the estimator is handed, for
    the reason the module's two regimen representations exist at all: a control that asked the
    estimator what it assigned would agree with it by construction.
    """
    node1, node2 = law.REGIMEN_ARMS[label]
    w = frame["W"].to_numpy().astype(int)
    # A unit censored at the first node has no ``L2`` and no second arm to compare against;
    # the follower mask below drops it before either is read, so any in-range filler will do.
    l2 = np.nan_to_num(frame["L2"].to_numpy()).astype(int)
    first = np.full(len(frame), float(node1)) if np.ndim(node1) == 0 else np.asarray(node1)[w]
    second = np.full(len(frame), float(node2)) if np.ndim(node2) == 0 else np.asarray(node2)[w, l2]
    return np.asarray(first, dtype=float), np.asarray(second, dtype=float)


def untargeted(frame: pd.DataFrame, label: str, configuration: str) -> float:
    r"""The sequential-regression plug-in for one regimen, with **no** fluctuation.

    The same backward recursion the estimator runs -- regress the outcome among the units
    that followed the plan and stayed under observation, carry that prediction back as the
    earlier node's pseudo-outcome, regress again, average -- and the same two nuisance
    learners the configuration hands the estimator.  What it leaves out is the one step in
    between: :math:`\bar Q^*_t = \text{expit}(\text{logit}\,\bar Q_t + \epsilon_t)`, solved
    against the cumulative inverse probability.  So the difference between this and
    :func:`fit` is the targeting step and nothing else.

    Written longhand here rather than by calling the estimator with the fluctuation disabled.
    A flag on the estimator would make the control a statement about a branch in the code it
    is supposed to be auditing; this is a second implementation of the plug-in, which is the
    same reason :data:`law.REGIMEN_ARMS` restates :data:`law.REGIMEN_SPEC`.
    """
    outcome, pseudo, _, _ = _learners(configuration)
    first, second = _plan_arms(frame, label)
    followed_one = (frame["C1"].to_numpy() == 1.0) & (frame["A1"].to_numpy() == first)
    followed_two = (
        followed_one & (frame["C2"].to_numpy() == 1.0) & (frame["A2"].to_numpy() == second)
    )
    baseline = frame[["W"]].to_numpy(dtype=float)
    history = np.column_stack([baseline[:, 0], np.nan_to_num(frame["L2"].to_numpy())])
    later = clone(outcome).fit(history[followed_two], frame["Y"].to_numpy()[followed_two])
    carried = np.asarray(later.predict_proba(history))[:, 1]
    earlier = clone(pseudo).fit(baseline[followed_one], carried[followed_one])
    return float(np.mean(earlier.predict(baseline)))


def _fit_replication(
    payload: tuple[str, str, int, int, int, int, str],
) -> list[dict[str, Any]]:
    property_name, cell_suffix, replicate, n, requested, seed, configuration = payload
    probs = NULL_PROBS if property_name == "type_i_error" else law.PROBS
    frame = sample(probs, n, seed)
    result = fit(frame, configuration)
    labels = ("static",) if property_name in {"type_i_error", "power"} else tuple(CONTRASTS)
    rows: list[dict[str, Any]] = []
    for label in labels:
        name = CONTRASTS[label]
        truth = NULL_TRUTH if property_name == "type_i_error" else float(law.TRUTH[name])
        role = (
            "control"
            if cell_suffix == "both_wrong"
            or (property_name == "root_n_and_efficiency" and n == min(RATE_SIZES))
            else "positive"
        )
        rows.append(
            replicate_row(
                property_name=property_name,
                cell=f"{label}__{cell_suffix}",
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
            # The standard error is the *targeted* fit's, which is what ``control_row``
            # documents and why: the plug-in has no influence curve of its own to report.
            left, right = CONTRASTS[label][len("ate_regimen[") : -1].split(" vs ")
            plug_in = untargeted(frame, left, configuration) - untargeted(
                frame, right, configuration
            )
            rows.append(
                control_row(
                    property_name="targeting_necessity",
                    cell=f"{label}__untargeted",
                    replicate=replicate,
                    n=n,
                    requested=requested,
                    truth=truth,
                    estimate=plug_in,
                    standard_error=float(result[name].std_error),
                    critical=CRITICAL,
                )
            )
    return rows


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
            # One spec, four cells: ``_fit_replication`` emits the plug-in beside the estimate
            # for each contrast, off the same draw, so the two arms are paired on replication
            # without a second sample or a shared-seed convention to get wrong.
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


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    outcomes = map_parallel(_fit_replication, _payloads(), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=tuple(CONTRASTS),
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
    """The shared verdicts, plus what this study claims beyond them.

    Double robustness, the size ladder and its small-sample control, the null, the power cell,
    the four rate rows and every calibration interval -- including the efficiency ratios, which
    the shared pass computes once against :data:`EFFICIENCY_SD` -- are
    :func:`~tests.studies.evidence.property_verdicts.apply_shared_verdicts`, exactly as they are
    for every other registered study, and so is the targeting family:
    :func:`~tests.studies.evidence.property_verdicts.necessity_verdicts` states the rule once
    for every study that pairs a positive arm against a step-removed control.  What is left
    here is this study's own declarations -- its labels, its arms, and the displacement
    threshold it fixed before the run -- plus calibration cells that come in three *kinds*
    rather than one.
    """
    margins = STUDY.margins
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("targeting_displacement",),
        rate_labels=tuple(CONTRASTS),
        efficiency_bounds=EFFICIENCY_SD,
    )

    calibration_verdicts(summary, margins=margins, efficiency_band=EFFICIENCY_RATIO_BAND)

    necessity_verdicts(
        summary,
        rows,
        family="targeting_necessity",
        labels=tuple(CONTRASTS),
        arms=("targeted", "untargeted"),
        column="targeting_displacement",
        threshold=TARGETING_DISPLACEMENT,
    )
    return finish(summary, rates)
