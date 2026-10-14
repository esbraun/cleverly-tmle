"""Independent repeated-sampling properties of repeated-outcome LTMLE (X32a).

Every binary family samples :func:`cleverly.datasets.make_longitudinal_repeated` at
``n = 2,000`` on its own stream ``stream_seed(STUDY, "property_sample", family,
configuration, replicate)``.  Every truth is enumerated: the binary truths come from
:func:`cleverly.datasets.repeated_truth`, and :mod:`tests.studies.ltmle_repeated_law` states
the control limits and the continuous law.  The labels name the parameters:
``never_t3`` is ``ey_regimen[never @ t=3]`` and ``rule_vs_never_t3`` is the rule's contrast
at ``t = 3``.  The five ``*_t3`` labels are the gated labels.

Declared before any run
-----------------------

==============================  =========================================================
family and cells                what is fitted, replications and size
==============================  =========================================================
C1 ``interval_calibration``     ``*_t3__estimated_mechanisms``: in sample, the mechanisms
                                estimated by a main-terms logistic GLM on the declared
                                design (correctly specified), quasibinomial GLM outcome
                                regressions.  ``simultaneous=True``.  2,000 replications.
                                The derived ``shrunken_se_control`` and ``noise_control``
                                arms read these rows
``horizon_calibration``         the same fits read at ``t = 1`` and ``t = 2``.  Role
                                ``diagnostic``: reported, never gated
C2 ``double_robustness``        ``*_t3__mechanism_correct``: the C1 mechanisms with an
                                outcome regression on ``W`` alone, which drops every
                                ``Y_{t-1}`` and ``L_t`` term.  ``both_wrong`` (static plans
                                only): that regression with intercept-only mechanisms.
                                1,000 replications
C3 ``interval_calibration``     ``*_t3__crossfit_glm``: C1 at five folds.  2,000
C4 ``simultaneous_coverage``    ``all_reported``: the default band of the C1 fits over all
                                fifteen reported parameters, and its pointwise control
C5 ``interval_calibration``     ``*_t3__crossfit_boosted``: five folds with a shallow
                                regularized booster (depth 2, 50 rounds, rate 0.1, L2
                                penalty 1, at least 20 rows a leaf, no early stopping) for
                                both outcome learners.  2,000
C6 ``interval_calibration``     ``*_t3__continuous_gaussian``: the three-value law
                                :data:`~tests.studies.ltmle_repeated_law.CONTINUOUS`,
                                least-squares outcome regressions, ``q_bounds = (0, 1)``,
                                C1 mechanisms.  2,000
C7 ``rule_necessity``           ``rule_t{2,3}__declared`` against ``__reversed_rule``, the
                                rule that treats after a zero outcome, both read against
                                the declared rule's truth.  1,000
N1 ``repeated_outcome_neces     ``{plan}_t{2,3}__repeated`` against ``__absorbing_reading``,
sity``                          an end-of-study fit of :math:`\\max_{s \\le h} Y_s` with the
                                earlier outcomes in the history.  The control must miss the
                                truth and reach :math:`E[\\max_{s \\le h} Y_s^d]`.  1,000
N2 ``outcome_history_necessi    ``{never,always,always_vs_never}_t3__repeated`` against
ty``                            ``__history_free``, an end-of-study fit with every ``Y_s``
                                left out of the history and saturated cell-mean outcome
                                regressions.  The control must miss the truth and reach the
                                coarse g-formula of
                                :meth:`~tests.studies.ltmle_repeated_law.RepeatedOutcomeLaw.g_formula`.
                                1,000
==============================  =========================================================

Gating, declared before any run:

* the SE-ratio band gates the ``t = 3`` means and contrasts only.  The earlier horizons
  publish their calibration under ``horizon_calibration`` with no verdict;
* every other positive cell answers to its family's shared rule, and each control to the
  opposite direction of the same instrument;
* a red SE-ratio or coverage row is finite-sample only when its SE-ratio 99% interval
  contains 1, its coverage interval is consistent with 0.95, and a diagnosis finds no
  defect.  Such a row then publishes under ``reporting`` with the owner
  ``X32-finite-sample``, and the run is not repeated.  Any other red row is a defect, which
  is fixed before a fresh declaration.

Budgets.  ``CALIBRATION_REPLICATES`` comes from :func:`calibration_pass_probability`:
2,000 is the smallest multiple of 400 at which a cell at exact calibration passes the
two-sided SE-ratio and coverage rule with probability at least 0.90.  No pilot ran, because
a pilot does not fit the fast tier.  :func:`pilot_se_ratios` is the hook: run it before the
declared run, and re-size the calibration budget by the same rule at the measured ratio.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom, norm
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from cleverly.datasets import make_longitudinal_repeated
from cleverly.longitudinal import LTMLE, DynamicRegimen
from cleverly.utils.parallel import map_parallel
from tests.discrete_law_longitudinal import CellMeans
from tests.parallel import STUDY_JOBS
from tests.studies import ltmle_repeated_law as law
from tests.studies.canonical_ltmle_repeated import (
    COLUMNS,
    G_BOUNDS,
    GATED_LABELS,
    REPORTED_LABELS,
    RULE_LABEL,
    STUDY,
    label_estimand,
    treat_after,
)
from tests.studies.evidence.inference import clopper_pearson
from tests.studies.evidence.properties import (
    REPLICATE_COLUMNS,
    replicate_row,
    se_ratio_interval,
)
from tests.studies.evidence.property_verdicts import (
    DIAGNOSTIC_ROLE,
    alternative_target_necessity_verdicts,
    apply_shared_verdicts,
    calibration_controls,
    calibration_verdicts,
    finish,
    necessity_verdicts,
    simultaneous_coverage_verdicts,
)
from tests.studies.evidence.seeds import stream_seed
from tests.studies.evidence.simultaneous import joint_coverage_rows
from tests.studies.fractional_glm import QuasiBinomialGLM

N = 2_000
CALIBRATION_REPLICATES = 2_000
DOUBLE_ROBUST_REPLICATES = 1_000
NECESSITY_REPLICATES = 1_000
FOLDS = 5
SHRUNKEN_SE_FACTOR = 0.70
#: The least paired displacement, in the positive arm's empirical standard deviations, of
#: every necessity family here: the standardized-bias margin.
NECESSITY_DISPLACEMENT = 0.25
#: The red-cell owner of a row the finite-sample rule admits.
FINITE_SAMPLE_OWNER = "X32-finite-sample"
CRITICAL = float(norm.ppf(1.0 - STUDY.margins.alpha / 2.0))

#: The positive calibration arms, in the order the table above declares them.
CALIBRATION_ARMS = (
    "estimated_mechanisms",
    "crossfit_glm",
    "crossfit_boosted",
    "continuous_gaussian",
)
STATIC_GATED = ("never_t3", "always_t3", "always_vs_never_t3")
RULE_LABELS = ("rule_t2", "rule_t3")
ABSORBING_LABELS = ("never_t2", "never_t3", "always_t2", "always_t3", "rule_t2", "rule_t3")
HISTORY_LABELS = STATIC_GATED
JOINT_LABEL = "all_reported"

_PLAN = {"never": "never", "always": "always", RULE_LABEL: law.RULE, "reversed": "reversed"}
REVERSED = "reversed"

# ---------------------------------------------------------------------------- truths

_, TRUTH = make_longitudinal_repeated(n=1, seed=0, backend="pandas")
TRUTH = {name: float(value) for name, value in TRUTH.items()}


def _law_value(model: law.RepeatedOutcomeLaw, estimand: str, **options: Any) -> float:
    """``estimand`` on ``model`` through :meth:`~law.RepeatedOutcomeLaw.g_formula`."""
    if estimand.startswith("ate_regimen["):
        comparison, horizon = estimand[len("ate_regimen[") : -1].rsplit(" @ t=", 1)
        left, right = comparison.split(" vs ")
        return _law_value(model, f"ey_regimen[{left} @ t={horizon}]", **options) - _law_value(
            model, f"ey_regimen[{right} @ t={horizon}]", **options
        )
    plan, horizon = estimand[len("ey_regimen[") : -1].rsplit(" @ t=", 1)
    return model.g_formula(_PLAN[plan], int(horizon), **options)


CONTINUOUS_TRUTH = {
    label: _law_value(law.CONTINUOUS, label_estimand(label)) for label in GATED_LABELS
}
ABSORBING_LIMIT = {
    label: _law_value(law.BINARY, label_estimand(label), absorbing=True)
    for label in ABSORBING_LABELS
}
HISTORY_FREE_LIMIT = {
    label: _law_value(law.BINARY, label_estimand(label), history="free") for label in HISTORY_LABELS
}
BOTH_WRONG_LIMIT = {
    label: _law_value(law.BINARY, label_estimand(label), history="baseline")
    for label in STATIC_GATED
}


def _eif_sd(model: law.RepeatedOutcomeLaw, label: str) -> float:
    estimand = label_estimand(label)
    if estimand.startswith("ate_regimen["):
        comparison, horizon = estimand[len("ate_regimen[") : -1].rsplit(" @ t=", 1)
        left, right = comparison.split(" vs ")
        return model.eif_sd(_PLAN[left], int(horizon), reference=_PLAN[right])
    plan, horizon = estimand[len("ey_regimen[") : -1].rsplit(" @ t=", 1)
    return model.eif_sd(_PLAN[plan], int(horizon))


#: The efficient influence curve's standard deviation of each gated label.  It scales the
#: noise control.  The study claims no efficiency, so no efficiency band reads it.
EFFICIENCY_SD = {label: _eif_sd(law.BINARY, label) for label in GATED_LABELS}

# ---------------------------------------------------------------------------- learners


class BaselineOnly(BaseEstimator):
    """``learner`` fitted on the first design column alone, which is ``W``.

    The misspecified outcome regression of the double-robustness family: it drops every
    ``Y_{t-1}`` and ``L_t`` term.
    """

    def __init__(self, learner: Any) -> None:
        self.learner = learner

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> BaselineOnly:
        self.fitted_ = clone(self.learner).fit(
            np.asarray(X, dtype=float)[:, :1], y, sample_weight=sample_weight
        )
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.fitted_.predict(np.asarray(X, dtype=float)[:, :1]))

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.fitted_.predict_proba(np.asarray(X, dtype=float)[:, :1]))


def booster(kind: str) -> Any:
    """The shallow regularized booster of C5, never the library default."""
    options = {
        "max_depth": 2,
        "max_iter": 50,
        "learning_rate": 0.1,
        "l2_regularization": 1.0,
        "min_samples_leaf": 20,
        "early_stopping": False,
        "random_state": 0,
    }
    if kind == "classifier":
        return HistGradientBoostingClassifier(**options)
    return HistGradientBoostingRegressor(**options)


def _learners(configuration: str) -> tuple[Any, Any, Any, Any]:
    if configuration == "crossfit_boosted":
        q, pseudo = booster("classifier"), booster("regressor")
    elif configuration == "continuous_gaussian":
        q, pseudo = LinearRegression(), LinearRegression()
    elif configuration in {"mechanism_correct", "both_wrong"}:
        q, pseudo = BaselineOnly(QuasiBinomialGLM()), BaselineOnly(QuasiBinomialGLM())
    else:
        q, pseudo = QuasiBinomialGLM(), QuasiBinomialGLM()
    if configuration == "both_wrong":
        g, c = DummyClassifier(strategy="prior"), DummyClassifier(strategy="prior")
    else:
        g, c = QuasiBinomialGLM(), QuasiBinomialGLM()
    return q, pseudo, g, c


def regimens(*, reversed_rule: bool = False) -> dict[str, Any]:
    """The three plans, and the reversed rule for the C7 control."""
    plans: dict[str, Any] = {
        "never": 0,
        "always": 1,
        RULE_LABEL: DynamicRegimen(
            RULE_LABEL, (1, treat_after("Y1"), treat_after("Y2")), rule_kind="known"
        ),
    }
    if reversed_rule:
        plans[REVERSED] = DynamicRegimen(
            REVERSED, (1, treat_zero("Y1"), treat_zero("Y2")), rule_kind="known"
        )
    return plans


def treat_zero(previous: str) -> Callable[[Any], np.ndarray]:
    """The reversed rule's node: treat exactly when the outcome ``previous`` was zero."""

    def node(history: Any) -> np.ndarray:
        return (np.asarray(history[previous], dtype=float) <= 0.0).astype(int)

    return node


def fit(
    frame: pd.DataFrame,
    configuration: str = "estimated_mechanisms",
    *,
    simultaneous: bool = False,
    reversed_rule: bool = False,
) -> Any:
    """The repeated-outcome fit of one configuration."""
    q, pseudo, g, c = _learners(configuration)
    folds = FOLDS if configuration in {"crossfit_glm", "crossfit_boosted"} else 1
    return LTMLE(
        regimens(reversed_rule=reversed_rule),
        reference="never",
        outcome_learner=q,
        pseudo_learner=pseudo,
        treatment_learner=g,
        censoring_learner=c,
        n_folds=folds,
        g_bounds=G_BOUNDS,
        q_bounds=(0.0, 1.0) if configuration == "continuous_gaussian" else None,
        simultaneous=simultaneous,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(frame, **COLUMNS)


def absorbing_fit(frame: pd.DataFrame, horizon: int) -> Any:
    r"""The N1 control: an end-of-study fit of :math:`\max_{s \le h} Y_s` at ``horizon``.

    The earlier outcomes stay in the history, so the mechanisms and the rule read what the
    repeated fit reads.  Only the outcome is the absorbing reading.

    The outcome regressions are saturated cell means rather than the positive arm's GLM.
    ``Y_s = 1`` implies ``M = 1`` for every later regression, so a logistic fit separates
    on every draw by construction.  Every node is binary, so cell means are the exact
    regressions of the absorbing reading, and the fit converges to
    :data:`ABSORBING_LIMIT` through the outcome model as well as through the mechanisms.
    """
    panel = frame.copy()
    outcomes = panel[[f"Y{s}" for s in range(1, horizon + 1)]]
    panel["M"] = outcomes.max(axis=1, skipna=False)
    _, _, g, c = _learners("estimated_mechanisms")
    plans = {
        "never": 0,
        "always": 1,
        RULE_LABEL: DynamicRegimen(
            RULE_LABEL,
            (1, *(treat_after(f"Y{s}") for s in range(1, horizon))),
            rule_kind="known",
        ),
    }
    blocks: list[list[str]] = [[], *([f"Y{s - 1}", f"L{s}"] for s in range(2, horizon + 1))]
    return LTMLE(
        plans,
        reference="never",
        outcome_learner=CellMeans(),
        pseudo_learner=CellMeans(),
        treatment_learner=g,
        censoring_learner=c,
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        panel,
        outcome="M",
        treatment=[f"A{s}" for s in range(1, horizon + 1)],
        baseline=["W"],
        time_varying=blocks,
        censoring=[f"C{s}" for s in range(1, horizon + 1)],
    )


def history_free_fit(frame: pd.DataFrame, horizon: int = 3) -> Any:
    """The N2 control: an end-of-study fit of ``Y_h`` with no ``Y_s`` in any history.

    Saturated cell-mean outcome regressions on ``[W, L_2, ..., L_t]`` are correct for that
    coarser history, so the fit converges to its coarse g-formula whatever the mechanisms.
    """
    _, _, g, c = _learners("estimated_mechanisms")
    return LTMLE(
        {"never": 0, "always": 1},
        reference="never",
        outcome_learner=CellMeans(),
        pseudo_learner=CellMeans(),
        treatment_learner=g,
        censoring_learner=c,
        n_folds=1,
        g_bounds=G_BOUNDS,
        simultaneous=False,
        max_iter=100,
        tol=1e-10,
        random_state=0,
    ).fit(
        frame,
        outcome=f"Y{horizon}",
        treatment=[f"A{s}" for s in range(1, horizon + 1)],
        baseline=["W"],
        time_varying=[[], *([f"L{s}"] for s in range(2, horizon + 1))],
        censoring=[f"C{s}" for s in range(1, horizon + 1)],
    )


def _end_of_study_name(label: str) -> str:
    """The name an end-of-study fit reports for a label: the horizon leaves the bracket."""
    return label_estimand(label).rsplit(" @ t=", 1)[0] + "]"


# ---------------------------------------------------------------------------- rows


def _row(
    family: str, cell: str, role: str, replicate: int, requested: int, truth: float, estimate: Any
) -> dict[str, Any]:
    return replicate_row(
        property_name=family,
        cell=cell,
        role=role,
        replicate=replicate,
        n=N,
        requested=requested,
        truth=truth,
        estimate=estimate,
        alpha=STUDY.margins.alpha,
    )


def sample(configuration: str, seed: int) -> pd.DataFrame:
    """One draw of the law a configuration reads."""
    if configuration == "continuous_gaussian":
        return law.CONTINUOUS.sample(N, seed)
    frame, _ = make_longitudinal_repeated(n=N, seed=seed, censoring=True, backend="pandas")
    return frame


def _replication(payload: tuple[str, str, int, int, int]) -> list[dict[str, Any]]:
    family, configuration, replicate, requested, seed = payload
    frame = sample(configuration, seed)
    rows: list[dict[str, Any]] = []
    if family == "interval_calibration":
        joint = configuration == "estimated_mechanisms"
        result = fit(frame, configuration, simultaneous=joint)
        truths = CONTINUOUS_TRUTH if configuration == "continuous_gaussian" else None
        for label in GATED_LABELS:
            name = label_estimand(label)
            truth = TRUTH[name] if truths is None else truths[label]
            rows.append(
                _row(
                    family,
                    f"{label}__{configuration}",
                    "positive",
                    replicate,
                    requested,
                    truth,
                    result[name],
                )
            )
        if joint:
            for label in REPORTED_LABELS:
                name = label_estimand(label)
                rows.append(
                    _row(
                        "horizon_calibration",
                        f"{label}__{configuration}",
                        DIAGNOSTIC_ROLE,
                        replicate,
                        requested,
                        TRUTH[name],
                        result[name],
                    )
                )
            rows.extend(
                joint_coverage_rows(
                    result,
                    {name: TRUTH[name] for name in result.estimates},
                    tuple(result.estimates),
                    label=JOINT_LABEL,
                    replicate=replicate,
                    n=N,
                    requested=requested,
                    pointwise_critical=CRITICAL,
                )
            )
        return rows
    if family == "double_robustness":
        result = fit(frame, configuration)
        labels = STATIC_GATED if configuration == "both_wrong" else GATED_LABELS
        role = "control" if configuration == "both_wrong" else "positive"
        for label in labels:
            name = label_estimand(label)
            rows.append(
                _row(
                    family,
                    f"{label}__{configuration}",
                    role,
                    replicate,
                    requested,
                    TRUTH[name],
                    result[name],
                )
            )
        return rows
    if family == "rule_necessity":
        result = fit(frame, reversed_rule=True)
        for label in RULE_LABELS:
            name = label_estimand(label)
            reversed_name = name.replace(RULE_LABEL, REVERSED)
            rows.append(
                _row(
                    family,
                    f"{label}__declared",
                    "positive",
                    replicate,
                    requested,
                    TRUTH[name],
                    result[name],
                )
            )
            rows.append(
                _row(
                    family,
                    f"{label}__reversed_rule",
                    "control",
                    replicate,
                    requested,
                    TRUTH[name],
                    result[reversed_name],
                )
            )
        return rows
    if family == "repeated_outcome_necessity":
        result = fit(frame)
        controls = {horizon: absorbing_fit(frame, horizon) for horizon in (2, 3)}
        for label in ABSORBING_LABELS:
            name = label_estimand(label)
            horizon = int(label.rsplit("_t", 1)[1])
            rows.append(
                _row(
                    family,
                    f"{label}__repeated",
                    "positive",
                    replicate,
                    requested,
                    TRUTH[name],
                    result[name],
                )
            )
            rows.append(
                _row(
                    family,
                    f"{label}__absorbing_reading",
                    "control",
                    replicate,
                    requested,
                    TRUTH[name],
                    controls[horizon][_end_of_study_name(label)],
                )
            )
        return rows
    if family == "outcome_history_necessity":
        result = fit(frame)
        control = history_free_fit(frame)
        for label in HISTORY_LABELS:
            name = label_estimand(label)
            rows.append(
                _row(
                    family,
                    f"{label}__repeated",
                    "positive",
                    replicate,
                    requested,
                    TRUTH[name],
                    result[name],
                )
            )
            rows.append(
                _row(
                    family,
                    f"{label}__history_free",
                    "control",
                    replicate,
                    requested,
                    TRUTH[name],
                    control[_end_of_study_name(label)],
                )
            )
        return rows
    raise KeyError(family)


#: Every fit set: ``(family, configuration, replications)``.
FIT_SETS: tuple[tuple[str, str, int], ...] = (
    *(("interval_calibration", arm, CALIBRATION_REPLICATES) for arm in CALIBRATION_ARMS),
    ("double_robustness", "mechanism_correct", DOUBLE_ROBUST_REPLICATES),
    ("double_robustness", "both_wrong", DOUBLE_ROBUST_REPLICATES),
    ("rule_necessity", "paired", NECESSITY_REPLICATES),
    ("repeated_outcome_necessity", "paired", NECESSITY_REPLICATES),
    ("outcome_history_necessity", "paired", NECESSITY_REPLICATES),
)


def _payloads(
    fit_sets: tuple[tuple[str, str, int], ...] = FIT_SETS,
) -> list[tuple[tuple[str, str, int, int, int]]]:
    payloads: list[tuple[tuple[str, str, int, int, int]]] = []
    for family, configuration, replicates in fit_sets:
        for replicate in range(replicates):
            seed = stream_seed(STUDY, "property_sample", family, configuration, replicate)
            payloads.append(((family, configuration, replicate, replicates, seed),))
    return payloads


def generate_property_rows(*, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    outcomes = map_parallel(_replication, _payloads(), n_jobs=n_jobs)
    rows = pd.DataFrame([row for result in outcomes for row in result])
    rows = pd.concat(
        [
            rows,
            calibration_controls(
                rows,
                STUDY,
                labels=GATED_LABELS,
                efficiency_bounds=EFFICIENCY_SD,
                calibration_n=N,
                shrunken_se_factor=SHRUNKEN_SE_FACTOR,
                critical=CRITICAL,
                positive_suffix="estimated_mechanisms",
            ),
        ],
        ignore_index=True,
    )
    return rows.loc[:, list(REPLICATE_COLUMNS)].sort_values(
        ["property", "cell", "replicate"], ignore_index=True
    )


# ---------------------------------------------------------------------------- verdicts


def _horizon_report(summary: pd.DataFrame, rows: pd.DataFrame) -> None:
    """Publish the earlier horizons' SE-ratio interval beside their coverage, gated by nothing."""
    margins = STUDY.margins
    family = summary["property"] == "horizon_calibration"
    for index in summary.index[family]:
        cell = str(summary.loc[index, "cell"])
        group = rows.loc[(rows["property"] == "horizon_calibration") & (rows["cell"] == cell)]
        interval = se_ratio_interval(
            group,
            replicates=margins.bootstrap_replicates,
            confidence_level=margins.confidence_level,
            seed=stream_seed(STUDY, "horizon_calibration", cell),
        )
        summary.loc[index, "se_ratio_ci_lower"] = interval.low
        summary.loc[index, "se_ratio_ci_upper"] = interval.high
    summary.loc[family, "passed"] = True
    summary.loc[family, "property_passed"] = True


def summarize_properties(rows: pd.DataFrame) -> pd.DataFrame:
    summary, rates = apply_shared_verdicts(
        rows,
        STUDY,
        extra_columns=("necessity_displacement",),
        rate_labels=(),
    )
    margins = STUDY.margins

    # The C1 arm and its two derived controls read the instrument's rule for each kind.  The
    # other positive arms keep the two-sided rule ``apply_shared_verdicts`` wrote, which is
    # the same rule.
    primary = (summary["property"] == "interval_calibration") & ~summary["cell"].str.endswith(
        tuple(f"__{arm}" for arm in CALIBRATION_ARMS[1:])
    )
    subset = summary.loc[primary].copy()
    calibration_verdicts(subset, margins=margins, positive_suffix="estimated_mechanisms")
    summary.loc[subset.index, "passed"] = subset["passed"]

    _horizon_report(summary, rows)
    simultaneous_coverage_verdicts(summary, margins=margins)
    necessity_verdicts(
        summary,
        rows,
        family="rule_necessity",
        labels=RULE_LABELS,
        arms=("declared", "reversed_rule"),
        column="necessity_displacement",
        threshold=NECESSITY_DISPLACEMENT,
    )
    alternative_target_necessity_verdicts(
        summary,
        rows,
        STUDY,
        family="repeated_outcome_necessity",
        labels=ABSORBING_LABELS,
        arms=("repeated", "absorbing_reading"),
        alternative_truths=ABSORBING_LIMIT,
        column="necessity_displacement",
        threshold=NECESSITY_DISPLACEMENT,
    )
    alternative_target_necessity_verdicts(
        summary,
        rows,
        STUDY,
        family="outcome_history_necessity",
        labels=HISTORY_LABELS,
        arms=("repeated", "history_free"),
        alternative_truths=HISTORY_FREE_LIMIT,
        column="necessity_displacement",
        threshold=NECESSITY_DISPLACEMENT,
    )
    return finish(summary, rates)


# ---------------------------------------------------------------------------- design


def calibration_pass_probability(
    replicates: int, *, ratio: float = 1.0, coverage: float = 0.95
) -> float:
    """The probability that a calibration cell passes the two-sided rule.

    The SE-ratio half reads a normal approximation: the empirical SD of ``R`` normal
    estimates has relative standard deviation ``1 / sqrt(2 (R - 1))``, and the bootstrap 99%
    interval has half-width ``z_{0.995}`` of those.  The mean reported SE is treated as exact.
    The coverage half is exact: it sums the binomial law of the covered count over the counts
    whose Clopper-Pearson interval lies inside the band.  The two halves are independent under
    this approximation, so the probability is their product.
    """
    margins = STUDY.margins
    low, high = margins.calibration_se_ratio
    z = float(norm.ppf(0.5 + margins.confidence_level / 2.0))
    spread = ratio / np.sqrt(2.0 * (replicates - 1))
    half = z * spread
    se_pass = float(
        norm.cdf((high - half - ratio) / spread) - norm.cdf((low + half - ratio) / spread)
    )
    band_low, band_high = margins.calibration_coverage
    counts = np.arange(replicates + 1)
    inside = np.zeros(replicates + 1, dtype=bool)
    for count in counts:
        interval = clopper_pearson(
            int(count), replicates, confidence_level=margins.confidence_level
        )
        inside[count] = interval.low >= band_low and interval.high <= band_high
    coverage_pass = float(np.sum(binom.pmf(counts[inside], replicates, coverage)))
    return max(se_pass, 0.0) * coverage_pass


def _pilot_replication(payload: tuple[str, int]) -> dict[str, tuple[float, float]]:
    configuration, replicate = payload
    frame = sample(configuration, stream_seed(STUDY, "pilot", configuration, replicate))
    result = fit(frame, configuration)
    return {
        label: (
            float(result[label_estimand(label)].psi),
            float(result[label_estimand(label)].std_error),
        )
        for label in GATED_LABELS
    }


def pilot_se_ratios(
    replicates: int = 200, *, configuration: str = "estimated_mechanisms", n_jobs: int = STUDY_JOBS
) -> dict[str, float]:
    """PILOT HOOK: the SE ratio of each gated label on ``replicates`` pilot draws.

    Not run by the fast tier.  Run it before the declared run on the stream
    ``stream_seed(STUDY, "pilot", configuration, replicate)``, which no published cell reads,
    and re-size :data:`CALIBRATION_REPLICATES` with :func:`calibration_pass_probability` at
    the measured ratio.  Record the pilot and the new budget in the declaration commit.
    """

    payloads = [((configuration, replicate),) for replicate in range(replicates)]
    draws = map_parallel(_pilot_replication, payloads, n_jobs=n_jobs)
    ratios: dict[str, float] = {}
    for label in GATED_LABELS:
        estimates = np.array([draw[label][0] for draw in draws])
        errors = np.array([draw[label][1] for draw in draws])
        ratios[label] = float(np.mean(errors) / np.std(estimates, ddof=1))
    return ratios


def time_fit_sets(seed: int = 0) -> dict[tuple[str, str], float]:
    """Seconds for one replication of each fit set, single process, for the budget table."""
    seconds: dict[tuple[str, str], float] = {}
    for family, configuration, _ in FIT_SETS:
        start = time.perf_counter()
        _replication((family, configuration, 0, 1, seed))
        seconds[family, configuration] = time.perf_counter() - start
    return seconds


def declared_truths() -> Mapping[str, Mapping[str, float]]:
    """Every truth and control limit a cell reads, by kind."""
    return {
        "binary": TRUTH,
        "continuous": CONTINUOUS_TRUTH,
        "absorbing": ABSORBING_LIMIT,
        "history_free": HISTORY_FREE_LIMIT,
        "both_wrong": BOTH_WRONG_LIMIT,
    }
