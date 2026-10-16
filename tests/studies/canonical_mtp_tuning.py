"""Registered evidence for tuned caps and their two targets on law Q (``mtp-tuning``).

The subject is ``Shift(1.0, cap=c(h))`` at both nodes of law Q, whose cap is tuned on the
data: a dose quantile (``DoseQuantile(0.5)``) or a linear quantile regression of the dose
(``SupportBoundary(LinearQuantileRegression(...), 0.5)``).  The population target (X40 plan
section 2.5) adds the tuning term :math:`\\sum_t \\partial_t\\Phi' \\mathrm{IF}_t` to the curve;
the fold-average target (section 2.4) reads the mean of the fold maps' values, a truth that
varies by replication.  The law and its quadrature truths are :mod:`tests.law_q_tuned`; this
module holds the oracle density, the declared learners and the record.  Law Q has no
comparator: no other implementation fits a tuned cap's tuning term.

=============================  ==========================================================
scenario                       subject
=============================  ==========================================================
``population_quantile_ltmle``  ``LTMLE`` at one fold, population target, ``DoseQuantile``
                               caps, the oracle density on the density route
``population_quantile_sdr``    the same draw, ``SDR``
=============================  ==========================================================

1,000 primary replications at n = 2,000: the primary phase exists so that the record has a
primary table.  The first declaration's 200 replications (cleverly-tmle fb70e6ef) could not pass
the bias-equivalence and coverage-floor gates at any plausible bias: both rows read red with
coverage 0.935 (99% interval 0.876 to 0.972) and SE ratio 0.92 (0.82 to 1.06).  The second
declaration (2026-10-16) sizes the primary at 1,000, which those gates can resolve.  The claims are the property cells of :mod:`tests.studies.mtp_tuning_properties`.

**Red-cell reading, declared before any run.**  As ``mtp-boundaries``, with the owner row
``mtp-tuning-calibration``.  Publication policy is ``reporting``.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, clone

from cleverly.interventions import DoseQuantile, LinearQuantileRegression, Shift, SupportBoundary
from cleverly.learners.density import bin_edges
from cleverly.longitudinal import LTMLE, SDR, DynamicRegimen
from tests import law_q_tuned as lawq
from tests.parallel import STUDY_JOBS
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.fractional_glm import QuasiBinomialGLM
from tests.studies.mtp_pricing_common import restricted_cubic
from tests.studies.oracle_density_bins import oracle_bins

PRIMARY_REPLICATES = 1_000
PRIMARY_N = 2_000
SEED = 20264031
RESAMPLING_SEED = 2026403101
G_BOUNDS = (0.001, 1.0)
Q_BOUNDS = (0.0, 1.0)
FAILURE_BOUND = 0.01
LABEL = "tuned_cap"

COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "continuous_treatment": ["A1", "A2"],
}

#: The tuner of each kind at nodes 1 and 2, in the oracle's terms.
TUNERS: dict[str, tuple[lawq.Tuner, lawq.Tuner]] = {
    "dose_quantile": (lawq.Tuner(), lawq.Tuner()),
    "quantile_regression": (lawq.Tuner(features=("W",)), lawq.Tuner(features=("W", "L2", "A1"))),
}


#: The declared kernel bandwidth of each kind, by the rule of X40 plan section 2.5 (design
#: review M6), read off :func:`tests.studies.mtp_tuning_properties.kernel_biases` at
#: n = 2,000.  At Silverman's bandwidth the dose quantile's ``f_1`` and ``f_2`` carry relative
#: biases of 1.86% and 1.79%, above the 1% ceiling, and undersmoothed every bias is at most
#: 0.48%.  The regression's largest bias at Silverman's bandwidth is 0.085% (``J_1``).
#: ``tests/unit/test_mtp_pricing_design.py`` recomputes the rule.
BANDWIDTH: dict[str, Any] = {"dose_quantile": "undersmoothed", "quantile_regression": "silverman"}


def caps(kind: str) -> tuple[Any, Any]:
    rule = BANDWIDTH[kind]
    if kind == "dose_quantile":
        return DoseQuantile(0.5, bandwidth=rule), DoseQuantile(0.5, bandwidth=rule)
    if kind == "quantile_regression":
        return (
            SupportBoundary(LinearQuantileRegression(["W"]), 0.5, bandwidth=rule),
            SupportBoundary(LinearQuantileRegression(["W", "L2", "A1"]), 0.5, bandwidth=rule),
        )
    raise KeyError(kind)


def regimen(kind: str, target: str) -> DynamicRegimen:
    first, second = caps(kind)
    return DynamicRegimen(
        LABEL, (Shift(1.0, cap=first, target=target), Shift(1.0, cap=second, target=target))
    )


@cache
def law(kind: str) -> lawq.LawQ:
    return lawq.LawQ(tuners=TUNERS[kind])


@cache
def population_truth(kind: str) -> float:
    """:math:`\\Psi_0 = \\Phi(\\theta_0)` by quadrature."""
    return float(law(kind).tuned_psi)


def fold_average_truth(result: Any, kind: str) -> float:
    """The mean over folds of :math:`\\Phi(\\theta^{(-k)})`, at each fold's fitted tuner."""
    (plan,) = [p for p in result.replay_recipe.plans if p.label.split(",")[0] == LABEL]
    by_node = {tuning.time: tuning for tuning in plan.tuning}
    first, second = by_node[1].fold_fits, by_node[2].fold_fits
    if not first or len(first) != len(second):
        raise ValueError("the fold-average truth needs every fold's tuner at both nodes")
    values = [
        law(kind).psi(theta=(np.asarray(a.theta, dtype=float), np.asarray(b.theta, dtype=float)))
        for a, b in zip(first, second, strict=True)
    ]
    return float(np.mean(values))


# ---------------------------------------------------------------------------- learners


def _tn_cdf(x: Any, mean: Any) -> np.ndarray:
    lo, hi = lawq.A_RANGE
    low = norm.cdf(lo - mean)
    return np.asarray((norm.cdf(np.clip(x, lo, hi) - mean) - low) / (norm.cdf(hi - mean) - low))


class OracleHazardQ(BaseEstimator):
    """The exact pooled-hazard probabilities of law Q's doses (``bin_design = "index"``).

    The history is ``[W]`` at node 1 and ``[W, L2, A1]`` at node 2, so a design of width 2
    is node 1's hazard and one of width 4 is node 2's.

    Parameters
    ----------
    edges1 : tuple of float
        The bin edges of node 1.
    edges2 : tuple of float
        The bin edges of node 2.
    """

    bin_design = "index"

    def __init__(self, edges1: tuple[float, ...] = (), edges2: tuple[float, ...] = ()) -> None:
        self.edges1 = edges1
        self.edges2 = edges2

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OracleHazardQ:
        del X, y, sample_weight
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        history, index = design[:, :-1], np.rint(design[:, -1]).astype(int)
        if design.shape[1] == 2:
            edges, mean = np.asarray(self.edges1, dtype=float), 0.5 * history[:, 0]
        elif design.shape[1] == 4:
            edges = np.asarray(self.edges2, dtype=float)
            mean = 0.4 * history[:, 2] + 0.6 * history[:, 1] + 0.3 * history[:, 0]
        else:
            raise ValueError(f"unexpected oracle design width {design.shape[1]}")
        last = len(edges) - 2
        lower = np.where(index == 0, -np.inf, edges[index])
        top = np.where(index == last, np.inf, edges[np.minimum(index + 1, last + 1)])
        below = _tn_cdf(lower, mean)
        survived = 1.0 - below
        hazard = np.divide(
            _tn_cdf(top, mean) - below, survived, out=np.ones_like(below), where=survived > 1e-15
        )
        hazard = np.clip(hazard, 1e-12, 1.0 - 1e-12)
        return np.column_stack([1.0 - hazard, hazard])


def oracle_density(frame: pd.DataFrame) -> OracleHazardQ:
    bins = oracle_bins(len(frame))
    return OracleHazardQ(
        tuple(float(v) for v in bin_edges(frame["A1"].to_numpy(dtype=float), bins)),
        tuple(float(v) for v in bin_edges(frame["A2"].to_numpy(dtype=float), bins)),
    )


class SplineFirst(BaseEstimator):
    """The node-1 outcome regression of law Q: a quasibinomial GLM on splines of ``A1`` and
    ``W`` (5 knots each, at the training quantiles 0.05 to 0.95) and their product.

    The design is ``[W, A1]``.
    """

    def _basis(self, X: Any, fit: bool) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if fit:
            self.knots_ = [np.quantile(x[:, j], (0.05, 0.275, 0.5, 0.725, 0.95)) for j in range(2)]
        splines = [restricted_cubic(x[:, j], self.knots_[j]) for j in range(2)]
        return np.column_stack([*splines, x[:, 0] * x[:, 1]])

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SplineFirst:
        self.model_ = QuasiBinomialGLM().fit(
            self._basis(X, True), np.asarray(y, dtype=float), sample_weight=sample_weight
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._basis(X, False)), dtype=float)


def classifier() -> Any:
    """The declared classifier of the cross-fitted classifier cell: a shallow boosted model.

    Provisional until the law-Q screen (:func:`tests.studies.mtp_tuning_properties.
    learner_screen`) runs; a smooth logistic cannot carry the ratio's jumps at the cap.
    """
    from tests.studies.mtp_pricing_common import boosted

    return boosted(2)


def fit(
    frame: pd.DataFrame,
    kind: str,
    target: str,
    *,
    estimator: str = "ltmle",
    n_folds: int = 1,
    ratio: str = "density",
    treatment_learner: Any = None,
    random_state: int = 0,
) -> Any:
    """``LTMLE`` or ``SDR`` of the tuned plan on law Q with the declared outcome learners."""
    settings: dict[str, Any] = {
        "q_bounds": Q_BOUNDS,
        "outcome_learner": QuasiBinomialGLM(),
        "pseudo_learner": SplineFirst(),
        "treatment_learner": clone(
            oracle_density(frame) if treatment_learner is None else treatment_learner
        ),
        "n_folds": n_folds,
        "g_bounds": G_BOUNDS,
        "ratio": ratio,
        "simultaneous": False,
        "random_state": random_state,
    }
    if ratio == "density":
        settings["density_bins"] = oracle_bins(len(frame))
    if estimator == "ltmle":
        model: Any = LTMLE([regimen(kind, target)], max_iter=100, tol=1e-10, **settings)
    elif estimator == "sdr":
        model = SDR([regimen(kind, target)], **settings)
    else:
        raise ValueError(estimator)
    return model.fit(frame, **COLUMNS)


# ---------------------------------------------------------------------------- record

SCENARIOS: dict[str, str] = {
    "population_quantile_ltmle": "ltmle",
    "population_quantile_sdr": "sdr",
}
ESTIMAND = f"ey_regimen[{LABEL}]"
ESTIMANDS: dict[str, tuple[str, ...]] = dict.fromkeys(SCENARIOS, (ESTIMAND,))

CALIBRATION_LABELS = (
    "population_dose_quantile_tau2",
    "population_quantile_regression_tau2",
    "population_dose_quantile_crossfit_tau2",
    "population_dose_quantile_crossfit_classifier_tau2",
    "fold_average_dose_quantile_tau2",
    "fold_average_quantile_regression_tau2",
)
#: Each calibration label's derived control beyond the shared two.
EXTRA_CONTROLS: dict[str, str] = {
    **{
        label: "term_dropped_control"
        for label in CALIBRATION_LABELS
        if label.startswith("population_")
    },
    **{
        label: "population_truth_control"
        for label in CALIBRATION_LABELS
        if label.startswith("fold_average_")
    },
}
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{cell}"
        for label in CALIBRATION_LABELS
        for cell in (
            "correctly_specified",
            "shrunken_se_control",
            "noise_control",
            EXTRA_CONTROLS[label],
        )
    ),
}

STUDY = StudyRecord(
    name="tuned caps and their population and fold-average targets",
    slug="mtp-tuning",
    artifacts=ROOT / "tests" / "canonical" / "mtp_tuning",
    document="docs/technical-reference/method-evidence/mtp-pricing.md",
    anchor="mtp-tuning",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    scenario_seed_owners={"population_quantile_sdr": "population_quantile_ltmle"},
    margins=Margins(),
    implementation="cleverly-mtp-tuning",
    modules=(
        "tests/studies/canonical_mtp_tuning.py",
        "tests/studies/mtp_tuning_properties.py",
        "tests/studies/mtp_pricing_common.py",
        "tests/studies/mtp_study_tools.py",
        "tests/discrete_law_longitudinal.py",
        "tests/discrete_law_longitudinal_multivalue.py",
        "tests/studies/canonical_categorical_ltmle.py",
        "tests/studies/canonical_ltmle.py",
        "tests/studies/canonical_mtp_pricing.py",
        "tests/studies/categorical_longitudinal_common.py",
        "tests/studies/mtp_pricing_properties.py",
        "tests/law_q_tuned.py",
        "tests/studies/oracle_density_bins.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_mtp_tuning",
    properties_module="tests.studies.mtp_tuning_properties",
    property_cells=PROPERTY_CELLS,
    calibration_efficiency_ratio=False,
    publication_policy="reporting",
    truth_varies_by_replicate=True,
    failure_bound={"cleverly-mtp-tuning": FAILURE_BOUND},
)

CONFIGURATION = {
    "construction": "ordinary, one fold",
    "density_bins": "oracle_bins(n): 320 at n = 2,000, index-only oracle design",
    "g_bounds": list(G_BOUNDS),
    "q_bounds": list(Q_BOUNDS),
    "outcome_learner": "QuasiBinomialGLM at node 2; SplineFirst at node 1",
    "treatment_learner": "OracleHazardQ (exact pooled-hazard bins)",
    "plan": (
        "Shift(1.0, cap=DoseQuantile(0.5, bandwidth='undersmoothed'), target='population') "
        "at both nodes"
    ),
}


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return lawq.sample(n, seed), {ESTIMAND: population_truth("dose_quantile")}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    estimate = result[ESTIMAND]
    low, high = estimate.ci
    reference = float(truth[ESTIMAND])
    first = result.fits[LABEL].steps[0]
    return [
        {
            "implementation": STUDY.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": result.n,
            "estimand": ESTIMAND,
            "truth": reference,
            "estimate": float(estimate.psi),
            "inference_estimate": float(estimate.psi),
            "std_error": float(estimate.std_error),
            "ci_lower": float(low),
            "ci_upper": float(high),
            "inference_scale": "identity",
            "covered": int(low <= reference <= high),
            "initial_estimate": float(np.mean(first.initial)),
        }
    ]


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, _ = key
    frame, truth = drawn
    result = fit(frame, "dose_quantile", "population", estimator=SCENARIOS[scenario])
    truths = [
        {
            "scenario": scenario,
            "replicate": replicate,
            "estimand": ESTIMAND,
            "truth": truth[ESTIMAND],
        }
    ]
    return truths, result_rows(result, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for replicate in range(replicates) for scenario in SCENARIOS],
        _replicate,
        draw=_draw,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        rows_at=1,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    truths = pd.DataFrame([row for rows, _ in outcomes.kept for row in rows])
    estimates = pd.DataFrame([row for _, rows in outcomes.kept for row in rows])
    return replication.result(outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)], truths=truths)


__all__ = ["STUDY", "draw_and_fit", "fit"]
