r"""Registered protocol for the canonical DR-TMLE comparison.

The law is the complete-data binary law in Benkeser et al. (2017).  The two
implementations receive the same realized rows and fold assignment.  Each side fits the
same declared GLM nuisance class; the comparison is about the corrected construction, not
about two unrelated learner libraries.

This module is intentionally importable without R or Docker.  The container is used only by
``tests/canonical/drtmle/regenerate.py`` when the registered evidence is regenerated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import quad
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold

from cleverly.data import CausalData
from cleverly.estimators import DRTMLE
from cleverly.learners.crossfit import Folds, check_integrity
from cleverly.utils.bounds import expit
from cleverly.utils.parallel import map_parallel
from cleverly.validation.score import DEFAULT_TOLERANCE, score_threshold
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.seeds import draw_replicate

DRTMLE_COMMIT = "538a3a264c1ca984b6d88978ca7f96165f43152c"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
PRIMARY_REPLICATES = 800
PRIMARY_N = 3000
SEED = 20260824
G_BOUNDS = (0.01, 0.99)
N_FOLDS = 10
#: Rounds of the three-equation alternation, matching the ``maxIter = 100`` the R runner
#: passes.  It is declared here because it is now reachable: the alternation used to run at a
#: hard-coded 50 while this study's manifest published ``max_iter: 100``, which is the
#: *inner* Newton cap and was never the loop's.
MAX_OUTER = 100
SCENARIOS = ("outcome_correct", "treatment_correct", "both_correct")
ESTIMANDS = ("ey0", "ey1", "ate")
#: The audit bar, applied identically to both implementations.
#:
#: **The library's own criterion and not a study-local number.**  This was ``1e-4``, defended
#: as stricter than the canonical package's ``1 / n`` default -- true at n = 3000, and the
#: wrong reference point for a claim about Cleverly, because
#: :func:`~cleverly.validation.score.score_threshold` ships ``1e-3 * se / sqrt(n)``, which is
#: about ``3.4e-7`` here.  Auditing at ``1e-4`` meant the study could not detect a score
#: failure the library itself reports on a user's fit.
#:
#: The *rule* is imported rather than restated, so the two sides are judged by one function
#: and the study cannot drift from the library one release later.  Only the tolerance is named
#: here, and naming it is what lets the manifest say which bar ran.
SCORE_AUDIT_TOLERANCE = DEFAULT_TOLERANCE

#: What each implementation writes on its own replicate rows.
FIT_DIAGNOSTIC_SOURCE_COLUMNS = (
    "implementation",
    "scenario",
    "replicate",
    "n",
    "score_max",
    "solver_reported",
    "solver_passed",
    "bound_active",
)

#: The published order.  ``score_threshold`` and ``score_passed`` are absent above because
#: :func:`extra_artifacts` derives them for **both** sides from one rule, which is the point:
#: a column each implementation filled in for itself is how the audit came to hold two
#: different quantities.
FIT_DIAGNOSTIC_COLUMNS = (
    *FIT_DIAGNOSTIC_SOURCE_COLUMNS[:5],
    "score_threshold",
    "score_passed",
    *FIT_DIAGNOSTIC_SOURCE_COLUMNS[5:],
)

STUDY = StudyRecord(
    name="DR-TMLE for binary complete data",
    slug="canonical-drtmle",
    artifacts=ROOT / "tests" / "canonical" / "drtmle",
    document="docs/technical-reference/method-evidence/canonical-dr-tmle.md",
    anchor="canonical-dr-tmle",
    scenarios=dict.fromkeys(SCENARIOS, ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=20260826,
    margins=Margins(),
    implementation="cleverly",
    reference="drtmle-r",
    publication_policy="reporting",
    extra_artifacts=("fit-diagnostics.csv",),
    modules=(
        "tests/studies/canonical_drtmle.py",
        "tests/studies/drtmle_properties.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_drtmle",
    properties_module="tests.studies.drtmle_properties",
    property_cells={
        "double_robustness": (
            "both_correct",
            "outcome_correct",
            "treatment_correct",
            "both_wrong",
        ),
        "double_robust_contraction": (
            "both_wrong_n1500",
            "both_wrong_n3000",
            "both_wrong_n6000",
            "outcome_correct_n1500",
            "outcome_correct_n3000",
            "outcome_correct_n6000",
            "rate_both_wrong",
            "rate_outcome_correct",
            "rate_treatment_correct",
            "treatment_correct_n1500",
            "treatment_correct_n3000",
            "treatment_correct_n6000",
        ),
        "root_n_and_efficiency": ("n_500", "n_1500", "n_4500"),
        "root_n_rate": ("empirical_sd", "reported_se"),
        "interval_calibration": ("correctly_specified",),
    },
)

REFERENCE_METADATA = {
    "drtmle_commit": DRTMLE_COMMIT,
    "drtmle_version": "1.1.2",
    "r_base_image": R_BASE_IMAGE,
}

CONFIGURATION = {
    "source_law": "Benkeser et al. (2017), Section 4",
    "cross_fit": True,
    "n_folds": N_FOLDS,
    "targeting_scheme": "pooled",
    "reduced_crossfit": "pooled",
    "reduction": "univariate",
    "guard": ["Q", "g"],
    "update_order": "drtmle",
    "qsteps": 2,
    # Two caps, named apart, because publishing one number for both is what this study did.
    # `max_outer` is the alternation's and matches the R runner's `maxIter`; `max_iter` is
    # the Newton cap inside one fluctuation and has no counterpart in the R package.
    "max_outer": MAX_OUTER,
    "max_iter": 100,
    "score_audit": "score_check's own bar, 1e-3 x se / sqrt(n), applied to both implementations",
    "g_bounds": list(G_BOUNDS),
    "nuisance_models": {
        "correct": "unpenalized logistic GLM with W1:W2",
        "misspecified": "unpenalized main-effects logistic GLM",
        "reduced_Q": "Gaussian GLM",
        "reduced_g": "binomial GLM",
    },
}


class ColumnLogistic(BaseEstimator, ClassifierMixin):
    """Unpenalized logistic regression on a fixed subset of design columns."""

    def __init__(self, columns: Sequence[int] | None = None) -> None:
        self.columns = columns

    def _select(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        if self.columns is None:
            return values
        return values[:, list(self.columns)]

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> ColumnLogistic:
        self.model_ = LogisticRegression(
            C=np.inf,
            max_iter=5000,
            solver="newton-cholesky",
            tol=1e-10,
            random_state=0,
        ).fit(self._select(design), target, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        return np.asarray(self.model_.predict_proba(self._select(design)), dtype=float)


class FixedFoldDRTMLE(DRTMLE):
    """Study-only DR-TMLE whose outer fold assignment is supplied by the sample."""

    def __init__(self, assignment: np.ndarray, **kwargs: Any) -> None:
        self._assignment = np.asarray(assignment, dtype=np.int64)
        super().__init__(**kwargs)

    def _folds(self, data: CausalData, seed: int | None = None) -> Folds:
        del seed
        if data.n != self._assignment.size:
            raise ValueError(
                f"fold assignment has {self._assignment.size} rows for a {data.n}-row fit"
            )
        folds = Folds(self._assignment.copy(), int(self._assignment.max()) + 1)
        check_integrity(folds, cluster=data.cluster)
        return folds


def _linear_predictor(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    return -w1 + 2.0 * w1 * w2


def truth() -> dict[str, float]:
    """Independent quadrature truth for the paper law."""

    def arm_mean(arm: float) -> float:
        def integrate(w1: float) -> float:
            return 0.5 * (float(expit(0.2 * arm - w1)) + float(expit(0.2 * arm + w1)))

        value, error = quad(integrate, -2.0, 2.0, epsabs=1e-13, epsrel=1e-13, limit=200)
        if error > 1e-11:
            raise RuntimeError(f"paper-law quadrature error {error:g} exceeds its audit bar")
        return value / 4.0

    ey0, ey1 = arm_mean(0.0), arm_mean(1.0)
    return {"ey0": ey0, "ey1": ey1, "ate": ey1 - ey0}


def fixed_folds(treatment: np.ndarray, seed: int) -> np.ndarray:
    """The exact zero-based fold vector shared with R."""
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    assignment = np.empty(len(treatment), dtype=np.int64)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(treatment)), treatment)):
        assignment[test] = fold
    return assignment


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario not in (*SCENARIOS, "both_wrong"):
        raise KeyError(scenario)
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-2.0, 2.0, size=n)
    w2 = rng.binomial(1, 0.5, size=n).astype(float)
    linear = _linear_predictor(w1, w2)
    a = rng.binomial(1, expit(linear)).astype(float)
    y = rng.binomial(1, expit(0.2 * a + linear)).astype(float)
    folds = fixed_folds(a, seed + 1)
    return (
        pd.DataFrame(
            {
                "Y": y,
                "A": a,
                "W1": w1,
                "W2": w2,
                "W12": w1 * w2,
                "fold": folds,
            }
        ),
        truth(),
    )


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def _learners(scenario: str) -> tuple[ColumnLogistic, ColumnLogistic]:
    outcome_correct = scenario in {"outcome_correct", "both_correct"}
    treatment_correct = scenario in {"treatment_correct", "both_correct"}
    # Outcome design is [A, W1, W2, W12]; treatment design is [W1, W2, W12].
    outcome_columns = None if outcome_correct else (0, 1, 2)
    treatment_columns = None if treatment_correct else (0, 1)
    return ColumnLogistic(outcome_columns), ColumnLogistic(treatment_columns)


def fit_cleverly(
    frame: pd.DataFrame, scenario: str, *, update_order: str = CONFIGURATION["update_order"]
) -> Any:
    """The registered fit, at one nuisance regime.

    ``update_order`` is the only setting a caller may vary, and it is here so the
    update-order comparison `docs/technical-reference/dr-tmle/targeting.md` publishes is a call to *this* function rather
    than a second copy of the construction beside it.  A copy is how two routes come to be
    compared at settings that differ in more than the route, which is the one thing that
    comparison cannot afford.  Every published row uses the default.
    """
    outcome, treatment = _learners(scenario)
    assignment = frame["fold"].to_numpy(dtype=np.int64)
    result = (
        FixedFoldDRTMLE(
            assignment,
            outcome_learner=outcome,
            treatment_learner=treatment,
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=ColumnLogistic(),
            cross_fit=True,
            n_folds=N_FOLDS,
            estimands=ESTIMANDS,
            simultaneous=False,
            g_bounds=G_BOUNDS,
            max_outer=MAX_OUTER,
            max_iter=CONFIGURATION["max_iter"],
            tol=1e-10,
            random_state=0,
            guard=tuple(CONFIGURATION["guard"]),
            reduction=CONFIGURATION["reduction"],
            reduced_crossfit=CONFIGURATION["reduced_crossfit"],
            update_order=update_order,
        )
        .fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2", "W12"],
        )
        .single()
    )
    score_check = result.diagnostics.score_equations()
    worst_score = max(abs(float(row.score)) for row in score_check.rows)
    if not np.isfinite(worst_score):
        raise RuntimeError("DR-TMLE empirical score audit is non-finite")
    raw = np.asarray(result.repeats[0].nuisance.propensity.values, dtype=float)
    bounded = result.repeats[0].nuisance.bounded_propensity(G_BOUNDS)
    if not np.array_equal(raw, bounded):
        raise RuntimeError("the propensity bound activated in the canonical comparison")
    return result


def cleverly_rows(
    frame: pd.DataFrame,
    reference_truth: Mapping[str, float],
    scenario: str,
    replicate: int,
    *,
    result: Any | None = None,
) -> list[dict[str, Any]]:
    result = fit_cleverly(frame, scenario) if result is None else result
    score_check = result.diagnostics.score_equations()
    score_max = max(abs(float(row.score)) for row in score_check.rows)
    initial = result.repeats[0].nuisance.outcome.arms
    initial_values = {
        "ey0": float(np.mean(initial[0.0])),
        "ey1": float(np.mean(initial[1.0])),
        "ate": float(np.mean(initial[1.0] - initial[0.0])),
    }
    rows: list[dict[str, Any]] = []
    for name in ESTIMANDS:
        estimate = result.estimates[name]
        target = float(reference_truth[name])
        low, high = estimate.ci
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": len(frame),
                "estimand": name,
                "truth": target,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= target <= high),
                "initial_estimate": initial_values[name],
                "score_max": score_max,
                # `score_passed` is **not** written here. It needs the fit's reported standard
                # error, which is per estimand, and the bar is one bar per replication -- so
                # `extra_artifacts` derives it once from the assembled rows, for the reference
                # as well, rather than each side deciding its own.
                "solver_reported": True,
                "solver_passed": _solver_passed(score_check),
                "bound_active": False,
            }
        )
    return rows


def _solver_passed(score_check: Any) -> bool:
    """Whether every actual fluctuation row completed without a recorded failure."""
    return all(not row.failure for row in score_check.rows if row.kind == "fluctuation")


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, reference_truth = draw_scenario(scenario, n, replicate)
    result = fit_cleverly(frame, scenario)
    payload_frame = frame.copy()
    nuisance = result.repeats[0].nuisance
    payload_frame["qn0"] = nuisance.outcome.arms[0.0]
    payload_frame["qn1"] = nuisance.outcome.arms[1.0]
    payload_frame["gn1"] = nuisance.propensity.arm(1.0)
    payload_frame.insert(0, "replicate", replicate)
    payload_frame.insert(0, "scenario", scenario)
    truth_row = {
        "scenario": scenario,
        "replicate": replicate,
        **{f"truth_{name}": value for name, value in reference_truth.items()},
    }
    return (
        payload_frame,
        truth_row,
        cleverly_rows(frame, reference_truth, scenario, replicate, result=result),
    )


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payloads = [
        ((scenario, replicate, n),) for scenario in SCENARIOS for replicate in range(replicates)
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    samples = pd.concat([frame for frame, _, _ in outcomes], ignore_index=True)
    truths = pd.DataFrame([item for _, item, _ in outcomes])
    rows = pd.DataFrame([row for _, _, fitted in outcomes for row in fitted])
    return samples, truths, rows


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One fit-health row per implementation, scenario, and replication.

    Two columns are derived here rather than taken from either side, and both used to be
    taken from the side that reported them.

    **The score bar is one bar.**  It is :func:`~cleverly.validation.score_check`'s own,
    ``1e-3 * se / sqrt(n)``, formed from each replication's *own* largest reported standard
    error and applied to both implementations.  Each side used to apply its own constant --
    ``1e-4`` here and ``1 / n`` in the R runner -- so the published counts were two
    quantities in one column.

    **The solver flag is not a comparison and now says so.**  R ``drtmle`` exposes no
    convergence flag, so the runner had no honest value to write and wrote ``TRUE``.  The
    published study then read "24 Cleverly failures against 0" off a column the reference
    could not fail.  ``solver_reported`` marks which side reports one at all, and the
    reference's ``solver_passed`` is left empty rather than filled with a pass.  What *is*
    comparable is ``score_passed``, and at the shared bar it runs the other way.
    """
    key = ["implementation", "scenario", "replicate"]
    diagnostics = (
        rows.loc[:, list(FIT_DIAGNOSTIC_SOURCE_COLUMNS)]
        .drop_duplicates()
        .sort_values(["scenario", "replicate", "implementation"], ignore_index=True)
    )
    if not diagnostics.groupby(key).size().eq(1).all():
        raise ValueError("fit diagnostics are not unique by implementation/scenario/replicate")

    # One bar per replication, from `score_check`'s own function rather than from a copy of its
    # arithmetic. The bar reads every estimand's standard error, so it is formed on the
    # replicate rows and joined back on: the diagnostics frame is one row per fit.
    thresholds = (
        rows.groupby(key, sort=False)
        .apply(
            lambda group: score_threshold(
                group["std_error"], int(group["n"].iloc[0]), tolerance=SCORE_AUDIT_TOLERANCE
            ),
            include_groups=False,
        )
        .rename("score_threshold")
    )
    diagnostics = diagnostics.merge(thresholds, on=key, how="left", validate="1:1")
    diagnostics["score_passed"] = (
        diagnostics["score_max"].astype(float) <= diagnostics["score_threshold"]
    )
    reference = diagnostics["implementation"].eq(STUDY.reference)
    diagnostics["solver_reported"] = ~reference
    diagnostics.loc[reference, "solver_passed"] = np.nan
    return {"fit-diagnostics.csv": diagnostics.loc[:, list(FIT_DIAGNOSTIC_COLUMNS)]}


def scientific_failures(
    artifacts: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """The fits either side left unsolved, split by whether the two sides can be compared.

    Two frames rather than one.  The score audit runs the same bar over both implementations
    and is reported as a comparison.  The solver flag exists on one side only, so it is
    reported alone rather than beside a column of manufactured passes.
    """
    diagnostics = artifacts["fit-diagnostics.csv"]
    reported = diagnostics["solver_reported"].astype(bool)
    return {
        "score audit (both implementations, shared bar)": diagnostics.loc[
            ~diagnostics["score_passed"].astype(bool)
        ],
        "solver health (subject only; the reference reports no flag)": diagnostics.loc[
            reported & ~diagnostics["solver_passed"].fillna(True).astype(bool)
        ],
    }
