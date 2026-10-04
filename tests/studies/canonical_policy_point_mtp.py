"""Registered evidence for point-treatment modified treatment policies (``policy-point-mtp``).

The subject is ordinary TMLE with ``policies=`` beyond the additive shift, on a bounded dose:

.. code-block:: text

    W1 ~ N(0, 1),  W2 ~ Bernoulli(1/2)
    A | W ~ N(2.8 + 0.5 W1 + 0.4 W2, 1) truncated to [0, 6]
    Y | A, W ~ Bernoulli(expit(-1.8 + 0.6 A - 0.06 A^2 + 0.5 W1 + 0.3 W2))

Four policies: the natural course, ``Scale(1.25, cap=5.5)``, a ``Piecewise`` policy that
leaves doses at or below 3 and lowers the rest by 0.5 (the shape of ``lmtp``'s piecewise
example and of Hoffman et al. 2024, Example 5), and a declared ``ModifiedPolicy`` that halves
the distance below 3 (``a -> (a + 3) / 2`` for ``a <= 3``), whose inverse and its derivative are
declared.  Every policy keeps the dose
inside the support, so every density ratio is bounded.  The truth is the g-formula
:math:`E\\{\\bar Q(d(A, W), W)\\}` by Gauss-Hermite quadrature over ``W1`` and composite
Gauss-Legendre quadrature over the dose, split at every point where a policy jumps, and checked
by doubling the order (agreement below ``1e-10``).

**Learners.**  The density is the oracle binned density of the truncated normal
(:class:`OraclePointHazard`, ``DENSITY_BINS`` bins) and the outcome regression the correctly
specified logistic model in ``[A, A^2, W1, W2]``.

**The comparator.**  ``lmtp`` 1.5.4 is handed the ratio array this package computed
(``PolicySet.policy_ratio``) and the policy dose, beside the replicate data, through
``tests/canonical/lmtp_mtp_adapter.R`` with one node; no lmtp density ratio is fitted.  The
estimators still differ with identical inputs: this package targets with the covariate
submodel over the observed and policy doses, and ``lmtp`` with the intercept fluctuation
weighted by the ratio.  The runner gives ``lmtp`` the same outcome design, a logistic
regression on the columns and the squared dose (``SL.glm.quadratic``), so the untargeted
estimates agree to rounding.  The paired rows are read under the default margins and are not
an exactness check.

Publication policy is ``gated``.  The red-cell route is declared before any run: a red primary
or paired row blocks the merge and gets a diagnosis.  A red property cell that reads as a
finite-sample limit is re-registered ``reporting`` with an owner row in "Red-cell owners"
before any re-run.  A replication that raises is never redrawn; the shared harness refuses a
cell that lost one, so the failing cell drops to its red-cell owner with its failure count
stated as a page limit and the run repeats without it, with no other change.  No budget,
margin, law, learner or seed changes after a verdict is seen.
"""

from __future__ import annotations

import itertools
import re
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.base import BaseEstimator

from cleverly.estimators import TMLE
from cleverly.interventions import ModifiedPolicy, Piece, Piecewise, Scale, Shift
from cleverly.learners.density import bin_edges
from cleverly.utils.parallel import map_parallel
from tests.parallel import STUDY_JOBS
from tests.studies.canonical_categorical_ltmle import (
    LMTP_SOURCE_COMMIT,
    LMTP_TARBALL_SHA256,
    LMTP_VERSION,
    R_BASE_IMAGE,
)
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate
from tests.studies.fractional_glm import QuasiBinomialGLM
from tests.studies.point_study_helpers import primary_rows

PRIMARY_REPLICATES = 1_000
PRIMARY_N = 2_000
SEED = 20261043
RESAMPLING_SEED = 2026104301
SCENARIO = "policy_point"
DENSITY_BINS = 160
LOWER, UPPER = 0.0, 6.0
CAP = 5.5
FACTOR = 1.25
KNEE = 3.0
DROP = 0.5
HALVE_BELOW = 3.0

LABELS = ("natural course", "x1.25", "piecewise", "halve below 3")
REFERENCE = "natural course"
ESTIMANDS = tuple(f"ey_policy[{label}]" for label in LABELS) + tuple(
    f"ate_policy[{label} vs {REFERENCE}]" for label in LABELS[1:]
)


def dose_mean(w1: Any, w2: Any) -> Any:
    return 2.8 + 0.5 * np.asarray(w1, dtype=float) + 0.4 * np.asarray(w2, dtype=float)


def outcome_mean(a: Any, w1: Any, w2: Any) -> Any:
    a = np.asarray(a, dtype=float)
    return expit(
        -1.8 + 0.6 * a - 0.06 * a**2 + 0.5 * np.asarray(w1, dtype=float) + 0.3 * np.asarray(w2)
    )


def truncated_pdf(a: Any, mean: Any) -> Any:
    a = np.asarray(a, dtype=float)
    mean = np.asarray(mean, dtype=float)
    mass = norm.cdf(UPPER - mean) - norm.cdf(LOWER - mean)
    return np.where((a >= LOWER) & (a <= UPPER), norm.pdf(a - mean) / mass, 0.0)


def sample(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w1 = rng.normal(size=n)
    w2 = rng.integers(0, 2, n).astype(float)
    mean = dose_mean(w1, w2)
    u = rng.uniform(norm.cdf(LOWER - mean), norm.cdf(UPPER - mean))
    a = np.clip(mean + norm.ppf(u), LOWER, UPPER)
    y = rng.binomial(1, outcome_mean(a, w1, w2)).astype(float)
    return pd.DataFrame({"W1": w1, "W2": w2, "A": a, "Y": y})


# ---------------------------------------------------------------------------- policies


def halve_map(a: Any, h: Any) -> Any:
    return (np.asarray(a, dtype=float) + HALVE_BELOW) / 2.0


def halve_inverse(b: Any, h: Any) -> Any:
    return 2.0 * np.asarray(b, dtype=float) - HALVE_BELOW


def halve_derivative(b: Any, h: Any) -> Any:
    return np.full_like(np.asarray(b, dtype=float), 2.0)


def policies() -> tuple[Any, ...]:
    return (
        Shift(0.0, cap=None),
        Scale(FACTOR, cap=CAP, name="x1.25"),
        Piecewise(
            ((-np.inf, KNEE, Shift(0.0, None)), (KNEE, np.inf, Shift(-DROP, None))),
            closed="right",
            name="piecewise",
        ),
        ModifiedPolicy(
            "halve below 3",
            pieces=(
                Piece(
                    -np.inf, HALVE_BELOW, halve_map, halve_inverse, halve_derivative, closed="right"
                ),
                Piece(HALVE_BELOW, np.inf, closed="right"),
            ),
            policy_kind="known",
        ),
    )


MAPS = {
    "natural course": lambda a: a,
    "x1.25": lambda a: np.where(a * FACTOR > CAP, a, a * FACTOR),
    "piecewise": lambda a: np.where(a > KNEE, a - DROP, a),
    "halve below 3": lambda a: np.where(a <= HALVE_BELOW, halve_map(a, None), a),
}
_BREAKS = (CAP / FACTOR, KNEE, HALVE_BELOW)


def _mean(label: str, order: int, hermite: int) -> float:
    points = sorted({LOWER, UPPER, *_BREAKS})
    base, base_weights = np.polynomial.legendre.leggauss(order)
    nodes = np.concatenate(
        [low + 0.5 * (high - low) * (base + 1.0) for low, high in itertools.pairwise(points)]
    )
    weights = np.concatenate(
        [0.5 * (high - low) * base_weights for low, high in itertools.pairwise(points)]
    )
    w1_nodes, w1_weights = np.polynomial.hermite_e.hermegauss(hermite)
    w1_weights = w1_weights / np.sqrt(2.0 * np.pi)
    total = 0.0
    for w2 in (0.0, 1.0):
        density = truncated_pdf(nodes[None, :], dose_mean(w1_nodes[:, None], w2))
        q = outcome_mean(MAPS[label](nodes)[None, :], w1_nodes[:, None], w2)
        inner = np.sum(weights[None, :] * density * q, axis=1)
        total += 0.5 * float(np.sum(w1_weights * inner))
    return total


@cache
def policy_truth(label: str) -> float:
    value = _mean(label, 96, 60)
    check = _mean(label, 192, 90)
    if abs(value - check) > 1e-10:
        raise AssertionError(f"the quadrature of {label!r} moved by {abs(value - check):.3g}")
    return value


def _truth() -> dict[str, float]:
    means = {label: policy_truth(label) for label in LABELS}
    out = {f"ey_policy[{label}]": value for label, value in means.items()}
    for label in LABELS[1:]:
        out[f"ate_policy[{label} vs {REFERENCE}]"] = means[label] - means[REFERENCE]
    return out


TRUTH = _truth()


# ---------------------------------------------------------------------------- learners


class OraclePointHazard(BaseEstimator):
    """The exact pooled-hazard probabilities of the dose, on the design ``[W1, W2, bin]``.

    Parameters
    ----------
    edges : tuple of float
        The bin edges.
    marginal : bool
        Whether to ignore the covariates, the misspecified mechanism.
    """

    def __init__(self, edges: tuple[float, ...], marginal: bool = False) -> None:
        self.edges = edges
        self.marginal = marginal

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> OraclePointHazard:
        del X, y, sample_weight
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        edges = np.asarray(self.edges, dtype=float)
        index = np.rint(design[:, 2]).astype(int)
        mean = np.full(len(design), 3.0) if self.marginal else dose_mean(design[:, 0], design[:, 1])
        last = len(edges) - 2
        lower = np.where(index == 0, LOWER, edges[index])
        upper = np.where(index == last, UPPER, edges[np.minimum(index + 1, last + 1)])
        mass = norm.cdf(UPPER - mean) - norm.cdf(LOWER - mean)
        below = (norm.cdf(lower - mean) - norm.cdf(LOWER - mean)) / mass
        reached = (norm.cdf(upper - mean) - norm.cdf(LOWER - mean)) / mass
        survived = 1.0 - below
        hazard = np.divide(
            reached - below, survived, out=np.ones_like(below), where=survived > 1e-15
        )
        hazard = np.clip(hazard, 1e-12, 1.0 - 1e-12)
        return np.column_stack([1.0 - hazard, hazard])


class QuadraticOutcome(BaseEstimator):
    """The correctly specified logistic regression on ``[A, A^2, W1, W2]``."""

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> QuadraticOutcome:
        self.model_ = QuasiBinomialGLM().fit(self._design(X), y, sample_weight=sample_weight)
        self.classes_ = np.array([0.0, 1.0])
        return self

    @staticmethod
    def _design(X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        return np.column_stack([matrix, matrix[:, 0] ** 2])

    def predict(self, X: Any) -> np.ndarray:
        return np.asarray(self.model_.predict(self._design(X)), dtype=float)

    def predict_proba(self, X: Any) -> np.ndarray:
        p = np.clip(self.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


# ---------------------------------------------------------------------------- the study

PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "double_robustness": tuple(
        f"x1_25__{configuration}"
        for configuration in ("both_correct", "outcome_correct", "density_correct", "both_wrong")
    ),
    "root_n_and_efficiency": tuple(f"x1_25__n_{size}" for size in (500, 2_000, 8_000)),
    "root_n_rate": ("x1_25__empirical_sd", "x1_25__reported_se"),
    "interval_calibration": tuple(
        f"{label}__{cell}"
        for label in ("x1_25", "piecewise", "halve", "classifier_route")
        for cell in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "type_i_error": ("x1_25__sharp_null",),
    "power": ("x1_25__alternative",),
    "targeting_necessity": ("x1_25__targeted", "x1_25__untargeted"),
    "inverse_necessity": ("halve__declared_inverse", "halve__inverse_dropped_control"),
}

STUDY = StudyRecord(
    name="point-treatment modified treatment policies beyond the additive shift",
    slug="policy-point-mtp",
    artifacts=ROOT / "tests" / "canonical" / "policy_point_mtp",
    document="docs/technical-reference/method-evidence/point-modified-treatment-policies.md",
    anchor="point-modified-treatment-policies",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-policy-tmle",
    reference="lmtp",
    modules=(
        "tests/studies/canonical_policy_point_mtp.py",
        "tests/studies/policy_point_mtp_properties.py",
        "tests/studies/canonical_categorical_ltmle.py",
        "tests/studies/fractional_glm.py",
        "tests/studies/point_study_helpers.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/canonical/lmtp_crossfit/Dockerfile",
        "tests/canonical/lmtp_crossfit_adapter.R",
        "tests/canonical/lmtp_policy_adapter.R",
        "tests/canonical/lmtp_mtp_adapter.R",
        "tests/canonical/policy_point_mtp_runner.R",
    ),
    runner_module="tests.studies.canonical_policy_point_mtp",
    properties_module="tests.studies.policy_point_mtp_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="gated",
)

REFERENCE_METADATA = {
    "lmtp_version": LMTP_VERSION,
    "lmtp_source_commit": LMTP_SOURCE_COMMIT,
    "lmtp_tarball_sha256": LMTP_TARBALL_SHA256,
    "r_base_image": R_BASE_IMAGE,
    "reference_construction": (
        "lmtp cf_tmle and theta_dr with one node, the policy dose and the density ratio this "
        "package computed (PolicySet.policy_ratio), written beside the replicate data; no lmtp "
        "density ratio is fitted"
    ),
    "reference_adapter": "tests/canonical/lmtp_mtp_adapter.R",
    "pairing": "read under the default margins; the point fluctuations differ by construction",
}

CONFIGURATION = {
    "construction": "ordinary",
    "cross_fit": False,
    "density_bins": DENSITY_BINS,
    "outcome_learner": "QuadraticOutcome",
    "density": "oracle binned density of the truncated normal dose",
    "policies": list(LABELS),
    "reference": REFERENCE,
}


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def edges_of(frame: pd.DataFrame) -> tuple[float, ...]:
    return tuple(
        float(value) for value in bin_edges(frame["A"].to_numpy(dtype=float), DENSITY_BINS)
    )


def fit(
    frame: pd.DataFrame,
    *,
    configuration: str = "both_correct",
    chosen: tuple[Any, ...] | None = None,
    ratio: str = "density",
    treatment_learner: Any = None,
) -> Any:
    """The one fit every primary and property row reads."""
    q_correct = configuration in {"both_correct", "outcome_correct"}
    g_correct = configuration in {"both_correct", "density_correct"}
    from sklearn.dummy import DummyClassifier

    outcome = QuadraticOutcome() if q_correct else DummyClassifier(strategy="prior")
    density = OraclePointHazard(edges_of(frame), marginal=not g_correct)
    return (
        TMLE(
            policies=policies() if chosen is None else chosen,
            outcome_learner=outcome,
            treatment_learner=density if treatment_learner is None else treatment_learner,
            cross_fit=False,
            simultaneous=False,
            density_bins=DENSITY_BINS,
            ratio=ratio,
            max_iter=100,
            tol=1e-10,
            random_state=0,
        )
        .fit(frame, outcome="Y", treatment="A", covariates=["W1", "W2"])
        .single()
    )


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return sample(n, seed), dict(TRUTH)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def initial_estimates(result: Any) -> dict[str, float]:
    """The untargeted plug-ins under public parameter names."""
    nuisance = result.nuisance
    means = {
        label: float(np.mean(nuisance.scaler.unscale_levels(nuisance.outcome.arms[float(index)])))
        for index, label in enumerate(LABELS)
    }
    out = {f"ey_policy[{label}]": value for label, value in means.items()}
    for label in LABELS[1:]:
        out[f"ate_policy[{label} vs {REFERENCE}]"] = means[label] - means[REFERENCE]
    return out


def pairing_columns(result: Any) -> dict[str, np.ndarray]:
    """The policy dose and the ratio this fit used, for the comparator to read."""
    evaluated = result.nuisance.policies
    out: dict[str, np.ndarray] = {}
    for index, label in enumerate(LABELS):
        out[f"shift__{slug(label)}__1__1"] = np.asarray(evaluated.shifted[:, index], dtype=float)
        out[f"ratio__{slug(label)}__1"] = np.asarray(evaluated.policy_ratio[:, index], dtype=float)
    return out


def _replicate(
    payload: tuple[str, int, int],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = payload
    frame, reference = draw_scenario(scenario, n, replicate)
    result = fit(frame)
    sample_frame = frame.copy()
    for name, values in pairing_columns(result).items():
        sample_frame[name] = values
    sample_frame.insert(0, "row", np.arange(len(sample_frame)))
    sample_frame.insert(0, "fold", np.zeros(len(sample_frame), dtype=int))
    sample_frame.insert(0, "replicate", replicate)
    sample_frame.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in reference.items()
    ]
    rows = primary_rows(
        result=result,
        truth=reference,
        implementation=STUDY.implementation,
        scenario=scenario,
        replicate=replicate,
        estimands=ESTIMANDS,
        initials=initial_estimates(result),
        n=len(frame),
    )
    return sample_frame, truth_rows, rows


def draw_and_fit(
    *, replicates: int, n: int, n_jobs: int = STUDY_JOBS
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outcomes = map_parallel(
        _replicate, [((SCENARIO, replicate, n),) for replicate in range(replicates)], n_jobs=n_jobs
    )
    samples = pd.concat([sample for sample, _, _ in outcomes], ignore_index=True)
    truth_rows = pd.DataFrame([row for _, rows, _ in outcomes for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes for row in rows])
    return samples, truth_rows, estimates.loc[:, list(REPLICATE_COLUMNS)]


__all__ = ["STUDY", "TRUTH", "draw_and_fit", "fit"]
