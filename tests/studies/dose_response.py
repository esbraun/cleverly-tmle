r"""Registered evidence for the kernel-smoothed dose-response curve (X93 S1).

The subject is :class:`cleverly.estimators.DoseResponseTMLE`: CV-TMLE of the smoothed curve
:math:`\theta_h(a_0) = \int_D w_{h,a_0}(a)\theta(a)\,da` at declared bandwidths and doses, with a
Wald interval for each pair and a multiplier band over the whole declared set
(``docs/technical-reference/dose-response.md``).  The law is
:mod:`tests.studies.dose_response_law`: a confounded continuous dose with a normal location
model and a binary or Gaussian outcome whose truths are Gauss-Hermite and Gauss-Legendre
integrals, exact to rounding.  :math:`D = [0, 5]` throughout.

============  ======  ==============================================================================
scenario      n       declaration and estimands
============  ======  ==============================================================================
``binary``    4,000   doses 0.25, 1.5, 2.5, 3.5, 4.75 by bandwidths 0.5, 1.0, 1.5: 15 pairs, the
                      default logistic fluctuation
``strata``    8,000   the binary law with ``V = 1{W1 > 0}`` as a fourth covariate and
                      ``strata=["V"]``; doses 0.25, 2.5, 4.75 at bandwidth 1.0 in each stratum:
                      6 estimands named ``<pair>[V=0]`` and ``<pair>[V=1]``
``gaussian``  4,000   the Gaussian law on the identity scale, ``fluctuation="linear"`` with the
                      declared :data:`Q_BOUNDS`; the five doses at bandwidth 1.0
``tail``      1,000   one pair, bandwidth 0.5 at :data:`TAIL_DOSE` = 0.4, whose expected Kish size
                      under the true density is 62.3 at n = 1,000, just above the default floor
                      of 50 (``tests/unit/test_dose_response_design.py`` recomputes it).  It
                      measures the interval at the floor the defaults allow
============  ======  ==============================================================================

Every fit is cross-fitted with the defaults: five folds, the default density bins
(``max(20, ceil(2 n^(1/3)))``: 32 at 4,000, 40 at 8,000 and 20 at 1,000), eight nodes per quadrature
piece, ``random_state=0``.  The learners are declared here, not the package defaults:

* outcome: :class:`DoseSieve`, a cubic B-spline of the dose with eight knots, tensored with
  ``(1, W)``, then ``StandardScaler`` and ``LogisticRegression(C=100)`` (binary) or
  ``LinearRegression`` (Gaussian).  :math:`\operatorname{logit}\bar Q_0 = f(a) + c(a) U` with
  :math:`U` linear in :math:`W` and :math:`c` linear in :math:`a`, so the sieve holds the
  :math:`W` part exactly and approximates :math:`f(a) = -0.3 + 0.9\sin\{1.1(a - 1.5)\}` by a
  spline;
* treatment density: the package's pooled-hazard binned density with the hazard classifier
  :class:`HazardDesign`, ``[bin indicators, W, W^2, bin * W, bin * W^2]`` with
  ``StandardScaler`` and ``LogisticRegression(C=100)``.

The Gaussian scenario declares ``q_bounds=(-15, 15)``.  A cross-fitted continuous outcome needs
a declared range, and an observed outcome outside it raises ``ValueError`` from
:class:`cleverly.utils.bounds.OutcomeScaler` before any learner fits, which is not a
recordable failure and would stop the run.  The design test bounds the chance of any row
outside it over the whole declared run below 1e-9.  The linear fluctuation and both Gaussian
learners are equivariant in the outcome's affine scale, so the width of the range does not
move an estimate.

**Size** (declared before any run, from :mod:`tests.studies.dose_response_pilot`).
:data:`PRIMARY_REPLICATES` is the shared rule
:func:`~tests.studies.repeated_sizing.primary_replicates` on the pilot, with one
implementation and X32b's clause shares, one bias group per scenario, read on the 1,000 pilot
draws of each scenario: 4,000.  The lowest pilot coverage is 0.931 (``strata``, dose 0.25 in
``V = 1``), and at n = 4,000 that cell read 0.908, which no budget sizes; ``strata`` therefore
runs at :data:`STRATA_N`.  The design test recomputes the size.

**Properties.** :mod:`tests.studies.dose_response_properties` declares the band family
(``simultaneous_coverage``), the calibration family (``interval_calibration``) and the
union-model family (``double_robustness``).

**Comparator.** Pending: npcausal ``ctseff`` (Kennedy, Ma, McHugh and Small 2017) estimates a
related target, a curve smoothed against the marginal dose density rather than Lebesgue
measure, which agrees with this one only where the curve is linear on each kernel's support.
X93 part S2 owns its runs on the local R queue.  The method merges on its Python evidence.

**Initial estimate.** The ``initial_estimate`` column is each pair's untargeted plug-in
:math:`n^{-1}\sum_i \int_D w(a)\hat{\bar Q}(a, W_i)\,da` from the same cross-fitted regression on
the same quadrature rule.  :func:`fit_with_initials` reads it by observing each call to
``target_pair`` that the fit makes, without changing any argument or result.

**Not covered.** Observation weights, an in-sample fit, clustered rows (refused), a missing
outcome (refused), the triweight and uniform kernels, the package's default learner library,
and the unsmoothed curve :math:`\theta(a_0)` (refused, roadmap F39).

The red-cell readings are declared in ``tests/canonical/dose_response/regenerate.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

import cleverly.estimators.dose_response as dose_engine
from cleverly.dose_response import DoseResponse, pair_name
from cleverly.estimators import DoseResponseTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import dose_response_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

PRIMARY_REPLICATES = 4_000
PRIMARY_N = 4_000
TAIL_N = 1_000
#: The stratified scenario's size.  At :data:`PRIMARY_N` its pilot read coverage 0.908 at dose
#: 0.25 in ``V = 1`` (Kish size 69, SE ratio 0.94, 1,000 draws), whose one-sided 80% lower bound
#: lies below the 0.90 floor, so no budget met the primary rule.  Raised before any run.
STRATA_N = 8_000
#: The declared share of replicates one cell may lose to recordable fitting failures.
FAILURE_BOUND = 0.01
SEED = 2026119301
RESAMPLING_SEED = 2026119302

SUPPORT = law.SUPPORT
DOSES = (0.25, 1.5, 2.5, 3.5, 4.75)
BANDWIDTHS = (0.5, 1.0, 1.5)
STRATA_DOSES = (0.25, 2.5, 4.75)
#: The tail pair: the smallest bandwidth at the dose whose expected Kish size at
#: :data:`TAIL_N` rows sits just above the default floor (the design test recomputes it).
TAIL_DOSE = 0.4
TAIL_BANDWIDTH = 0.5
#: The Gaussian outcome's declared range (see the module docstring).
Q_BOUNDS = (-15.0, 15.0)

BINARY = "binary"
STRATA = "strata"
GAUSSIAN = "gaussian"
TAIL = "tail"

COVARIATES = ("W1", "W2", "W3")
STRATA_COVARIATES = ("W1", "W2", "W3", "V")

# ---------------------------------------------------------------------------- learners


class DoseSieve(BaseEstimator, TransformerMixin):
    """``[a, W] -> B(a) (x) (1, W)``: a cubic B-spline of the dose with dose-varying slopes.

    Column 0 of the design is the dose, as the package builds every dose-response design.
    With ``interactions=False`` the design is ``B(a)`` alone, a regression that ignores
    :math:`W`.

    Parameters
    ----------
    n_knots : int
        The spline's knots, uniform over the training doses.
    degree : int
        The spline's degree.
    interactions : bool
        Whether to tensor the spline with the covariates.
    """

    def __init__(self, n_knots: int = 8, degree: int = 3, interactions: bool = True) -> None:
        self.n_knots = n_knots
        self.degree = degree
        self.interactions = interactions

    def fit(self, X: Any, y: Any = None) -> DoseSieve:
        design = np.asarray(X, dtype=float)
        self.spline_ = SplineTransformer(
            n_knots=self.n_knots, degree=self.degree, extrapolation="linear"
        ).fit(design[:, :1])
        return self

    def transform(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        basis = self.spline_.transform(design[:, :1])
        if not self.interactions:
            return np.asarray(basis)
        covariates = design[:, 1:]
        return np.column_stack(
            [basis] + [basis * covariates[:, [j]] for j in range(covariates.shape[1])]
        )


class HazardDesign(BaseEstimator, TransformerMixin):
    """The pooled-hazard design ``[W, bin, indicators]`` to a hazard classifier's features.

    The package's pooled design holds the ``p`` covariates, the bin index and the drop-first
    bin indicators.  ``"full"`` returns ``[indicators, W, W^2, bin * W, bin * W^2]`` with the
    bin index divided by 10; ``"marginal"`` returns the indicators alone, a hazard that ignores
    :math:`W`, so the density is the binned marginal density of the dose.

    Parameters
    ----------
    p : int
        The number of covariates in the fit.
    form : {"full", "marginal"}
        The feature set.
    """

    def __init__(self, p: int = 3, form: str = "full") -> None:
        self.p = p
        self.form = form

    def fit(self, X: Any, y: Any = None) -> HazardDesign:
        return self

    def transform(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        p = int(self.p)
        indicators = design[:, p + 1 :]
        if self.form == "marginal":
            return indicators
        if self.form != "full":
            raise ValueError(f"unknown hazard design {self.form!r}")
        w = design[:, :p]
        bin_index = design[:, p : p + 1] / 10.0
        return np.column_stack([indicators, w, w * w, bin_index * w, bin_index * w * w])


def outcome_learner(*, gaussian: bool = False, interactions: bool = True) -> Pipeline:
    """The declared outcome regression, or with ``interactions=False`` one that ignores W."""
    sieve = DoseSieve(interactions=interactions)
    if gaussian:
        return make_pipeline(sieve, LinearRegression())
    return make_pipeline(sieve, StandardScaler(), LogisticRegression(C=100.0, max_iter=2000))


def hazard_learner(p: int, *, form: str = "full") -> Pipeline:
    """The declared hazard classifier over ``p`` covariates, or the marginal one."""
    return make_pipeline(
        HazardDesign(p=p, form=form), StandardScaler(), LogisticRegression(C=100.0, max_iter=3000)
    )


# ---------------------------------------------------------------------------- scenarios


@dataclass(frozen=True)
class Scenario:
    """One primary scenario.

    Parameters
    ----------
    curve : DoseResponse
        The declaration.
    n : int
        The sample size.
    gaussian : bool
        Whether the outcome is the Gaussian one.
    strata : bool
        Whether ``V`` is a covariate and the fit stratifies on it.
    """

    curve: DoseResponse
    n: int
    gaussian: bool = False
    strata: bool = False

    @property
    def covariates(self) -> tuple[str, ...]:
        """The covariate columns of the fit."""
        return STRATA_COVARIATES if self.strata else COVARIATES

    @property
    def estimands(self) -> tuple[str, ...]:
        """The names the fit reports, in its report order: every stratum's pairs in turn."""
        names = tuple(pair_name(h, a0) for h, a0 in self.curve.pairs)
        if not self.strata:
            return names
        return tuple(f"{name}[V={v}]" for v in (0, 1) for name in names)


SCENARIOS: dict[str, Scenario] = {
    BINARY: Scenario(DoseResponse(DOSES, BANDWIDTHS, SUPPORT), PRIMARY_N),
    STRATA: Scenario(DoseResponse(STRATA_DOSES, (1.0,), SUPPORT), STRATA_N, strata=True),
    GAUSSIAN: Scenario(DoseResponse(DOSES, (1.0,), SUPPORT), PRIMARY_N, gaussian=True),
    TAIL: Scenario(DoseResponse((TAIL_DOSE,), (TAIL_BANDWIDTH,), SUPPORT), TAIL_N),
}
ESTIMANDS = {scenario: spec.estimands for scenario, spec in SCENARIOS.items()}
TRUTH: dict[str, dict[str, float]] = {
    scenario: law.truths(spec.curve, gaussian=spec.gaussian, strata=spec.strata)
    for scenario, spec in SCENARIOS.items()
}

#: Every published property cell, by family (see the properties module).
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "simultaneous_coverage": (
        "binary_curve__simultaneous_band",
        "binary_curve__pointwise_joint_control",
    ),
    "interval_calibration": (
        "interior_h1_a2_5__correctly_specified",
        "interior_h1_a2_5__shrunken_se_control",
        "interior_h1_a2_5__noise_control",
    ),
    "double_robustness": (
        "h1_a3_5__outcome_correct_density_wrong",
        "h1_a3_5__both_wrong",
    ),
}

STUDY = StudyRecord(
    name="kernel-smoothed dose-response CV-TMLE",
    slug="dose-response",
    artifacts=ROOT / "tests" / "canonical" / "dose_response",
    document="docs/technical-reference/method-evidence/dose-response.md",
    anchor="kernel-smoothed-dose-response-cv-tmle",
    scenarios=dict(ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    scenario_n={STRATA: STRATA_N, TAIL: TAIL_N},
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-dose-response",
    reference=None,
    pending_reference="npcausal ctseff",
    pending_reason=(
        "X93 part S2: npcausal ctseff estimates a related target (a curve smoothed against the "
        "marginal dose density), compared at a law that is linear on each kernel's support; "
        "its runs are queued on the local R machine"
    ),
    comparison="related target",
    failure_bound={"cleverly-dose-response": FAILURE_BOUND},
    modules=(
        "tests/studies/dose_response.py",
        "tests/studies/dose_response_properties.py",
        "tests/studies/dose_response_law.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
        "tests/studies/evidence/simultaneous.py",
    ),
    runner_module="tests.studies.dose_response",
    properties_module="tests.studies.dose_response_properties",
    property_cells=PROPERTY_CELLS,
)

CONFIGURATION = {
    "construction": "CV-TMLE of the kernel-smoothed curve (one pooled fluctuation per pair)",
    "support": list(SUPPORT),
    "kernel": "epanechnikov",
    "nodes_per_piece": 8,
    "n_folds": 5,
    "density_bins": "default: max(20, ceil(2 n^(1/3)))",
    "declarations": {
        scenario: {
            "doses": list(spec.curve.doses),
            "bandwidths": list(spec.curve.bandwidths),
            "n": spec.n,
            "covariates": list(spec.covariates),
            "strata": ["V"] if spec.strata else [],
        }
        for scenario, spec in SCENARIOS.items()
    },
    "fluctuation": {BINARY: "logistic", STRATA: "logistic", GAUSSIAN: "linear", TAIL: "logistic"},
    "q_bounds": {GAUSSIAN: list(Q_BOUNDS)},
    "learners": (
        "outcome: DoseSieve (cubic B-spline of the dose, 8 knots, tensor (1, W)) with "
        "StandardScaler and LogisticRegression(C=100), or LinearRegression for the Gaussian "
        "outcome; density: pooled-hazard binned density with HazardDesign "
        "[indicators, W, W^2, bin*W, bin*W^2], StandardScaler, LogisticRegression(C=100)"
    ),
    "initial_estimate": "the untargeted plug-in of the same cross-fitted regression",
    "variance_method": "ic",
}


def estimator(scenario: str, **overrides: Any) -> DoseResponseTMLE:
    """The declared estimator of ``scenario``; ``overrides`` replace a learner or the curve."""
    spec = SCENARIOS[scenario]
    settings: dict[str, Any] = {
        "outcome_learner": outcome_learner(gaussian=spec.gaussian),
        "treatment_learner": hazard_learner(len(spec.covariates)),
        "fluctuation": "linear" if spec.gaussian else "logistic",
        "q_bounds": Q_BOUNDS if spec.gaussian else None,
        "random_state": 0,
    }
    curve = overrides.pop("curve", spec.curve)
    settings.update(overrides)
    return DoseResponseTMLE(curve, **settings)


def fit(frame: pd.DataFrame, scenario: str, **overrides: Any) -> Any:
    """The one fit of each primary scenario.  ``overrides`` replace a learner or the curve."""
    spec = SCENARIOS[scenario]
    return estimator(scenario, **overrides).fit(
        frame,
        outcome="Y",
        treatment="A",
        covariates=list(spec.covariates),
        strata=["V"] if spec.strata else None,
    )


def fit_with_initials(
    frame: pd.DataFrame, scenario: str, **overrides: Any
) -> tuple[Any, dict[str, float]]:
    """:func:`fit`, and each reported estimate's untargeted plug-in on the outcome's scale.

    The fit targets every reported estimate on the declared rule first, in report order, and
    then again on the doubled rule.  The observer records the weighted plug-in
    :math:`\\sum_i \\omega_i \\int_D w\\hat{\\bar Q}(\\cdot, W_i) / \\sum_i \\omega_i` of each of
    the first calls, which is the untargeted value the fluctuation starts from, and passes
    every call through unchanged.
    """
    spec = SCENARIOS[scenario]
    original: Callable[..., Any] = dose_engine.target_pair
    seen: list[float] = []

    def observed(
        y: np.ndarray,
        q_obs: np.ndarray,
        q_nodes: np.ndarray,
        h_obs: np.ndarray,
        h_nodes: np.ndarray,
        pair_weights: np.ndarray,
        row_weights: np.ndarray,
        **options: Any,
    ) -> Any:
        omega = np.asarray(row_weights, dtype=float)
        seen.append(float(np.sum(omega * (q_nodes @ pair_weights)) / np.sum(omega)))
        return original(y, q_obs, q_nodes, h_obs, h_nodes, pair_weights, row_weights, **options)

    with mock.patch.object(dose_engine, "target_pair", observed):
        result = fit(frame, scenario, **overrides)
    names = list(result.estimates)
    if len(seen) != 2 * len(names):
        raise AssertionError(f"expected {2 * len(names)} fluctuations, observed {len(seen)}")
    lower, upper = Q_BOUNDS if spec.gaussian else (0.0, 1.0)
    return result, {
        name: lower + (upper - lower) * value for name, value in zip(names, seen, strict=False)
    }


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return law.sample(n, seed, gaussian=SCENARIOS[scenario].gaussian), dict(TRUTH[scenario])


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def result_rows(
    result: Any,
    truth: Mapping[str, float],
    scenario: str,
    replicate: int,
    initial: Mapping[str, float],
    n: int,
) -> list[dict[str, Any]]:
    """One row per estimand on the identity scale."""
    rows = []
    for name in ESTIMANDS[scenario]:
        estimate = result[name]
        std_error, low, high = reported_inference(estimate)
        reference = float(truth[name])
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": std_error,
                "ci_lower": low,
                "ci_upper": high,
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": float(initial[name]),
            }
        )
    return rows


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, n = key
    frame, truth = drawn
    result, initial = fit_with_initials(frame, scenario)
    return result_rows(result, truth, scenario, replicate, initial, n)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Every primary row, each scenario at its declared size.  No sample is kept."""
    sizes = STUDY.sizes(n)
    outcomes = replication.fit_primary(
        [
            (scenario, replicate, sizes[scenario])
            for scenario in SCENARIOS
            for replicate in range(replicates)
        ],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    rows = pd.DataFrame([row for result in outcomes.kept for row in result])
    return replication.result(outcomes, rows.loc[:, list(REPLICATE_COLUMNS)])
