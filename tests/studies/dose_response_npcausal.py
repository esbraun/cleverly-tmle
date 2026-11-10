r"""Declared comparison of the smoothed dose-response curve with npcausal ``ctseff`` (X93 S2).

**Status: declared, not yet run.**  :data:`BANDWIDTHS` and :data:`PRIMARY_REPLICATES` are
:func:`tests.studies.dose_response_npcausal_pilot.declaration_from_pilot`'s reading of the
committed local R pilot (``python -m tests.canonical.dose_response_npcausal.pilot reference``):
1,000 draws, on which real ``ctseff`` failed 77, 9, 1 and 0 calls at the four candidate
bandwidths, every one with ``locpoly``'s "'bandwidth' must be a scalar or an array of length
'gridsize'".  Those are exactly the calls the transcription fails or counts as knife-edge
(pilot draw 11 at bandwidth 1 among them), so the rule selects 2.0 and 2.5 and sizes 4,800
replications, the values it read off the transcription before.  The record is registered with
the declared run's artifacts.

The subject is :class:`cleverly.estimators.DoseResponseTMLE`, the CV-TMLE of
:math:`\theta_h(a_0) = \int_D w_{h,a_0}(a)\theta(a)\,da` with the local-linear equivalent kernel
on Lebesgue measure over :math:`D = [-1.5, 6.5]`.  The reference is ``npcausal::ctseff``
(Kennedy, Ma, McHugh and Small 2017) at commit ``56a5ac1``: a local-linear kernel regression of
a doubly robust pseudo-outcome on the dose, which smooths :math:`\theta` against the marginal
dose density.  The two targets differ in general.  On the law of
:mod:`tests.studies.dose_response_npcausal_law` the curve is a line, so both equal
:math:`\theta(a_0) = 0.2 + 0.5 a_0` at every dose and bandwidth, and the study compares target
recovery on that shared value, not the two smoothing measures (``comparison="related
target"``).

=====================  =====  ===================================================================
scenario               n      declaration and estimands
=====================  =====  ===================================================================
``linear_gaussian``    4,000  doses 1.5, 2.5 and 3.5 at two consecutive triweight bandwidths of
                              :data:`CANDIDATE_BANDWIDTHS`: six pairs, each kernel's support
                              inside :math:`D`
=====================  =====  ===================================================================

**Candidate bandwidths.**  ``ctseff`` always computes its leave-one-out bandwidth risk, and at
a small bandwidth an isolated extreme dose makes the end node of its binned local-linear fit
singular: the risk is ``NA``, ``which.min`` selects nothing, and the call errors (review finding
B1).  The rate falls with the bandwidth.  :data:`CANDIDATE_BANDWIDTHS` is 1.0, 1.5, 2.0 and 2.5,
declared before any R output: the two first-declared values and two wider fallbacks, with
:math:`D` widened to hold every candidate's kernel at every dose.  The rule picks the first
consecutive pair whose pilot failure rate keeps the reference inside its 1% failure bound
(:func:`~tests.studies.dose_response_npcausal_pilot.declaration_from_pilot`).  The doses and
learners do not move.

**Matched kernels.**  ``DoseResponse`` offers compact kernels and ``ctseff`` a Gaussian one, so
one number cannot be "the same bandwidth" for both.  The subject uses the triweight, the compact
kernel nearest the Gaussian, and each run passes ``ctseff`` the Gaussian bandwidth
:math:`h_G = h\,R(K_G)/R(K_T)` (:func:`gaussian_bandwidth`), with :math:`R(K) = \int K^2` the
kernel's roughness: :math:`R(K_T) = 350/429` and :math:`R(K_G) = 1/(2\sqrt\pi)`.  At an interior
dose both estimators then have the leading variance :math:`R(K)\,E_W\{\sigma^2/g_0(a_0 \mid W)\}
/(nh)`, so the RMSE leg compares two estimators of one target at one variance order, not two
smoothing amounts.  A pair keeps one name, ``pair_name(h, a0)`` with the triweight :math:`h`,
for both implementations, and the R runner computes :math:`h_G` from that name.  ``bw.seq`` is
that one value, so ``ctseff``'s leave-one-out bandwidth selection has one candidate and selects
nothing, though it still computes the risk.

**Nuisances.**  Both implementations fit the outcome regression on the main terms and pairwise
products of :math:`(a, W)`: ``SL.glm.interaction`` in R and
``PolynomialFeatures(2, interaction_only=True)`` with ``LinearRegression`` here, the same
design, which holds the law's :math:`\bar Q_0` exactly.  ``ctseff`` fits the dose mean and
variance with the same library on :math:`W`, which holds both, and estimates the standardized
residual density by a kernel density estimate.  The subject fits the dose density with the S1
pooled-hazard learner (:func:`tests.studies.dose_response.hazard_learner`).  The library is one
learner, so ``SuperLearner``'s cross-validated weight is 1 and its prediction is that learner's
full-sample fit: the R fit does not depend on the fold draw.  Unmatched by construction, and
declared: the subject cross-fits over five folds and ``ctseff`` fits in sample; the two density
estimators differ; ``ctseff``'s point estimate is a binned local-linear fit (``KernSmooth::
locpoly``) and its standard error a sandwich, the subject's a pooled fluctuation with an
influence-curve standard error; ``ctseff``'s interval uses 1.96 and the subject's
:math:`z_{0.975}`.

**Evaluation grid.**  ``ctseff`` reports its curve on ``seq(a.rng[1], a.rng[2], length.out =
n.pts)``.  The runner passes ``a.rng`` as the observed dose range widened outward to multiples of
:data:`GRID_STEP` = 1/8 and ``n.pts`` to match, so the grid holds every declared dose exactly
(dyadic arithmetic) and spans the data, as ``ctseff``'s default range does.  A grid that stopped
short of the data would leave ``ctseff``'s smoothing splines to extrapolate (linearly) at the
outer doses.  :func:`npcausal_grid` mirrors the runner.

**Outcome range.**  The subject's linear fluctuation declares :data:`Q_BOUNDS`.  An outcome
outside it raises before any learner fits, which would stop the run, so the design test bounds
the chance of any such row over the declared run below 1e-9.

**Size.**  :data:`PRIMARY_REPLICATES` is the largest of three readings on the pilots, each a
multiple of 400: the shared rule :func:`~tests.studies.repeated_sizing.primary_replicates` at
X32b's clause shares with the subject's coverage and biases for both implementations, the same
rule with each implementation's own pilot rows, and the count at which the paired legs of every
cell outside the pre-registered red set pass together with probability 0.95
(:func:`~tests.studies.dose_response_npcausal_pilot.paired_replicates`).  The rule shares
nothing with the bandwidth selection beyond the selected pair.

**Publication.**  ``reporting``, declared before any run.  At dose 3.5 the subject's mean
reported standard error runs 10% to 11% above the exact efficient standard deviation at every
candidate bandwidth, against 1% to 4% at doses 1.5 and 2.5.  The reflected pilot fits
(:math:`A \mapsto 5 - A`) put the excess at the upper end of the dose axis again, so it comes
from the pooled-hazard density, not from the law.  At bandwidth 1 and 1.5 the spread stays near
efficient and the excess is an over-report, which the calibration leg reads; at 2.5 the spread
rises to meet the reported SE, so the SE is calibrated and the RMSE leg reads the excess
instead.  The design test pre-registers that red set by signature (:data:`EXPECTED_RED_DOSE`,
:data:`UPPER_DOSE_SE_RATIO`); any other red fails it and forces a diagnosis.

**Not covered.** The unsmoothed curve, a nonlinear curve (where the two targets part), binary
outcomes, boundary doses, the Epanechnikov and uniform kernels, bandwidth selection, the band,
strata, weights, clusters and the default learner library.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import cleverly.estimators.dose_response as dose_engine
from cleverly.dose_response import DoseResponse, KernelName, pair_name
from cleverly.estimators import DoseResponseTMLE
from tests.parallel import STUDY_JOBS
from tests.studies import dose_response_npcausal_law as law
from tests.studies.dose_response import hazard_learner
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

NPCAUSAL_COMMIT = "56a5ac117a29258b67b94874be662a171b5131f7"
NPCAUSAL_VERSION = "0.1.0"
R_BASE_IMAGE = (
    "rocker/r-ver:4.5.2@sha256:fd4ccdd3a4a6f7ef805e2daeee2a0fe3bf126bc231f36351223baecf5a595a4c"
)
#: The image the incremental studies build from ``tests/canonical/npcausal_incremental``.
#: It installs ``SuperLearner``, ``earth``, ``gam`` and ``ranger``, which ``ctseff`` requires;
#: ``KernSmooth`` is an R recommended package, which ``rocker/r-ver`` builds with
#: ``--with-recommended-packages``.
NPCAUSAL_IMAGE = "cleverly-npcausal:0.1.0-56a5ac1"
#: The one learner of every ``SuperLearner`` call ``ctseff`` makes.
SL_LIBRARY = "SL.glm.interaction"

#: The pilot rule's count on the committed R pilot (see the module docstring).
PRIMARY_REPLICATES = 4_800
PRIMARY_N = 4_000
#: The declared share of replicates one cell may lose to recordable fitting failures.
FAILURE_BOUND = 0.01
SEED = 2026119331
RESAMPLING_SEED = 2026119332

SUPPORT = law.SUPPORT
DOSES = (1.5, 2.5, 3.5)
#: The triweight bandwidths the pilots fit, in the order the selection rule reads them.
CANDIDATE_BANDWIDTHS = (1.0, 1.5, 2.0, 2.5)
#: The consecutive pair of :data:`CANDIDATE_BANDWIDTHS` the selection rule picks on the R pilot.
BANDWIDTHS = (2.0, 2.5)
KERNEL: KernelName = "triweight"
#: :math:`R(K) = \int K^2` of the triweight :math:`\tfrac{35}{32}(1 - u^2)^3` and of the
#: standard normal density.
TRIWEIGHT_ROUGHNESS = 350.0 / 429.0
GAUSSIAN_ROUGHNESS = 1.0 / (2.0 * math.sqrt(math.pi))
#: ``ctseff``'s evaluation grid step.
GRID_STEP = 0.125
#: The Gaussian outcome's declared range (see the module docstring).
Q_BOUNDS = (-25.0, 25.0)
#: The dose whose paired RMSE and calibration legs are pre-registered red (module docstring).
EXPECTED_RED_DOSE = 3.5
#: The pre-registered signature's floor on the subject's mean reported SE over the exact
#: efficient SD at :data:`EXPECTED_RED_DOSE`; the pilot reads 1.10 to 1.11 there.
UPPER_DOSE_SE_RATIO = 1.05

SCENARIO = "linear_gaussian"
COVARIATES = ("W1", "W2", "W3")
#: The name ``pair_name`` gives a pair; the R runner parses the same pattern.
PAIR_NAME = re.compile(r"^dose_response\[h=([0-9.]+),a=([0-9.]+)\]$")


def gaussian_bandwidth(h: float) -> float:
    """The Gaussian bandwidth of equal roughness per unit bandwidth: ``h R(K_G) / R(K_T)``.

    The R runner computes ``h * (1 / (2 * sqrt(pi))) / (350 / 429)``, the same operations in
    the same order, so the two agree to the last bit.
    """
    return h * GAUSSIAN_ROUGHNESS / TRIWEIGHT_ROUGHNESS


#: Every candidate triweight bandwidth -> the Gaussian bandwidth ``ctseff`` receives.
NPCAUSAL_BANDWIDTHS: dict[float, float] = {h: gaussian_bandwidth(h) for h in CANDIDATE_BANDWIDTHS}


def pairs_of(curve: DoseResponse) -> dict[str, tuple[float, float, float]]:
    """Estimand -> ``(dose, triweight bandwidth, Gaussian bandwidth)`` for each pair of ``curve``."""
    return {pair_name(h, a0): (a0, h, gaussian_bandwidth(h)) for h, a0 in curve.pairs}


CURVE = DoseResponse(DOSES, BANDWIDTHS, SUPPORT, kernel=KERNEL)
#: The curve the pilots fit: every declared dose at every candidate bandwidth.
PILOT_CURVE = DoseResponse(DOSES, CANDIDATE_BANDWIDTHS, SUPPORT, kernel=KERNEL)
ESTIMANDS = tuple(pair_name(h, a0) for h, a0 in CURVE.pairs)
PAIRS = pairs_of(CURVE)
PILOT_PAIRS = pairs_of(PILOT_CURVE)
TRUTH: dict[str, float] = {name: float(law.curve(a0)) for name, (a0, _, _) in PILOT_PAIRS.items()}

IMPLEMENTATION = "cleverly-dose-response"
REFERENCE = "npcausal"

STUDY = StudyRecord(
    name="kernel-smoothed dose-response CV-TMLE against npcausal ctseff",
    slug="dose-response-npcausal",
    artifacts=ROOT / "tests" / "canonical" / "dose_response_npcausal",
    document="docs/technical-reference/method-evidence/dose-response-npcausal.md",
    anchor="kernel-smoothed-dose-response-cv-tmle-against-npcausal-ctseff",
    scenarios={SCENARIO: ESTIMANDS},
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=REFERENCE,
    failure_bound={IMPLEMENTATION: FAILURE_BOUND, REFERENCE: FAILURE_BOUND},
    comparison="related target",
    publication_policy="reporting",
    modules=(
        "tests/studies/dose_response_npcausal.py",
        "tests/studies/dose_response_npcausal_properties.py",
        "tests/studies/dose_response_npcausal_law.py",
        "tests/studies/dose_response.py",
        "tests/studies/dose_response_law.py",
        "tests/studies/evidence/comparison.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.dose_response_npcausal",
    properties_module="tests.studies.dose_response_npcausal_properties",
    property_cells={},
)

REFERENCE_METADATA = {
    "npcausal_commit": NPCAUSAL_COMMIT,
    "npcausal_version": NPCAUSAL_VERSION,
    "r_base_image": R_BASE_IMAGE,
    "image": NPCAUSAL_IMAGE,
    "reference_parameter": (
        "ctseff: local-linear kernel regression of the doubly robust pseudo-outcome on the dose, "
        "Gaussian kernel; equal to theta(a0) at this law, where theta is linear"
    ),
    "reference_call": (
        "ctseff(y, a, x = cbind(W1, W2, W3), bw.seq = <one Gaussian bandwidth>, "
        "n.pts = 8 (hi - lo) + 1, a.rng = c(lo, hi), sl.lib = 'SL.glm.interaction'), with lo "
        "and hi the observed dose range widened to multiples of 1/8; one call per bandwidth"
    ),
    "bandwidths": {f"{h:g}": NPCAUSAL_BANDWIDTHS[h] for h in BANDWIDTHS},
    "candidate_bandwidths": list(CANDIDATE_BANDWIDTHS),
    "reference_nuisances": "in sample, one learner per SuperLearner call (weight 1)",
    "standard_error": "res$se / sqrt(n), the interval res$ci.ll and res$ci.ul (1.96)",
}

CONFIGURATION = {
    "construction": "CV-TMLE of the kernel-smoothed curve (one pooled fluctuation per pair)",
    "support": list(SUPPORT),
    "kernel": KERNEL,
    "doses": list(DOSES),
    "bandwidths": list(BANDWIDTHS),
    "npcausal_bandwidths": {f"{h:g}": NPCAUSAL_BANDWIDTHS[h] for h in BANDWIDTHS},
    "nodes_per_piece": 8,
    "n_folds": 5,
    "density_bins": "default: max(20, ceil(2 n^(1/3)))",
    "fluctuation": "linear",
    "q_bounds": list(Q_BOUNDS),
    "learners": (
        "outcome: PolynomialFeatures(2, interaction_only=True) of (a, W) with LinearRegression; "
        "density: pooled-hazard binned density with the S1 HazardDesign "
        "[indicators, W, W^2, bin*W, bin*W^2], StandardScaler, LogisticRegression(C=100)"
    ),
    "initial_estimate": "the untargeted plug-in of the same cross-fitted regression",
    "variance_method": "ic",
}

# ---------------------------------------------------------------------------- the subject


def outcome_learner() -> Pipeline:
    """Main terms and pairwise products of ``[a, W]``: the design of ``SL.glm.interaction``."""
    return make_pipeline(
        PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
        LinearRegression(),
    )


def estimator(**overrides: Any) -> DoseResponseTMLE:
    """The declared estimator; ``overrides`` replace a learner or the curve."""
    settings: dict[str, Any] = {
        "outcome_learner": outcome_learner(),
        "treatment_learner": hazard_learner(len(COVARIATES)),
        "fluctuation": "linear",
        "q_bounds": Q_BOUNDS,
        "random_state": 0,
    }
    curve = overrides.pop("curve", CURVE)
    settings.update(overrides)
    return DoseResponseTMLE(curve, **settings)


def fit_with_initials(frame: pd.DataFrame, **overrides: Any) -> tuple[Any, dict[str, float]]:
    """The declared fit, and each pair's untargeted plug-in on the outcome's scale.

    The fit targets every pair on the declared rule and then on the doubled rule.  The observer
    records the weighted plug-in of each of the first calls to ``target_pair``, the value the
    fluctuation starts from, and passes every call through unchanged (the S1 construction).
    """
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
        result = estimator(**overrides).fit(
            frame, outcome="Y", treatment="A", covariates=list(COVARIATES)
        )
    names = list(result.estimates)
    if len(seen) != 2 * len(names):
        raise AssertionError(f"expected {2 * len(names)} fluctuations, observed {len(seen)}")
    lower, upper = Q_BOUNDS
    return result, {
        name: lower + (upper - lower) * value for name, value in zip(names, seen, strict=False)
    }


# ---------------------------------------------------------------------------- the reference


def npcausal_grid(a: np.ndarray) -> np.ndarray:
    """The ``a.vals`` grid the runner gives ``ctseff``: the dose range widened to 1/8 steps."""
    low = math.floor(float(np.min(a)) / GRID_STEP) * GRID_STEP
    high = math.ceil(float(np.max(a)) / GRID_STEP) * GRID_STEP
    points = round((high - low) / GRID_STEP) + 1
    # R's ``seq(from, to, length.out = k)`` is ``from + i * ((to - from) / (k - 1))``.
    return low + np.arange(points) * ((high - low) / (points - 1))


# ---------------------------------------------------------------------------- the draws


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if scenario != SCENARIO:
        raise KeyError(scenario)
    return law.sample(n, seed), {name: TRUTH[name] for name in ESTIMANDS}


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
    """One row per pair on the identity scale."""
    rows = []
    for name in ESTIMANDS:
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


def _reference_input(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """The replicate's sample and truth rows, which need no fit, and no subject rows."""
    scenario, replicate, _ = key
    frame, reference = drawn
    sample = frame.copy()
    sample.insert(0, "replicate", replicate)
    sample.insert(0, "scenario", scenario)
    truth_rows = [
        {"scenario": scenario, "replicate": replicate, "estimand": name, "truth": value}
        for name, value in reference.items()
    ]
    return sample, truth_rows, []


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    scenario, replicate, n = key
    frame, reference = drawn
    sample, truth_rows, _ = _reference_input(key, drawn)
    result, initial = fit_with_initials(frame)
    return sample, truth_rows, result_rows(result, reference, scenario, replicate, initial, n)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Every primary row and the samples and truths the reference fits."""
    outcomes = replication.fit_primary(
        [(SCENARIO, replicate, n) for replicate in range(replicates)],
        _replicate,
        draw=_draw,
        reference_input=_reference_input,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    samples = pd.concat([sample for sample, _, _ in outcomes.kept], ignore_index=True)
    truth_rows = pd.DataFrame([row for _, rows, _ in outcomes.kept for row in rows])
    estimates = pd.DataFrame([row for _, _, rows in outcomes.kept for row in rows])
    return replication.result(
        outcomes,
        estimates.loc[:, list(REPLICATE_COLUMNS)],
        samples=samples,
        truths=truth_rows,
    )
