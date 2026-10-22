"""Registered evidence for the point-treatment refutation verdict rules (X82 follow-up SP).

The subject is not an estimator but three decision rules that :func:`cleverly.refute` prints
beside its statistics.

=====================  ==========================================================================
rule                   the reading
=====================  ==========================================================================
placebo ``t``          the one-sample ``t`` of 20 placebo refits against the exact null,
                       two-sided at 0.05 under ``t_19`` (:func:`~cleverly.validation._readings.placebo_reading`)
subset ``rho``         the scatter of 20 subset refits against ``SE sqrt((n/m - 1) n/(n-1))``,
                       two-sided at 0.05 under ``chi^2_19`` (:func:`~cleverly.validation._readings.subset_reading`)
negative control ``z`` the fit refitted on a declared negative-control outcome, ``z`` against the
                       no-effect value, two-sided at 0.05
=====================  ==========================================================================

The X82 plan (rev 5, sections 8.1 to 8.6, plans repository) declares the design. Each cell
checks that a rule is not grossly miscalibrated on one configuration. It does not certify size
to two decimals.

**The law.** Twenty standard normal covariates, an unmeasured standard normal ``U``, and a binary
treatment with ``logit P(A = 1 | W, U) = 0.4 W1 - 0.3 W2 + 0.05 (W3 + ... + W20) + 0.5 U``. The
outcome never reads ``U``, so ``U`` confounds nothing the fit targets. It drives the treatment and
the confounded negative control only. The Gaussian outcome is
``Y = A + 0.5 W1 + 0.4 W2 + 0.05 (W3 + ... + W20) + e`` with ``e ~ N(0, 1)``. The binary outcome
is ``Y ~ Bernoulli(expit(-2.65 + 0.5 A + 0.5 W1 + 0.4 W2 + 0.05 (W3 + ... + W20)))``, whose
prevalence is 0.100. Each law carries two negative controls of its outcome family:

- ``NC``, a valid control: the outcome's covariate part, no treatment term, no ``U``.
- ``NCU``, a confounded control: the same plus ``delta U`` (Gaussian) or ``delta U`` on the logit
  (binary). ``U`` drives the treatment, so the control's adjusted contrast is not zero.

Both covariate indices are linear in ``W``, so ``(eta, L)`` is bivariate normal and every
population quantity below is a three-dimensional Gauss-Hermite sum (:func:`quadrature`).

**The cells** (plan 8.5). ``n = 2,000``, five-fold cross-fitting, ``R = 20`` refits.

====  ===================================================  ========  ========================
cell  configuration                                        outcome   size readings
====  ===================================================  ========  ========================
T1    ``TMLE`` ATE, GLM nuisances                          Gaussian  placebo, subset, control
T2    ``TMLE`` ATE, GLM nuisances                          binary    placebo, subset, control
T3    ``TMLE`` ATE, depth-2 ``HistGradientBoosting``       binary    placebo
C1    ``CTMLE`` greedy, GLM nuisances                      binary    placebo
D1    ``DRTMLE``, declared complete-data mechanism, GLM    binary    placebo, subset, control
D2    ``DRTMLE``, estimated mechanism, GLM                 binary    placebo
====  ===================================================  ========  ========================

**Size cells.** 200 trials. A reading passes when it rejects in at most 17 trials.

**Power cells.** Trials 0 to 49 of each size cell also run the defects below. A power reading
passes when it detects its defect in at least 38 of 50 trials. Each defect is sized from the
declared law so that its per-trial power is at least 0.95 (``tests/unit/test_refuter_calibration_point_design.py``).
For the stale-row placebo the 0.95 is approximate: the design test sizes the population shift,
and the shift of one trial's placebo varies about it by roughly ``sqrt(q)`` permutation SD. The
count rule passes with probability 0.999 at a per-trial power of 0.90, so the slack is harmless.

====================  ==========================================================================
defect                construction
====================  ==========================================================================
stale rows            the placebo keeps the recorded treatment on a declared share ``q`` of rows
                      and permutes the rest among themselves (:data:`STALE_SHARE`). On D1 the
                      stale rows also make the declared arm shares wrong for the placebo
                      column, which is a second defect the placebo should catch
half SE, double SE    the subset ``rho`` of the same trial recomputed with the reported SE halved
                      and doubled. ``rho`` is linear in ``1 / SE``, so this is exact and costs no fit
confounded control    the negative-control fit on ``NCU`` instead of ``NC`` (:data:`CONTROL_DELTA`)
====================  ==========================================================================

The power cells are the controls of the size cells: a rule that never rejects passes its size
cell and fails its power cell.

**Fit configuration.** Every trial reads the configuration off its fitted result
(:func:`~cleverly.validation._interpretation.fit_configuration` with ``replicates = 20``),
including each learner's fingerprint with the booster's resolved ``early_stopping``. The run
stops if any axis but the smallest arm event count varies across a cell's trials. ``cells.csv``
records the one configuration of each cell and the range of its event count. A registry row
licensed by this study covers that configuration.

**Red reading** (declared before the run). A size red is a defect suspect until diagnosed. A red
with no defect goes to reporting with an owner, and the rule's registry row stays descriptive.
No rerun buys a pass. A power red with no defect means the defect was sized too small for this
cell's configuration; it keeps the rule descriptive and goes to reporting.
"""

from __future__ import annotations

import contextlib
import importlib
import math
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

from cleverly.estimators import CTMLE, DRTMLE, TMLE
from cleverly.validation import refute
from cleverly.validation._interpretation import fit_configuration
from tests.parallel import STUDY_JOBS
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

N = 2_000
#: The row count of every fit, as the shared driver reads it.
PRIMARY_N = N
N_FOLDS = 5
COVARIATES = tuple(f"W{j}" for j in range(1, 21))
SEED = 20268251
RESAMPLING_SEED = 20268252

#: Refits per placebo and subset reading: the library default (X82 rev 5).
REFITS = 20
SIZE_TRIALS = 200
POWER_TRIALS = 50
#: A size reading passes at no more than this many rejections in :data:`SIZE_TRIALS`.
SIZE_MAX_REJECTIONS = 17
#: A power reading passes at no fewer than this many detections in :data:`POWER_TRIALS`.
POWER_MIN_DETECTIONS = 38
#: The level every rule reads.
LEVEL = 0.05

#: The treatment law: ``logit P(A = 1 | W, U) = eta(W) + GAMMA_U U``.
TREATMENT_STRONG = (0.4, -0.3)
GAMMA_U = 0.5
#: The outcome's covariate index ``L(W) = 0.5 W1 + 0.4 W2 + WEAK (W3 + ... + W20)``.
OUTCOME_STRONG = (0.5, 0.4)
#: The coefficient of each of the 18 weak confounders, in both indices.
WEAK = 0.05
GAUSSIAN_EFFECT = 1.0
BINARY_INTERCEPT = -2.65
BINARY_EFFECT = 0.5

#: The share of rows whose placebo keeps the recorded treatment, by outcome family. The design
#: test requires the population shift of the placebo estimand to reach 0.850 placebo SD.
STALE_SHARE = {"gaussian": 0.05, "binomial": 0.35}
#: The strength of ``U`` in the confounded control, by family. The design test requires its
#: adjusted contrast to reach 3.605 control SE.
CONTROL_DELTA = {"gaussian": 0.45, "binomial": 1.5}

#: The declared Gaussian outcome bounds. A cross-fitted continuous fit needs declared bounds;
#: these never bind on this law (:func:`draw_law` refuses a sample outside them).
GAUSSIAN_Q_BOUNDS = (-12.0, 12.0)

#: The booster of cell T3, as the plan names it. ``early_stopping="auto"`` resolves to off at
#: ``n = 2,000``. Each trial records the resolved value in its fit configuration
#: (:func:`trial_configuration`), and the design test checks the resolution.
BOOSTER = {"max_depth": 2, "max_iter": 100}

SIZE_FAMILY = "refuter_size"
POWER_FAMILY = "refuter_power"

READINGS_ALL = ("placebo", "subset", "negative_control")
READINGS_PLACEBO = ("placebo",)


@dataclass(frozen=True)
class Cell:
    """One configuration of plan 8.5.

    Parameters
    ----------
    family : str
        ``"gaussian"`` or ``"binomial"``: the outcome law.
    readings : tuple of str
        The size readings the cell reads.
    estimator : str
        ``"tmle"``, ``"tmle_booster"``, ``"ctmle_greedy"``, ``"drtmle_known"`` or
        ``"drtmle_estimated"``.
    """

    family: str
    readings: tuple[str, ...]
    estimator: str

    @property
    def defects(self) -> tuple[str, ...]:
        """The power readings of this cell, one per defect."""
        out = ["placebo_stale_rows"]
        if "subset" in self.readings:
            out += ["subset_half_se", "subset_double_se"]
        if "negative_control" in self.readings:
            out.append("negative_control_confounded")
        return tuple(out)


CELLS: dict[str, Cell] = {
    "T1": Cell("gaussian", READINGS_ALL, "tmle"),
    "T2": Cell("binomial", READINGS_ALL, "tmle"),
    "T3": Cell("binomial", READINGS_PLACEBO, "tmle_booster"),
    "C1": Cell("binomial", READINGS_PLACEBO, "ctmle_greedy"),
    "D1": Cell("binomial", READINGS_ALL, "drtmle_known"),
    "D2": Cell("binomial", READINGS_PLACEBO, "drtmle_estimated"),
}

PROPERTY_CELLS = {
    SIZE_FAMILY: tuple(
        f"{name}__{reading}" for name, cell in CELLS.items() for reading in cell.readings
    ),
    POWER_FAMILY: tuple(
        f"{name}__{defect}" for name, cell in CELLS.items() for defect in cell.defects
    ),
}

#: The primary phase: the T1 and T2 fits' ATE against its exact truth. It is a sanity row on
#: the law; no verdict of this study reads it.
SCENARIOS = {"gaussian": ("ate",), "binomial": ("ate",)}
PRIMARY_REPLICATES = 200

STUDY = StudyRecord(
    name="point-treatment refutation verdict rules",
    slug="refuter-calibration-point",
    artifacts=ROOT / "tests" / "canonical" / "refuter_calibration_point",
    document="docs/technical-reference/method-evidence/refuter-calibration-point.md",
    anchor="point-treatment-refutation-verdict-rules",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-refute-point",
    reference=None,
    failure_bound={"cleverly-refute-point": 0.01},
    modules=(
        "tests/studies/refuter_calibration_point.py",
        "tests/studies/refuter_calibration_point_properties.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.refuter_calibration_point",
    properties_module="tests.studies.refuter_calibration_point_properties",
    property_cells=PROPERTY_CELLS,
    property_artifacts=("cells.csv",),
    publication_policy="reporting",
)

CONFIGURATION = {
    "folds": N_FOLDS,
    "refits": REFITS,
    "size_trials": SIZE_TRIALS,
    "power_trials": POWER_TRIALS,
    "size_pass": f"rejections <= {SIZE_MAX_REJECTIONS} of {SIZE_TRIALS}",
    "power_pass": f"detections >= {POWER_MIN_DETECTIONS} of {POWER_TRIALS}",
    "treatment": "logit g = 0.4 W1 - 0.3 W2 + 0.05 (W3 + ... + W20) + 0.5 U",
    "gaussian_outcome": "Y = A + 0.5 W1 + 0.4 W2 + 0.05 (W3 + ... + W20) + N(0, 1)",
    "binary_outcome": "logit P(Y = 1) = -2.65 + 0.5 A + 0.5 W1 + 0.4 W2 + 0.05 (W3 + ... + W20)",
    "stale_share": dict(STALE_SHARE),
    "control_delta": dict(CONTROL_DELTA),
    "cells": {
        name: {"family": cell.family, "readings": list(cell.readings), "estimator": cell.estimator}
        for name, cell in CELLS.items()
    },
}


# ---- the law ------------------------------------------------------------------------------


def treatment_index(covariates: np.ndarray) -> np.ndarray:
    """``eta(W)``, the treatment's covariate index."""
    weak = WEAK * covariates[:, 2:].sum(axis=1)
    return TREATMENT_STRONG[0] * covariates[:, 0] + TREATMENT_STRONG[1] * covariates[:, 1] + weak


def outcome_index(covariates: np.ndarray) -> np.ndarray:
    """``L(W)``, the outcome's covariate index."""
    weak = WEAK * covariates[:, 2:].sum(axis=1)
    return OUTCOME_STRONG[0] * covariates[:, 0] + OUTCOME_STRONG[1] * covariates[:, 1] + weak


#: Gauss-Hermite nodes and weights for a standard normal.
_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(40)
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()


def propensity(eta: np.ndarray) -> np.ndarray:
    """``P(A = 1 | W)``: the law's mechanism with ``U`` integrated out."""
    return np.asarray((expit(eta[:, None] + GAMMA_U * _NODES[None, :]) * _WEIGHTS).sum(axis=1))


def draw_law(family: str, n: int, seed: int) -> pd.DataFrame:
    """One sample of ``n`` rows: ``W1..W20, A, Y, NC, NCU`` and the declared mechanism ``p0, p1``."""
    rng = np.random.default_rng(seed)
    covariates = rng.standard_normal((n, len(COVARIATES)))
    latent = rng.standard_normal(n)
    eta = treatment_index(covariates)
    treatment = rng.binomial(1, expit(eta + GAMMA_U * latent)).astype(float)
    index = outcome_index(covariates)
    delta = CONTROL_DELTA[family]
    if family == "gaussian":
        outcome = GAUSSIAN_EFFECT * treatment + index + rng.standard_normal(n)
        control = index + rng.standard_normal(n)
        confounded = index + delta * latent + rng.standard_normal(n)
        low, high = GAUSSIAN_Q_BOUNDS
        if outcome.min() <= low or outcome.max() >= high:
            raise RuntimeError("a Gaussian draw left the declared q_bounds")
    elif family == "binomial":
        base = BINARY_INTERCEPT + index
        outcome = rng.binomial(1, expit(base + BINARY_EFFECT * treatment)).astype(float)
        control = rng.binomial(1, expit(base)).astype(float)
        confounded = rng.binomial(1, expit(base + delta * latent)).astype(float)
    else:
        raise ValueError(f"unknown family {family!r}")
    frame = pd.DataFrame(covariates, columns=list(COVARIATES))
    g = propensity(eta)
    frame["A"] = treatment
    frame["Y"] = outcome
    frame["NC"] = control
    frame["NCU"] = confounded
    frame["p0"] = 1.0 - g
    frame["p1"] = g
    return frame


@dataclass(frozen=True)
class Quadrature:
    """The law's population quantities as Gauss-Hermite sums over ``(eta, L, U)``.

    ``(eta, L)`` is bivariate normal, because both indices are linear in ``W``.
    """

    weights: np.ndarray
    eta: np.ndarray
    index: np.ndarray
    g: np.ndarray
    posterior: Mapping[int, np.ndarray]

    def mean(self, values: np.ndarray) -> float:
        return float(np.sum(self.weights * values))


def quadrature(nodes: int = 40) -> Quadrature:
    """Build the quadrature of the declared law."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    weak_variance = (len(COVARIATES) - 2) * WEAK**2
    var_eta = TREATMENT_STRONG[0] ** 2 + TREATMENT_STRONG[1] ** 2 + weak_variance
    var_l = OUTCOME_STRONG[0] ** 2 + OUTCOME_STRONG[1] ** 2 + weak_variance
    cov = (
        TREATMENT_STRONG[0] * OUTCOME_STRONG[0]
        + TREATMENT_STRONG[1] * OUTCOME_STRONG[1]
        + weak_variance
    )
    l11 = math.sqrt(var_eta)
    l21 = cov / l11
    l22 = math.sqrt(var_l - l21**2)
    z1, z2 = np.meshgrid(x, x, indexing="ij")
    eta = (l11 * z1).ravel()
    index = (l21 * z1 + l22 * z2).ravel()
    weights = np.outer(w, w).ravel()
    given_u = expit(eta[:, None] + GAMMA_U * x[None, :])
    g = (given_u * w).sum(axis=1)
    posterior = {}
    for arm, likelihood in ((1, given_u), (0, 1.0 - given_u)):
        joint = likelihood * w
        posterior[arm] = joint / joint.sum(axis=1, keepdims=True)
    return Quadrature(weights, eta, index, g, {**posterior, -1: x[None, :]})


def outcome_means(family: str, law: Quadrature) -> tuple[np.ndarray, np.ndarray]:
    """``E[Y | A = 1, W]`` and ``E[Y | A = 0, W]`` on the quadrature."""
    if family == "gaussian":
        return law.index + GAUSSIAN_EFFECT, law.index
    base = BINARY_INTERCEPT + law.index
    return expit(base + BINARY_EFFECT), expit(base)


def truth(family: str, law: Quadrature | None = None) -> float:
    """The exact ATE of the family's outcome."""
    law = quadrature() if law is None else law
    mu1, mu0 = outcome_means(family, law)
    return law.mean(mu1 - mu0)


def prevalence(law: Quadrature | None = None) -> float:
    """The binary outcome's marginal prevalence."""
    law = quadrature() if law is None else law
    mu1, mu0 = outcome_means("binomial", law)
    return law.mean(law.g * mu1 + (1.0 - law.g) * mu0)


def placebo_shift(family: str, share: float, n: int = N, law: Quadrature | None = None) -> float:
    """The stale-row placebo's population contrast in units of the placebo SD.

    A share ``q`` of rows keeps ``A``; the rest carry a draw independent of ``(W, Y)`` with the
    marginal share ``pbar``. Then ``E[Y | A* = a, W]`` mixes ``E[Y | A = a, W]`` and
    ``E[Y | W]`` with the weights ``P(kept | A* = a, W)``. The SD is the efficiency SD of the
    ATE under the placebo law, ``sqrt(E[Var(Y | W)] / (pbar (1 - pbar) n))``.
    """
    law = quadrature() if law is None else law
    mu1, mu0 = outcome_means(family, law)
    g = law.g
    pbar = law.mean(g)
    marginal = g * mu1 + (1.0 - g) * mu0
    if family == "gaussian":
        variance = 1.0 + g * (1.0 - g) * GAUSSIAN_EFFECT**2
    else:
        variance = marginal * (1.0 - marginal)
    star = share * g + (1.0 - share) * pbar
    treated = (share * g * mu1 + (1.0 - share) * pbar * marginal) / star
    control = (share * (1.0 - g) * mu0 + (1.0 - share) * (1.0 - pbar) * marginal) / (1.0 - star)
    shift = law.mean(treated - control)
    sd = math.sqrt(law.mean(variance) / (pbar * (1.0 - pbar) * n))
    return shift / sd


def control_shift(family: str, delta: float, n: int = N, law: Quadrature | None = None) -> float:
    """The confounded control's adjusted contrast in units of its efficiency SD at ``n``."""
    law = quadrature() if law is None else law
    nodes = law.posterior[-1]
    g = law.g
    if family == "gaussian":

        def mean(arm: int) -> np.ndarray:
            return law.index + delta * (law.posterior[arm] * nodes).sum(axis=1)

        def variance(arm: int) -> np.ndarray:
            first = (law.posterior[arm] * nodes).sum(axis=1)
            second = (law.posterior[arm] * nodes**2).sum(axis=1)
            return 1.0 + delta**2 * (second - first**2)

    else:
        given_u = expit(BINARY_INTERCEPT + law.index[:, None] + delta * nodes)

        def mean(arm: int) -> np.ndarray:
            return np.asarray((law.posterior[arm] * given_u).sum(axis=1))

        def variance(arm: int) -> np.ndarray:
            value = mean(arm)
            return value * (1.0 - value)

    m1, m0 = mean(1), mean(0)
    contrast = law.mean(m1 - m0)
    efficiency = (
        law.mean(variance(1) / g + variance(0) / (1.0 - g)) + law.mean((m1 - m0) ** 2) - contrast**2
    )
    return contrast / math.sqrt(efficiency / n)


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary sample from an explicit seed for the published-seed audit."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return draw_law(scenario, n, seed), {"ate": truth(scenario)}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


# ---- the fits -----------------------------------------------------------------------------


class DeclaredOnly(BaseEstimator, ClassifierMixin):
    """A treatment learner that must never run: the declared mechanism replaces it."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> DeclaredOnly:
        raise AssertionError("a fit on a declared mechanism fitted a treatment learner")

    def predict_proba(self, design: Any) -> Any:
        raise AssertionError("a fit on a declared mechanism fitted a treatment learner")


def _logistic() -> LogisticRegression:
    return LogisticRegression(C=np.inf, max_iter=2_000)


def _booster() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(**BOOSTER)


def estimator(cell: Cell, seed: int) -> Any:
    """The cell's estimator, seeded with the trial's fit seed."""
    common: dict[str, Any] = {
        "cross_fit": True,
        "n_folds": N_FOLDS,
        "estimands": ("ate",),
        "simultaneous": False,
        "random_state": seed,
    }
    if cell.estimator == "tmle":
        if cell.family == "gaussian":
            return TMLE(
                outcome_learner=LinearRegression(),
                treatment_learner=_logistic(),
                q_bounds=GAUSSIAN_Q_BOUNDS,
                **common,
            )
        return TMLE(outcome_learner=_logistic(), treatment_learner=_logistic(), **common)
    if cell.estimator == "tmle_booster":
        return TMLE(outcome_learner=_booster(), treatment_learner=_booster(), **common)
    if cell.estimator == "ctmle_greedy":
        return CTMLE(
            strategy="greedy", outcome_learner=_logistic(), treatment_learner=_logistic(), **common
        )
    if cell.estimator in ("drtmle_known", "drtmle_estimated"):
        known = cell.estimator == "drtmle_known"
        return DRTMLE(
            outcome_learner=_logistic(),
            treatment_learner=DeclaredOnly() if known else _logistic(),
            reduced_outcome_learner=LinearRegression(),
            reduced_treatment_learner=_logistic(),
            **common,
        )
    raise ValueError(f"unknown estimator {cell.estimator!r}")


def fit_cell(name: str, frame: pd.DataFrame, seed: int) -> Any:
    """Fit cell ``name`` on one sample."""
    cell = CELLS[name]
    keywords: dict[str, Any] = {"outcome": "Y", "treatment": "A", "covariates": list(COVARIATES)}
    if cell.estimator == "drtmle_known":
        keywords["treatment_probabilities"] = {0.0: "p0", 1.0: "p1"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator(cell, seed).fit(frame, **keywords).single()


# ---- the readings -------------------------------------------------------------------------


@dataclass(frozen=True)
class Reading:
    """One rule's statistic on one trial, and whether the rule rejects at 0.05."""

    statistic: float
    pvalue: float
    rejected: bool


def _reading(test: Any) -> Reading:
    statistic = float("nan") if test.statistic is None else float(test.statistic)
    pvalue = float("nan") if test.statistic_pvalue is None else float(test.statistic_pvalue)
    # ``nan`` is the placebo's constant-zero case, which does not reject. A constant nonzero
    # offset carries p = 0 and rejects (``PlaceboReading.rejects``).
    return Reading(statistic, pvalue, bool(pvalue < LEVEL))


def rescaled_subset(rho: float, factor: float, refits: int = REFITS) -> Reading:
    """The subset reading with the reported SE multiplied by ``factor``.

    ``rho = spread / (SE c)``, so the mutated ``rho`` is ``rho / factor``, and the two-sided
    ``chi^2_{R-1}`` probability follows as in :func:`~cleverly.validation._readings.subset_reading`.
    """
    mutated = rho / factor
    statistic = (refits - 1) * mutated**2
    low = stats.chi2.cdf(statistic, refits - 1)
    pvalue = float(min(1.0, 2.0 * min(low, 1.0 - low)))
    return Reading(mutated, pvalue, bool(pvalue < LEVEL))


@contextlib.contextmanager
def stale_rows(share: float) -> Iterator[None]:
    """Patch the placebo so a share of rows keeps the recorded treatment.

    The kept rows are a uniform draw of ``round(share n)`` rows; the other rows' treatments are
    permuted among themselves, so the marginal law of the placebo column is unchanged.
    """
    module = importlib.import_module("cleverly.validation.refute")
    original: Callable[..., np.ndarray] = module._permuted_treatment

    def mutated(data: Any, rng: np.random.Generator) -> np.ndarray:
        treatment = np.asarray(data.treatment, dtype=float).copy()
        n = treatment.size
        kept = rng.choice(n, size=round(share * n), replace=False)
        moved = np.setdiff1d(np.arange(n), kept)
        treatment[moved] = rng.permutation(treatment[moved])
        return treatment

    module._permuted_treatment = mutated
    try:
        yield
    finally:
        module._permuted_treatment = original


def trial_seed(stream: str, name: str, trial: int) -> int:
    """The seed of one named stream of one property trial.

    The streams are ``"property_sample"`` (the draw), ``"property_fit"`` (the fit's folds and
    learners), ``"refute_size"``, ``"refute_stale"`` and ``"refute_control"`` (the refits).
    """
    return stream_seed(STUDY, stream, name, trial)


def trial_configuration(result: Any) -> dict[str, Any]:
    """The fit configuration of one trial, read off the fitted result, as one flat row."""
    configuration = replace(fit_configuration(result, "ate"), replicates=REFITS)
    row = asdict(configuration)
    row["learners"] = " | ".join(configuration.learners)
    return row


@dataclass(frozen=True)
class Trial:
    """One trial's readings by name, and the configuration of its fit."""

    readings: dict[str, Reading]
    configuration: dict[str, Any]


def trial_readings(name: str, frame: pd.DataFrame, trial: int, *, power: bool) -> Trial:
    """Every size reading of one trial, its power readings when ``power``, and its fit."""
    cell = CELLS[name]
    result = fit_cell(name, frame, trial_seed("property_fit", name, trial))
    configuration = trial_configuration(result)
    tests = ["placebo"]
    if "subset" in cell.readings:
        tests.append("subset")
    if "negative_control" in cell.readings:
        tests.append("negative_control_outcome")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = refute(
            result,
            tests=tuple(tests),
            n_replicates=REFITS,
            negative_control_outcome=frame["NC"].to_numpy(dtype=float)
            if "negative_control" in cell.readings
            else None,
            random_state=trial_seed("refute_size", name, trial),
        )
    out = {"placebo": _reading(report["placebo"])}
    if "subset" in cell.readings:
        out["subset"] = _reading(report["subset"])
    if "negative_control" in cell.readings:
        out["negative_control"] = _reading(report["negative_control_outcome"])
    if not power:
        return Trial(out, configuration)
    with warnings.catch_warnings(), stale_rows(STALE_SHARE[cell.family]):
        warnings.simplefilter("ignore")
        stale = refute(
            result,
            tests=("placebo",),
            n_replicates=REFITS,
            random_state=trial_seed("refute_stale", name, trial),
        )
    out["placebo_stale_rows"] = _reading(stale["placebo"])
    if "subset" in cell.readings:
        rho = out["subset"].statistic
        out["subset_half_se"] = rescaled_subset(rho, 0.5)
        out["subset_double_se"] = rescaled_subset(rho, 2.0)
    if "negative_control" in cell.readings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            confounded = refute(
                result,
                tests=("negative_control_outcome",),
                negative_control_outcome=frame["NCU"].to_numpy(dtype=float),
                random_state=trial_seed("refute_control", name, trial),
            )
        out["negative_control_confounded"] = _reading(confounded["negative_control_outcome"])
    return Trial(out, configuration)


# ---- the primary phase --------------------------------------------------------------------


def cleverly_rows(
    frame: pd.DataFrame, truth_map: Mapping[str, float], scenario: str, replicate: int, seed: int
) -> list[dict[str, Any]]:
    """The T1 (Gaussian) or T2 (binary) fit's ATE as one primary row."""
    result = fit_cell("T1" if scenario == "gaussian" else "T2", frame, seed)
    estimate = result["ate"]
    low, high = (float(value) for value in estimate.ci)
    reference = float(truth_map["ate"])
    return [
        {
            "implementation": STUDY.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": result.n,
            "estimand": "ate",
            "truth": reference,
            "estimate": float(estimate.psi),
            "inference_estimate": float(estimate.psi),
            "std_error": float(estimate.std_error),
            "ci_lower": low,
            "ci_upper": high,
            "inference_scale": "identity",
            "covered": int(low <= reference <= high),
            "initial_estimate": math.nan,
        }
    ]


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, index, n = key
    return draw_scenario(scenario, n, index)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, index, _ = key
    seed = replicate_seed(STUDY, scenario, index)
    frame, truth_map = drawn
    return cleverly_rows(frame, truth_map, scenario, index, seed)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication."""
    outcomes = replication.fit_primary(
        [(scenario, index, n) for scenario in STUDY.scenarios for index in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for records in outcomes.kept for row in records])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])
