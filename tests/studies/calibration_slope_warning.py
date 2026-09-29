"""Registered evidence for the calibration-slope warning rule (RM15).

The subject is not an estimator of a causal parameter but a diagnostic: the calibration slope
that :func:`cleverly.validation.nuisance.nuisance_diagnostics` reports for a propensity model,
and the rule that turns it into a finding.  The slope is a logistic recalibration of the
treatment on ``logit(g_hat)`` with one intercept per validation fold, and its standard error is
the sandwich.  The rule warns when the Bonferroni interval over the tested models lies above 0
and excludes 1.

Every law here has four standard normal covariates unless it states one, a binary treatment,
and the outcome ``Y ~ Bernoulli(expit(0.5 A + 0.5 W1 - 0.25))``.  The outcome learner returns
that known regression, so the outcome report is calibrated and each fit tests two models.  The
primary scenarios give the treatment learner the known propensity too, so the slope of each fold
is Cox's recalibration of a fixed rule on rows it never saw, and its truth is 1.

The RM15 plan in ``docs/roadmap.md`` declared this study, its laws, its budget, its verdicts and
its red-cell route before any run.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

from cleverly.estimators import TMLE
from cleverly.exceptions import PositivityWarning
from cleverly.utils.bounds import logit
from cleverly.utils.parallel import map_parallel
from cleverly.validation.nuisance import (
    CALIBRATION_FAMILY_ALPHA,
    _critical_value,
    nuisance_diagnostics,
)
from tests.parallel import STUDY_JOBS
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

PRIMARY_REPLICATES = 10_000
PRIMARY_N = 2_000
SEED = 20261501
RESAMPLING_SEED = 20261502
N_FOLDS = 3

#: The primary estimand: the propensity report's calibration slope, whose truth is 1.
ESTIMAND = "propensity_calibration_slope"

#: The two-sided 95% multiplier of each primary row's interval.
TWO_SIDED = float(stats.norm.ppf(0.975))

#: The band the package applied before RM15, which the controls read.
FIXED_BAND = (0.7, 1.4)

#: How closely a known learner must reproduce its law on the observed rows.
KNOWN_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Law:
    """One treatment law and the treatment learner the study fits to it.

    Parameters
    ----------
    covariates : int
        Number of standard normal covariates ``W1`` to ``Wp``.
    coefficients : tuple of float
        ``logit g0 = sum_j coefficients[j] W_{j+1}``.  Empty means ``g0 = 1/2``.
    learner : str
        ``"known"`` returns ``g0``.  ``"logistic"`` fits an unpenalized main-effects logistic
        model on ``columns`` and multiplies its logit by ``scale``.
    columns : tuple of int
        The covariate columns the fitted model reads.
    scale : float
        The multiplier of the fitted logit.  2 makes the predictions too extreme and 1/2 too
        moderate.
    limit_slope : float
        The recalibration slope of the learner's limit, ``1 / scale``.
    """

    covariates: int
    coefficients: tuple[float, ...]
    learner: str
    columns: tuple[int, ...] = ()
    scale: float = 1.0
    limit_slope: float = 1.0


LAWS: dict[str, Law] = {
    "calibrated_weak": Law(4, (0.15,), "known"),
    "calibrated_strong": Law(4, (1.0, -0.5), "known"),
    "correct_weak": Law(1, (0.15,), "logistic", columns=(0,)),
    "correct_nested_weak": Law(4, (0.15,), "logistic", columns=(0, 1, 2, 3)),
    "randomized": Law(4, (), "logistic", columns=(0, 1, 2, 3)),
    "overconfident_moderate": Law(
        4, (0.4, -0.2), "logistic", columns=(0, 1), scale=2.0, limit_slope=0.5
    ),
    "underconfident_strong": Law(
        4, (1.0, -0.5), "logistic", columns=(0, 1), scale=0.5, limit_slope=2.0
    ),
}

SCENARIOS = {"calibrated_weak": (ESTIMAND,), "calibrated_strong": (ESTIMAND,)}

#: The laws with a fixed-band control on the same fits.  ``calibrated_strong`` has none: the
#: plan's pilot put the band's rate there at 0.000, so it could not fail the instrument.
BAND_CONTROLLED = ("calibrated_weak", "correct_weak", "correct_nested_weak", "randomized")

#: The laws whose rule rate is a false-warning rate, and the two whose rate is detection.
WARNING_LAWS = (
    "calibrated_weak",
    "calibrated_strong",
    "correct_weak",
    "correct_nested_weak",
    "randomized",
)
POWER_LAWS = ("overconfident_moderate", "underconfident_strong")

PROPERTY_CELLS = {
    "warning_rate": (
        *(f"{law}__rule" for law in WARNING_LAWS),
        *(f"{law}__fixed_band" for law in BAND_CONTROLLED),
    ),
    "power": tuple(f"{law}__rule" for law in POWER_LAWS),
}

STUDY = StudyRecord(
    name="calibration-slope warning",
    slug="calibration-slope-warning",
    artifacts=ROOT / "tests" / "canonical" / "calibration_slope_warning",
    document="docs/technical-reference/method-evidence/calibration-slope-warning.md",
    anchor="calibration-slope-warning",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-calibration-slope-rule",
    reference=None,
    modules=(
        "tests/studies/calibration_slope_warning.py",
        "tests/studies/calibration_slope_warning_properties.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.calibration_slope_warning",
    properties_module="tests.studies.calibration_slope_warning_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
)

CONFIGURATION = {
    "fit": "TMLE(n_folds=3, simultaneous=False, estimands=('ate',), random_state=<sample seed>) "
    "at n = 2000; the outcome learner returns the known regression, so each fit tests two "
    "models (K = 2)",
    "outcome": "Y ~ Bernoulli(expit(0.5 A + 0.5 W1 - 0.25))",
    "laws": {
        name: {
            "covariates": law.covariates,
            "logit_g0_coefficients": list(law.coefficients),
            "treatment_learner": law.learner,
            "learner_columns": list(law.columns),
            "logit_scale": law.scale,
            "limit_slope": law.limit_slope,
        }
        for name, law in LAWS.items()
    },
    "primary_row": "estimate = calibration_slope of the propensity report; std_error = "
    "calibration_slope_se; interval = estimate +/- 1.959964 std_error; truth 1",
    "rule": f"a finding when slope +/- z_{{1 - {CALIBRATION_FAMILY_ALPHA} / (2K)}} se lies above 0 "
    "and excludes 1",
    "warning_rate_row": "truth 1; covered = the rule's interval covers 1; rejected = the report "
    "has a calibration-slope finding",
    "fixed_band_control": "the same fits; rejected = the pooled one-intercept recalibration "
    "slope of the propensity or the outcome lies outside [0.7, 1.4], as the package computed "
    "it before RM15",
    "power_row": "truth = the limit slope 1 / scale; covered = the rule's interval covers it; "
    "rejected = the report has a calibration-slope finding",
    "warnings": "PositivityWarning is filtered in each fit; the strong and tempered laws raise "
    "it on the truncation bounds, which the diagnostics do not read",
}


def g0(law: Law, covariates: np.ndarray) -> np.ndarray:
    """The law's propensity at each row, which the known treatment learner returns."""
    return np.asarray(KnownPropensity(law.coefficients).predict_proba(covariates)[:, 1])


def qbar(treatment: np.ndarray, w1: np.ndarray) -> np.ndarray:
    """The known outcome regression."""
    return np.asarray(expit(0.5 * treatment + 0.5 * w1 - 0.25), dtype=float)


def draw_law(law_name: str, n: int, seed: int) -> pd.DataFrame:
    """One sample of ``n`` rows from the named law."""
    law = LAWS[law_name]
    rng = np.random.default_rng(seed)
    covariates = rng.standard_normal((n, law.covariates))
    treatment = rng.binomial(1, g0(law, covariates)).astype(float)
    outcome = rng.binomial(1, qbar(treatment, covariates[:, 0])).astype(float)
    frame = pd.DataFrame(covariates, columns=[f"W{j + 1}" for j in range(law.covariates)])
    frame["A"] = treatment
    frame["Y"] = outcome
    return frame


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary sample from an explicit seed for the published-seed audit."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return draw_law(scenario, n, seed), {ESTIMAND: 1.0}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


class KnownPropensity(BaseEstimator, ClassifierMixin):
    """Returns ``expit(sum_j c_j W_j)`` whatever it was trained on."""

    def __init__(self, coefficients: tuple[float, ...] = ()) -> None:
        self.coefficients = coefficients

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> KnownPropensity:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        index = np.zeros(values.shape[0])
        for column, coefficient in enumerate(self.coefficients):
            index = index + coefficient * values[:, column]
        p = expit(index)
        return np.column_stack([1.0 - p, p])


class KnownOutcome(BaseEstimator, ClassifierMixin):
    """Returns ``expit(0.5 A + 0.5 W1 - 0.25)`` from the ``[A, W]`` outcome design."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> KnownOutcome:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        values = np.asarray(design, dtype=float)
        p = qbar(values[:, 0], values[:, 1])
        return np.column_stack([1.0 - p, p])


class ScaledLogistic(BaseEstimator, ClassifierMixin):
    """An unpenalized logistic model on ``columns`` whose fitted logit is multiplied by ``scale``."""

    def __init__(self, columns: tuple[int, ...] = (), scale: float = 1.0) -> None:
        self.columns = columns
        self.scale = scale

    def _select(self, design: Any) -> np.ndarray:
        return np.asarray(design, dtype=float)[:, list(self.columns)]

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> ScaledLogistic:
        self.model_ = LogisticRegression(
            C=np.inf, solver="newton-cholesky", max_iter=1000, tol=1e-10
        ).fit(self._select(design), target, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        p = expit(self.scale * self.model_.decision_function(self._select(design)))
        return np.column_stack([1.0 - p, p])


def treatment_learner(law: Law) -> Any:
    if law.learner == "known":
        return KnownPropensity(law.coefficients)
    if law.learner == "logistic":
        return ScaledLogistic(law.columns, law.scale)
    raise ValueError(f"unknown treatment learner {law.learner!r}")


def fit(frame: pd.DataFrame, law_name: str, seed: int) -> Any:
    """The declared cross-fitted TMLE, with the known outcome regression."""
    law = LAWS[law_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PositivityWarning)
        result = (
            TMLE(
                outcome_learner=KnownOutcome(),
                treatment_learner=treatment_learner(law),
                n_folds=N_FOLDS,
                simultaneous=False,
                estimands=("ate",),
                random_state=seed,
            )
            .fit(frame, outcome="Y", treatment="A")
            .single()
        )
    covariates = frame[[f"W{j + 1}" for j in range(law.covariates)]].to_numpy(dtype=float)
    outcome = np.asarray(result.nuisance.outcome.observed, dtype=float)
    expected = qbar(frame["A"].to_numpy(dtype=float), covariates[:, 0])
    if np.max(np.abs(outcome - expected)) > KNOWN_TOLERANCE:
        raise RuntimeError("the known outcome learner does not reproduce the law's regression")
    if law.learner == "known":
        propensity = np.asarray(result.nuisance.propensity.arm(1.0), dtype=float)
        if np.max(np.abs(propensity - g0(law, covariates))) > KNOWN_TOLERANCE:
            raise RuntimeError("the known treatment learner does not reproduce the law's g0")
    return result


def pooled_slope(predicted: np.ndarray, actual: np.ndarray, weights: np.ndarray) -> float:
    """The one-intercept recalibration slope the package reported before RM15.

    The same clip and the same Newton solver as the removed ``_calibration_slope``.
    """
    from cleverly.fluctuation.iterative import _newton_logistic

    p = np.clip(np.asarray(predicted, dtype=float), 1e-12, 1.0 - 1e-12)
    x = np.column_stack([np.ones_like(p), logit(p)])
    coefficients, _converged, _detail = _newton_logistic(
        x, np.asarray(actual, dtype=float), np.zeros_like(p), weights
    )
    return float(coefficients[1])


@dataclass(frozen=True)
class Reading:
    """What one fit gives every row the study writes."""

    slope: float
    std_error: float
    tested: int
    finding: bool
    band: bool


def read(result: Any) -> Reading:
    """The propensity slope, its rule, and the fixed band, from one fit."""
    report = nuisance_diagnostics(result)
    propensity = report["propensity"].metrics
    # The package's own count of tested models, so the coverage below reads the level the
    # rule used rather than a second statement of it.
    tested = len(report._calibration_tested)
    finding = any("calibration slope" in note for note in report.findings)
    data = result.data
    slopes = (
        pooled_slope(result.nuisance.propensity.arm(1.0), data.treatment, data.weights),
        pooled_slope(result.nuisance.outcome.observed, data.outcome, data.weights),
    )
    band = any(not FIXED_BAND[0] <= slope <= FIXED_BAND[1] for slope in slopes)
    return Reading(
        slope=float(propensity["calibration_slope"]),
        std_error=float(propensity["calibration_slope_se"]),
        tested=tested,
        finding=finding,
        band=band,
    )


def rule_covers(reading: Reading, truth: float) -> bool:
    """Whether the rule's Bonferroni interval covers ``truth``, at the package's own level."""
    half = _critical_value(reading.tested) * reading.std_error
    return bool(reading.slope - half <= truth <= reading.slope + half)


def cleverly_rows(
    frame: pd.DataFrame, truth: Mapping[str, float], scenario: str, replicate: int, seed: int
) -> list[dict[str, Any]]:
    """Convert one fit to the shared replication schema."""
    result = fit(frame, scenario, seed)
    reading = read(result)
    reference = float(truth[ESTIMAND])
    low = reading.slope - TWO_SIDED * reading.std_error
    high = reading.slope + TWO_SIDED * reading.std_error
    return [
        {
            "implementation": STUDY.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": result.n,
            "estimand": ESTIMAND,
            "truth": reference,
            "estimate": reading.slope,
            "inference_estimate": reading.slope,
            "std_error": reading.std_error,
            "ci_lower": float(low),
            "ci_upper": float(high),
            "inference_scale": "identity",
            "covered": int(low <= reference <= high),
            "initial_estimate": math.nan,
        }
    ]


def _replicate(payload: tuple[str, int, int]) -> list[dict[str, Any]]:
    scenario, index, n = payload
    seed = replicate_seed(STUDY, scenario, index)
    frame, truth = draw_scenario(scenario, n, index)
    return cleverly_rows(frame, truth, scenario, index, seed)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> pd.DataFrame:
    """Draw and fit every declared primary replication."""
    payloads = [
        ((scenario, index, n),) for scenario in STUDY.scenarios for index in range(replicates)
    ]
    outcomes = map_parallel(_replicate, payloads, n_jobs=n_jobs)
    built = pd.DataFrame([row for records in outcomes for row in records])
    return built.loc[:, list(REPLICATE_COLUMNS)]
