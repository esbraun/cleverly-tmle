"""Registered evidence for the step-down max-T adjustment of ``variable_importance`` (X125).

Declared, not run. The record, the laws, the cells, the margins, the seeds and the sizing rule
are fixed here before the pilot (design rev 2; the sizing rule, rev 3). The record is not in the register: a registered
study must publish its artifacts, and the declared run has not happened. Until it lands,
``cleverly.variable_importance.PENDING_REASON`` keeps every configuration's adjustment label
experimental, and :data:`EXEMPT_REASON` waits for the registration commit, which adds it to
``cleverly._support_rows.EXEMPT_STUDIES``.

The claim. ``variable_importance(adjustment="max_t")`` controls the family-wise error rate
(FWER) of its adjusted p-values asymptotically (van der Laan, Dudoit and Pollard 2004, the
step-down paper, Theorems 1, 3 and 6, with a consistent estimate of the null law, the correlation of the
studentized curves; Chernozhukov, Chetverikov and Kato 2013, Theorem 5.1, reading the same
draws as a Gaussian-multiplier bootstrap). The studentization by estimated standard errors is
a derivation by Slutsky's lemma at fixed ``m``, and the use of each row's own p-value through
``norm.isf(p / 2)`` is a derivation that changes a null statistic by ``o_P(1)``. Because the
false discovery proportion is at most ``1{V >= 1}``, the FDR is at most the FWER.
Benjamini-Hochberg has no theorem for two-sided tests with correlation of either sign, so its
cells are diagnostic, here and after any switch (design section 10).

Scenarios: six laws (:mod:`tests.studies.variable_importance_law`) by two configurations, in
sample and cross-fitted with five folds, at ``n = 4,000``. Each ``__cross_fitted`` scenario
fits the samples of its law's ``__in_sample`` scenario (``scenario_seed_owners``). The two
``collinear_pair`` scenarios are reported. Every fit uses ``covariates=["W"]``, the eleven
candidates, ``LogisticRegression(C=1e6, max_iter=2000)`` for both learners, ``random_state=0``,
``keep_fits=False`` and the shipped ``multiplier_seed=0``.

Gates read by code on every replicate, where a failure stops the run
(:func:`check_shape`, :func:`check_identity`): the shape of the result (twelve rows, their
parameter names, inference, curves of length ``n``), the step-down recomputed by this study's
own loop on draws it makes itself equal to cleverly's adjusted p-values (``rtol = 1e-12``),
and Benjamini-Hochberg recomputed by a rank loop equal to cleverly's ``_bh_adjust`` (and to
scipy's ``false_discovery_control``).

Order of work (design rev 3): the pilot
(``python -m tests.studies.variable_importance_properties --pilot --raw DIR``) writes
:data:`PILOT_PATH`; the pilot extension (``--pilot-extension --raw DIR``) reads the
calibration law on 4,000 draws and writes :data:`PILOT_EXTENSION_PATH`; the declaration commit
sets :data:`DECLARED_REPLICATES` from both through :func:`declared_replicates`; then the
declared run (``python -m tests.canonical.variable_importance.regenerate``).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from sklearn.linear_model import LogisticRegression

from cleverly import variable_importance
from cleverly.estimators import TMLE
from tests.parallel import STUDY_JOBS
from tests.studies import variable_importance_law as law_module
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

SLUG = "variable-importance"
#: Entropy unused anywhere under ``tests/`` (the design test checks it).
SEED = 2026112601
RESAMPLING_SEED = 2026112602
#: The one sample size of the record (design section 6).
N = 4_000
#: Rev 3: 1,600, not 1,000, so the 120 gated primary coverage floors fit the family pass
#: (review B1): at 1,000 they alone carried a 3% to 12% chance of a false red.
PRIMARY_REPLICATES = 1_600
#: The configurations and the laws, in scenario order.
CONFIGURATIONS: tuple[str, ...] = ("in_sample", "cross_fitted")
LAW_NAMES: tuple[str, ...] = tuple(law_module.LAWS)
#: The laws whose property cells are gated; ``collinear_pair`` is reported.
GATED_LAWS: tuple[str, ...] = tuple(name for name, law in law_module.LAWS.items() if law.gated)
#: The parameters every scenario reports, in fit order.
ESTIMANDS: tuple[str, ...] = law_module.NAMES
#: The candidate columns passed to ``variable_importance``.
CANDIDATES: tuple[str, ...] = law_module.CANDIDATES
#: The fold count of the cross-fitted configuration.
FOLDS = 5


def scenario_name(law: str, configuration: str) -> str:
    """The scenario of one law and configuration."""
    return f"{law}__{configuration}"


def split_scenario(scenario: str) -> tuple[str, str]:
    """The law and the configuration of a scenario."""
    law, configuration = scenario.rsplit("__", 1)
    if law not in law_module.LAWS or configuration not in CONFIGURATIONS:
        raise KeyError(f"{SLUG} has no scenario {scenario!r}")
    return law, configuration


SCENARIOS: tuple[str, ...] = tuple(
    scenario_name(law, configuration) for law in LAW_NAMES for configuration in CONFIGURATIONS
)
REPORTED_SCENARIOS = frozenset(
    scenario_name("collinear_pair", configuration) for configuration in CONFIGURATIONS
)
SCENARIO_SEED_OWNERS: dict[str, str] = {
    scenario_name(law, "cross_fitted"): scenario_name(law, "in_sample") for law in LAW_NAMES
}

#: The levels of the family-wise error rate, the relative margin, and the power floor.
ALPHAS: tuple[float, ...] = (0.05, 0.10)
DELTA = 0.5
POWER_ALPHA = 0.05
#: The shrunken-standard-error control divides each statistic by this factor (design 7).
SHRUNKEN_FACTOR = 0.8

#: The declared pilot: replicates per law, on its salted stream; pilot draws enter no verdict.
PILOT_REPLICATES = 800
PILOT_STREAM = "variable-importance/pilot"
PILOT_PATH = ROOT / "tests" / "studies" / "variable_importance_pilot.csv"
#: The provisional property count of the design (section 9), sized on the oracle planning
#: values. The declaration commit replaces it by :func:`declared_replicates` on the pilot.
PROVISIONAL_REPLICATES = 3_200
#: The declared property count of a gated law. ``None`` until the declaration commit: the
#: declared run refuses to start without it.
DECLARED_REPLICATES: int | None = None
#: The property count of the reported law and of the adjustment witness.
REPORTED_REPLICATES = 1_000
WITNESS_REPLICATES = 1_000
#: Above this no R sizes the family; the row thread stops and reports (design section 6).
#: Rev 3 raised it from 4,800: the arm-pooled predictive rule plans the calibration cells at
#: their measured SE ratio, which needs more R than a cell planned at a ratio of 1.
MAXIMUM_REPLICATES = 8_000
#: The pilot extension (design rev 3, section 9a): this many fresh draws of the calibration
#: law, replicates :data:`PILOT_REPLICATES` onward on their own stream, so each calibration
#: arm is read on at least 3,000 draws (``guide/studies.md``, "Coverage readings") before the
#: predictive rule sizes it. It leaves out the pilot's 800, which triggered it (review S3).
PILOT_EXTENSION_REPLICATES = 4_000
#: The extension's own stream.
PILOT_EXTENSION_STREAM = "variable-importance/pilot-extension"
PILOT_EXTENSION_PATH = ROOT / "tests" / "studies" / "variable_importance_pilot_extension.csv"
#: A calibration arm whose plug-in pass at the at-target R falls below this needs the pilot
#: extension before the declaration (``guide/studies.md``, "Calibration budgets across arms").
EXTENSION_TRIGGER = 0.90
#: The smoke run: the first replicates of two scenarios on the pilot stream, printed only.
SMOKE_SCENARIOS = (
    scenario_name("independent_null", "in_sample"),
    scenario_name("nulls_beside_signals", "cross_fitted"),
)
SMOKE_REPLICATES = 12
#: The CPU ceilings of the two runner briefs, in vCPU-hours.
PILOT_CPU_CEILING_HOURS = 8.0
RUN_CPU_CEILING_HOURS = 50.0
#: Above this projected size of any one committed property file, in MB, the runner stops and
#: reports (``COMPACT_BYTES`` of the evidence writer; GitHub refuses a file over 100 MB).
ARTIFACT_CEILING_MB = 90.0

#: The per-replicate artifact: one row per replicate, configuration and adjustment set, with
#: each parameter's estimate, standard error, residual margin, raw, max-T and BH p-values, and
#: the replicate's correlation once (its 66 upper-triangle entries; review R2-6).
#: The per-replicate record of the property fits, one file per configuration so that each
#: stays under GitHub's 100 MB limit at the declared R.
REPLICATES_FILES: dict[str, str] = {
    configuration: f"variable-importance-replicates-{configuration.replace('_', '-')}.csv"
    for configuration in CONFIGURATIONS
}

#: The registration commit adds this entry to ``cleverly._support_rows.EXEMPT_STUDIES``.
EXEMPT_REASON = (
    "It measures the step-down max-T adjustment of variable_importance, a joint claim over "
    "separate candidate fits that no support key reads. "
    "cleverly.variable_importance.variable_importance_support, not a support row, records the "
    "configurations the study licenses."
)


def estimator(configuration: str) -> TMLE:
    """The study's TMLE for one configuration (design section 6)."""
    options: dict[str, Any] = {"cross_fit": False}
    if configuration == "cross_fitted":
        options = {"cross_fit": True, "n_folds": FOLDS}
    elif configuration != "in_sample":
        raise KeyError(configuration)
    return TMLE(
        outcome_learner=LogisticRegression(C=1e6, max_iter=2000),
        treatment_learner=LogisticRegression(C=1e6, max_iter=2000),
        simultaneous=False,
        random_state=0,
        **options,
    )


def fit(frame: pd.DataFrame, configuration: str, *, adjust: bool = True) -> Any:
    """One ``variable_importance`` run at the study's settings."""
    return variable_importance(
        frame,
        outcome="Y",
        candidates=list(CANDIDATES),
        covariates=["W"],
        estimator=estimator(configuration),
        adjust_for_other_candidates=adjust,
        keep_fits=False,
    )


def row_name(entry: Any) -> str:
    """The law's name of an entry: the candidate, or ``Z:1`` and ``Z:2``."""
    if entry.candidate != "Z":
        return str(entry.candidate)
    levels = {"ate[1.0 vs 0.0]": "Z:1", "ate[2.0 vs 0.0]": "Z:2"}
    return levels[entry.parameter]


#: The parameter names a run must report, with their counts (the shape gate).
EXPECTED_PARAMETERS = {"ate": 10, "ate[1.0 vs 0.0]": 1, "ate[2.0 vs 0.0]": 1}


def check_shape(result: Any, n: int) -> None:
    """The shape gate: twelve inferential rows of the expected names, curves of length ``n``.

    Raises
    ------
    AssertionError
        When the result breaks the shape, which stops the run.
    """
    entries = list(result)
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry.parameter] = counts.get(entry.parameter, 0) + 1
    if counts != EXPECTED_PARAMETERS:
        raise AssertionError(f"the run reported parameters {counts}")
    if sorted(row_name(entry) for entry in entries) != sorted(ESTIMANDS):
        raise AssertionError("the run's rows are not the law's twelve parameters")
    for entry in entries:
        estimate = entry.estimate
        if not estimate.supplies_inference or entry.adjusted_pvalue is None:
            raise AssertionError(f"{row_name(entry)} supplies no inference")
        if np.shape(estimate.influence_curve) != (n,):
            raise AssertionError(
                f"{row_name(entry)}'s curve has shape {estimate.influence_curve.shape}"
            )


def fit_order(result: Any) -> list[Any]:
    """The entries in the order the run fitted them, which its draws follow."""
    by_name = {row_name(entry): entry for entry in result}
    return [by_name[name] for name in ESTIMANDS]


def independent_max_t(result: Any) -> np.ndarray:
    """The step-down max-T recomputed apart from cleverly, in fit order.

    The correlation is the centred covariance of the curves; the draws are this module's own
    Cholesky draws from ``multiplier_seed``; the step-down is the law module's explicit loop;
    the floor is each row's own p-value.
    """
    entries = fit_order(result)
    curves = np.column_stack([entry.estimate.influence_curve for entry in entries])
    exact = law_module.correlation(np.cov(curves, rowvar=False, ddof=1))
    rng = np.random.default_rng(result.multiplier_seed)
    normals = rng.standard_normal((result.multiplier_draws, len(entries)))
    draws = np.abs(normals @ np.linalg.cholesky(exact).T)
    pvalues = np.array([entry.estimate.pvalue for entry in entries])
    statistics = stats.norm.isf(pvalues / 2.0)
    return np.minimum(np.maximum(law_module.step_down(statistics, draws), pvalues), 1.0)


def rank_bh(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values by an explicit loop over the ranks."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = sorted(range(m), key=lambda j: (p[j], j))
    out = np.empty(m)
    running = 1.0
    for rank in range(m, 0, -1):
        row = order[rank - 1]
        running = min(running, p[row] * m / rank)
        out[row] = min(running, 1.0)
    return out


def by_adjusted(pvalues: Sequence[float]) -> np.ndarray:
    """Benjamini-Yekutieli adjusted p-values: BH times the harmonic number, capped at one."""
    p = np.asarray(pvalues, dtype=float)
    harmonic = float(np.sum(1.0 / np.arange(1, p.size + 1)))
    return np.minimum(rank_bh(p) * harmonic, 1.0)


def check_identity(result: Any) -> dict[str, np.ndarray]:
    """The identity gate: cleverly's adjusted p-values against this module's procedures.

    Returns, in fit order, the raw p-values and the max-T, BH and BY adjusted p-values.

    Raises
    ------
    AssertionError
        When the max-T or the BH identity fails, which stops the run.
    """
    from cleverly.variable_importance import _bh_adjust

    if result.error_rate != "fwer":
        raise AssertionError("the study fits the shipped default, adjustment='max_t'")
    entries = fit_order(result)
    pvalues = np.array([entry.estimate.pvalue for entry in entries])
    shipped = np.array([entry.adjusted_pvalue for entry in entries])
    max_t = independent_max_t(result)
    if not np.allclose(shipped, max_t, rtol=1e-12, atol=0.0):
        raise AssertionError(f"max-T identity failed: {shipped} against {max_t}")
    bh = rank_bh(pvalues)
    if not np.allclose(_bh_adjust(pvalues), bh, rtol=1e-12, atol=1e-15):
        raise AssertionError("Benjamini-Hochberg identity failed")
    control = getattr(stats, "false_discovery_control", None)
    if control is not None and not np.allclose(control(pvalues), bh, rtol=1e-10, atol=1e-15):
        raise AssertionError("Benjamini-Hochberg disagrees with scipy")
    return {"p": pvalues, "max_t": max_t, "bh": bh, "by": by_adjusted(pvalues)}


# ------------------------------------------------------------------------ primaries


@functools.cache
def _truths(law: str, adjust: bool) -> tuple[float, ...]:
    model = law_module.LAWS[law]
    values = model.exact()[0] if adjust else model.exact_unadjusted()
    return tuple(float(value) for value in values)


def truths(law: str, *, adjust: bool = True) -> dict[str, float]:
    """The exact truth of each parameter, under either adjustment set."""
    return dict(zip(ESTIMANDS, _truths(law, adjust), strict=True))


@functools.cache
def exact_covariance(law: str) -> np.ndarray:
    """The exact covariance of the twelve efficient influence curves, in fit order."""
    covariance = np.array(law_module.LAWS[law].exact()[1], dtype=float)
    covariance.setflags(write=False)
    return covariance


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One sample of a scenario's law from an explicit seed, with its exact truths."""
    law, _ = split_scenario(scenario)
    return law_module.LAWS[law].sample(n, seed), truths(law)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replication ``replicate`` of a scenario, from this study's seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def cleverly_rows(
    result: Any, truth: Mapping[str, float], *, scenario: str, replicate: int, n: int
) -> list[dict[str, Any]]:
    """The primary rows of one run, one per parameter in fit order."""
    rows = []
    for entry in fit_order(result):
        name = row_name(entry)
        std_error, low, high = reported_inference(entry.estimate)
        reference = float(truth[name])
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": n,
                "estimand": name,
                "truth": reference,
                "estimate": float(entry.estimate.psi),
                "inference_estimate": float(entry.estimate.psi),
                "std_error": std_error,
                "ci_lower": low,
                "ci_upper": high,
                "inference_scale": "identity",
                "covered": bool(low <= reference <= high),
                "initial_estimate": math.nan,
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
    result = fit(frame, split_scenario(scenario)[1])
    check_shape(result, n)
    check_identity(result)
    return cleverly_rows(result, truth, scenario=scenario, replicate=replicate, n=n)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    """Draw and fit every declared primary replication of every scenario."""
    outcomes = replication.fit_primary(
        [(scenario, index, n) for scenario in SCENARIOS for index in range(replicates)],
        _replicate,
        draw=_draw,
        rows_at=None,
        n_jobs=n_jobs,
        implementation=STUDY.implementation,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    built = pd.DataFrame([row for records in outcomes.kept for row in records])
    return replication.result(outcomes, built.loc[:, list(REPLICATE_COLUMNS)])


# ------------------------------------------------------------------------ the sizing rule
# Design section 9: the smallest multiple of 400 whose family-wise pass probability, the
# product over gated cells of the per-cell pass probabilities at planning values, reaches
# 0.95. FWER cells are planned each from its own value (the larger of the oracle's and the
# pilot's), unpooled (review R2-4); power rows from the oracle; the calibration cells from the
# pilot's coverage through the SE-ratio pass curve, with kappa fitted to the pilot once it
# exists.

SIZING_TARGET = 0.95
SIZING_STEP = 400
#: The 99% Clopper-Pearson interval of every proportion gate.
CONFIDENCE = 0.99
Z995 = float(stats.norm.ppf(0.995))

#: The oracle planning values at n = 2,000 (``probes/maxt.py``, 8,000 datasets, 10,000 draws),
#: which the design sized on. The design test recomputes them at n = 4,000.
PLANNING_FWER: dict[tuple[str, float], float] = {
    ("independent_null", 0.05): 0.0502,
    ("independent_null", 0.10): 0.1001,
    ("negative_pairs_null", 0.05): 0.0513,
    ("negative_pairs_null", 0.10): 0.1009,
    ("independent_mixed", 0.05): 0.0488,
    ("independent_mixed", 0.10): 0.0974,
    ("nulls_beside_signals", 0.05): 0.0522,
    ("nulls_beside_signals", 0.10): 0.1049,
    ("dobriban_like", 0.05): 0.0454,
    ("dobriban_like", 0.10): 0.0881,
}
#: The oracle power of the gated power rows at level 0.05.
PLANNING_POWER: dict[tuple[str, str], float] = {
    ("independent_mixed", "X1"): 0.998,
    ("independent_mixed", "X2"): 0.998,
    ("independent_mixed", "X3"): 0.998,
    ("independent_mixed", "X4"): 0.998,
    ("independent_mixed", "Z:2"): 0.944,
    ("nulls_beside_signals", "X1"): 0.97,
    ("nulls_beside_signals", "X2"): 0.994,
    ("nulls_beside_signals", "X3"): 0.993,
    ("nulls_beside_signals", "X4"): 0.995,
    ("dobriban_like", "X3"): 0.917,
    ("dobriban_like", "X4"): 0.921,
}
#: The complete-null laws, whose FWER is gated in a band and carries the controls.
CALIBRATION_LAWS: tuple[str, ...] = ("independent_null", "negative_pairs_null")
#: The laws whose FWER is gated under a ceiling.
CEILING_LAWS: tuple[str, ...] = ("independent_mixed", "nulls_beside_signals", "dobriban_like")
#: The interval-calibration rows, and their law.
CALIBRATION_ROWS: tuple[str, ...] = ("X1", "X5")
CALIBRATION_LAW = "nulls_beside_signals"
#: The rows of the ``one_sided`` power control (each has beta < 0).
ONE_SIDED_ROWS: tuple[tuple[str, str], ...] = (
    ("independent_mixed", "X4"),
    ("nulls_beside_signals", "X4"),
)
#: Planning values of the controls: the unadjusted and shrunken-SE FWER on the complete nulls
#: (oracle 0.42 to 0.46 and about 0.20), the one-sided power of a negative effect (0), the
#: shrunken-SE calibration coverage (0.88), the noise control's SE ratio (about 1.41).
PLANNING_CONTROLS = {"unadjusted": 0.42, "shrunken_se": 0.20, "one_sided": 0.0}
#: The SE-ratio pass curve's kappa before the pilot: 0.95 at R = 2,400 (``guide/studies.md``).
SE_RATIO_BAND = (0.93, 1.07)


def _cp(k: np.ndarray, replicates: int) -> tuple[np.ndarray, np.ndarray]:
    tail = (1.0 - CONFIDENCE) / 2.0
    low = np.where(k > 0, stats.beta.ppf(tail, k, replicates - k + 1), 0.0)
    high = np.where(k < replicates, stats.beta.ppf(1 - tail, k + 1, replicates - k), 1.0)
    return low, high


def pass_band(p: float, replicates: int, low: float, high: float) -> float:
    """The chance the 99% CP interval of a binomial share lies inside ``[low, high]``."""
    k = np.arange(replicates + 1)
    lower, upper = _cp(k, replicates)
    return float(stats.binom.pmf(k[(lower >= low) & (upper <= high)], replicates, p).sum())


def pass_ceiling(p: float, replicates: int, ceiling: float) -> float:
    """The chance the upper limit is at or below ``ceiling``."""
    k = np.arange(replicates + 1)
    _, upper = _cp(k, replicates)
    return float(stats.binom.pmf(k[upper <= ceiling], replicates, p).sum())


def pass_floor(p: float, replicates: int, floor: float) -> float:
    """The chance the lower limit is at or above ``floor``."""
    k = np.arange(replicates + 1)
    lower, _ = _cp(k, replicates)
    return float(stats.binom.pmf(k[lower >= floor], replicates, p).sum())


def pass_above(p: float, replicates: int, threshold: float) -> float:
    """The chance the lower limit is strictly above ``threshold``."""
    k = np.arange(replicates + 1)
    lower, _ = _cp(k, replicates)
    return float(stats.binom.pmf(k[lower > threshold], replicates, p).sum())


def pass_below(p: float, replicates: int, threshold: float) -> float:
    """The chance the upper limit is strictly below ``threshold``."""
    k = np.arange(replicates + 1)
    _, upper = _cp(k, replicates)
    return float(stats.binom.pmf(k[upper < threshold], replicates, p).sum())


def se_ratio_pass(replicates: int, kappa: float, ratio: float = 1.0) -> float:
    """The chance the SE ratio's 99% interval lies in :data:`SE_RATIO_BAND`, as a normal."""
    sd = kappa / math.sqrt(replicates)
    low, high = SE_RATIO_BAND
    upper = stats.norm.cdf((high - Z995 * sd - ratio) / sd)
    lower = stats.norm.cdf((low + Z995 * sd - ratio) / sd)
    return float(max(0.0, upper - lower))


#: The SE-ratio kappa before the pilot, fitted to a pass of 0.95 at R = 2,400.
PLANNING_KAPPA = float(brentq(lambda kappa: se_ratio_pass(2_400, kappa) - 0.95, 0.3, 3.0))


def cell_passes(
    replicates: int,
    *,
    fwer: Mapping[tuple[str, float], float] = PLANNING_FWER,
    power: Mapping[tuple[str, str], float] = PLANNING_POWER,
    coverage: Mapping[str, float] | None = None,
    kappa: float = PLANNING_KAPPA,
    configuration: str = "in_sample",
    calibration: bool = True,
) -> dict[str, float]:
    """Each gated cell's pass probability for one configuration at ``replicates``.

    The three-level rows of the cross-fitted configuration have no support row and are not
    gated (design section 7), so its power cells leave them out. The calibration cells are
    planned at target here, at coverage ``coverage[row]`` (0.95 by default) and an SE ratio
    of 1; ``calibration=False`` leaves them out for :func:`calibration_pass` to plan from the
    pilot extension.
    """
    passes: dict[str, float] = {}
    for law in CALIBRATION_LAWS:
        for alpha in ALPHAS:
            passes[f"fwer_calibration/{law}/{alpha}"] = pass_band(
                fwer[(law, alpha)], replicates, alpha * (1 - DELTA), alpha * (1 + DELTA)
            )
            for kind in ("unadjusted", "shrunken_se"):
                passes[f"family_wise_error/{law}/{alpha}/{kind}"] = pass_above(
                    PLANNING_CONTROLS[kind], replicates, alpha * (1 + DELTA)
                )
    for law in CEILING_LAWS:
        for alpha in ALPHAS:
            passes[f"family_wise_error/{law}/{alpha}"] = pass_ceiling(
                fwer[(law, alpha)], replicates, alpha * (1 + DELTA)
            )
    for (law, row), value in power.items():
        if gated_power(law, row, configuration):
            passes[f"power/{law}/{row}"] = pass_floor(value, replicates, 0.80)
    for law, row in ONE_SIDED_ROWS:
        passes[f"power/{law}/{row}/one_sided"] = pass_below(
            PLANNING_CONTROLS["one_sided"], replicates, 0.80
        )
    for row in CALIBRATION_ROWS:
        if calibration:
            planned = 0.95 if coverage is None else coverage[row]
            passes[f"interval_calibration/{row}"] = se_ratio_pass(replicates, kappa) * pass_band(
                planned, replicates, 0.92, 0.98
            )
        passes[f"interval_calibration/{row}/shrunken_se_control"] = pass_below(
            0.88, replicates, 0.92
        )
        passes[f"interval_calibration/{row}/noise_control"] = 1.0 - se_ratio_pass(
            replicates, kappa, ratio=math.sqrt(2.0)
        )
    return passes


# Design rev 3, section 9a: the calibration cells. Each (configuration, row) is an arm, read on
# the pilot extension. The arm-pooled predictive rule (``guide/studies.md``, "Calibration
# budgets across arms") pools the arms' coverages, and separately their SE ratios, by a
# normal-normal model with tau at its one-sided 80% Q-profile upper bound (as
# ``band_t_reference`` pools coverages), within each configuration only: an in-sample fit's
# SE ratio sits systematically below a cross-fitted one's, so the configurations are not
# exchangeable (review S1). It draws each arm's planning value from the predictive law, then
# the study's own Monte Carlo noise, and counts the draws in which every arm passes both its
# coverage band and its SE-ratio band.

#: The calibration arms, in a fixed order.
CALIBRATION_ARMS: tuple[tuple[str, str], ...] = tuple(
    (configuration, row) for configuration in CONFIGURATIONS for row in CALIBRATION_ROWS
)
SIZING_DRAWS = 40_000
#: The Q-profile level of the pooled tau bound.
POOLING_LEVEL = 0.80


@dataclasses.dataclass(frozen=True)
class ArmReading:
    """One calibration arm's reading on the pilot or its extension.

    Parameters
    ----------
    coverage : float
        The share of intervals that cover the truth.
    se_ratio : float
        The mean standard error over the standard deviation of the estimates.
    kappa : float
        The bootstrap spread of the SE ratio times root R.
    replicates : int
        The draws behind the reading.
    """

    coverage: float
    se_ratio: float
    kappa: float
    replicates: int


def read_arms(readings: pd.DataFrame) -> dict[tuple[str, str], ArmReading]:
    """Each calibration arm's reading from pilot-format readings (``pilot_readings``)."""

    def value(kind: str, configuration: str, row: str) -> pd.Series:
        hit = readings.loc[
            (readings["kind"] == kind)
            & (readings["configuration"] == configuration)
            & (readings["row"] == row)
        ]
        if len(hit) != 1:
            raise KeyError(f"{kind} of {row} {configuration}: {len(hit)} readings")
        return hit.iloc[0]

    arms = {}
    for configuration, row in CALIBRATION_ARMS:
        coverage = value("coverage_by_configuration", configuration, row)
        arms[(configuration, row)] = ArmReading(
            coverage=float(coverage["value"]),
            se_ratio=float(value("se_ratio", configuration, row)["value"]),
            kappa=float(value("se_ratio_kappa_by_configuration", configuration, row)["value"]),
            replicates=int(coverage["replicates"]),
        )
    return arms


def tau_bound(
    y: np.ndarray, v: np.ndarray, level: float = POOLING_LEVEL
) -> tuple[float, float, float]:
    """The normal-normal pooling of readings ``y`` with variances ``v``: tau, mean, its SE.

    Tau is the one-sided ``level`` Q-profile upper bound of the between-arm spread.
    """

    def q(t2: float) -> float:
        w = 1.0 / (v + t2)
        mu = float(np.sum(w * y) / np.sum(w))
        return float(np.sum(w * (y - mu) ** 2))

    critical = float(stats.chi2.ppf(1.0 - level, len(y) - 1))
    t2 = 0.0 if q(0.0) <= critical else float(brentq(lambda t: q(t) - critical, 0.0, 1.0))
    w = 1.0 / (v + t2)
    return math.sqrt(t2), float(np.sum(w * y) / np.sum(w)), float(1.0 / math.sqrt(np.sum(w)))


def _predictive(
    y: np.ndarray, v: np.ndarray, rng: np.random.Generator, draws: int, *, pooled: bool
) -> np.ndarray:
    if not pooled:
        return y + np.sqrt(v) * rng.standard_normal((draws, len(y)))
    tau, mu, sd = tau_bound(y, v)
    centre = mu + sd * rng.standard_normal(draws)
    shrink = tau**2 / (v + tau**2)
    return (
        centre[:, None]
        + shrink * (y - centre[:, None])
        + np.sqrt(v * shrink) * rng.standard_normal((draws, len(y)))
    )


def calibration_pass(
    replicates: int,
    arms: Mapping[tuple[str, str], ArmReading],
    *,
    pooled: bool = True,
    draws: int = SIZING_DRAWS,
) -> float:
    """The chance every calibration arm passes its coverage and SE-ratio bands at ``replicates``.

    Each arm's SE-ratio interval is read as a normal of spread ``kappa / sqrt(replicates)``,
    with the arm's own kappa, as :func:`se_ratio_pass` reads it.
    """
    rng = np.random.default_rng(stream_seed(STUDY, "sizing", "calibration", replicates))
    readings = [arms[arm] for arm in CALIBRATION_ARMS]
    count = np.array([arm.replicates for arm in readings], dtype=float)
    coverage = np.array([arm.coverage for arm in readings])
    ratio = np.array([arm.se_ratio for arm in readings])
    kappa = np.array([arm.kappa for arm in readings])
    planned_coverage = np.empty((draws, len(readings)))
    planned_ratio = np.empty((draws, len(readings)))
    for configuration in CONFIGURATIONS:
        group = np.array([arm[0] == configuration for arm in CALIBRATION_ARMS])
        planned_coverage[:, group] = _predictive(
            coverage[group],
            coverage[group] * (1 - coverage[group]) / count[group],
            rng,
            draws,
            pooled=pooled,
        )
        planned_ratio[:, group] = _predictive(
            ratio[group], kappa[group] ** 2 / count[group], rng, draws, pooled=pooled
        )
    hits = rng.binomial(replicates, np.clip(planned_coverage, 0.0, 1.0))
    low, high = _cp(hits, replicates)
    covered = (low >= 0.92) & (high <= 0.98)
    spread = kappa / math.sqrt(replicates)
    observed = planned_ratio + spread * rng.standard_normal(planned_ratio.shape)
    band_low, band_high = SE_RATIO_BAND
    within = (observed - Z995 * spread >= band_low) & (observed + Z995 * spread <= band_high)
    return float(np.mean((covered & within).all(axis=1)))


def plug_in_arm_pass(replicates: int, arm: ArmReading) -> float:
    """One calibration arm's pass probability at its own reading, without pooling."""
    return se_ratio_pass(replicates, arm.kappa, arm.se_ratio) * pass_band(
        arm.coverage, replicates, 0.92, 0.98
    )


#: The planning coverage of the primary rows, by configuration: the rev-3 probe's mean over
#: the twelve rows of ``nulls_beside_signals`` on 3,200 fresh draws (0.947 in sample, 0.952
#: cross-fitted; ``rows/X125/probes/coverage_rev3_summary.csv`` in the plans repo), rounded
#: down. Every gated scenario's twelve primary rows are planned at it.
PRIMARY_PLANNING_COVERAGE: dict[str, float] = {"in_sample": 0.946, "cross_fitted": 0.952}
#: The primary coverage floor: the 99% Clopper-Pearson lower bound at least this
#: (``Margins().coverage_floor``, read by ``tests/studies/evidence/performance.py``).
PRIMARY_COVERAGE_FLOOR = 0.90


def primary_pass(replicates: int = PRIMARY_REPLICATES) -> float:
    """The chance every gated primary coverage cell passes its floor at ``replicates``.

    The bias and SE-ratio gates of the primaries pass with probability indistinguishable from
    1 at these readings (bias at most 0.03 SD against 0.25, SE ratio more than five SDs inside
    [0.80, 1.20]; review rev 3, table), so the coverage floor stands for them.
    """
    gated_laws = [law for law in LAW_NAMES if law in GATED_LAWS]
    return float(
        np.prod(
            [
                pass_floor(
                    PRIMARY_PLANNING_COVERAGE[configuration], replicates, PRIMARY_COVERAGE_FLOOR
                )
                ** (len(gated_laws) * len(ESTIMANDS))
                for configuration in CONFIGURATIONS
            ]
        )
    )


def family_pass(
    replicates: int,
    *,
    arms: Mapping[tuple[str, str], ArmReading] | None = None,
    **planning: Any,
) -> float:
    """The family-wise pass probability over both configurations at ``replicates``.

    Without ``arms`` the calibration cells are planned at target; with them, by
    :func:`calibration_pass`. Both include the primary coverage floors (:func:`primary_pass`).
    """
    product = float(
        np.prod(
            [
                np.prod(
                    list(
                        cell_passes(
                            replicates,
                            configuration=name,
                            calibration=arms is None,
                            **planning,
                        ).values()
                    )
                )
                for name in CONFIGURATIONS
            ]
        )
    )
    product *= primary_pass()
    return product if arms is None else product * calibration_pass(replicates, arms)


def size(**planning: Any) -> tuple[int | None, float]:
    """The smallest multiple of 400 up to :data:`MAXIMUM_REPLICATES` passing at 0.95."""
    for replicates in range(SIZING_STEP, MAXIMUM_REPLICATES + 1, SIZING_STEP):
        probability = family_pass(replicates, **planning)
        if probability >= SIZING_TARGET:
            return replicates, probability
    return None, family_pass(MAXIMUM_REPLICATES, **planning)


def read_pilot(path: Path = PILOT_PATH) -> pd.DataFrame | None:
    """The committed pilot readings, or ``None`` before the pilot has run."""
    return pd.read_csv(path) if path.exists() else None


def read_extension(path: Path = PILOT_EXTENSION_PATH) -> pd.DataFrame | None:
    """The committed pilot-extension readings, or ``None`` before the extension has run."""
    return pd.read_csv(path) if path.exists() else None


def planning_fwer(*readings: pd.DataFrame | None) -> dict[tuple[str, float], float]:
    """Each gated FWER cell's planning value: the largest of the oracle and every reading."""
    fwer = dict(PLANNING_FWER)
    for frame in readings:
        if frame is None:
            continue
        for row in frame.loc[frame["kind"] == "fwer"].itertuples():
            key = (str(row.law), float(row.alpha))
            fwer[key] = max(fwer[key], float(row.value))
    return fwer


def declared_replicates(
    pilot: pd.DataFrame | None = None, extension: pd.DataFrame | None = None
) -> tuple[int, float]:
    """The declared R and its family pass, from the committed pilot and its extension.

    Each FWER cell takes the largest of its oracle, pilot and extension values, unpooled.
    Without the extension, the calibration cells are planned at target (coverage 0.95, SE
    ratio 1, the pilot's largest kappa); an arm whose plug-in pass at that R is below
    :data:`EXTENSION_TRIGGER` stops the declaration until the extension runs. With the
    extension, :func:`calibration_pass` plans the four arms from it.

    Raises
    ------
    RuntimeError
        If an arm needs the pilot extension, or if no R up to :data:`MAXIMUM_REPLICATES`
        sizes the family: the row thread stops and reports before any declaration (design
        section 6).
    """
    planning: dict[str, Any] = {"fwer": planning_fwer(pilot, extension)}
    if pilot is not None:
        kappas = pilot.loc[pilot["kind"] == "se_ratio_kappa", "value"]
        if not kappas.empty:
            planning["kappa"] = float(kappas.max())
    if extension is not None:
        planning["arms"] = read_arms(extension)
    replicates, probability = size(**planning)
    if replicates is None:
        raise RuntimeError(
            f"no R up to {MAXIMUM_REPLICATES} sizes the family (pass {probability:.3f}); "
            "stop and report before any declaration"
        )
    if extension is None and pilot is not None:
        low = {
            f"{row} {configuration}": round(plug_in_arm_pass(replicates, arm), 3)
            for (configuration, row), arm in read_arms(pilot).items()
            if plug_in_arm_pass(replicates, arm) < EXTENSION_TRIGGER
        }
        if low:
            raise RuntimeError(
                f"at the at-target R = {replicates}, calibration arms {low} pass below "
                f"{EXTENSION_TRIGGER}: run the pilot extension (--pilot-extension) first"
            )
    return replicates, probability


# ------------------------------------------------------------------------ the record

#: Every study-specific module the runner and the property module reach, and the evidence
#: modules they read.
MODULES: tuple[str, ...] = (
    "tests/studies/variable_importance.py",
    "tests/studies/variable_importance_law.py",
    "tests/studies/variable_importance_properties.py",
    "tests/studies/evidence/properties.py",
    "tests/studies/evidence/property_verdicts.py",
    "tests/studies/evidence/schema.py",
    "tests/studies/evidence/seeds.py",
)


# ------------------------------------------------------------------------ the property cells
# Design section 7. A cell name never holds a dot or a colon: ``alpha05``, ``Z2``.

FWER_CALIBRATION = "fwer_calibration"
FAMILY_WISE_ERROR = "family_wise_error"
POWER = "power"
CALIBRATION = "interval_calibration"
FALSE_DISCOVERY = "false_discovery"
CORRELATION_ERROR = "correlation_error"
ADJUSTMENT_WITNESS = "adjustment_witness"
#: The families every row of which is ``diagnostic``.
REPORTED_FAMILIES: tuple[str, ...] = (FALSE_DISCOVERY, CORRELATION_ERROR, ADJUSTMENT_WITNESS)
#: The FWER controls on the complete nulls.
FWER_CONTROLS: tuple[str, ...] = ("unadjusted", "shrunken_se")
#: The report-only multiple-testing procedures, at each level ``q`` in :data:`ALPHAS`.
PROCEDURES: tuple[str, ...] = ("bh", "by", "max_t")
#: The moderate rows whose max-T power is reported (design section 7).
MODERATE_ROWS: tuple[tuple[str, str], ...] = (
    ("independent_mixed", "Z:1"),
    ("dobriban_like", "X1"),
    ("dobriban_like", "X2"),
)
#: The positive arm's suffix of an ``interval_calibration`` cell: every learner role is
#: correctly specified on these laws.
CALIBRATION_SUFFIX = "correctly_specified"
DIAGNOSTIC = "diagnostic"


def alpha_label(alpha: float) -> str:
    """``alpha05`` for 0.05, ``alpha10`` for 0.10."""
    return f"alpha{round(alpha * 100):02d}"


def q_label(q: float) -> str:
    """``q05`` for 0.05, ``q10`` for 0.10."""
    return f"q{round(q * 100):02d}"


def row_label(name: str) -> str:
    """A parameter's name in a cell: ``Z:2`` becomes ``Z2``."""
    return name.replace(":", "")


def calibration_label(row: str, configuration: str) -> str:
    """The ``interval_calibration`` label of a row and configuration, ``X1_in_sample``."""
    return f"{row}_{configuration}"


def is_null(law: str, name: str) -> bool:
    """Whether a parameter is an exact null of the declared estimand on a law."""
    return abs(truths(law)[name]) <= law_module.NULL_TOLERANCE


def gated_power(law: str, row: str, configuration: str) -> bool:
    """Whether a power row is gated: the three-level rows of a cross-fitted run are not."""
    return (law, row) in PLANNING_POWER and not (
        configuration == "cross_fitted" and row.startswith("Z")
    )


def law_cells(law: str, configuration: str) -> list[tuple[str, str, str]]:
    """Every ``(family, cell, role)`` one replicate of a law fills for one configuration.

    ``interval_calibration`` lists only its positive arm; the derived controls are
    :func:`tests.studies.evidence.property_verdicts.calibration_derived` of it.
    """
    cells: list[tuple[str, str, str]] = []
    reported = law not in GATED_LAWS
    for alpha in ALPHAS:
        stem = f"{law}__{configuration}__{alpha_label(alpha)}"
        if law in CALIBRATION_LAWS:
            cells.append((FWER_CALIBRATION, stem, "positive"))
            cells += [(FAMILY_WISE_ERROR, f"{stem}__{kind}", "control") for kind in FWER_CONTROLS]
        else:
            cells.append((FAMILY_WISE_ERROR, stem, DIAGNOSTIC if reported else "positive"))
    for name in ESTIMANDS:
        stem = f"{law}__{configuration}__{row_label(name)}"
        if (law, name) in PLANNING_POWER:
            role = "positive" if gated_power(law, name, configuration) else DIAGNOSTIC
            cells.append((POWER, stem, role))
        elif (law, name) in MODERATE_ROWS or (reported and not is_null(law, name)):
            cells.append((POWER, stem, DIAGNOSTIC))
        if (law, name) in ONE_SIDED_ROWS:
            cells.append((POWER, f"{stem}__one_sided", "control"))
    if law == "nulls_beside_signals":
        for name in CALIBRATION_ROWS:
            label = calibration_label(name, configuration)
            cells.append((CALIBRATION, f"{label}__{CALIBRATION_SUFFIX}", "positive"))
    for procedure in PROCEDURES:
        for q in ALPHAS:
            cell = f"{law}__{configuration}__{procedure}__{q_label(q)}"
            cells.append((FALSE_DISCOVERY, cell, DIAGNOSTIC))
    cells.append((CORRELATION_ERROR, f"{law}__{configuration}", DIAGNOSTIC))
    return cells


#: The law whose first property draws the adjustment witness refits with ``False``.
WITNESS_LAW = "nulls_beside_signals"


def witness_cells(configuration: str) -> list[tuple[str, str, str]]:
    """The adjustment witness's cells for one configuration: every row, diagnostic."""
    return [
        (ADJUSTMENT_WITNESS, f"{configuration}__{row_label(name)}", DIAGNOSTIC)
        for name in ESTIMANDS
    ]


def calibration_labels() -> tuple[str, ...]:
    """Every ``interval_calibration`` label, in configuration order."""
    return tuple(
        calibration_label(name, configuration)
        for configuration in CONFIGURATIONS
        for name in CALIBRATION_ROWS
    )


def property_cells() -> dict[str, tuple[str, ...]]:
    """Every family's cells, the derived calibration controls included."""
    out: dict[str, list[str]] = {}
    for configuration in CONFIGURATIONS:
        for law in LAW_NAMES:
            for family, cell, _ in law_cells(law, configuration):
                out.setdefault(family, []).append(cell)
        for family, cell, _ in witness_cells(configuration):
            out.setdefault(family, []).append(cell)
    for label in calibration_labels():
        out[CALIBRATION] += [f"{label}__shrunken_se_control", f"{label}__noise_control"]
    return {family: tuple(cells) for family, cells in out.items()}


PROPERTY_CELLS = property_cells()


STUDY = StudyRecord(
    name="variable importance with step-down max-T adjustment",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "variable_importance",
    document="docs/technical-reference/method-evidence/variable-importance.md",
    anchor="variable-importance-with-step-down-max-t-adjustment",
    scenarios=dict.fromkeys(SCENARIOS, ESTIMANDS),
    replicates=PRIMARY_REPLICATES,
    n=N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-variable-importance",
    reference=None,
    failure_bound={"cleverly-variable-importance": 0.01},
    scenario_seed_owners=SCENARIO_SEED_OWNERS,
    reported_scenarios=REPORTED_SCENARIOS,
    modules=MODULES,
    runner_module="tests.studies.variable_importance",
    properties_module="tests.studies.variable_importance_properties",
    publication_policy="gated",
    property_cells=PROPERTY_CELLS,
    property_artifacts=tuple(REPLICATES_FILES.values()),
)

CONFIGURATION = {
    "fit": (
        "variable_importance(frame, outcome='Y', candidates=X1..X10 and Z, covariates=['W'], "
        "adjust_for_other_candidates=True, keep_fits=False) with TMLE(outcome_learner and "
        "treatment_learner LogisticRegression(C=1e6, max_iter=2000), simultaneous=False, "
        "random_state=0), in sample or with n_folds=5, the adjustment at its shipped default "
        "(max_t, 10,000 draws, multiplier_seed=0)"
    ),
    "laws": "tests/studies/variable_importance_law.py, six laws, exact truths and covariance",
    "pilot": f"{PILOT_REPLICATES} replicates per law on stream {PILOT_STREAM!r}",
    "margins": f"FWER band and ceiling at alpha (1 +/- {DELTA}); power floor 0.80",
}
