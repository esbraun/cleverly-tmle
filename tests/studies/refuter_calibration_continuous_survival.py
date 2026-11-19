r"""Registered evidence for the refutation verdict rules on continuous-time survival fits (X101 PR 2).

The subject is not an estimator but two decision rules that :func:`cleverly.refute` prints
beside its statistics on a :class:`~cleverly.survival.ContinuousSurvivalResult`
(:mod:`cleverly.validation._continuous_survival_refute`).

=====================  ==========================================================================
rule                   the reading
=====================  ==========================================================================
placebo ``t``          the one-sample ``t`` of 20 placebo refits, each on the recorded treatment
                       permuted among the units, against the exact null 0 of a risk difference,
                       two-sided at 0.05 under ``t_19``
                       (:func:`~cleverly.validation._readings.placebo_reading`)
subset ``rho``         the scatter of 20 refits on 700 of the 1,000 units, each keeping its
                       original outer fold, against ``SE sqrt((n/m - 1) n/(n-1))``, two-sided at
                       0.05 under ``chi^2_19`` (:func:`~cleverly.validation._readings.subset_reading`)
=====================  ==========================================================================

The X101 plan (rev 2, section 5, plans repository) declares the design. Each cell checks that a
rule is not grossly miscalibrated on one configuration. It does not certify size to two
decimals.

**The fit.** The default :class:`~cleverly.survival.ContinuousSurvivalTMLE`: the default
hazard, censoring and treatment libraries, cross-fitted with 10 outer folds and selection in 5
inner folds, at :math:`n = 1{,}000`, targeting the three quartiles of the observed event times
(:data:`~tests.studies.continuous_survival_law.TIMES`), the configuration the
``continuous-time-survival`` record gates. The rules read the risk difference of each of the two
causes at the two gated times, the median and the upper quartile (1.12 and 2.32): four
estimands, :data:`ESTIMANDS`.

**The size law** is the registered base law of :mod:`tests.studies.continuous_survival_law`
(Hage et al. 2025 scenario 1 with the arm-specific term :math:`\beta Z X_4`, two causes,
censoring independent of everything, about half treated).

**The power law** raises the treatment's effect so that each of the four differences is at
least 4 efficiency SE at :math:`n = 1{,}000` (plan finding M2): :math:`\beta` = :data:`POWER_BETA`
and the cause-2 arm coefficient :data:`POWER_CAUSE2_ARM`. The plan names the arm-specific term
alone. Raising :math:`\beta` alone reaches 4 SE on the cause-2 difference at the median only at
:math:`\beta = 3.5`, where the true treated survival at the last target time falls to
:math:`5 \times 10^{-13}` for units with :math:`X_1 = 1` and :math:`X_3` just below 1, below
the survival floor of :math:`10^{-12}` the engine refused at when this design was set, so almost
every fit would have raised. The engine now refuses only below
:data:`~cleverly.survival.engine.SURVIVAL_UNDERFLOW` (:math:`10^{-250}`), since X6 part A
computes the clever covariate without cancellation, so that reason no longer binds; the
registered coefficients stand until the design is revisited. The design test checks both facts
from the exact law.

**Size cells** (``refuter_size``, 8 cells). 600 trials on the size law. Each trial draws one
sample, fits it once, and runs one placebo test and one subset test of 20 refits each. All eight
readings of a trial come from those 41 fits: the four :func:`~cleverly.validation.refute` calls of
a trial share one seed, so they draw the same permutations and subsets, and :func:`shared_refits`
fits each perturbed sample once. A reading passes when it rejects in at most 68 trials, the largest
count whose 99% Clopper-Pearson upper endpoint stays below that of the X82 rule's 17 of 200. The
tolerated size is larger than the X82 count's: at a true rejection rate of 0.10 a cell passes with
probability 0.875 (0.285 at 17 of 200). The 200-trial pilot sized the count: its predictive family
pass reads 0.39 at 200 trials, 0.82 at 400 and 0.92 at 600.

**Power cells** (``refuter_power``, 12 cells). A power reading passes when it detects its
defect in at least 38 of 50 trials.

====================  ==========================================================================
defect                construction
====================  ==========================================================================
stale rows            50 trials on the power law: the placebo keeps the recorded treatment on a
                      declared share of rows (:data:`STALE_SHARE`) and permutes the rest among
                      themselves (:func:`stale_rows`). 41 fits per trial, read on all four
                      estimands. The design test sizes the share for per-trial power 0.95
half SE, double SE    the subset ``rho`` of size trials 0 to 49 recomputed with the reported SE
                      halved and doubled. ``rho`` is linear in ``1 / SE``, so this is exact and
                      costs no fit
====================  ==========================================================================

**Reporting cells** (no verdict; :data:`REPORTING_CELLS`). 100 trials each, the placebo only,
on the size law with censoring that depends on the arm and :math:`X_3`
(``dependent_censoring``), and with about 25% treated (``imbalanced``). They publish as
``reporting-cells.csv`` and ``reporting-replicates.csv``; the registry scope names neither as
measured by a gate.

**Seeds.** Size trials draw from the named stream :data:`STREAM`, power trials from
:data:`POWER_STREAM`, reporting cells from :data:`REPORTING_STREAM` and the pilot from
:data:`PILOT_STREAM`; each stream has its own ``sample``, ``fit`` and ``refute`` labels
(:func:`trial_seed`).

**Fit configuration.** Every trial reads the configuration off its fitted result
(:func:`~cleverly.validation._interpretation.fit_configuration` with ``replicates = 20``) for
each estimand. The run stops if any axis but the smallest event count and the event share
varies across a cell's trials. ``cells.csv`` records the one configuration of each estimand and
law, and the ranges of its event count and event share; a registry row licensed by this study
covers that configuration and that event-share range.

**Pending.** :data:`DECLARATION_PENDING` names any pre-run gate the pilot leaves short; a
declared run is refused until it is empty. It is empty: the 200-trial pilot after X6 part A
lost no trial and reads a family pass of 0.92 at 600 trials.

**Red reading** (declared before the run). A size red is a defect suspect until diagnosed. A red
with no defect goes to reporting with an owner, and the rule's registry row stays descriptive.
No rerun buys a pass. A power red with no defect means the defect was sized too small for this
configuration; it keeps the rule descriptive and goes to reporting.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import math
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy import stats

from cleverly.survival import ContinuousSurvivalTMLE
from cleverly.validation import refute
from cleverly.validation._interpretation import fit_configuration
from tests.parallel import STUDY_JOBS
from tests.studies import continuous_survival_law as law
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

N = 1_000
#: The row count of every fit, as the shared driver reads it.
PRIMARY_N = N
#: The outer cross-fitting folds and the inner selection folds: the estimator's defaults.
N_FOLDS = 10
LEARNER_FOLDS = 5
SEED = 2026111401
RESAMPLING_SEED = 2026111402

#: Refits per placebo and subset reading: the library default.
REFITS = 20
#: The share of units each subset refit reads: ``refute``'s default.
SUBSET_FRACTION = 0.7
#: The placebo draw every placebo reading of this study ran: ``refute``'s default, which
#: permutes the recorded treatment column. The registry rows cover this draw only.
PLACEBO_DRAW: Final = "permute"
SIZE_TRIALS = 600
POWER_TRIALS = 50
#: Trials per reporting cell.
REPORTING_TRIALS = 100
#: A size reading passes at no more than this many rejections in :data:`SIZE_TRIALS`.
SIZE_MAX_REJECTIONS = 68
#: A power reading passes at no fewer than this many detections in :data:`POWER_TRIALS`.
POWER_MIN_DETECTIONS = 38
#: The level every rule reads.
LEVEL = 0.05

#: The named seed streams of the size trials, the power trials, the reporting cells and the
#: pilot. No other study reads them.
STREAM = "x101-refuter"
POWER_STREAM = "x101-refuter-power"
REPORTING_STREAM = "x101-refuter-reporting"
PILOT_STREAM = "x101-refuter-pilot"

#: The power law's coefficients: the arm-specific term of the cause-1 log hazard and the arm
#: coefficient of the cause-2 log hazard (the size law reads 1.0 and -0.5).
POWER_BETA = 2.0
POWER_CAUSE2_ARM = -1.0
#: Each calibrated difference on the power law is at least this many efficiency SE at ``N``.
POWER_EFFECT_SE = 4.0
#: The share of rows whose placebo keeps the recorded treatment. The design test requires a
#: per-trial power of at least 0.95 on every estimand of the power law.
STALE_SHARE = 0.35
#: Why a declared run is refused, or ``""`` once the design test's pre-run gates read green.
#: ``regenerate.py`` refuses a declared run while it is set, and the design test requires it
#: exactly while a gate is short (plan 5: the family pass must read 0.90 before the
#: declaration).
DECLARATION_PENDING = ""
#: The treated share of the ``imbalanced`` reporting cell.
IMBALANCED_SHARE = 0.25

#: The gated target times: the median and upper quartile of the observed event times.
GATED_TIMES = law.TIMES[1:]
#: ``key -> parameter name`` of the four risk differences every rule reads.
ESTIMANDS: dict[str, str] = {
    f"c{code}_t{t:g}": law.difference_name(cause, t)
    for code, cause in law.CAUSES.items()
    for t in GATED_TIMES
}

SIZE_FAMILY = "refuter_size"
POWER_FAMILY = "refuter_power"
SIZE_READINGS = ("placebo", "subset")
#: The power readings of each estimand: the stale-row placebo on the power law and the two
#: rescaled subset readings of the size trials.
POWER_READINGS = ("placebo_stale_rows", "subset_half_se", "subset_double_se")
SUBSET_DEFECTS = ("subset_half_se", "subset_double_se")

#: ``cell -> law`` of the reporting cells: the placebo only, no verdict.
REPORTING_CELLS: dict[str, str] = {
    "dependent_censoring": "dependent_censoring",
    "imbalanced": "imbalanced",
}

PROPERTY_CELLS = {
    SIZE_FAMILY: tuple(f"{key}__{reading}" for key in ESTIMANDS for reading in SIZE_READINGS),
    POWER_FAMILY: tuple(f"{key}__{defect}" for key in ESTIMANDS for defect in POWER_READINGS),
}

#: The primary phase: the size law's default fit, the four risk differences against their
#: exact truths. It is a sanity row on the law; no verdict of this study reads it.
SCENARIOS = {"size": tuple(ESTIMANDS.values())}
PRIMARY_REPLICATES = 200

SUBJECT = "cleverly-refute-continuous-survival"

STUDY = StudyRecord(
    name="continuous-time survival refutation verdict rules",
    slug="refuter-calibration-continuous-survival",
    artifacts=ROOT / "tests" / "canonical" / "refuter_calibration_continuous_survival",
    document="docs/technical-reference/method-evidence/refuter-calibration-continuous-survival.md",
    anchor="continuous-time-survival-refutation-verdict-rules",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=SUBJECT,
    reference=None,
    failure_bound={SUBJECT: 0.01},
    nuisance_count=3,
    modules=(
        "tests/studies/refuter_calibration_continuous_survival.py",
        "tests/studies/refuter_calibration_continuous_survival_properties.py",
        "tests/studies/continuous_survival_law.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.refuter_calibration_continuous_survival",
    properties_module="tests.studies.refuter_calibration_continuous_survival_properties",
    property_cells=PROPERTY_CELLS,
    property_artifacts=("cells.csv", "reporting-cells.csv", "reporting-replicates.csv"),
    publication_policy="reporting",
)

CONFIGURATION = {
    "folds": N_FOLDS,
    "learner_folds": LEARNER_FOLDS,
    "learners": "cox.default_library() for each hazard; logistic + lasso C in (1, 0.1, 0.03)",
    "times": list(law.TIMES),
    "estimands": dict(ESTIMANDS),
    "refits": REFITS,
    "subset_fraction": SUBSET_FRACTION,
    "placebo_draw": PLACEBO_DRAW,
    "size_trials": SIZE_TRIALS,
    "power_trials": POWER_TRIALS,
    "reporting_trials": REPORTING_TRIALS,
    "size_pass": f"rejections <= {SIZE_MAX_REJECTIONS} of {SIZE_TRIALS}",
    "power_pass": f"detections >= {POWER_MIN_DETECTIONS} of {POWER_TRIALS}",
    "size_law": "tests/studies/continuous_survival_law.py, base law",
    "power_law": f"base law with beta = {POWER_BETA} and cause-2 arm coefficient "
    f"{POWER_CAUSE2_ARM}",
    "stale_share": STALE_SHARE,
    "reporting_cells": {
        "dependent_censoring": "censoring rate 0.15 exp(0.5 X3 + 0.5 Z)",
        "imbalanced": f"treatment logit shifted so that {IMBALANCED_SHARE} are treated",
    },
    "streams": {
        "size": STREAM,
        "power": POWER_STREAM,
        "reporting": REPORTING_STREAM,
        "pilot": PILOT_STREAM,
    },
}


# ---- the laws -----------------------------------------------------------------------------

#: The laws a trial draws from.
LAWS = ("size", "power", "dependent_censoring", "imbalanced")


def draw_law(name: str, n: int, seed: int) -> pd.DataFrame:
    """One sample of the size law, the power law or a reporting cell's law."""
    if name == "size":
        return law.sample(n, seed)
    if name == "power":
        return law.sample(n, seed, beta=POWER_BETA, cause2_arm=POWER_CAUSE2_ARM)
    if name == "dependent_censoring":
        return law.sample(n, seed, dependent_censoring=True)
    if name == "imbalanced":
        return law.sample(n, seed, treated_share=IMBALANCED_SHARE)
    raise ValueError(f"unknown law {name!r}")


def power_truth() -> dict[str, float]:
    """The exact risk differences of :data:`ESTIMANDS` on the power law."""
    truth = law.truth(GATED_TIMES, beta=POWER_BETA, cause2_arm=POWER_CAUSE2_ARM)
    return {key: truth[name] for key, name in ESTIMANDS.items()}


def efficiency_se(*, power: bool, n: int = N) -> dict[str, float]:
    """The efficiency SE at ``n`` of each difference of :data:`ESTIMANDS` on one law."""
    beta, cause2_arm = (POWER_BETA, POWER_CAUSE2_ARM) if power else (law.BETA, law.CAUSE2_ARM)
    out = {}
    for key in ESTIMANDS:
        code, t = int(key[1]), float(key.split("_t")[1])
        sd = law.efficiency_sd(code, t, beta=beta, cause2_arm=cause2_arm)["difference"]
        out[key] = sd / math.sqrt(n)
    return out


def stale_shift(share: float, *, nodes: int = 200) -> dict[str, float]:
    r"""The identified risk difference of each estimand under the stale-row placebo law.

    On the power law a share ``q`` of rows keeps :math:`Z` and the rest carry a draw
    independent of :math:`(X, T)` with the marginal share :math:`\bar p`. Censoring is
    independent of everything, so the placebo arm's cumulative incidence given :math:`X` mixes
    the two arms' with the weight :math:`w_a(X) = P(Z = 1 \mid Z^* = a, X)`:
    :math:`w_1 = g (q + (1 - q) \bar p) / (q g + (1 - q) \bar p)` and
    :math:`w_0 = (1 - q) g (1 - \bar p) / (q (1 - g) + (1 - q)(1 - \bar p))`, and the
    difference is :math:`E_X[(w_1 - w_0)(F_j(t \mid 1, X) - F_j(t \mid 0, X))]`. It vanishes
    at ``q = 0``, the exact placebo null.
    """
    from scipy.special import expit

    x3, w3 = law._x3_rule(nodes)
    weights = np.concatenate([w3 / 3.0] * 3)
    x1 = np.concatenate([np.full_like(x3, a) for a, _ in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))])
    x2 = np.concatenate([np.full_like(x3, b) for _, b in ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))])
    x3 = np.concatenate([x3] * 3)
    g = expit(law.OMEGA[0] * x1 + law.OMEGA[1] * x2 + law.OMEGA[2] * x3)
    pbar = float(weights @ g)
    q = share
    w1 = g * (q + (1 - q) * pbar) / (q * g + (1 - q) * pbar)
    w0 = (1 - q) * g * (1 - pbar) / (q * (1 - g) + (1 - q) * (1 - pbar))
    out = {}
    for key in ESTIMANDS:
        code, t = int(key[1]), float(key.split("_t")[1])
        risks = []
        for z in (1, 0):
            h1, h2 = law.rates(x1, x2, x3, z, beta=POWER_BETA, cause2_arm=POWER_CAUSE2_ARM)
            h = h1 + h2
            risks.append((h1 if code == 1 else h2) / h * (1.0 - np.exp(-h * t)))
        out[key] = float(weights @ ((w1 - w0) * (risks[0] - risks[1])))
    return out


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary sample from an explicit seed for the published-seed audit."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return draw_law(scenario, n, seed), {name: law.TRUTH[name] for name in ESTIMANDS.values()}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


# ---- the fit ------------------------------------------------------------------------------


def estimator(seed: int) -> ContinuousSurvivalTMLE:
    """The default estimator, cross-fitted, seeded with the trial's fit seed."""
    return ContinuousSurvivalTMLE(
        law.TIMES,
        arms=law.ARMS,
        reference=law.REFERENCE,
        causes=law.CAUSES,
        cross_fit=True,
        n_folds=N_FOLDS,
        learner_folds=LEARNER_FOLDS,
        simultaneous=False,
        random_state=seed,
    )


def fit(frame: pd.DataFrame, seed: int) -> Any:
    """Fit one sample."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator(seed).fit(
            frame, time="time", event="event", treatment="Z", baseline=list(law.BASELINE)
        )


# ---- shared refits and the stale-row defect ----------------------------------------------


class _Refitted:
    """The estimates of one refit, which is all the refutation adapter reads of it.

    A full result holds its stored data and targeting record; forty of them would hold
    several hundred megabytes for no reading.
    """

    def __init__(self, result: Any) -> None:
        self.estimates = {name: result[name] for name in result.estimates}

    def __getitem__(self, name: str) -> Any:
        return self.estimates[name]


@dataclass
class RefitLog:
    """How many refits a :func:`shared_refits` block was asked for, and how many it ran."""

    calls: int = 0
    fits: int = 0
    keys: set[str] = field(default_factory=set)


def _refit_key(estimator: Any, data: Any, random_state: Any, folds: Any) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(repr((id(estimator), random_state, tuple(data.names))).encode())
    for array in (data.time, data.event, data.treatment, data.baseline):
        values = np.ascontiguousarray(np.asarray(array))
        digest.update(repr((values.dtype.str, values.shape)).encode())
        digest.update(values.tobytes())
    if folds is None:
        digest.update(b"no folds")
    else:
        digest.update(np.ascontiguousarray(np.asarray(folds, dtype=np.int64)).tobytes())
    return digest.hexdigest()


@contextlib.contextmanager
def shared_refits() -> Iterator[RefitLog]:
    """Fit each perturbed sample once across the refutation calls of one block.

    The four :func:`~cleverly.validation.refute` calls of a trial differ only in the estimand
    and share the seed, so they draw the same subsets and permutations and ask for the same
    40 refits. Inside the block the first call of each ``(estimator, data, seed, folds)`` runs
    :meth:`~cleverly.survival.ContinuousSurvivalTMLE.refit` and every later call reuses its
    estimates, so each reading is bit for bit the reading of a lone call.
    """
    original: Callable[..., Any] = ContinuousSurvivalTMLE.refit
    cache: dict[str, _Refitted] = {}
    log = RefitLog()

    def cached(
        self: ContinuousSurvivalTMLE,
        data: Any,
        *,
        random_state: int | None,
        _frozen_folds: Any = None,
    ) -> Any:
        key = _refit_key(self, data, random_state, _frozen_folds)
        log.calls += 1
        if key not in cache:
            log.fits += 1
            log.keys.add(key)
            cache[key] = _Refitted(
                original(self, data, random_state=random_state, _frozen_folds=_frozen_folds)
            )
        return cache[key]

    ContinuousSurvivalTMLE.refit = cached  # type: ignore[method-assign]
    try:
        yield log
    finally:
        ContinuousSurvivalTMLE.refit = original  # type: ignore[method-assign]


@contextlib.contextmanager
def stale_rows(share: float) -> Iterator[None]:
    """Patch the continuous-time placebo so a share of rows keeps the recorded treatment.

    The kept rows are a uniform draw of ``round(share n)`` rows; the other rows' treatments
    are permuted among themselves, so the placebo column keeps the recorded arm counts.
    """
    module: Any = importlib.import_module("cleverly.validation._continuous_survival_refute")
    original: Callable[..., np.ndarray] = module._placebo_treatment

    def mutated(treatment: np.ndarray, rng: np.random.Generator, mode: str) -> np.ndarray:
        if mode != "permute":
            raise AssertionError("the stale-row defect patches the permutation placebo only")
        out = np.asarray(treatment).copy()
        n = out.size
        kept = rng.choice(n, size=round(share * n), replace=False)
        moved = np.setdiff1d(np.arange(n), kept)
        out[moved] = rng.permutation(out[moved])
        return out

    module._placebo_treatment = mutated
    try:
        yield
    finally:
        module._placebo_treatment = original


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


def trial_seed(stream: str, label: str, cell: str, trial: int) -> int:
    """The seed of one label of one trial on one named stream.

    The labels are ``"sample"`` (the draw), ``"fit"`` (the fit's folds and learners) and
    ``"refute"`` (the permutations and subsets of every refutation call of the trial, and the
    refits' seed). ``cell`` names the law or reporting cell within the stream.
    """
    return stream_seed(STUDY, stream, label, cell, trial)


def trial_configuration(result: Any, estimand: str) -> dict[str, Any]:
    """The fit configuration of one estimand of one trial, as one flat row."""
    configuration = replace(fit_configuration(result, estimand), replicates=REFITS)
    row = asdict(configuration)
    row["learners"] = " | ".join(configuration.learners)
    return row


def refute_all(
    result: Any, tests: tuple[str, ...], seed: int, *, stale_share: float | None = None
) -> tuple[dict[str, Any], RefitLog]:
    """One :func:`~cleverly.validation.refute` report per estimand, all from shared refits."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(warnings.catch_warnings())
        warnings.simplefilter("ignore")
        if stale_share is not None:
            stack.enter_context(stale_rows(stale_share))
        log = stack.enter_context(shared_refits())
        reports = {
            key: refute(
                result,
                estimand=name,
                tests=tests,
                n_replicates=REFITS,
                subset_fraction=SUBSET_FRACTION,
                placebo_mode=PLACEBO_DRAW,
                random_state=seed,
            )
            for key, name in ESTIMANDS.items()
        }
    return reports, log


@dataclass(frozen=True)
class Trial:
    """One trial's readings by ``estimand__reading``, and each estimand's configuration."""

    readings: dict[str, Reading]
    configurations: dict[str, dict[str, Any]]
    fits: int


#: ``group -> (stream, law, tests, stale share)`` of the trials :func:`trial_readings` runs.
GROUPS: dict[str, tuple[str, str, tuple[str, ...], float | None]] = {
    "size": (STREAM, "size", ("placebo", "subset"), None),
    "power": (POWER_STREAM, "power", ("placebo",), STALE_SHARE),
    "dependent_censoring": (REPORTING_STREAM, "dependent_censoring", ("placebo",), None),
    "imbalanced": (REPORTING_STREAM, "imbalanced", ("placebo",), None),
}


def draw_trial(group: str, trial: int, stream: str | None = None) -> pd.DataFrame:
    """The sample of one trial of ``group``, on its declared stream or on ``stream``."""
    declared, law_name, _, _ = GROUPS[group]
    return draw_law(law_name, N, trial_seed(stream or declared, "sample", group, trial))


def trial_readings(
    group: str, frame: pd.DataFrame, trial: int, *, stream: str | None = None
) -> tuple[Trial, Any, dict[str, Any]]:
    """Every reading of one trial of ``group``, the fit, and the four reports.

    A size trial reads the placebo and the subset on every estimand, and the two rescaled
    subset readings when ``trial < POWER_TRIALS``. A power trial reads the stale-row placebo.
    A reporting trial reads the placebo.
    """
    declared, _, tests, stale = GROUPS[group]
    chosen = stream or declared
    result = fit(frame, trial_seed(chosen, "fit", group, trial))
    configurations = {key: trial_configuration(result, name) for key, name in ESTIMANDS.items()}
    reports, log = refute_all(
        result, tests, trial_seed(chosen, "refute", group, trial), stale_share=stale
    )
    readings: dict[str, Reading] = {}
    for key, report in reports.items():
        if group == "power":
            readings[f"{key}__placebo_stale_rows"] = _reading(report["placebo"])
            continue
        readings[f"{key}__placebo"] = _reading(report["placebo"])
        if group != "size":
            continue
        subset = _reading(report["subset"])
        readings[f"{key}__subset"] = subset
        if trial < POWER_TRIALS:
            readings[f"{key}__subset_half_se"] = rescaled_subset(subset.statistic, 0.5)
            readings[f"{key}__subset_double_se"] = rescaled_subset(subset.statistic, 2.0)
    return Trial(readings, configurations, 1 + log.fits), result, reports


# ---- the primary phase --------------------------------------------------------------------


def cleverly_rows(
    frame: pd.DataFrame, truth_map: Mapping[str, float], scenario: str, replicate: int, seed: int
) -> list[dict[str, Any]]:
    """The four risk differences of one size-law fit as primary rows."""
    result = fit(frame, seed)
    rows = []
    for name in ESTIMANDS.values():
        estimate = result[name]
        low, high = (float(value) for value in estimate.ci)
        reference = float(truth_map[name])
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": int(result.n),
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": low,
                "ci_upper": high,
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": float(result.initial_estimates[name]),
            }
        )
    return rows


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
