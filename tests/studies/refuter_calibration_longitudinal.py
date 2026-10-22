"""Registered evidence for the longitudinal refutation verdict rules (X82 follow-up SL).

The subject is not an estimator but three decision rules that :func:`cleverly.refute` prints
beside its statistics on a longitudinal fit.

=====================  ==========================================================================
rule                   the reading
=====================  ==========================================================================
placebo ``t``          the one-sample ``t`` of 20 Markov placebo refits against the exact null,
                       two-sided at 0.05 under ``t_19`` (:func:`~cleverly.validation._readings.placebo_reading`)
subset ``rho``         the scatter of 20 subset refits against ``SE sqrt((n/m - 1) n/(n-1))``,
                       two-sided at 0.05 under ``chi^2_19`` (:func:`~cleverly.validation._readings.subset_reading`)
negative control ``z`` the fit refitted on a declared negative-control outcome, ``z`` against the
                       no-effect value, two-sided at 0.05
=====================  ==========================================================================

The X82 plan (rev 5, sections 7.2 and 8.1 to 8.6, plans repository) declares the design. Each
cell checks that a rule is not grossly miscalibrated on one configuration. It does not certify
size to two decimals.

**Law E** (end of study, cells S1, I1, B1, B2, B3). Two nodes and no censoring. ``W1`` is
standard normal, ``W2`` is Bernoulli(0.5), and ``U`` is an unmeasured standard normal.

.. code-block:: text

    logit P(A1 = 1 | W, U)        = -0.2 + 0.6 W2 + 0.5 U
    L2                            = 0.5 W1 + 0.3 W2 + 0.5 A1 + N(0, 1)
    logit P(A2 = 1 | W, A1, L2)   = -0.3 + 0.4 W1 - 0.3 W2 + 0.8 A1 + 0.5 L2
    Y                             = 0.5 W1 + 0.4 W2 + 0.5 A1 + 0.4 L2 + 0.5 A2 + N(0, 1)

``U`` reads ``A1`` and nothing that the fit targets. It drives the treatment and the confounded
negative control only. ``A1`` reads ``W`` only through the binary ``W2``, so a main-terms
logistic GLM of ``A1`` on ``(W1, W2)`` is saturated in the law with ``U`` integrated out. The
other GLMs are correctly specified: ``A2`` is logistic in its history, ``L2`` and ``Y`` are
linear, and so the iterated regression of ``Y`` on ``(W, A1)`` is linear too. The contrast
``always`` against ``never`` is 1.2 exactly. Each sample carries two Gaussian negative
controls:

- ``NC``, a valid control: the outcome's baseline part ``0.5 W1 + 0.4 W2`` plus ``N(0, 1)``,
  with no treatment term and no ``U``.
- ``NCU``, a confounded control: ``NC + delta U`` with ``delta`` :data:`CONTROL_DELTA`.
  ``U`` drives ``A1``, so the control's adjusted contrast is not zero.

**Law S** (survival, cell L1) is the registered law of
:func:`cleverly.datasets.make_longitudinal_survival`: two nodes, an absorbing event at each,
monotone censoring and a misspecifying ``tanh`` term in the second hazard. The registered
survival studies ``ltmle-survival`` and ``ltmle-survival-crossfit`` read it.

**The cells** (plan 8.5). ``n = 1,000``, ``T = 2``, ``R = 20`` refits.

====  ==========================================================  ===========  =====================
cell  configuration                                               law          size readings
====  ==========================================================  ===========  =====================
L1    ``LTMLE`` static regimens, GLM nuisances, five folds         S, risk at   placebo, subset
                                                                  ``t = 2``
S1    ``SDR`` static regimens, GLM nuisances, five folds           E            placebo, subset,
                                                                               control
I1    ``IPSI``, odds ``x2`` against the natural course, GLM,       E            placebo, subset,
      five folds                                                               control
B1    ``SequentialGComputation``, GLM, in sample                  E            placebo, subset,
                                                                               control
B2    ``SequentialIPW`` (Hajek), GLM, in sample                   E            placebo, subset,
                                                                               control
B3    ``ParametricGFormula``, GLM models, 10 draws per unit       E            placebo, subset,
                                                                               control
====  ==========================================================  ===========  =====================

**L1 reads no negative control.** The plan's cell lists one, but the C2 code refuses a
negative-control outcome on a survival fit (``refute.py``, ``_refuse_negative_control_design``):
no replacement keeps the risk sets. That refusal is the shipped behavior, so no survival row
of the control rule can exist yet.

**The baselines run in sample.** ``ParametricGFormula`` requires ``n_folds = 1``, and the two
other baselines are the in-sample estimators their evidence page describes. The fold count is
in each cell's recorded configuration.

**The g-formula's Monte Carlo size** (plan 8.7) is the shipped default of 10 histories per unit.
A fit at another ``n_draws`` is outside the cell's scope.

**Size cells.** 200 trials. A reading passes when it rejects in at most 17 trials.

**Power cells.** Trials 0 to 49 of each size cell also run the defects below. A power reading
passes when it detects its defect in at least 38 of 50 trials. Each defect is sized from the
declared law so that its per-trial power is at least 0.95. The sizing probe
``tests/diagnostics/x82_sl_sizing/`` fits each cell's own estimator on large samples of the
defective law, and ``tests/unit/test_refuter_calibration_longitudinal_design.py`` reads it.

====================  ==========================================================================
defect                construction
====================  ==========================================================================
stale rows            the Markov placebo keeps the recorded trajectory on a declared share of
                      rows (:data:`STALE_SHARE`). The kept rows carry their own at-risk mask, so
                      the masked cells stay ``nan`` exactly where the observed treatment is
half SE, double SE    the subset ``rho`` of the same trial recomputed with the reported SE halved
                      and doubled. ``rho`` is linear in ``1 / SE``, so this is exact and costs no fit
confounded control    the negative-control fit on ``NCU`` instead of ``NC`` (law E only)
====================  ==========================================================================

The power cells are the controls of the size cells: a rule that never rejects passes its size
cell and fails its power cell.

**Fit configuration.** Every trial reads the configuration off its fitted result
(:func:`~cleverly.validation._interpretation.fit_configuration` with ``replicates = 20``). The
run stops if any axis but the smallest event count varies across a cell's trials. ``cells.csv``
records the one configuration of each cell and the range of its event count.

**Red reading** (declared before the run). A size red is a defect suspect until diagnosed. A red
with no defect goes to reporting with an owner, and the rule's registry row stays descriptive.
No rerun buys a pass. A power red with no defect means the defect was sized too small for this
cell's configuration; it keeps the rule descriptive and goes to reporting.
"""

from __future__ import annotations

import contextlib
import math
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit

from cleverly.datasets import make_longitudinal_survival
from cleverly.datasets.longitudinal import survival_truth
from cleverly.interventions import Incremental
from cleverly.learners import GLM
from cleverly.longitudinal import (
    IPSI,
    LTMLE,
    SDR,
    ParametricGFormula,
    SequentialGComputation,
    SequentialIPW,
)
from cleverly.validation import refute
from cleverly.validation._interpretation import fit_configuration
from cleverly.validation._markov_placebo import MarkovKernel
from tests.parallel import STUDY_JOBS
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

N = 1_000
#: The row count of every fit, as the shared driver reads it.
PRIMARY_N = N
#: The fold count of the three cross-fitted cells.
N_FOLDS = 5
SEED = 20268253
RESAMPLING_SEED = 20268254

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

# ---- law E --------------------------------------------------------------------------------

#: ``logit P(A1 = 1 | W, U) = A1_INTERCEPT + A1_W2 W2 + GAMMA_U U``.
A1_INTERCEPT = -0.2
A1_W2 = 0.6
GAMMA_U = 0.5
#: ``L2 = L2_COEF . (W1, W2, A1) + N(0, 1)``.
L2_COEF = (0.5, 0.3, 0.5)
#: ``logit P(A2 = 1 | ...) = A2_COEF . (1, W1, W2, A1, L2)``.
A2_COEF = (-0.3, 0.4, -0.3, 0.8, 0.5)
#: ``Y = Y_COEF . (W1, W2, A1, L2, A2) + N(0, 1)``.
Y_COEF = (0.5, 0.4, 0.5, 0.4, 0.5)
#: The exact contrast ``always`` against ``never`` on law E.
END_OF_STUDY_ATE = Y_COEF[2] + Y_COEF[3] * L2_COEF[2] + Y_COEF[4]
#: The strength of ``U`` in the confounded control. The sizing probe requires its adjusted
#: contrast to reach 3.605 control SE on every law-E cell.
CONTROL_DELTA = 1.25
#: The declared outcome bounds of the cross-fitted law-E fits. A cross-fitted continuous fit
#: needs declared bounds; these never bind (:func:`draw_end_of_study` refuses a sample
#: outside them, for ``Y``, ``NC`` and ``NCU`` alike).
Q_BOUNDS = (-12.0, 12.0)

#: The share of rows whose placebo keeps the recorded trajectory, by law. The sizing probe
#: requires the population shift of each cell's placebo to reach 0.850 placebo SD.
STALE_SHARE = {"end_of_study": 0.10, "survival": 0.40}

#: The regimens of the static cells, and the incremental plans of I1.
REGIMENS = {"always": 1, "never": 0}
REFERENCE = "never"
IPSI_PLAN = "odds x2"
IPSI_REFERENCE = "natural course"
#: The g-formula's simulated histories per unit: the shipped default.
GFORMULA_DRAWS = 10

SIZE_FAMILY = "refuter_size"
POWER_FAMILY = "refuter_power"

READINGS_ALL = ("placebo", "subset", "negative_control")
READINGS_NO_CONTROL = ("placebo", "subset")


@dataclass(frozen=True)
class Cell:
    """One configuration of plan 8.5.

    Parameters
    ----------
    law : str
        ``"end_of_study"`` (law E) or ``"survival"`` (law S).
    readings : tuple of str
        The size readings the cell reads.
    estimator : str
        ``"ltmle"``, ``"sdr"``, ``"ipsi"``, ``"ice"``, ``"ipw"`` or ``"gformula"``.
    """

    law: str
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

    @property
    def estimand(self) -> str:
        """The alias every reading of the cell refutes."""
        if self.estimator == "ipsi":
            return f"ate_regimen[{IPSI_PLAN} vs {IPSI_REFERENCE}]"
        if self.law == "survival":
            return "ate_regimen[always vs never @ t=2]"
        return "ate_regimen[always vs never]"


CELLS: dict[str, Cell] = {
    "L1": Cell("survival", READINGS_NO_CONTROL, "ltmle"),
    "S1": Cell("end_of_study", READINGS_ALL, "sdr"),
    "I1": Cell("end_of_study", READINGS_ALL, "ipsi"),
    "B1": Cell("end_of_study", READINGS_ALL, "ice"),
    "B2": Cell("end_of_study", READINGS_ALL, "ipw"),
    "B3": Cell("end_of_study", READINGS_ALL, "gformula"),
}

PROPERTY_CELLS = {
    SIZE_FAMILY: tuple(
        f"{name}__{reading}" for name, cell in CELLS.items() for reading in cell.readings
    ),
    POWER_FAMILY: tuple(
        f"{name}__{defect}" for name, cell in CELLS.items() for defect in cell.defects
    ),
}

#: The primary phase: the S1 and L1 fits against their exact truths. It is a sanity row on
#: each law; no verdict of this study reads it.
SCENARIOS = {"end_of_study": ("ate",), "survival": ("ate",)}
PRIMARY_REPLICATES = 200
#: The cell each primary scenario fits.
PRIMARY_CELL = {"end_of_study": "S1", "survival": "L1"}

STUDY = StudyRecord(
    name="longitudinal refutation verdict rules",
    slug="refuter-calibration-longitudinal",
    artifacts=ROOT / "tests" / "canonical" / "refuter_calibration_longitudinal",
    document="docs/technical-reference/method-evidence/refuter-calibration-longitudinal.md",
    anchor="longitudinal-refutation-verdict-rules",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-refute-longitudinal",
    reference=None,
    failure_bound={"cleverly-refute-longitudinal": 0.01},
    modules=(
        "tests/studies/refuter_calibration_longitudinal.py",
        "tests/studies/refuter_calibration_longitudinal_properties.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.refuter_calibration_longitudinal",
    properties_module="tests.studies.refuter_calibration_longitudinal_properties",
    property_cells=PROPERTY_CELLS,
    property_artifacts=("cells.csv",),
    publication_policy="reporting",
)

CONFIGURATION = {
    "nodes": 2,
    "folds": {"L1": N_FOLDS, "S1": N_FOLDS, "I1": N_FOLDS, "B1": 1, "B2": 1, "B3": 1},
    "refits": REFITS,
    "size_trials": SIZE_TRIALS,
    "power_trials": POWER_TRIALS,
    "size_pass": f"rejections <= {SIZE_MAX_REJECTIONS} of {SIZE_TRIALS}",
    "power_pass": f"detections >= {POWER_MIN_DETECTIONS} of {POWER_TRIALS}",
    "law_e_a1": "logit g1 = -0.2 + 0.6 W2 + 0.5 U",
    "law_e_l2": "L2 = 0.5 W1 + 0.3 W2 + 0.5 A1 + N(0, 1)",
    "law_e_a2": "logit g2 = -0.3 + 0.4 W1 - 0.3 W2 + 0.8 A1 + 0.5 L2",
    "law_e_y": "Y = 0.5 W1 + 0.4 W2 + 0.5 A1 + 0.4 L2 + 0.5 A2 + N(0, 1)",
    "law_e_controls": "NC = 0.5 W1 + 0.4 W2 + N(0, 1); NCU = NC + 1.25 U",
    "law_s": "cleverly.datasets.make_longitudinal_survival (censoring on)",
    "stale_share": dict(STALE_SHARE),
    "control_delta": CONTROL_DELTA,
    "gformula_draws": GFORMULA_DRAWS,
    "cells": {
        name: {"law": cell.law, "readings": list(cell.readings), "estimator": cell.estimator}
        for name, cell in CELLS.items()
    },
}


# ---- the laws -----------------------------------------------------------------------------


def draw_end_of_study(n: int, seed: int, delta: float = CONTROL_DELTA) -> pd.DataFrame:
    """One sample of law E: ``W1, W2, A1, L2, A2, Y, NC, NCU``.

    Raises
    ------
    RuntimeError
        If ``Y``, ``NC`` or ``NCU`` leaves the declared :data:`Q_BOUNDS`.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.standard_normal(n)
    w2 = rng.binomial(1, 0.5, n).astype(float)
    latent = rng.standard_normal(n)
    a1 = rng.binomial(1, expit(A1_INTERCEPT + A1_W2 * w2 + GAMMA_U * latent)).astype(float)
    l2 = L2_COEF[0] * w1 + L2_COEF[1] * w2 + L2_COEF[2] * a1 + rng.standard_normal(n)
    a2_index = A2_COEF[0] + A2_COEF[1] * w1 + A2_COEF[2] * w2 + A2_COEF[3] * a1 + A2_COEF[4] * l2
    a2 = rng.binomial(1, expit(a2_index)).astype(float)
    base = Y_COEF[0] * w1 + Y_COEF[1] * w2
    outcome = base + Y_COEF[2] * a1 + Y_COEF[3] * l2 + Y_COEF[4] * a2 + rng.standard_normal(n)
    control = base + rng.standard_normal(n)
    confounded = control + delta * latent
    low, high = Q_BOUNDS
    for values in (outcome, control, confounded):
        if values.min() <= low or values.max() >= high:
            raise RuntimeError("a law-E draw left the declared q_bounds")
    return pd.DataFrame(
        {
            "W1": w1,
            "W2": w2,
            "A1": a1,
            "L2": l2,
            "A2": a2,
            "Y": outcome,
            "NC": control,
            "NCU": confounded,
        }
    )


def draw_survival(n: int, seed: int) -> pd.DataFrame:
    """One sample of law S, the registered censored survival law."""
    frame, _ = make_longitudinal_survival(n, seed=seed)
    return pd.DataFrame(frame)


def draw_law(law: str, n: int, seed: int) -> pd.DataFrame:
    """One sample of ``law``."""
    if law == "end_of_study":
        return draw_end_of_study(n, seed)
    if law == "survival":
        return draw_survival(n, seed)
    raise ValueError(f"unknown law {law!r}")


#: Gauss-Hermite nodes and weights for a standard normal.
_NODES, _WEIGHTS = np.polynomial.hermite_e.hermegauss(60)
_WEIGHTS = _WEIGHTS / _WEIGHTS.sum()


def a1_posterior_means() -> dict[tuple[int, int], float]:
    """``E[U | W2 = w, A1 = a]`` on law E, by ``(w, a)``."""
    out = {}
    for w2 in (0, 1):
        given_u = expit(A1_INTERCEPT + A1_W2 * w2 + GAMMA_U * _NODES)
        for arm, likelihood in ((1, given_u), (0, 1.0 - given_u)):
            joint = likelihood * _WEIGHTS
            out[(w2, arm)] = float(np.sum(joint * _NODES) / np.sum(joint))
    return out


def a1_propensity(w2: int) -> float:
    """``P(A1 = 1 | W2 = w2)`` on law E, with ``U`` integrated out."""
    return float(np.sum(expit(A1_INTERCEPT + A1_W2 * w2 + GAMMA_U * _NODES) * _WEIGHTS))


def confounded_contrast(cell: str, delta: float = CONTROL_DELTA) -> float:
    """The exact contrast of ``NCU`` that cell ``cell`` targets on law E.

    ``U`` reads ``A1`` only, so ``E[NCU | W, A1, L2, A2] = E[NC | W] + delta E[U | W2, A1]``.
    A static regimen then reads ``delta E_W2[E[U | W2, 1] - E[U | W2, 0]]``. The incremental
    plan that multiplies the odds of ``A1`` by ``k`` moves the share treated at node 1 from
    ``g(W2)`` to ``q(W2) = k g / (k g + 1 - g)``, so its contrast against the natural course is
    ``delta E_W2[(q - g)(E[U | W2, 1] - E[U | W2, 0])]``. The node-2 tilt does not move the
    control, which is free of ``A2`` given ``(W, A1)``.
    """
    means = a1_posterior_means()
    total = 0.0
    for w2 in (0, 1):
        gap = means[(w2, 1)] - means[(w2, 0)]
        if CELLS[cell].estimator == "ipsi":
            g = a1_propensity(w2)
            q = 2.0 * g / (2.0 * g + 1.0 - g)
            gap *= q - g
        total += 0.5 * gap
    return delta * total


def truth(law: str) -> float:
    """The exact contrast of each primary scenario's fit."""
    if law == "end_of_study":
        return END_OF_STUDY_ATE
    if law == "survival":
        return survival_truth(1.0, 1.0, 2) - survival_truth(0.0, 0.0, 2)
    raise ValueError(f"unknown law {law!r}")


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary sample from an explicit seed for the published-seed audit."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return draw_law(scenario, n, seed), {"ate": truth(scenario)}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


# ---- the fits -----------------------------------------------------------------------------

#: The node declaration of each law.
COLUMNS: dict[str, dict[str, Any]] = {
    "end_of_study": {
        "outcome": "Y",
        "treatment": ["A1", "A2"],
        "baseline": ["W1", "W2"],
        "time_varying": [[], ["L2"]],
    },
    "survival": {
        "outcome": ["Y1", "Y2"],
        "treatment": ["A1", "A2"],
        "baseline": ["W1", "W2"],
        "time_varying": [[], ["L2"]],
        "censoring": ["C1", "C2"],
    },
}


def estimator(cell: Cell, seed: int) -> Any:
    """The cell's estimator, seeded with the trial's fit seed."""
    common: dict[str, Any] = {"simultaneous": False, "random_state": seed}
    gaussian = GLM(family="gaussian")
    if cell.estimator == "ltmle":
        return LTMLE(
            REGIMENS,
            reference=REFERENCE,
            outcome_learner=GLM(),
            pseudo_learner=GLM(),
            treatment_learner=GLM(),
            censoring_learner=GLM(),
            n_folds=N_FOLDS,
            **common,
        )
    if cell.estimator == "sdr":
        return SDR(
            REGIMENS,
            reference=REFERENCE,
            outcome_learner=gaussian,
            pseudo_learner=GLM(family="gaussian"),
            treatment_learner=GLM(),
            n_folds=N_FOLDS,
            q_bounds=Q_BOUNDS,
            **common,
        )
    if cell.estimator == "ipsi":
        return IPSI(
            [Incremental(2.0, name=IPSI_PLAN), Incremental(1.0, name=IPSI_REFERENCE)],
            reference=IPSI_REFERENCE,
            outcome_learner=gaussian,
            pseudo_learner=GLM(family="gaussian"),
            treatment_learner=GLM(),
            n_folds=N_FOLDS,
            q_bounds=Q_BOUNDS,
            **common,
        )
    if cell.estimator == "ice":
        return SequentialGComputation(
            REGIMENS, reference=REFERENCE, outcome_learner=gaussian, **common
        )
    if cell.estimator == "ipw":
        return SequentialIPW(REGIMENS, reference=REFERENCE, treatment_learner=GLM(), **common)
    if cell.estimator == "gformula":
        return ParametricGFormula(
            REGIMENS,
            reference=REFERENCE,
            n_draws=GFORMULA_DRAWS,
            covariate_types={"L2": "normal"},
            covariate_learner=GLM(),
            outcome_learner=gaussian,
            treatment_learner=GLM(),
            **common,
        )
    raise ValueError(f"unknown estimator {cell.estimator!r}")


def fit_cell(name: str, frame: pd.DataFrame, seed: int) -> Any:
    """Fit cell ``name`` on one sample."""
    cell = CELLS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator(cell, seed).fit(frame, **COLUMNS[cell.law])


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
    """Patch the Markov placebo so a share of rows keeps the recorded trajectory.

    The kept rows are a uniform draw of ``round(share n)`` rows. Each keeps its observed
    treatment at every node, ``nan`` included, so the at-risk mask of the placebo matrix is
    unchanged. The other rows keep their kernel draws.
    """
    original = MarkovKernel.placebo

    def mutated(self: MarkovKernel, data: Any, rng: np.random.Generator) -> np.ndarray:
        drawn = np.array(original(self, data, rng), dtype=float)
        observed = np.asarray(data.treatment, dtype=float)
        n = drawn.shape[0]
        kept = rng.choice(n, size=round(share * n), replace=False)
        drawn[kept] = observed[kept]
        return drawn

    MarkovKernel.placebo = mutated  # type: ignore[method-assign]
    try:
        yield
    finally:
        MarkovKernel.placebo = original  # type: ignore[method-assign]


def trial_seed(stream: str, name: str, trial: int) -> int:
    """The seed of one named stream of one property trial.

    The streams are ``"property_sample"`` (the draw), ``"property_fit"`` (the fit's folds,
    learners and simulation draws), ``"refute_size"``, ``"refute_stale"`` and
    ``"refute_control"`` (the refits).
    """
    return stream_seed(STUDY, stream, name, trial)


def trial_configuration(result: Any, estimand: str) -> dict[str, Any]:
    """The fit configuration of one trial, read off the fitted result, as one flat row."""
    configuration = replace(fit_configuration(result, estimand), replicates=REFITS)
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
    estimand = cell.estimand
    result = fit_cell(name, frame, trial_seed("property_fit", name, trial))
    configuration = trial_configuration(result, estimand)
    control = "negative_control" in cell.readings
    tests = ["placebo", "subset"]
    if control:
        tests.append("negative_control_outcome")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = refute(
            result,
            estimand=estimand,
            tests=tuple(tests),
            n_replicates=REFITS,
            negative_control_outcome=frame["NC"].to_numpy(dtype=float) if control else None,
            random_state=trial_seed("refute_size", name, trial),
        )
    out = {"placebo": _reading(report["placebo"]), "subset": _reading(report["subset"])}
    if control:
        out["negative_control"] = _reading(report["negative_control_outcome"])
    if not power:
        return Trial(out, configuration)
    with warnings.catch_warnings(), stale_rows(STALE_SHARE[cell.law]):
        warnings.simplefilter("ignore")
        stale = refute(
            result,
            estimand=estimand,
            tests=("placebo",),
            n_replicates=REFITS,
            random_state=trial_seed("refute_stale", name, trial),
        )
    out["placebo_stale_rows"] = _reading(stale["placebo"])
    rho = out["subset"].statistic
    out["subset_half_se"] = rescaled_subset(rho, 0.5)
    out["subset_double_se"] = rescaled_subset(rho, 2.0)
    if control:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            confounded = refute(
                result,
                estimand=estimand,
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
    """The S1 (law E) or L1 (law S) contrast as one primary row."""
    name = PRIMARY_CELL[scenario]
    result = fit_cell(name, frame, seed)
    estimate = result[CELLS[name].estimand]
    low, high = (float(value) for value in estimate.ci)
    reference = float(truth_map["ate"])
    return [
        {
            "implementation": STUDY.implementation,
            "scenario": scenario,
            "replicate": replicate,
            "n": int(result.data.n),
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
