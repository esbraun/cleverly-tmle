r"""Registered evidence for collaborative TMLE of a risk curve (X91 part B).

The subject is :class:`~cleverly.longitudinal.SurvivalCTMLE`: for each static arm of a
binary baseline treatment and each horizon, the greedy collaborative selection of a
main-term logistic treatment model and a pooled main-term logistic censoring model of
Stitelman and van der Laan (2010), with the printed penalty (24).  This module holds the
declaration and the runner of the study ``survival-ctmle`` (``X91-B-design.md`` section 8).
It declares no property cell, so it registers under ``reporting`` and the validation grid does
not list it; its page is ``docs/technical-reference/method-evidence/survival-ctmle.md``.

**Law** (the source's Sections 12 to 14).  :math:`W_1, \dots, W_5` are multivariate normal,
mean 0, variance 1, covariance 0.2, each set to :math:`\pm 2` beyond :math:`\pm 2`;
:math:`P(A = 1 \mid W) = \operatorname{expit}(0.4 + 0.4 W_1 + 0.4 W_2 - 0.5 W_3 +
\log(\mathrm{OR}) W_4)` with :math:`\mathrm{OR} \in \{1, 10, 15\}`; the event hazard is
:math:`\operatorname{expit}(0.25 - 0.6 (W_1 + W_2 + W_3) - A)` and the censoring hazard
:math:`\operatorname{expit}(-3 - 0.1 (W_1 + W_2 + W_3 + A))`, after the event, both constant in
:math:`t`.  Follow-up ends at :math:`t = 2`, through ``TimeToEvent`` on the integer grid
:math:`\{1, 2\}`.  Scenario 2 sets the :math:`W_1` coefficient of the treatment model to
:math:`\log 15` (source p. 29).

**Truth.** :math:`\psi_a(2) = E_W\{1 - \operatorname{expit}(0.25 - a - 0.6 \Sigma)\}^2`,
:math:`\Sigma = W_1 + W_2 + W_3`, over the clipped law (:func:`truth`); the reported risk is
:math:`F_a(2) = 1 - \psi_a(2)`.  The clip puts a point mass at :math:`\pm 2` on each
coordinate, so :func:`clipped_sum_law` splits the three coordinates into below, inside and
above :math:`[-2, 2]`.  On each of the 27 patterns the inside coordinates are integrated by
Gauss-Legendre quadrature against their normal density, the outside ones contribute their
conditional normal orthant probability and :math:`\pm 2` to :math:`\Sigma`.  The integrand is
smooth on every piece, so the result is accurate to about :math:`10^{-8}` (the total mass is one
to :math:`10^{-9}`, and 20, 30 and 60 nodes agree to :math:`10^{-9}`).  The clipped law gives
:math:`\psi_1(2) = 0.461490`, which rounds to 0.461; :math:`10^9` Monte Carlo draws give
0.461502 with standard error :math:`8 \times 10^{-6}`.  The source's Table 1 prints 0.462, which
the unclipped law (0.46180) rounds to.  :func:`check_truth` asserts the value lies within
:math:`10^{-3}` of 0.462.

**Arms.** ``SurvivalCTMLE``; hazard ``LTMLE`` with main-term logistic treatment and censoring
models on every :math:`W` (the source's "TMLE given the true :math:`g` model"); and hazard
``LTMLE`` with both models without :math:`W_4`.  The outcome model is correct (``GLM`` on
:math:`W`, :math:`A` and the grid time, ``hazard_pooling="pooled"``) or misspecified
(:math:`A` and :math:`W_5` only).

**Targets.** Arm 1's selection for :math:`F_1(2)`, arm 0's for :math:`F_0(2)`, and the risk
difference at 2 from both.

**Cells.** Primary: three odds ratios by two outcome models, scenario 1, :math:`n = 1{,}000`,
in sample.  Secondary: scenario 2 at OR 10 and 15; :math:`n = 2{,}000` at OR 15; one
cross-fitted cell (``n_folds=2``) at OR 10 with the outcome model correct.  Reported for the
source comparison: :math:`n = 500` at each OR; OR 15 with tertile-binarized covariates passed
as columns (``or15_binarized``, without the source's FDR step; the outcome model on the
indicators is not the correct one); OR 15 with ``g_bounds`` lower bound :math:`10^{-8}`.

**Gates** (declared before any declared draw).  :data:`GATED_SCENARIOS` names every gated
cell, and the record's rows hold those cells and the subject alone, because the shared driver
judges every row against its truth.  (a) The bias of :math:`F_1(2)` and of the risk difference
within the standardized-bias margin at the binding confidence bound, in every gated cell.  A
review probe had read a risk-difference bias of 0.018 against an SD of 0.035 at OR 15, so the
declaration left that gate to the pilot.  The pilot read standardized biases of -0.016
(``or15_qcorrect``), 0.030 (``s2_or15_qcorrect``) and -0.012 (``or15_qcorrect_n2000``) over
1,000 draws each, inside the 0.25 margin, so the risk difference is gated at OR 15 as well.
(b) The one-sided upper bound of the RMSE ratio of ``SurvivalCTMLE`` over the full-:math:`g`
hazard ``LTMLE``, for :math:`F_1(2)` in ``or15_qcorrect``, at most 1
(:func:`rmse_comparison`, :func:`scientific_failures`).  Both hazard ``LTMLE`` arms are fitted
on every gated replicate and carried on the subject's rows; they are this library, not a
comparator of the grid row.  (c) The subject's fit-failure share under :data:`FAILURE_BOUND`.

The study declares no coverage gate and no SE-ratio gate (F18): its :class:`Margins` set the
coverage floor to zero and the SE-ratio band wide open, so plug-in standard errors against the
empirical SD and plug-in Wald coverage are reported only.  Reported and not gated: the risk of
arm 0 (``arm0-risk.csv``), every other RMSE ratio (``rmse-comparison.csv``), the share selecting
:math:`W_4` into :math:`g_1` and :math:`A` into :math:`g_2`, the restart counts, clips, dropped
terms and run time (``selection-readings.csv``).  The study's censoring is about 4.7% per
interval, so :math:`g_2` moves are rare here; ``tests/unit/test_survival_ctmle.py`` check 3 is
the :math:`g_2` witness.  The reported cells (:data:`REPORTED_SCENARIOS`: the misspecified
outcome model and the source comparison) are fitted by the property phase,
``tests/studies/survival_ctmle_properties.py``, with all three implementations, and publish as
``reported-cells.csv`` and ``reported-replicates.csv``.

**Size and seeds.** :data:`PILOT_REPLICATES` sizes the pilot,
``tests/studies/survival_ctmle_pilot.py``, which draws on its own seed streams.  The shared rule
(the smallest multiple of 400 that reaches the pass share) on that pilot gives
:data:`PRIMARY_REPLICATES`, 1,600.  The reported cells take :data:`REPORTED_REPLICATES`, 400
each: they carry no verdict, and the share of the run they would cost at 1,600 buys no claim.
:data:`SEED` and :data:`RESAMPLING_SEED` are used by no other study; replicate ``r`` of a gated
cell draws from ``replicate_seed(STUDY, scenario, r)`` and of a reported cell from
``stream_seed(STUDY, "reported", scenario, r)``.
"""

from __future__ import annotations

import inspect
import itertools
import time
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.legendre import leggauss
from scipy.special import expit
from scipy.stats import multivariate_normal, norm
from sklearn.linear_model import LogisticRegression

from cleverly.learners import GLM
from cleverly.longitudinal import LTMLE, LongitudinalData, SurvivalCTMLE
from cleverly.validation._failures import recordable
from tests.parallel import STUDY_JOBS
from tests.studies.evidence import failures as evidence_failures
from tests.studies.evidence import replication
from tests.studies.evidence.inference import bootstrap, upper_bound
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed
from tests.studies.stratified_incremental_msm_properties import Columns

#: The pilot's replicate count per scenario; the primary count follows from it.
PILOT_REPLICATES = 1_000
#: The declared primary replicate count, set from the committed pilot
#: (``tests/studies/survival_ctmle_pilot.py``) by the shared rule before any declared draw.
PRIMARY_REPLICATES = 1_600
PRIMARY_N = 1_000
#: The declared share of replicates one cell may lose to recordable fitting failures.
FAILURE_BOUND = 0.01
SEED = 20271917
RESAMPLING_SEED = 2027191701
#: The source's outcome horizon and grid.
HORIZON = 2
GRID = (1.0, 2.0)
COVARIATES = ("W1", "W2", "W3", "W4", "W5")
ARMS = {"a1": 1, "a0": 0}
REFERENCE = "a0"
ESTIMAND_NAMES = (
    "risk_regimen[a1 @ t=2]",
    "risk_regimen[a0 @ t=2]",
    "ate_regimen[a1 vs a0 @ t=2]",
)

#: The reported selection readings, one row per scenario and arm (:func:`extra_artifacts`).
SELECTION_FILE = "selection-readings.csv"

#: The three implementations each replicate fits.
CTMLE_ARM = "cleverly-survival-ctmle"
FULL_G_ARM = "cleverly-hazard-ltmle-full-g"
NO_W4_ARM = "cleverly-hazard-ltmle-no-w4"
IMPLEMENTATIONS = (CTMLE_ARM, FULL_G_ARM, NO_W4_ARM)

#: ``scenario -> (odds ratio, source scenario, outcome model, n, settings)``.  The settings
#: are ``n_folds``, ``binarized`` and ``g_lower``.
SCENARIOS: dict[str, tuple[float, int, str, int, dict[str, Any]]] = {
    **{
        f"or{odds:g}_q{model}": (odds, 1, model, PRIMARY_N, {})
        for odds in (1.0, 10.0, 15.0)
        for model in ("correct", "wrong")
    },
    **{f"s2_or{odds:g}_qcorrect": (odds, 2, "correct", PRIMARY_N, {}) for odds in (10.0, 15.0)},
    **{f"s2_or{odds:g}_qwrong": (odds, 2, "wrong", PRIMARY_N, {}) for odds in (10.0, 15.0)},
    "or15_qcorrect_n2000": (15.0, 1, "correct", 2_000, {}),
    "or10_qcorrect_crossfit": (10.0, 1, "correct", PRIMARY_N, {"n_folds": 2}),
    **{f"or{odds:g}_qcorrect_n500": (odds, 1, "correct", 500, {}) for odds in (1.0, 10.0, 15.0)},
    "or15_binarized": (15.0, 1, "correct", PRIMARY_N, {"binarized": True}),
    "or15_qcorrect_glower": (15.0, 1, "correct", PRIMARY_N, {"g_lower": 1e-8}),
}
PRIMARY_SCENARIOS = tuple(
    name
    for name, (_, source, _, n, settings) in SCENARIOS.items()
    if source == 1 and n == PRIMARY_N and not settings
)
#: The cell gate (b) reads.
RMSE_GATE_SCENARIO = "or15_qcorrect"
#: Every gated cell: the outcome model correct, at a declared size, on the source's
#: covariates.  The ``n = 500``, binarized and ``g_lower`` cells are the source comparison and
#: are reported only, and so is every cell with the outcome model misspecified.
GATED_SCENARIOS = (
    "or1_qcorrect",
    "or10_qcorrect",
    "or15_qcorrect",
    "s2_or10_qcorrect",
    "s2_or15_qcorrect",
    "or15_qcorrect_n2000",
    "or10_qcorrect_crossfit",
)
REPORTED_SCENARIOS = tuple(name for name in SCENARIOS if name not in GATED_SCENARIOS)
#: The estimands gate (a) reads in each gated cell: the risk of arm 1 and the risk difference.
#: The pilot settled the risk difference at OR 15 (the module docstring gives its readings).
GATED_ESTIMANDS = dict.fromkeys(GATED_SCENARIOS, (ESTIMAND_NAMES[0], ESTIMAND_NAMES[2]))
#: No coverage or SE-ratio gate (F18): the plug-in interval is a diagnostic.  A floor of zero
#: cannot fail, and the open SE-ratio band cannot bind; the bias and RMSE margins stand.
MARGINS = Margins(coverage_floor=0.0, se_ratio_sanity=(0.0, 1e6))

#: The two hazard ``LTMLE`` arms every replicate also fits.  They are reported beside the
#: subject on its own rows, never as a comparator of the grid row: they are this library.
COMPARATORS = (FULL_G_ARM, NO_W4_ARM)
#: Gate (b): in :data:`RMSE_GATE_SCENARIO`, the one-sided upper bound of the subject's RMSE
#: over the full-:math:`g` arm's, for the risk of arm 1, at most this ceiling.
RMSE_GATE_ESTIMAND = ESTIMAND_NAMES[0]
RMSE_GATE_CEILING = 1.0
#: The name gate (b) fails under, in the run's gates and its failure accounting.
RMSE_LEG = "RMSE against the full-g arm"
#: Each gated cell's RMSE of the subject against both hazard ``LTMLE`` arms (gate (b)).
COMPARISON_FILE = "rmse-comparison.csv"
#: The subject's risk of arm 0 in each gated cell, reported and not gated.
ARM0_FILE = "arm0-risk.csv"
#: The reported cells' replicate count.  They are fitted by the property phase
#: (``tests/studies/survival_ctmle_properties.py``) on their own seed stream.
REPORTED_REPLICATES = 400
#: The property phase's two files: the reported cells' summary and their replicate rows.
REPORTED_CELLS_FILE = "reported-cells.csv"
REPORTED_REPLICATES_FILE = "reported-replicates.csv"


#: The Genz integrator behind ``multivariate_normal.cdf`` is randomized.  A fixed generator
#: makes the truth the same bits in every process that imports this module.  SciPy 1.15
#: renamed the keyword from ``random_state`` to ``rng``.
_FIXED: dict[str, Any] = (
    {"rng": 0}
    if "rng" in inspect.signature(multivariate_normal.cdf).parameters
    else {"random_state": 0}
)


def clipped_sum_law(nodes: int = 30, eps: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    r"""Points and masses of :math:`\Sigma = \sum_{j \le 3} \operatorname{clip}(Z_j, -2, 2)`.

    :math:`Z` is trivariate normal, unit variances, covariance 0.2.  Each coordinate is
    below, inside or above :math:`[-2, 2]`.  On each of the 27 patterns the inside coordinates
    take ``nodes`` Gauss-Legendre points per axis weighted by their normal density, and the
    outside ones contribute :math:`\pm 2` each and their conditional orthant probability given
    the inside ones (``norm.cdf`` for one, ``multivariate_normal.cdf`` at ``eps`` for more).
    """
    covariance = np.full((3, 3), 0.2)
    np.fill_diagonal(covariance, 1.0)
    unit, unit_weights = leggauss(nodes)
    axis, axis_weights = 2.0 * unit, 2.0 * unit_weights
    points, masses = [], []
    for pattern in itertools.product((-1, 0, 1), repeat=3):
        inside = [j for j, side in enumerate(pattern) if side == 0]
        outside = [j for j, side in enumerate(pattern) if side != 0]
        sign = np.array([pattern[j] for j in outside], dtype=float)
        k = len(inside)
        if k:
            grid = np.stack(np.meshgrid(*([axis] * k), indexing="ij"), -1).reshape(-1, k)
            weight = np.stack(np.meshgrid(*([axis_weights] * k), indexing="ij"), -1)
            inner = covariance[np.ix_(inside, inside)]
            mass = np.prod(weight.reshape(-1, k), axis=1) * np.atleast_1d(
                multivariate_normal(np.zeros(k), inner).pdf(grid)
            )
        else:
            grid, mass = np.zeros((1, 0)), np.ones(1)
        if outside:
            if k:
                coef = covariance[np.ix_(outside, inside)] @ np.linalg.inv(inner)
                mean = grid @ coef.T
                spread = (
                    covariance[np.ix_(outside, outside)]
                    - coef @ covariance[np.ix_(inside, outside)]
                )
            else:
                mean = np.zeros((1, len(outside)))
                spread = covariance[np.ix_(outside, outside)]
            # P(side * Z > 2) = P(-side * Z < -2), one orthant below -2 on every coordinate.
            mean, spread = -sign * mean, spread * np.outer(sign, sign)
            if len(outside) == 1:
                orthant = norm.cdf(-2.0, loc=mean[:, 0], scale=np.sqrt(spread[0, 0]))
            else:
                corner = np.full(len(outside), -2.0)
                orthant = np.array(
                    [
                        multivariate_normal.cdf(
                            corner, mean=row, cov=spread, abseps=eps, releps=eps, **_FIXED
                        )
                        for row in mean
                    ]
                )
            mass = mass * orthant
        points.append(grid.sum(axis=1) + 2.0 * sign.sum())
        masses.append(mass)
    return np.concatenate(points), np.concatenate(masses)


def truth(nodes: int = 30, eps: float = 1e-10) -> dict[str, float]:
    """The risk of each arm and their difference at ``t = 2``, over the clipped law."""
    total, mass = clipped_sum_law(nodes, eps)
    survival = {
        arm: float(mass @ (1.0 - expit(0.25 - arm - 0.6 * total)) ** HORIZON) for arm in (0, 1)
    }
    return {
        "risk_regimen[a1 @ t=2]": 1.0 - survival[1],
        "risk_regimen[a0 @ t=2]": 1.0 - survival[0],
        "ate_regimen[a1 vs a0 @ t=2]": survival[0] - survival[1],
    }


TRUTH = truth()


def check_truth() -> None:
    """The source's Table 1 survival of arm 1 at ``t = 2``, 0.462, within ``1e-3``."""
    survival = 1.0 - TRUTH["risk_regimen[a1 @ t=2]"]
    assert abs(survival - 0.462) < 1e-3, survival


STUDY = StudyRecord(
    name="collaborative TMLE of a risk curve",
    slug="survival-ctmle",
    artifacts=ROOT / "tests" / "canonical" / "survival_ctmle",
    document="docs/technical-reference/method-evidence/survival-ctmle.md",
    anchor="collaborative-tmle-of-a-risk-curve",
    scenarios={name: GATED_ESTIMANDS[name] for name in GATED_SCENARIOS},
    scenario_n={
        name: SCENARIOS[name][3] for name in GATED_SCENARIOS if SCENARIOS[name][3] != PRIMARY_N
    },
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=MARGINS,
    implementation=CTMLE_ARM,
    reference=None,
    failure_bound={CTMLE_ARM: FAILURE_BOUND},
    extra_artifacts=(SELECTION_FILE, COMPARISON_FILE, ARM0_FILE),
    property_artifacts=(REPORTED_CELLS_FILE, REPORTED_REPLICATES_FILE),
    modules=(
        "tests/studies/survival_ctmle.py",
        "tests/studies/survival_ctmle_properties.py",
        "tests/studies/stratified_incremental_msm_properties.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.survival_ctmle",
    properties_module="tests.studies.survival_ctmle_properties",
    # No property cell, so no validation row: the study reports its verdicts off the grid
    # (``method-benchmarking.md``, "Adding a method row").  Its claim is points only (F18).
    publication_policy="reporting",
)

CONFIGURATION = {
    "construction": "Stitelman and van der Laan (2010), Sections 12 to 14",
    "horizon": HORIZON,
    "grid": list(GRID),
    "engines": {
        CTMLE_ARM: "SurvivalCTMLE, selection_folds=5, penalty=True",
        FULL_G_ARM: (
            "LTMLE(survival_model='hazard'), GLM treatment model and logistic censoring model "
            "(C=1e6, with the grid time) on every W"
        ),
        NO_W4_ARM: "the same without W4 in either model",
    },
    "outcome_models": {
        "correct": "GLM on W, A and the grid time, hazard_pooling='pooled'",
        "wrong": "GLM on A and W5, hazard_pooling='pooled'",
    },
    "truth": (
        "the clipped law with its point masses at -2 and 2 exact: Gauss-Legendre, 30 nodes "
        "per inside axis, and conditional normal orthant probabilities at 1e-10"
    ),
    "gated_scenarios": list(GATED_SCENARIOS),
    "reported_scenarios": list(REPORTED_SCENARIOS),
    "reported_replicates": REPORTED_REPLICATES,
    "gates": "bias of F1(2) and the RD; RMSE ratio against the full-g arm at OR 15; no coverage "
    "or SE-ratio gate (F18)",
    "variance_method": "plug-in influence curve (SurvivalCTMLE: working-mechanism diagnostic)",
}


# --------------------------------------------------------------------------- the law


def sample(n: int, seed: int, *, odds: float, source: int, binarized: bool = False) -> Any:
    """One draw of the source's law as a time-to-event frame."""
    rng = np.random.default_rng(seed)
    covariance = np.full((5, 5), 0.2)
    np.fill_diagonal(covariance, 1.0)
    w = np.clip(rng.multivariate_normal(np.zeros(5), covariance, size=n), -2.0, 2.0)
    w1 = np.log(15.0) if source == 2 else 0.4
    logit = 0.4 + w1 * w[:, 0] + 0.4 * w[:, 1] - 0.5 * w[:, 2] + np.log(odds) * w[:, 3]
    a = rng.binomial(1, expit(logit))
    total = w[:, :3].sum(axis=1)
    time_ = np.full(n, GRID[-1])
    event = np.zeros(n, dtype=int)
    alive = np.ones(n, dtype=bool)
    for k, at in enumerate(GRID, start=1):
        hit = alive & (rng.uniform(size=n) < expit(0.25 - 0.6 * total - a))
        time_[hit], event[hit] = at, 1
        alive &= ~hit
        if k < len(GRID):
            gone = alive & (rng.uniform(size=n) < expit(-3.0 - 0.1 * (total + a)))
            time_[gone] = at
            alive &= ~gone
    frame = pd.DataFrame(w, columns=list(COVARIATES))
    if binarized:
        columns = {}
        for name in COVARIATES:
            low, high = np.quantile(frame[name], [1.0 / 3.0, 2.0 / 3.0])
            columns[f"{name}_t2"] = (frame[name] > low).astype(float)
            columns[f"{name}_t3"] = (frame[name] > high).astype(float)
        frame = pd.DataFrame(columns)
    frame["A"], frame["time"], frame["event"] = a, time_, event
    return frame


def container(frame: pd.DataFrame) -> LongitudinalData:
    baseline = [name for name in frame.columns if name not in ("A", "time", "event")]
    return LongitudinalData.from_time_to_event(
        frame, time="time", event="event", treatment="A", baseline=baseline, grid=GRID
    )


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    odds, source, _, _, settings = SCENARIOS[scenario]
    frame = sample(n, seed, odds=odds, source=source, binarized=bool(settings.get("binarized")))
    return frame, dict(TRUTH)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


# ------------------------------------------------------------------------- the fits


def _outcome(model: str, width: int) -> Any:
    """The hazard model on the pooled design ``[W..., A, time]``."""
    if model == "correct":
        return GLM(family="binomial")
    return Columns(GLM(family="binomial"), keep=(4, width - 2))


def fit(frame: pd.DataFrame, scenario: str, implementation: str) -> Any:
    """One implementation's fit of one replicate."""
    _, _, model, _, settings = SCENARIOS[scenario]
    data = container(frame)
    p = len(data.baseline_names)
    common: dict[str, Any] = {
        "reference": REFERENCE,
        "horizons": [HORIZON],
        "outcome_learner": _outcome(model, p + 2),
        "hazard_pooling": "pooled",
        "n_folds": int(settings.get("n_folds", 1)),
        "g_bounds": (float(settings.get("g_lower", 0.01)), 1.0),
        "random_state": 0,
    }
    if implementation == CTMLE_ARM:
        return SurvivalCTMLE(ARMS, selection_folds=5, **common).fit(data)
    keep = tuple(range(p))
    if implementation == NO_W4_ARM:
        dropped = {i for i, name in enumerate(data.baseline_names) if name.startswith("W4")}
        keep = tuple(i for i in range(p) if i not in dropped)
    return LTMLE(
        ARMS,
        survival_model="hazard",
        simultaneous=False,
        treatment_learner=Columns(GLM(family="binomial"), keep=keep),
        # The pooled censoring design is [W..., A, time], and node 1 holds no censoring, so
        # the grid time separates it: a large-C logistic regression keeps that node's
        # retention near one where GLM would raise.
        censoring_learner=Columns(LogisticRegression(C=1e6, max_iter=2000), keep=(*keep, p, p + 1)),
        **common,
    ).fit(data)


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int, implementation: str
) -> list[dict[str, Any]]:
    """One row per estimand on the identity scale, read through the plug-in accessors."""
    rows = []
    for name in ESTIMAND_NAMES:
        estimate = result[name]
        std_error, low, high = reported_inference(estimate)
        reference = float(truth[name])
        rows.append(
            {
                "implementation": implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": result.n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": std_error,
                "ci_lower": low,
                "ci_upper": high,
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": float(estimate.psi),
            }
        )
    return rows


#: The reading of each arm's selection that a ``SurvivalCTMLE`` row carries, per arm label.
SELECTION_READINGS = ("w4_in_g1", "a_in_g2", "n_steps", "clipped", "dropped_terms", "selected")
#: The columns the subject's rows carry beside the published schema.
EXTRA_COLUMNS = (
    "seconds",
    *(f"{reading}_{arm}" for arm in ARMS for reading in SELECTION_READINGS),
)
#: The run time of each hazard ``LTMLE`` arm on the subject's replicate.
SECONDS_COLUMNS = tuple(f"seconds[{arm}]" for arm in COMPARATORS)
#: The columns a primary row carries beside the published schema: the selection readings,
#: the run times, both hazard ``LTMLE`` arms' estimates of the row's estimand, and the
#: subject's risk of arm 0, which no row of the record states.
PRIMARY_COLUMNS = (
    *EXTRA_COLUMNS,
    *SECONDS_COLUMNS,
    *(f"estimate[{arm}]" for arm in COMPARATORS),
    *(f"failure[{arm}]" for arm in COMPARATORS),
    "a0_truth",
    "a0_estimate",
    "a0_std_error",
    "a0_covered",
)


def selection_readings(result: Any) -> dict[str, float]:
    """The reported, ungated selection readings of one ``SurvivalCTMLE`` fit at ``t = 2``."""
    readings: dict[str, float] = {}
    for arm in ARMS:
        record = result.selection[(arm, HORIZON)]
        g1, g2 = record.selected_terms
        values = {
            "w4_in_g1": float(any(name.startswith("W4") for name in g1)),
            "a_in_g2": float("A" in g2),
            "n_steps": float(record.n_steps[record.selected]),
            "clipped": float(record.clipped[record.selected]),
            "dropped_terms": float(record.dropped_terms[record.selected]),
            "selected": float(record.selected),
        }
        readings.update({f"{name}_{arm}": value for name, value in values.items()})
    return readings


def extra_artifacts(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The three reported tables of the primary rows, read from the subject's own rows.

    ``selection-readings.csv``: per scenario and arm, the share of fits that select
    :math:`W_4` into :math:`g_1` and :math:`A` into :math:`g_2`, the mean restart count,
    selected position, clips and dropped terms of the selected candidate, and the mean seconds
    per fit of each implementation.  ``rmse-comparison.csv`` (:func:`rmse_comparison`) holds
    gate (b).  ``arm0-risk.csv`` (:func:`arm0_risk`) reports the risk of arm 0.
    """
    return {
        SELECTION_FILE: selection_table(rows),
        COMPARISON_FILE: rmse_comparison(rows),
        ARM0_FILE: arm0_risk(rows),
    }


def _one_per_replicate(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per scenario and replicate: the columns every estimand's row repeats."""
    subject = rows.loc[rows["implementation"] == CTMLE_ARM]
    return subject.drop_duplicates(["scenario", "replicate"])


def selection_table(rows: pd.DataFrame) -> pd.DataFrame:
    """``selection-readings.csv``; none of it is a verdict."""
    fits = _one_per_replicate(rows)
    records = []
    for scenario, group in fits.groupby("scenario", sort=True):
        for arm in ARMS:
            records.append(
                {
                    "scenario": scenario,
                    "arm": arm,
                    "fits": len(group),
                    "share_w4_in_g1": float(group[f"w4_in_g1_{arm}"].mean()),
                    "share_a_in_g2": float(group[f"a_in_g2_{arm}"].mean()),
                    "mean_n_steps": float(group[f"n_steps_{arm}"].mean()),
                    "mean_selected": float(group[f"selected_{arm}"].mean()),
                    "total_clipped": int(group[f"clipped_{arm}"].sum()),
                    "total_dropped_terms": int(group[f"dropped_terms_{arm}"].sum()),
                }
            )
    readings = pd.DataFrame(records)
    seconds = fits.groupby("scenario")[["seconds", *SECONDS_COLUMNS]].mean()
    seconds.columns = [f"mean_seconds[{name}]" for name in IMPLEMENTATIONS]
    return readings.merge(seconds.reset_index(), on="scenario", how="left")


def rmse_comparison(rows: pd.DataFrame) -> pd.DataFrame:
    """Each gated cell's RMSE of the subject against both hazard ``LTMLE`` arms.

    Per cell, estimand and comparator: both RMSEs on the replicates where the comparator
    returned, the ratio, and its one-sided upper bound at the declared confidence level from a
    paired percentile bootstrap of :data:`MARGINS`' size, on a stream named for the cell.  Only
    the row of :data:`RMSE_GATE_SCENARIO`, :data:`RMSE_GATE_ESTIMAND` and the full-:math:`g`
    arm is gated (``gated``), against :data:`RMSE_GATE_CEILING`; every other row is reported.
    """
    records = []
    subject = rows.loc[rows["implementation"] == CTMLE_ARM]
    for (scenario, estimand), group in subject.groupby(["scenario", "estimand"], sort=True):
        truth_ = float(group["truth"].iloc[0])
        for arm in COMPARATORS:
            kept = group.loc[np.isfinite(group[f"estimate[{arm}]"].astype(float))]
            gated = (scenario, estimand, arm) == (
                RMSE_GATE_SCENARIO,
                RMSE_GATE_ESTIMAND,
                FULL_G_ARM,
            )
            base = {
                "scenario": scenario,
                "estimand": estimand,
                "comparator": arm,
                "replicates": len(kept),
                "comparator_failures": len(group) - len(kept),
            }
            if len(kept) < 2:
                # Too few paired replicates to bound a ratio: no reading, and a gated row fails.
                records.append(
                    {
                        **base,
                        **dict.fromkeys(
                            ("rmse_subject", "rmse_comparator", "rmse_ratio", "rmse_ratio_upper"),
                            np.nan,
                        ),
                        "ceiling": RMSE_GATE_CEILING if gated else np.nan,
                        "gated": gated,
                        "passed": False if gated else np.nan,
                    }
                )
                continue
            arrays = {
                "subject": kept["estimate"].to_numpy(dtype=float) - truth_,
                "comparator": kept[f"estimate[{arm}]"].to_numpy(dtype=float) - truth_,
            }

            def ratio(draw: dict[str, np.ndarray]) -> np.ndarray:
                return np.sqrt(
                    np.mean(draw["subject"] ** 2, axis=1) / np.mean(draw["comparator"] ** 2, axis=1)
                )

            samples = bootstrap(
                arrays,
                {"ratio": ratio},
                replicates=MARGINS.bootstrap_replicates,
                seed=stream_seed(STUDY, "rmse", str(scenario), str(estimand), arm),
            )
            point = float(ratio({name: values[None, :] for name, values in arrays.items()})[0])
            upper = upper_bound(samples["ratio"], confidence_level=MARGINS.confidence_level)
            records.append(
                {
                    **base,
                    "rmse_subject": float(np.sqrt(np.mean(arrays["subject"] ** 2))),
                    "rmse_comparator": float(np.sqrt(np.mean(arrays["comparator"] ** 2))),
                    "rmse_ratio": point,
                    "rmse_ratio_upper": upper,
                    "ceiling": RMSE_GATE_CEILING if gated else np.nan,
                    "gated": gated,
                    "passed": bool(upper <= RMSE_GATE_CEILING) if gated else np.nan,
                }
            )
    return pd.DataFrame(records)


def arm0_risk(rows: pd.DataFrame) -> pd.DataFrame:
    """The subject's risk of arm 0 per gated cell: bias, spread, RMSE, mean SE, coverage."""
    records = []
    for scenario, group in _one_per_replicate(rows).groupby("scenario", sort=True):
        error = group["a0_estimate"] - group["a0_truth"]
        records.append(
            {
                "scenario": scenario,
                "estimand": ESTIMAND_NAMES[1],
                "replicates": len(group),
                "truth": float(group["a0_truth"].iloc[0]),
                "bias": float(error.mean()),
                "empirical_sd": float(error.std(ddof=1)),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "mean_std_error": float(group["a0_std_error"].mean()),
                "coverage": float(group["a0_covered"].mean()),
            }
        )
    return pd.DataFrame(records)


def scientific_failures(extra_frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Gate (b): the gated row of ``rmse-comparison.csv`` when its bound passes the ceiling."""
    table = extra_frames[COMPARISON_FILE]
    gated = table.loc[table["gated"].astype(bool)]
    if len(gated) != 1:
        raise RuntimeError(f"{STUDY.slug}: expected one gated RMSE row, found {len(gated)}")
    return {RMSE_LEG: gated.loc[gated["passed"].ne(True)]}


def account_failures(
    failures: pd.DataFrame | None, artifacts: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Gate (b) with each subject fit lost in :data:`RMSE_GATE_SCENARIO` counted as failing it.

    Decision Q1: the RMSE ratio reads the surviving replicates only, so a lost fit of the gated
    cell could have moved it, and each one turns the audit red.  A fit lost in another cell
    leaves the gated ratio whole; gate (c) bounds those through the failure bound.
    """
    lost = evidence_failures.expand(failures, STUDY)
    lost = lost.loc[
        (lost["implementation"] == STUDY.implementation) & (lost["scenario"] == RMSE_GATE_SCENARIO)
    ]
    return evidence_failures.audits(
        scientific_failures(artifacts),
        missing={RMSE_LEG: len(lost[["scenario", "replicate"]].drop_duplicates())},
    )


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _timed(frame: pd.DataFrame, scenario: str, implementation: str) -> tuple[Any, float]:
    start = time.perf_counter()
    result = fit(frame, scenario, implementation)
    return result, time.perf_counter() - start


def _comparator(frame: pd.DataFrame, scenario: str, arm: str) -> tuple[Any | None, float, str]:
    """A hazard ``LTMLE`` arm's fit, its seconds, and the error class when it raised.

    The arms are reported beside the subject and gate only through :func:`rmse_comparison`,
    which counts the replicates they lost.  Only a recordable fitting failure is a reading;
    any other raise is a defect and stops the run.
    """
    start = time.perf_counter()
    try:
        result = fit(frame, scenario, arm)
    except Exception as error:
        if not recordable(error):
            raise
        return None, time.perf_counter() - start, type(error).__name__
    return result, time.perf_counter() - start, ""


def _comparator_estimate(result: Any | None, estimand: str) -> float:
    """The arm's estimate, ``NaN`` when it raised or returned a non-finite value."""
    if result is None:
        return np.nan
    psi = float(result[estimand].psi)
    return psi if np.isfinite(psi) else np.nan


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    """The subject's gated rows, each carrying both hazard ``LTMLE`` arms' estimates."""
    scenario, replicate, _ = key
    frame, truth_ = drawn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result, seconds = _timed(frame, scenario, CTMLE_ARM)
        others = {arm: _comparator(frame, scenario, arm) for arm in COMPARATORS}
    arm0 = result[ESTIMAND_NAMES[1]]
    _, low, high = reported_inference(arm0)
    extra: dict[str, Any] = {
        "seconds": seconds,
        **selection_readings(result),
        **{f"seconds[{arm}]": spent for arm, (_, spent, _) in others.items()},
        **{f"failure[{arm}]": failed for arm, (_, _, failed) in others.items()},
        "a0_truth": float(truth_[ESTIMAND_NAMES[1]]),
        "a0_estimate": float(arm0.psi),
        "a0_std_error": reported_inference(arm0)[0],
        "a0_covered": int(low <= truth_[ESTIMAND_NAMES[1]] <= high),
    }
    rows = []
    for row in result_rows(result, truth_, scenario, replicate, CTMLE_ARM):
        if row["estimand"] not in GATED_ESTIMANDS[scenario]:
            continue
        comparators = {
            f"estimate[{arm}]": _comparator_estimate(fitted, row["estimand"])
            for arm, (fitted, _, _) in others.items()
        }
        rows.append({**row, **extra, **comparators})
    return rows


def draw_and_fit(
    *,
    replicates: int,
    n: int,
    n_jobs: int = STUDY_JOBS,
    scenarios: tuple[str, ...] | None = None,
) -> replication.PhaseResult:
    """Every gated cell of ``scenarios`` (all by default), each at its declared size."""
    sizes = STUDY.sizes(n)
    chosen = tuple(STUDY.scenarios) if scenarios is None else scenarios
    outcomes = replication.fit_primary(
        [
            (scenario, replicate, sizes[scenario])
            for scenario in chosen
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
    return replication.result(outcomes, rows.loc[:, [*REPLICATE_COLUMNS, *PRIMARY_COLUMNS]])
