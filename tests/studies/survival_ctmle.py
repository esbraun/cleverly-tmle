r"""Declared evidence for collaborative TMLE of a risk curve (X91 part B), before any run.

The subject is :class:`~cleverly.longitudinal.SurvivalCTMLE`: for each static arm of a
binary baseline treatment and each horizon, the greedy collaborative selection of a
main-term logistic treatment model and a pooled main-term logistic censoring model of
Stitelman and van der Laan (2010), with the printed penalty (24).  This module holds the
declaration and the runner of the study ``survival-ctmle``.  It is **not registered**: the
pilot that sizes it, the registration, the grid row and the support rows come in a later
change, after the design review (``X91-B-design.md`` section 8).

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

**Gates** (declared, not yet measured).  :data:`GATED_SCENARIOS` names every gated cell; every
other cell is reported only.  (a) The bias of :math:`F_1(2)` within the standardized-bias
margin at the binding confidence bound in each gated cell, and the bias of the risk difference
in the gated cells at OR 1 and 10.  Whether the risk difference is gated at OR 15 is left to the
pilot: a review probe read a bias of 0.018 against an SD of 0.035 there, twice the margin, so
the pilot either measures it inside the margin or the declaration reports it only, before any
declared draw.  (b) The RMSE of ``SurvivalCTMLE`` at most the full-:math:`g` hazard ``LTMLE``'s
at the binding bound, outcome model correct, OR 15 only.  (c) The fit-failure share under
:data:`FAILURE_BOUND`.  The study declares no coverage gate and no SE-ratio gate (F18): its
:class:`Margins` set the coverage floor to zero and the SE-ratio band wide open, so plug-in
standard errors against the empirical SD and plug-in Wald coverage are reported only.  The
misspecified cells, the share selecting :math:`W_4` into :math:`g_1` and :math:`A` into
:math:`g_2`, the restart counts, clips, dropped terms and run time are reported and not gated;
:func:`draw_and_fit` carries them on the subject's rows and :func:`extra_artifacts` publishes
them as ``selection-readings.csv``.  The study's censoring is about 4.7% per interval, so :math:`g_2` moves are rare
here; ``tests/unit/test_survival_ctmle.py`` check 3 is the :math:`g_2` witness.

**Size and seeds.** :data:`PILOT_REPLICATES` sizes the pilot.  The primary replicate count is
the shared rule (the smallest multiple of 400 that reaches the pass share) on that pilot,
declared before any declared draw.  :data:`SEED` and :data:`RESAMPLING_SEED` are used by no
other study; replicate ``r`` of a scenario draws from ``replicate_seed(STUDY, scenario, r)``.
"""

from __future__ import annotations

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
from tests.parallel import STUDY_JOBS
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed
from tests.studies.stratified_incremental_msm_properties import Columns

#: The pilot's replicate count per scenario; the primary count follows from it.
PILOT_REPLICATES = 400
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
#: The estimands gate (a) reads in each gated cell.  The risk difference at OR 15 waits on the
#: pilot (the module docstring says why), so those cells gate the risk of arm 1 alone.
GATED_ESTIMANDS = {
    name: (
        (ESTIMAND_NAMES[0],)
        if SCENARIOS[name][0] == 15.0
        else (ESTIMAND_NAMES[0], ESTIMAND_NAMES[2])
    )
    for name in GATED_SCENARIOS
}
#: No coverage or SE-ratio gate (F18): the plug-in interval is a diagnostic.  A floor of zero
#: cannot fail, and the open SE-ratio band cannot bind; the bias and RMSE margins stand.
MARGINS = Margins(coverage_floor=0.0, se_ratio_sanity=(0.0, 1e6))


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
                            corner, mean=row, cov=spread, abseps=eps, releps=eps
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
    scenarios=dict.fromkeys(SCENARIOS, ESTIMAND_NAMES),
    scenario_n={name: n for name, (_, _, _, n, _) in SCENARIOS.items() if n != PRIMARY_N},
    replicates=PILOT_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=MARGINS,
    implementation=CTMLE_ARM,
    reference=None,
    failure_bound={CTMLE_ARM: FAILURE_BOUND},
    extra_artifacts=(SELECTION_FILE,),
    modules=(
        "tests/studies/survival_ctmle.py",
        "tests/studies/stratified_incremental_msm_properties.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.survival_ctmle",
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
    "gates": "bias and RMSE only; no coverage or SE-ratio gate (F18)",
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
    """``selection-readings.csv``: per scenario and arm, the selection shares and run time.

    The share of fits that select :math:`W_4` into :math:`g_1` and :math:`A` into :math:`g_2`,
    the mean restart count, selected position, clips and dropped terms of the selected
    candidate, and the mean seconds per fit of each implementation.  None is a verdict.
    """
    subject = rows.loc[rows["implementation"] == CTMLE_ARM].drop_duplicates(
        ["scenario", "replicate"]
    )
    records = []
    for scenario, group in subject.groupby("scenario", sort=True):
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
    seconds = (
        rows.drop_duplicates(["implementation", "scenario", "replicate"])
        .groupby(["scenario", "implementation"])["seconds"]
        .mean()
        .unstack("implementation")
        .rename(columns=lambda name: f"mean_seconds[{name}]")
        .reset_index()
    )
    return {SELECTION_FILE: readings.merge(seconds, on="scenario", how="left")}


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _fit_rows(
    frame: pd.DataFrame,
    truth_: Mapping[str, float],
    scenario: str,
    replicate: int,
    implementation: str,
) -> list[dict[str, Any]]:
    """One implementation's rows, with its run time and, for the subject, its readings."""
    start = time.perf_counter()
    result = fit(frame, scenario, implementation)
    extra: dict[str, float] = {"seconds": time.perf_counter() - start}
    extra.update(dict.fromkeys(EXTRA_COLUMNS[1:], np.nan))
    if implementation == CTMLE_ARM:
        extra.update(selection_readings(result))
    return [
        {**row, **extra} for row in result_rows(result, truth_, scenario, replicate, implementation)
    ]


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth_ = drawn
    rows: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for implementation in IMPLEMENTATIONS:
            rows.extend(_fit_rows(frame, truth_, scenario, replicate, implementation))
    return rows


def draw_and_fit(
    *,
    replicates: int,
    n: int,
    n_jobs: int = STUDY_JOBS,
    scenarios: tuple[str, ...] | None = None,
) -> replication.PhaseResult:
    """Every row of ``scenarios`` (all by default), each at its declared size."""
    sizes = STUDY.sizes(n)
    chosen = tuple(SCENARIOS) if scenarios is None else scenarios
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
    return replication.result(outcomes, rows.loc[:, [*REPLICATE_COLUMNS, *EXTRA_COLUMNS]])


def smoke(scenarios: tuple[str, ...] = ("or15_qcorrect", "or1_qwrong"), replicates: int = 2) -> Any:
    """A cheap run of a few cells, for a runner to check the module before the pilot.

    Prints each row and the selection readings; it writes nothing and registers nothing.
    """
    check_truth()
    out = []
    for scenario in scenarios:
        n = STUDY.n_of(scenario)
        for replicate in range(replicates):
            out.extend(_replicate((scenario, replicate, n), draw_scenario(scenario, n, replicate)))
    frame = pd.DataFrame(out)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(
            frame[
                [
                    "implementation",
                    "scenario",
                    "replicate",
                    "estimand",
                    "truth",
                    "estimate",
                    "std_error",
                    "covered",
                    "seconds",
                ]
            ]
        )
        print(extra_artifacts(frame)[SELECTION_FILE])
    return frame


if __name__ == "__main__":
    smoke()
