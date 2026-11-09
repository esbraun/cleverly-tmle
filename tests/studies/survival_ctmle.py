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
:math:`\Sigma = W_1 + W_2 + W_3`, by Gauss-Hermite quadrature over the clipped law
(:func:`truth`); the reported risk is :math:`F_a(2) = 1 - \psi_a(2)`.  The quadrature is
accurate to about :math:`10^{-5}`, since the clip makes the integrand non-smooth.  The clipped
law gives :math:`\psi_1(2) = 0.46149`, which rounds to 0.461; the source's Table 1 prints
0.462, which the unclipped law (0.46180) rounds to.  :func:`check_truth` asserts the value lies
within :math:`10^{-3}` of 0.462.

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
as columns (without the source's FDR step); OR 15 with ``g_bounds`` lower bound
:math:`10^{-8}`.

**Gates** (declared, not yet measured).  (a) The bias of :math:`F_1(2)` and of the risk
difference within a declared margin at the binding confidence bound in every cell with the
outcome model correct.  (b) The RMSE of ``SurvivalCTMLE`` at most the full-:math:`g` hazard
``LTMLE``'s at the binding bound, outcome model correct, OR 15 only.  (c) The fit-failure share
under :data:`FAILURE_BOUND`.  Plug-in standard errors against the empirical SD, plug-in Wald
coverage (F18), the misspecified cells, the share selecting :math:`W_4` into :math:`g_1` and
:math:`A` into :math:`g_2`, the restart counts, clips, dropped terms and run time are reported
and not gated.  The study's censoring is about 4.7% per interval, so :math:`g_2` moves are rare
here; ``tests/unit/test_survival_ctmle.py`` check 3 is the :math:`g_2` witness.

**Size and seeds.** :data:`PILOT_REPLICATES` sizes the pilot.  The primary replicate count is
the shared rule (the smallest multiple of 400 that reaches the pass share) on that pilot,
declared before any declared draw.  :data:`SEED` and :data:`RESAMPLING_SEED` are used by no
other study; replicate ``r`` of a scenario draws from ``replicate_seed(STUDY, scenario, r)``.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit
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
    "or15_qcorrect_binarized": (15.0, 1, "correct", PRIMARY_N, {"binarized": True}),
    "or15_qcorrect_glower": (15.0, 1, "correct", PRIMARY_N, {"g_lower": 1e-8}),
}
PRIMARY_SCENARIOS = tuple(
    name
    for name, (_, source, _, n, settings) in SCENARIOS.items()
    if source == 1 and n == PRIMARY_N and not settings
)
#: The cell gate (b) reads.
RMSE_GATE_SCENARIO = "or15_qcorrect"


def truth(nodes: int = 80) -> dict[str, float]:
    """The risk of each arm and their difference at ``t = 2``, by quadrature."""
    covariance = np.full((3, 3), 0.2)
    np.fill_diagonal(covariance, 1.0)
    factor = np.linalg.cholesky(covariance)
    points, weights = hermegauss(nodes)
    weights = weights / weights.sum()
    grid = np.stack(np.meshgrid(points, points, points, indexing="ij"), axis=-1).reshape(-1, 3)
    mass = np.prod(
        np.stack(np.meshgrid(weights, weights, weights, indexing="ij"), axis=-1).reshape(-1, 3),
        axis=1,
    )
    total = np.clip(grid @ factor.T, -2.0, 2.0).sum(axis=1)
    survival = {
        arm: float(np.sum(mass * (1.0 - expit(0.25 - arm - 0.6 * total)) ** HORIZON))
        for arm in (0, 1)
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
    margins=Margins(),
    implementation=CTMLE_ARM,
    reference=None,
    failure_bound={CTMLE_ARM: FAILURE_BOUND},
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
    "truth": "Gauss-Hermite quadrature, 80 nodes per axis, over the clipped law",
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


def selection_rows(result: Any, scenario: str, replicate: int) -> list[dict[str, Any]]:
    """The reported, ungated selection readings of one ``SurvivalCTMLE`` fit."""
    rows = []
    for (arm, horizon), record in result.selection.items():
        g1, g2 = record.selected_terms
        rows.append(
            {
                "scenario": scenario,
                "replicate": replicate,
                "arm": arm,
                "horizon": horizon,
                "selected": record.selected,
                "w4_in_g1": any(name.startswith("W4") for name in g1),
                "a_in_g2": "A" in g2,
                "n_steps": record.n_steps[record.selected],
                "clipped": record.clipped[record.selected],
                "dropped_terms": record.dropped_terms[record.selected],
            }
        )
    return rows


def _draw(key: tuple[str, int, int]) -> tuple[pd.DataFrame, dict[str, float]]:
    scenario, replicate, n = key
    return draw_scenario(scenario, n, replicate)


def _replicate(
    key: tuple[str, int, int], drawn: tuple[pd.DataFrame, dict[str, float]]
) -> list[dict[str, Any]]:
    scenario, replicate, _ = key
    frame, truth_ = drawn
    rows: list[dict[str, Any]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for implementation in IMPLEMENTATIONS:
            result = fit(frame, scenario, implementation)
            rows.extend(result_rows(result, truth_, scenario, replicate, implementation))
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
    return replication.result(outcomes, rows.loc[:, list(REPLICATE_COLUMNS)])


def smoke(scenarios: tuple[str, ...] = ("or15_qcorrect", "or1_qwrong"), replicates: int = 2) -> Any:
    """A cheap run of a few cells, for a runner to check the module before the pilot.

    Prints each row and the selection readings; it writes nothing and registers nothing.
    """
    check_truth()
    out = []
    for scenario in scenarios:
        n = STUDY.n_of(scenario)
        for replicate in range(replicates):
            frame, truth_ = draw_scenario(scenario, n, replicate)
            for implementation in IMPLEMENTATIONS:
                start = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = fit(frame, scenario, implementation)
                seconds = time.perf_counter() - start
                for row in result_rows(result, truth_, scenario, replicate, implementation):
                    out.append({**row, "seconds": seconds})
                if implementation == CTMLE_ARM:
                    for row in selection_rows(result, scenario, replicate):
                        print(row)
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
    return frame


if __name__ == "__main__":
    smoke()
