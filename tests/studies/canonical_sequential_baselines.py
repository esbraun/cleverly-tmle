r"""Registered evidence for sequential g-computation and IPW (``sequential-baselines``, X41).

The subjects are :class:`~cleverly.longitudinal.SequentialGComputation` (ICE) and
:class:`~cleverly.longitudinal.SequentialIPW` with ``normalization="hajek"`` (Hajek) and
``"horvitz_thompson"`` (HT), each with :class:`~cleverly.learners.GLM` nuisances.  The laws are
in :mod:`tests.studies.sequential_baselines_laws`, and the property families in
:mod:`tests.studies.sequential_baselines_properties`.

This study is Python only.  It has no reference implementation: the R ``ltmle`` 1.3-0 parity
(plan cell P) is a separate study that a later pull request registers, so this record does
not depend on it.  The fast-tier fixture
W9 already checks the two ``ltmle`` points on one panel.

**Primary scenarios.**  The end-of-study laws ``D2`` and ``G``, one scenario per estimator, at
:data:`PRIMARY_N` with :data:`PRIMARY_REPLICATES` replicates.  The three scenarios of one law
read one draw per replicate (``scenario_seed_owners``), so the estimators are paired.  The shared
independent-performance tests read each row: the standardized bias inside 0.25 SD, the coverage
floor 0.90 and the SE sanity band.  :data:`PRIMARY_N` is the top rung of :data:`LADDER`.

========================  =================================================================
scenario                  estimands
========================  =================================================================
``d2_<estimator>``        ``ey_regimen`` of ``static_12``, ``static_00``, ``rule`` and
                          ``policy``, and ``ate_regimen[static_12 vs static_00]``
``g_<estimator>``         ``ey_regimen`` of ``treat_all``, ``treat_none`` and ``rule``, and
                          ``ate_regimen[treat_all vs treat_none]``
========================  =================================================================

**Learners.**  On the ``D2`` family the outcome and pseudo-outcome regressions are
``GLM(interactions=2)`` and every mechanism is ``GLM()`` (main terms).  On ``G``, ``X`` and
``GM`` the outcome regressions are ``GLM(family="gaussian")`` and every mechanism is ``GLM()``.
``q_bounds`` is the exact support of the outcome.  ``g_bounds`` is the shipped default
``(0.01, 1)``.  The bound acts on the cumulative product of the mechanism factors along a
follower.  The design test computes its true minimum on ``D2`` and on a large ``G`` draw
(about 0.019 on ``G``), so the bound binds only where a fitted product crosses 0.01.

**Red-cell reading** (declared before any run).  The study publishes under ``reporting``.  The
declared red set is :data:`EXPECTED_RED`: the K1 cells of law ``X``, where every main-terms GLM
is wrong and the interval covers the misspecified limit, so the calibration rule must fail
(the design test computes that power).  A red row outside :data:`EXPECTED_RED` is a defect
suspect.  It is a finite-sample red only when the 99% interval of its SE ratio contains 1, its
coverage is consistent with 0.95 and the diagnosis finds no defect; it then goes to
``reporting`` with an owner, declared before any repeat run.  A defect is fixed and the study
re-runs under a fresh declaration.  A replicate that raises a recordable error is recorded and
never redrawn; the declared failure bound is :data:`FAILURE_BOUND` per cell.  No budget,
margin, law, learner or seed moves after a verdict is seen.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

from cleverly.interventions import ModifiedPolicy, Piece, Shift, Stochastic
from cleverly.learners import GLM
from cleverly.longitudinal import DynamicRegimen, SequentialGComputation, SequentialIPW
from tests.parallel import STUDY_JOBS
from tests.studies import sequential_baselines_laws as laws
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

SEED = 20261441
RESAMPLING_SEED = 2026144101

#: The three sizes of the ladder.  ``tests/unit/test_sequential_baselines_design.py`` pins that
#: every node's MLE exists with probability at least 0.999 at the smallest.
LADDER = (2_000, 4_000, 8_000)
#: The size of the reporting-only adaptive cells (A).  One draw fits the default learner library
#: twice, about 180 CPU-seconds at n = 1,000, so A has one size.
ADAPTIVE_SIZES = (1_000,)
#: The size of the ``D2C`` calibration cell.  The design probe found a singular ICE fit on
#: 227 of 2,000 draws at n = 2,000 and 13 of 2,000 at 4,000 (a sparse node-2 cell among the
#: event-free followers), and none of 2,000 at 8,000.
D2C_N = 8_000
#: The size of every other calibration cell.
CALIBRATION_N = 2_000
PRIMARY_N = LADDER[-1]
PRIMARY_REPLICATES = 1_000
#: The declared share of replicates one cell may lose to recordable fitting failures.
FAILURE_BOUND = 0.01

IMPLEMENTATION = "cleverly-sequential-baselines"
#: The estimators, by their short names in scenario and cell labels.
ESTIMATORS = ("ice", "hajek", "ht")
NORMALIZATION = {"hajek": "hajek", "ht": "horvitz_thompson"}
G_BOUNDS = (0.01, 1.0)

# ------------------------------------------------------------------------------- regimens


def _labels(value: Any) -> Any:
    return np.asarray(value, dtype=float).astype(int)


def d2_regimens(*, binary: bool = False) -> dict[str, Any]:
    """The ``D2`` plans of :data:`tests.studies.sequential_baselines_laws.D2_PLANS`.

    ``binary=True`` gives the plans of the binary law ``D2B``.
    """
    if binary:
        return {
            "always": 1,
            "never": 0,
            "rule": DynamicRegimen(
                "rule",
                (lambda h: _labels(h["L1"]), lambda h: _labels(h["L2"])),
                rule_kind="known",
            ),
        }
    policy = Stochastic(
        lambda h: pd.DataFrame(np.tile(laws.D2_POLICY, (len(h), 1)), columns=[0, 1, 2]),
        "q2",
        density_kind="known",
    )
    return {
        "static_12": (1, 2),
        "static_00": (0, 0),
        "rule": DynamicRegimen(
            "rule",
            (lambda h: _labels(h["L1"]), lambda h: 2 * _labels(h["L2"])),
            rule_kind="known",
        ),
        "policy": DynamicRegimen("policy", (1, policy)),
    }


def g_regimens() -> dict[str, Any]:
    """The ``G`` and ``X`` plans: treat at all three times, at none, and the rule :math:`A_t = V`."""

    def rule(h: Any) -> Any:
        return _labels(h["V"])

    return {
        "treat_all": 1,
        "treat_none": 0,
        "rule": DynamicRegimen("rule", (rule, rule, rule), rule_kind="known"),
    }


def _gm_down(node: int) -> ModifiedPolicy:
    column = f"L{node}"

    def boundary(h: Any) -> Any:
        return laws.gm_boundary(h[column])

    def lower(a: Any, h: Any) -> Any:
        return np.asarray(a, dtype=float) - laws.GM_STEP

    def inverse(b: Any, h: Any) -> Any:
        return np.asarray(b, dtype=float) + laws.GM_STEP

    def derivative(b: Any, h: Any) -> Any:
        return np.ones_like(np.asarray(b, dtype=float))

    return ModifiedPolicy(
        f"down at {node}",
        pieces=(Piece(-np.inf, boundary), Piece(boundary, np.inf, lower, inverse, derivative)),
        policy_kind="known",
    )


def gm_regimens() -> dict[str, Any]:
    """The ``GM`` plans: the natural course and the policy that lowers each dose."""
    natural = Shift(0.0, cap=None)
    return {
        "natural": DynamicRegimen("natural", (natural, natural)),
        "down": DynamicRegimen("down", (_gm_down(1), _gm_down(2))),
    }


# --------------------------------------------------------------------------------- fits


class FixedMechanismGLM(BaseEstimator, ClassifierMixin):
    """A main-terms :class:`~cleverly.learners.GLM` that the stacked curve does not read.

    The fit and the predictions are the GLM's, so an IPW estimate with this mechanism equals
    the estimate with ``GLM()`` bit for bit.  The estimator does not recognize the wrapper as a
    GLM, so the fit reports the fixed-mechanism diagnostic curve under ``plugin_std_error``.
    That curve is cell K2's standard error.

    Parameters
    ----------
    tol : float
        The GLM's step tolerance.
    """

    def __init__(self, tol: float = 1e-10) -> None:
        self.tol = tol

    def fit(self, x: Any, y: Any, sample_weight: Any = None) -> FixedMechanismGLM:
        self.model_ = GLM(tol=self.tol).fit(x, y, sample_weight=sample_weight)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, x: Any) -> Any:
        return self.model_.predict_proba(x)

    def predict(self, x: Any) -> Any:
        return self.model_.predict(x)


class StatusMismatch(AssertionError):
    """A fit stamped an inference status that its cell does not read.

    An ``AssertionError`` is not a recordable fitting failure
    (:func:`cleverly.validation._failures.recordable`), so it stops the run.  A replicate row
    reads ``plugin_std_error`` under any status, so a silently degraded status would calibrate
    the wrong curve, and the rows could not show it afterwards.
    """


#: The status of a fit whose every regression is a ``GLM`` the stacked curve reads.
INFERENTIAL = "influence_curve"
#: The status of a fit with a regression the curve does not read (K2 and cell A).
DIAGNOSTIC = "adaptive_baseline_plugin"


def require_status(result: Any, names: Any, expected: str) -> None:
    """Raise :class:`StatusMismatch` unless the fit and each read parameter carry ``expected``."""
    if result.inference_status != expected:
        raise StatusMismatch(f"fit status {result.inference_status!r}, expected {expected!r}")
    for name in names:
        estimate = result[name]
        supplies = expected == INFERENTIAL
        if estimate.inference != expected or bool(estimate.supplies_inference) != supplies:
            raise StatusMismatch(
                f"{name}: status {estimate.inference!r} (supplies inference "
                f"{estimate.supplies_inference}), expected {expected!r}"
            )


def law_settings(law: str) -> dict[str, Any]:
    """The regimens, reference, learners and node declaration of one law."""
    if law in ("D2", "D2S", "D2C", "D2B"):
        target = laws.DISCRETE_LAWS[law]
        binary = law == "D2B"
        return {
            "regimens": d2_regimens(binary=binary),
            "reference": "never" if binary else "static_00",
            "outcome": GLM(interactions=2),
            "columns": target.fit_columns(),
            "q_bounds": None,
            "horizons": [1, 2] if target.events else None,
        }
    if law in ("G", "X"):
        continuous = laws.G if law == "G" else laws.X
        return {
            "regimens": g_regimens(),
            "reference": "treat_none",
            "outcome": GLM(family="gaussian"),
            "columns": continuous.columns(),
            "q_bounds": continuous.q_bounds(),
            "horizons": None,
        }
    if law == "GM":
        return {
            "regimens": gm_regimens(),
            "reference": "natural",
            "outcome": GLM(family="gaussian"),
            "columns": dict(laws.GM_COLUMNS),
            "q_bounds": laws.gm_outcome_support(),
            "horizons": None,
        }
    raise KeyError(law)


def fit(
    estimator: str,
    law: str,
    frame: pd.DataFrame,
    *,
    weights: str | None = None,
    mechanism: Any = None,
    outcome: Any = None,
) -> Any:
    """One fit of ``estimator`` (``"ice"``, ``"hajek"`` or ``"ht"``) on a draw of ``law``.

    ``mechanism`` and ``outcome`` replace the declared GLMs (cells K2 and A).
    """
    settings = law_settings(law)
    common: dict[str, Any] = {
        "reference": settings["reference"],
        "g_bounds": G_BOUNDS,
        "q_bounds": settings["q_bounds"],
        "horizons": settings["horizons"],
        "simultaneous": False,
        "random_state": 0,
    }
    treatment = GLM() if mechanism is None else mechanism
    with warnings.catch_warnings():
        # The shipped warnings (positivity, extrapolation) are diagnostics; none changes a
        # number, and the design test reads the bound-binding share directly.
        warnings.simplefilter("ignore")
        if estimator == "ice":
            model: Any = SequentialGComputation(
                settings["regimens"],
                outcome_learner=settings["outcome"] if outcome is None else outcome,
                treatment_learner=treatment,
                censoring_learner=treatment,
                **common,
            )
        else:
            model = SequentialIPW(
                settings["regimens"],
                treatment_learner=treatment,
                censoring_learner=treatment,
                normalization=NORMALIZATION[estimator],
                **common,
            )
        return model.fit(frame, weights=weights, **settings["columns"])


# ------------------------------------------------------------------------------ estimands

D2_ESTIMANDS = (
    "ey_regimen[static_12]",
    "ey_regimen[static_00]",
    "ey_regimen[rule]",
    "ey_regimen[policy]",
    "ate_regimen[static_12 vs static_00]",
)
G_ESTIMANDS = (
    "ey_regimen[treat_all]",
    "ey_regimen[treat_none]",
    "ey_regimen[rule]",
    "ate_regimen[treat_all vs treat_none]",
)


def d2_truths() -> dict[str, float]:
    """The exact ``D2`` truths of :data:`D2_ESTIMANDS`."""
    levels = {label: laws.D2.truth(plan) for label, plan in laws.D2_PLANS.items()}
    out = {f"ey_regimen[{label}]": value for label, value in levels.items()}
    out["ate_regimen[static_12 vs static_00]"] = levels["static_12"] - levels["static_00"]
    return out


def g_estimand_truths(law: laws.ContinuousLaw) -> dict[str, float]:
    """The exact truths of :data:`G_ESTIMANDS` on ``G`` or ``X``."""
    values = laws.g_truths(law)
    out = {f"ey_regimen[{label}]": values[label] for label in laws.G_PLANS}
    out["ate_regimen[treat_all vs treat_none]"] = values["contrast"]
    return out


SCENARIOS: dict[str, tuple[str, ...]] = {
    **{f"d2_{estimator}": D2_ESTIMANDS for estimator in ESTIMATORS},
    **{f"g_{estimator}": G_ESTIMANDS for estimator in ESTIMATORS},
}
SEED_OWNERS = {
    **{f"d2_{estimator}": "d2_ice" for estimator in ESTIMATORS},
    **{f"g_{estimator}": "g_ice" for estimator in ESTIMATORS},
}

#: The declared red set: the K1 cells (``misspecification_control`` on law ``X``).
EXPECTED_RED: tuple[str, ...] = tuple(
    f"misspecification_control/{estimator}_x_contrast__main_terms" for estimator in ESTIMATORS
)

#: Every published property cell, by family.  :mod:`tests.studies.sequential_baselines_properties`
#: declares how each is drawn and read.
CALIBRATION_LABELS: tuple[str, ...] = (
    *(
        f"{estimator}_d2_{name}"
        for estimator in ESTIMATORS
        for name in ("static_12", "static_00", "rule", "policy", "contrast")
    ),
    *(
        f"{estimator}_g_{name}"
        for estimator in ESTIMATORS
        for name in ("treat_all", "treat_none", "rule", "contrast")
    ),
    *(
        f"{estimator}_d2s_{name}"
        for estimator in ESTIMATORS
        for name in ("risk_h1", "risk_h2", "contrast_h2")
    ),
    *(f"{estimator}_d2c_relapse_h2" for estimator in ESTIMATORS),
    "ice_gm_down",
    "ice_gm_contrast",
    "ice_d2w_static_12",
    "hajek_d2w_static_12",
)
LADDER_LABELS: tuple[str, ...] = tuple(
    f"{estimator}_{law}_contrast" for law in ("d2", "g") for estimator in ESTIMATORS
)
G_REPORTED = ("treat_all", "treat_none", "contrast")
#: The K2 targets of each law.  On ``D2`` the multinomial node 2 carries the mechanism term;
#: Hajek's term there moves its SE by about 1%, HT's by 17% to 46% (design review S2).
K2_TARGETS: dict[str, tuple[str, ...]] = {
    "G": G_REPORTED,
    "D2": ("static_12", "policy", "rule"),
}
PROPERTY_CELLS: dict[str, tuple[str, ...]] = {
    "interval_calibration": tuple(
        f"{label}__{kind}"
        for label in CALIBRATION_LABELS
        for kind in ("correctly_specified", "shrunken_se_control", "noise_control")
    ),
    "root_n_and_efficiency": tuple(
        f"{label}__n_{size}" for label in LADDER_LABELS for size in LADDER
    ),
    "root_n_rate": tuple(
        f"{label}__{suffix}"
        for label in LADDER_LABELS
        for suffix in ("empirical_sd", "reported_se")
    ),
    "misspecification_control": tuple(
        f"{estimator}_x_contrast__main_terms" for estimator in ESTIMATORS
    ),
    "fixed_mechanism_se": tuple(
        f"{estimator}_{law.lower()}_{name}__fixed_mechanism"
        for law, names in K2_TARGETS.items()
        for estimator in ("hajek", "ht")
        for name in names
    ),
    "first_node_se": tuple(f"ice_g_{name}__first_node" for name in G_REPORTED),
    "adaptive_learners": tuple(
        f"{estimator}_x_contrast__n_{size}"
        for estimator in ("ice", "hajek")
        for size in ADAPTIVE_SIZES
    ),
}

STUDY = StudyRecord(
    name="sequential g-computation and IPW",
    slug="sequential-baselines",
    artifacts=ROOT / "tests" / "canonical" / "sequential_baselines_study",
    document="docs/technical-reference/method-evidence/sequential-baselines.md",
    anchor="sequential-g-computation-and-ipw",
    scenarios=dict(SCENARIOS),
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    scenario_seed_owners=dict(SEED_OWNERS),
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    calibration_efficiency_ratio=False,
    modules=(
        "tests/studies/canonical_sequential_baselines.py",
        "tests/studies/sequential_baselines_properties.py",
        "tests/studies/sequential_baselines_laws.py",
        "tests/studies/evidence/inference.py",
        "tests/studies/evidence/performance.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.canonical_sequential_baselines",
    properties_module="tests.studies.sequential_baselines_properties",
    property_cells=PROPERTY_CELLS,
    publication_policy="reporting",
    failure_bound={IMPLEMENTATION: FAILURE_BOUND},
)

CONFIGURATION = {
    "estimators": {
        "ice": "SequentialGComputation",
        "hajek": "SequentialIPW(normalization='hajek')",
        "ht": "SequentialIPW(normalization='horvitz_thompson')",
    },
    "outcome_learner": {
        "D2 family": "GLM(interactions=2)",
        "G, X, GM": "GLM(family='gaussian')",
    },
    "mechanism_learner": "GLM()",
    "g_bounds": list(G_BOUNDS),
    "q_bounds": {
        "G": list(laws.G.q_bounds()),
        "X": list(laws.X.q_bounds()),
        "GM": list(laws.gm_outcome_support()),
    },
    "ladder": list(LADDER),
    "calibration_n": CALIBRATION_N,
    "failure_bound": FAILURE_BOUND,
    "expected_red": list(EXPECTED_RED),
    "reference": "none; the R ltmle parity is a separate study",
}

# --------------------------------------------------------------------------------- primary


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """One draw of ``scenario``'s law, with its exact truths."""
    if scenario.startswith("d2_"):
        return laws.D2.sample(n, seed), d2_truths()
    if scenario.startswith("g_"):
        return laws.G.sample(n, seed), g_estimand_truths(laws.G)
    raise KeyError(scenario)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def result_rows(
    result: Any, truth: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    rows = []
    for name in SCENARIOS[scenario]:
        estimate = result[name]
        low, high = estimate.ci
        reference = float(truth[name])
        rows.append(
            {
                "implementation": IMPLEMENTATION,
                "scenario": scenario,
                "replicate": replicate,
                "n": result.n,
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": float(estimate.std_error),
                "ci_lower": float(low),
                "ci_upper": float(high),
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": float(estimate.psi),
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
    frame, truth = drawn
    law, estimator = scenario.split("_", 1)
    result = fit(estimator, law.upper(), frame)
    require_status(result, SCENARIOS[scenario], INFERENTIAL)
    return result_rows(result, truth, scenario, replicate)


def draw_and_fit(*, replicates: int, n: int, n_jobs: int = STUDY_JOBS) -> replication.PhaseResult:
    outcomes = replication.fit_primary(
        [(scenario, replicate, n) for replicate in range(replicates) for scenario in SCENARIOS],
        _replicate,
        draw=_draw,
        n_jobs=n_jobs,
        implementation=IMPLEMENTATION,
        rows_at=None,
        seed_of=lambda key: replicate_seed(STUDY, key[0], key[1]),
    )
    estimates = pd.DataFrame([row for rows in outcomes.kept for row in rows])
    return replication.result(outcomes, estimates.loc[:, list(REPLICATE_COLUMNS)])
