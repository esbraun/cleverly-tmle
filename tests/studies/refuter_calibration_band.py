"""Registered evidence for the band-rank placebo rule (X82 follow-up SJ).

The subject is the placebo ``t`` rule of :func:`cleverly.refute` on a continuous dose whose
declared support is a function of the covariates or the history. On such a dose the placebo
draws each dose's position inside its band and maps it onto the receiving unit's band
(:mod:`cleverly.validation._band_placebo`). ``X82-J-plan.md`` rev 2 (plans repository,
section 5) declares the design. Each cell checks that the rule is not grossly miscalibrated on
one configuration. It does not certify size to two decimals.

=====================  ==========================================================================
rule                   the reading
=====================  ==========================================================================
placebo ``t``          the one-sample ``t`` of 20 band-rank placebo refits against the exact null,
                       two-sided at 0.05 under ``t_19`` (:func:`~cleverly.validation._readings.placebo_reading`)
=====================  ==========================================================================

**Law P** (cell PJ1). Two covariates ``W1, W2 ~ U(-1, 1)``. The dose ``A`` has the declared band
``[L(W), U(W)]`` with ``L = W1 + 0.5 W2`` and width ``U - L = 2 + 0.75 W1``, so the band's
location and its width read ``W``. The position ``u = (A - L) / (U - L)`` is a normal with mean
``0.5 + 0.2 W2`` and SD 0.25, truncated to ``[0, 1]``. The outcome is
``Y = A + 0.8 W1 + 0.6 W2 + N(0, 1)``. The policy ``up`` adds 0.5 where the result stays below
``U(W)`` and keeps the dose elsewhere. The contrast is ``up`` against the natural course.

**Law L** (cell LJ1), the shape of the pricing law of :func:`cleverly.datasets.make_sequential_pricing`.
Three nodes, no censoring. ``W ~ U(-1, 1)``. At node ``t`` the engagement ``L_t`` is binary, and
the price ``A_t`` has the declared band with lower end ``2 + L_t + 0.5 A_{t-1}`` and width
``2 + L_t`` (``A_0 = 6``). The band reads the engagement and the earlier price, so the placebo
maps node ``t`` on the placebo history (``X82-J-plan.md`` section 3.2 step 4).

.. code-block:: text

    logit P(L_t = 1 | past)   = 0.3 + 0.6 W - (u_{t-1} - 0.5) + 0.8 L_{t-1}     (u_0 = 0.5, L_0 = 0)
    u_t | past                ~ N(0.5 + 0.15 W + 0.3 (u_{t-1} - 0.5), 0.4^2) truncated to [0, 1]
    A_t                       = 2 + L_t + 0.5 A_{t-1} + u_t (2 + L_t)
    logit P(Y = 1 | past)     = 0.3 - 0.5 (A_3 - 7.2) + 0.9 L_3 + 0.3 W - 0.1 (A_1 + A_2 - 13.5)

The policy ``up`` adds 0.25 to each price where the result stays below that node's band top, and
keeps the price elsewhere. The contrast is ``up`` against the natural course.

**The cells** (plan section 5). ``n = 2,000``, five-fold cross-fitting, ``R = 20`` refits.

====  ==============================================================================  =====
cell  configuration                                                                   law
====  ==============================================================================  =====
PJ1   ``TMLE``, the policy ``up``, OLS outcome, classifier ratio with the quadratic      P
      logistic :class:`QuadraticLogistic`
LJ1   ``LTMLE``, ``T = 3``, the regimen ``up``, logistic GLM outcome and pseudo-outcome  L
      regressions, classifier ratio with :class:`QuadraticLogistic`
====  ==============================================================================  =====

``ratio="density"`` is not covered and stays descriptive.

**Size cells.** 200 trials. A reading passes when it rejects in at most 17 trials.

**Power cells.** Trials 0 to 49 of each size cell also run the stale-row defect. The power
reading passes when it detects the defect in at least 38 of 50 trials. A declared share of units
keeps its own dose (PJ1) or its whole observed trajectory (LJ1), and the other units keep their
band-rank draws (:func:`stale_rows`). The share is sized so that the placebo moves by at least
0.850 placebo SD (per-trial power 0.95 at ``R = 20``). The sizing probe
``tests/diagnostics/x82_sj_sizing/`` measures the shift, and
``tests/unit/test_refuter_calibration_band_design.py`` reads it.

**Fit configuration.** Every trial reads the configuration off its fitted result
(:func:`~cleverly.validation._interpretation.fit_configuration` with ``replicates = 20`` and
``placebo_draw`` set to the cell's draw). The run stops if any axis but the smallest event count
varies across a cell's trials. ``cells.csv`` records the one configuration of each cell.

**Red reading** (declared before the run). A size red is a defect suspect until diagnosed. A red
with no defect goes to reporting with an owner, and the rule's registry row stays descriptive. A
red with one pre-identified cause, a classifier that cannot follow the band on the placebo law
(``X82-J-plan.md`` section 4), goes to diagnosis, which tests that cause first. No rerun buys a
pass. A power red with no defect means the defect was sized too small for this cell; it keeps
the rule descriptive and goes to reporting.
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
from scipy.special import expit
from scipy.stats import truncnorm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from cleverly.estimators import TMLE
from cleverly.interventions import ModifiedPolicy, Piece, Shift, Support
from cleverly.learners import GLM
from cleverly.longitudinal import LTMLE, DynamicRegimen
from cleverly.validation import refute
from cleverly.validation._band_placebo import PanelBands, PointBands
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
SEED = 20268831
RESAMPLING_SEED = 20268832

#: Refits per placebo reading: the library default (X82 rev 5).
REFITS = 20
SIZE_TRIALS = 200
POWER_TRIALS = 50
#: A size reading passes at no more than this many rejections in :data:`SIZE_TRIALS`.
SIZE_MAX_REJECTIONS = 17
#: A power reading passes at no fewer than this many detections in :data:`POWER_TRIALS`.
POWER_MIN_DETECTIONS = 38
#: The level every rule reads.
LEVEL = 0.05

# ---- law P --------------------------------------------------------------------------------

#: ``L(W) = P_LOWER . (W1, W2)``.
P_LOWER = (1.0, 0.5)
#: ``U(W) - L(W) = P_WIDTH[0] + P_WIDTH[1] W1``.
P_WIDTH = (2.0, 0.75)
#: The position's mean ``P_POSITION[0] + P_POSITION[1] W2`` and its SD, before truncation.
P_POSITION = (0.5, 0.2)
P_POSITION_SD = 0.25
#: ``Y = A + P_OUTCOME . (W1, W2) + N(0, 1)``.
P_OUTCOME = (0.8, 0.6)
P_EFFECT = 1.0
#: The policy's shift on law P.
P_SHIFT = 0.5
#: The declared outcome bounds. A cross-fitted continuous fit needs declared bounds; these
#: never bind on this law (:func:`draw_point` refuses a sample outside them).
P_Q_BOUNDS = (-15.0, 15.0)

# ---- law L --------------------------------------------------------------------------------

NODES = 3
#: The price before the first node.
L_START = 6.0
#: The lower end ``L_LOWER[0] + L_LOWER[1] L_t + L_LOWER[2] A_{t-1}``.
L_LOWER = (2.0, 1.0, 0.5)
#: The width ``L_WIDTH[0] + L_WIDTH[1] L_t``.
L_WIDTH = (2.0, 1.0)
#: ``logit P(L_t = 1) = L_ENGAGED . (1, W, u_{t-1} - 0.5, L_{t-1})``.
L_ENGAGED = (0.3, 0.6, -1.0, 0.8)
#: The position's mean ``L_POSITION . (1, W, u_{t-1} - 0.5)`` and its SD, before truncation.
L_POSITION = (0.5, 0.15, 0.3)
L_POSITION_SD = 0.4
#: ``logit P(Y = 1) = L_RENEW[0] + L_RENEW[1] (A_3 - 7.2) + L_RENEW[2] L_3 + L_RENEW[3] W
#: + L_RENEW[4] (A_1 + A_2 - 13.5)``.
L_RENEW = (0.3, -0.5, 0.9, 0.3, -0.1)
#: The policy's shift on law L.
L_SHIFT = 0.25

#: The share of units whose placebo keeps the recorded dose or trajectory, by cell. The sizing
#: probe requires the shift of each cell's placebo to reach 0.850 placebo SD.
STALE_SHARE = {"PJ1": 0.10, "LJ1": 0.15}

SIZE_FAMILY = "refuter_size"
POWER_FAMILY = "refuter_power"

READINGS = ("placebo",)
DEFECTS = ("placebo_stale_rows",)


@dataclass(frozen=True)
class Cell:
    """One configuration of plan section 5.

    Parameters
    ----------
    law : str
        ``"point"`` (law P) or ``"panel"`` (law L).
    estimand : str
        The alias every reading of the cell refutes.
    placebo_draw : str
        The draw the placebo reading runs, as ``FitConfiguration.placebo_draw`` names it.
    """

    law: str
    estimand: str
    placebo_draw: str

    @property
    def readings(self) -> tuple[str, ...]:
        """The size readings of this cell."""
        return READINGS

    @property
    def defects(self) -> tuple[str, ...]:
        """The power readings of this cell, one per defect."""
        return DEFECTS


CELLS: dict[str, Cell] = {
    "PJ1": Cell("point", "ate_policy[up vs natural course]", "band_permute"),
    "LJ1": Cell("panel", "ate_regimen[up vs natural]", "band_markov"),
}

PROPERTY_CELLS = {
    SIZE_FAMILY: tuple(
        f"{name}__{reading}" for name, cell in CELLS.items() for reading in cell.readings
    ),
    POWER_FAMILY: tuple(
        f"{name}__{defect}" for name, cell in CELLS.items() for defect in cell.defects
    ),
}

#: The primary phase: the PJ1 fit's contrast against its exact truth. It is a sanity row on
#: law P; no verdict of this study reads it.
SCENARIOS = {"point": ("ate",)}
PRIMARY_REPLICATES = 200

STUDY = StudyRecord(
    name="band-rank placebo verdict rule",
    slug="refuter-calibration-band",
    artifacts=ROOT / "tests" / "canonical" / "refuter_calibration_band",
    document="docs/technical-reference/method-evidence/refuter-calibration-band.md",
    anchor="band-rank-placebo-verdict-rule",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-refute-band",
    reference=None,
    failure_bound={"cleverly-refute-band": 0.01},
    modules=(
        "tests/studies/refuter_calibration_band.py",
        "tests/studies/refuter_calibration_band_properties.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.refuter_calibration_band",
    properties_module="tests.studies.refuter_calibration_band_properties",
    property_cells=PROPERTY_CELLS,
    property_artifacts=("cells.csv",),
    publication_policy="reporting",
)

CONFIGURATION = {
    "nodes": {"PJ1": None, "LJ1": NODES},
    "folds": N_FOLDS,
    "refits": REFITS,
    "size_trials": SIZE_TRIALS,
    "power_trials": POWER_TRIALS,
    "size_pass": f"rejections <= {SIZE_MAX_REJECTIONS} of {SIZE_TRIALS}",
    "power_pass": f"detections >= {POWER_MIN_DETECTIONS} of {POWER_TRIALS}",
    "law_p_band": "[W1 + 0.5 W2, W1 + 0.5 W2 + 2 + 0.75 W1], W ~ U(-1, 1)^2",
    "law_p_position": "N(0.5 + 0.2 W2, 0.25^2) truncated to [0, 1]",
    "law_p_outcome": "Y = A + 0.8 W1 + 0.6 W2 + N(0, 1)",
    "law_p_policy": "a + 0.5 where a + 0.5 < U(W), else a",
    "law_l_band": "lower 2 + L_t + 0.5 A_{t-1} (A_0 = 6), width 2 + L_t, W ~ U(-1, 1)",
    "law_l_engaged": "logit = 0.3 + 0.6 W - (u_{t-1} - 0.5) + 0.8 L_{t-1}",
    "law_l_position": "N(0.5 + 0.15 W + 0.3 (u_{t-1} - 0.5), 0.4^2) truncated to [0, 1]",
    "law_l_outcome": ("logit = 0.3 - 0.5 (A_3 - 7.2) + 0.9 L_3 + 0.3 W - 0.1 (A_1 + A_2 - 13.5)"),
    "law_l_policy": "a + 0.25 at each node where a + 0.25 < the node's band top, else a",
    "stale_share": dict(STALE_SHARE),
    "cells": {
        name: {"law": cell.law, "estimand": cell.estimand, "placebo_draw": cell.placebo_draw}
        for name, cell in CELLS.items()
    },
}


# ---- law P --------------------------------------------------------------------------------


def _column(history: Any, name: str) -> np.ndarray:
    return np.asarray(history[name], dtype=float)


def point_lower(history: Any) -> np.ndarray:
    """``L(W)``, the lower end of law P's band."""
    return P_LOWER[0] * _column(history, "W1") + P_LOWER[1] * _column(history, "W2")


def point_width(history: Any) -> np.ndarray:
    """``U(W) - L(W)``, the width of law P's band."""
    return P_WIDTH[0] + P_WIDTH[1] * _column(history, "W1")


def point_upper(history: Any) -> np.ndarray:
    """``U(W)``, the upper end of law P's band."""
    return point_lower(history) + point_width(history)


def _truncated_positions(rng: np.random.Generator, mean: np.ndarray, sd: float) -> np.ndarray:
    return np.asarray(
        truncnorm.rvs((0.0 - mean) / sd, (1.0 - mean) / sd, loc=mean, scale=sd, random_state=rng),
        dtype=float,
    )


def draw_point(n: int, seed: int) -> pd.DataFrame:
    """One sample of law P: ``W1, W2, A, Y``.

    Raises
    ------
    RuntimeError
        If ``Y`` leaves the declared :data:`P_Q_BOUNDS`.
    """
    rng = np.random.default_rng(seed)
    w1 = rng.uniform(-1.0, 1.0, n)
    w2 = rng.uniform(-1.0, 1.0, n)
    history = {"W1": w1, "W2": w2}
    position = _truncated_positions(rng, P_POSITION[0] + P_POSITION[1] * w2, P_POSITION_SD)
    dose = point_lower(history) + position * point_width(history)
    outcome = P_EFFECT * dose + P_OUTCOME[0] * w1 + P_OUTCOME[1] * w2 + rng.standard_normal(n)
    low, high = P_Q_BOUNDS
    if outcome.min() <= low or outcome.max() >= high:
        raise RuntimeError("a law-P draw left the declared q_bounds")
    return pd.DataFrame({"W1": w1, "W2": w2, "A": dose, "Y": outcome})


#: Gauss-Legendre nodes and weights on ``[-1, 1]`` for the uniform covariates.
_LEGENDRE, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(64)


def point_truth() -> float:
    """The exact contrast ``up`` against the natural course on law P.

    ``Y`` is linear in ``A`` with slope 1, so the contrast is ``0.5 P(A + 0.5 < U(W))``. Given
    ``W`` the event is ``u < 1 - 0.5 / (U - L)``, whose probability is the truncated normal's
    distribution function. The covariates are integrated by a 64-point Gauss-Legendre rule.
    """
    w1, w2 = np.meshgrid(_LEGENDRE, _LEGENDRE, indexing="ij")
    weights = np.outer(_LEGENDRE_WEIGHTS, _LEGENDRE_WEIGHTS) / 4.0
    width = P_WIDTH[0] + P_WIDTH[1] * w1
    mean = P_POSITION[0] + P_POSITION[1] * w2
    sd = P_POSITION_SD
    moved = truncnorm.cdf(
        1.0 - P_SHIFT / width, (0.0 - mean) / sd, (1.0 - mean) / sd, loc=mean, scale=sd
    )
    return float(P_EFFECT * P_SHIFT * np.sum(weights * moved))


# ---- law L --------------------------------------------------------------------------------


def _previous(history: Any, node: int) -> Any:
    return L_START if node == 1 else _column(history, f"A{node - 1}")


def panel_lower(history: Any, node: int) -> np.ndarray:
    """The lower end of node ``node``'s band on law L."""
    return np.asarray(
        L_LOWER[0]
        + L_LOWER[1] * _column(history, f"L{node}")
        + L_LOWER[2] * _previous(history, node),
        dtype=float,
    )


def panel_width(history: Any, node: int) -> np.ndarray:
    """The width of node ``node``'s band on law L."""
    return np.asarray(L_WIDTH[0] + L_WIDTH[1] * _column(history, f"L{node}"), dtype=float)


def panel_upper(history: Any, node: int) -> np.ndarray:
    """The upper end of node ``node``'s band on law L."""
    return panel_lower(history, node) + panel_width(history, node)


def draw_panel(n: int, seed: int) -> pd.DataFrame:
    """One sample of law L: ``W, L1, A1, L2, A2, L3, A3, Y``."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(-1.0, 1.0, n)
    frame: dict[str, np.ndarray] = {"W": w}
    engaged = np.zeros(n)
    position = np.full(n, 0.5)
    for node in range(1, NODES + 1):
        index = (
            L_ENGAGED[0]
            + L_ENGAGED[1] * w
            + L_ENGAGED[2] * (position - 0.5)
            + L_ENGAGED[3] * engaged
        )
        engaged = rng.binomial(1, expit(index)).astype(float)
        frame[f"L{node}"] = engaged
        mean = L_POSITION[0] + L_POSITION[1] * w + L_POSITION[2] * (position - 0.5)
        position = _truncated_positions(rng, mean, L_POSITION_SD)
        frame[f"A{node}"] = panel_lower(frame, node) + position * panel_width(frame, node)
    index = (
        L_RENEW[0]
        + L_RENEW[1] * (frame["A3"] - 7.2)
        + L_RENEW[2] * frame["L3"]
        + L_RENEW[3] * w
        + L_RENEW[4] * (frame["A1"] + frame["A2"] - 13.5)
    )
    frame["Y"] = rng.binomial(1, expit(index)).astype(float)
    return pd.DataFrame(frame)


def draw_law(law: str, n: int, seed: int) -> pd.DataFrame:
    """One sample of ``law``."""
    if law == "point":
        return draw_point(n, seed)
    if law == "panel":
        return draw_panel(n, seed)
    raise ValueError(f"unknown law {law!r}")


def draw_from_seed(scenario: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary sample from an explicit seed for the published-seed audit."""
    if scenario not in SCENARIOS:
        raise KeyError(f"{STUDY.slug} has no scenario {scenario!r}")
    return draw_point(n, seed), {"ate": point_truth()}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


# ---- the policies -------------------------------------------------------------------------


def _shift_pieces(shift: float, upper: Any) -> tuple[Piece, Piece]:
    """``a + shift`` below the knee ``upper(h) - shift``, and the dose kept above it."""

    def knee(history: Any) -> np.ndarray:
        return np.asarray(upper(history), dtype=float) - shift

    return (
        Piece(
            -np.inf,
            knee,
            lambda a, h: np.asarray(a, dtype=float) + shift,
            lambda b, h: np.asarray(b, dtype=float) - shift,
            lambda b, h: np.ones_like(np.asarray(b, dtype=float)),
        ),
        Piece(knee, np.inf),
    )


NATURAL = Shift(0.0, cap=None)
#: Law P's policy.
POINT_POLICY = ModifiedPolicy("up", pieces=_shift_pieces(P_SHIFT, point_upper), policy_kind="known")


def _node_policy(node: int) -> ModifiedPolicy:
    return ModifiedPolicy(
        f"up at {node}",
        pieces=_shift_pieces(L_SHIFT, lambda h: panel_upper(h, node)),
        policy_kind="known",
    )


#: Law L's regimens.
REGIMENS = {
    "natural": DynamicRegimen("natural", (NATURAL,) * NODES),
    "up": DynamicRegimen("up", tuple(_node_policy(node) for node in range(1, NODES + 1))),
}
REFERENCE = "natural"


# ---- the fits -----------------------------------------------------------------------------


def _knees(dose: np.ndarray, lower: np.ndarray, upper: np.ndarray, shift: float) -> list[Any]:
    """The policy's two breaks on the stacked dose: ``a >= L + shift`` and ``a >= U - shift``."""
    return [dose >= lower + shift, dose >= upper - shift]


class QuadraticLogistic(BaseEstimator, ClassifierMixin):
    """The classifier of the ratio: a logistic regression on a quadratic design.

    The stacked design is ``[A, H]``. The features are every term of degree one and two in
    ``[A, H]``, and the two indicators of the policy's breaks on the node's declared band
    (:func:`_knees`). The design is standardized before an L-BFGS fit with ``C = 1e6``.

    Parameters
    ----------
    law : str
        ``"point"`` (stacked ``[A, W1, W2]``) or ``"panel"`` (stacked
        ``[A_t, W, L_1..L_t, A_1..A_{t-1}]``).
    """

    def __init__(self, law: str = "point") -> None:
        self.law = law

    def _design(self, X: Any) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        dose, history = x[:, 0], x[:, 1:]
        if self.law == "point":
            if history.shape[1] != 2:
                raise ValueError(f"law P stacks [A, W1, W2], got {x.shape[1]} columns")
            frame = {"W1": history[:, 0], "W2": history[:, 1]}
            knees = _knees(dose, point_lower(frame), point_upper(frame), P_SHIFT)
        else:
            if history.shape[1] % 2:
                raise ValueError(f"unexpected history width {history.shape[1]}")
            node = history.shape[1] // 2
            frame = {"W": history[:, 0]}
            for earlier in range(1, node + 1):
                frame[f"L{earlier}"] = history[:, earlier]
            for earlier in range(1, node):
                frame[f"A{earlier}"] = history[:, node + earlier]
            knees = _knees(dose, panel_lower(frame, node), panel_upper(frame, node), L_SHIFT)
        quadratic = PolynomialFeatures(2, include_bias=False).fit_transform(x)
        return np.column_stack([quadratic, *knees]).astype(float)

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> QuadraticLogistic:
        design = self._design(X)
        self.scaler_ = StandardScaler().fit(design)
        self.model_ = LogisticRegression(C=1e6, max_iter=2_000).fit(
            self.scaler_.transform(design), np.asarray(y), sample_weight=sample_weight
        )
        self.n_iter_ = int(np.max(self.model_.n_iter_))
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(
            self.model_.predict_proba(self.scaler_.transform(self._design(X))), dtype=float
        )


#: The node declaration of law L.
PANEL_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": [f"A{node}" for node in range(1, NODES + 1)],
    "baseline": ["W"],
    "time_varying": [[f"L{node}"] for node in range(1, NODES + 1)],
}


def _panel_support(node: int) -> Support:
    return Support(lambda h: panel_lower(h, node), lambda h: panel_upper(h, node))


PANEL_SUPPORTS = {f"A{node}": _panel_support(node) for node in range(1, NODES + 1)}


def estimator(cell: Cell, seed: int) -> Any:
    """The cell's estimator, seeded with the trial's fit seed."""
    if cell.law == "point":
        return TMLE(
            policies=[NATURAL, POINT_POLICY],
            ratio="classifier",
            outcome_learner=LinearRegression(),
            treatment_learner=QuadraticLogistic("point"),
            cross_fit=True,
            n_folds=N_FOLDS,
            q_bounds=P_Q_BOUNDS,
            simultaneous=False,
            random_state=seed,
        )
    return LTMLE(
        REGIMENS,
        reference=REFERENCE,
        ratio="classifier",
        outcome_learner=GLM(),
        pseudo_learner=GLM(),
        treatment_learner=QuadraticLogistic("panel"),
        n_folds=N_FOLDS,
        simultaneous=False,
        random_state=seed,
    )


def fit_cell(name: str, frame: pd.DataFrame, seed: int) -> Any:
    """Fit cell ``name`` on one sample."""
    cell = CELLS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cell.law == "point":
            return (
                estimator(cell, seed)
                .fit(
                    frame,
                    outcome="Y",
                    treatment="A",
                    covariates=["W1", "W2"],
                    treatment_support=Support(point_lower, point_upper),
                )
                .single()
            )
        return estimator(cell, seed).fit(
            frame, continuous_treatment=PANEL_SUPPORTS, **PANEL_COLUMNS
        )


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


@contextlib.contextmanager
def stale_rows(share: float) -> Iterator[None]:
    """Patch the band-rank placebo so a share of units keeps the recorded treatment.

    The kept units are a uniform draw of ``round(share n)`` units. On a point fit each keeps its
    own dose. On a panel each keeps its whole observed trajectory, and its kernel draw holds its
    own positions. The other units keep their band-rank draws, so every placebo dose stays in
    its band.
    """
    point_draw = PointBands.draw
    panel_draw = PanelBands.placebo

    def point(self: PointBands, data: Any, rng: np.random.Generator, mode: Any) -> np.ndarray:
        doses = np.array(point_draw(self, data, rng, mode), dtype=float)
        n = doses.shape[0]
        kept = rng.choice(n, size=round(share * n), replace=False)
        doses[kept] = np.asarray(data.treatment, dtype=float)[kept]
        return doses

    def panel(
        self: PanelBands, data: Any, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        doses, drawn = panel_draw(self, data, rng)
        doses = np.array(doses, dtype=float)
        drawn = np.array(drawn, dtype=float)
        n = doses.shape[0]
        kept = rng.choice(n, size=round(share * n), replace=False)
        doses[kept] = np.asarray(data.treatment, dtype=float)[kept]
        own = np.asarray(self.kernel.trajectories, dtype=float)
        drawn[kept] = np.where(np.isnan(drawn[kept]), np.nan, own[kept])
        return doses, drawn

    PointBands.draw = point  # type: ignore[method-assign]
    PanelBands.placebo = panel  # type: ignore[method-assign]
    try:
        yield
    finally:
        PointBands.draw = point_draw  # type: ignore[method-assign]
        PanelBands.placebo = panel_draw  # type: ignore[method-assign]


def trial_seed(stream: str, name: str, trial: int) -> int:
    """The seed of one named stream of one property trial.

    The streams are ``"property_sample"`` (the draw), ``"property_fit"`` (the fit's folds and
    learners), ``"refute_size"`` and ``"refute_stale"`` (the refits).
    """
    return stream_seed(STUDY, stream, name, trial)


def trial_configuration(result: Any, cell: Cell) -> dict[str, Any]:
    """The fit configuration of one trial, read off the fitted result, as one flat row."""
    configuration = replace(
        fit_configuration(result, cell.estimand),
        replicates=REFITS,
        placebo_draw=cell.placebo_draw,
    )
    row = asdict(configuration)
    row["learners"] = " | ".join(configuration.learners)
    return row


@dataclass(frozen=True)
class Trial:
    """One trial's readings by name, and the configuration of its fit."""

    readings: dict[str, Reading]
    configuration: dict[str, Any]


def _placebo(result: Any, cell: Cell, seed: int) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return refute(
            result,
            estimand=cell.estimand,
            tests=("placebo",),
            n_replicates=REFITS,
            random_state=seed,
        )["placebo"]


def trial_readings(name: str, frame: pd.DataFrame, trial: int, *, power: bool) -> Trial:
    """The size reading of one trial, its power reading when ``power``, and its fit."""
    cell = CELLS[name]
    result = fit_cell(name, frame, trial_seed("property_fit", name, trial))
    configuration = trial_configuration(result, cell)
    out = {"placebo": _reading(_placebo(result, cell, trial_seed("refute_size", name, trial)))}
    if power:
        with stale_rows(STALE_SHARE[name]):
            stale = _placebo(result, cell, trial_seed("refute_stale", name, trial))
        out["placebo_stale_rows"] = _reading(stale)
    return Trial(out, configuration)


# ---- the primary phase --------------------------------------------------------------------


def cleverly_rows(
    frame: pd.DataFrame, truth_map: Mapping[str, float], scenario: str, replicate: int, seed: int
) -> list[dict[str, Any]]:
    """The PJ1 fit's contrast as one primary row."""
    result = fit_cell("PJ1", frame, seed)
    estimate = result[CELLS["PJ1"].estimand]
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
