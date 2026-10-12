r"""The declared laws of ``ctmle-oat-per-arm-strata-missing``, and the conditions they meet.

Fixed before any registered run.  The laws extend the X17 active laws of
:mod:`tests.studies.oat_per_arm_laws`.  ``W = (W1, W2, W3)`` is iid ``N(0, 1)``, ``W3`` enters
only the treatment law and the observation laws, so it is an instrument for the outcome, and
each arm reads its own combination of ``(W1, W2)``.  The outcome is binary.

=====================  ============================================================================
law                    the added piece
=====================  ============================================================================
``strata_missing``     ``S`` in ``{0, 1, 2}`` with ``P = (0.5, 0.3, 0.2)``, independent of ``W``.
                       ``logit Qbar_0(a, W, S)`` is the X17 ``binary_active`` index plus
                       ``0.3 S`` at ``a = 0`` and ``0.6 - 0.4 S`` at ``a = 1``.
                       ``logit g_0(1 | W, S) = 0.5 W1 + (1.6 - 0.6 S) W3 + 0.3 S``.
                       ``logit pi_0(a, W, S) = 1.4 + 0.4 a - 0.5 W1 + 0.7 W3 - 0.3 S``
``strata_complete``    ``strata_missing`` with ``pi_0 = 1``
``missing_outcome``    ``strata_missing`` at ``S = 0`` in every piece, fitted without strata
``three_arm_missing``  X17 ``three_arm_active`` with
                       ``logit pi_0(a, W) = 1.4 + 0.3 1{a=1} - 0.3 1{a=2} - 0.5 W1 + 0.6 W3``
``missing_treatment``  ``missing_outcome`` with ``logit P(Delta_A = 1 | A, W) = 1.6 - 0.6 A + 0.5 W3``
=====================  ============================================================================

No coefficient moved from the plan's candidates (``X29-plan.md``, section 3.4).  Each condition
of that section holds at the candidates, as ``_x29_law_design.log`` records and
``tests/unit/test_ctmle_oat_per_arm_strata_missing_study_design.py`` recomputes.  The values
at ``10^6`` draws:

=====  =========================================================================  ===============
cond.  value                                                                      target
=====  =========================================================================  ===============
a      strata design gap per arm 0.0379, 0.0675                                   >= 0.02
b      missing design gap: ``strata_missing`` 0.0646, 0.0713; ``missing_outcome``  >= 0.02
       0.0677, 0.0602; ``three_arm_missing`` 0.0282, 0.0452, 0.0480;
       ``missing_treatment`` 0.121, 0.091; natural course 0.102 to 0.122;
       recording 0.083, 0.097
d      smallest treatment factor 0.0281 (``three_arm_missing``), largest 0.892;    inside
       smallest observation factor 0.150, treatment observation 0.748
e      largest learner distance 1.6e-4 (``three_arm_missing``)                    <= 0.01
f      stratum ``ate`` 0.2315, 0.0981, -0.0395; every arm mean in [0.1, 0.9]       spacing >= 0.03
g      P(smallest stratum >= 400 at ``n = 2,250``) 0.9965                         >= 0.99
r      pass probability at ``d = 0.10``: 0.950 at ``R = 800``, 0.777 at 500         >= 0.90
=====  =========================================================================  ===============

Condition (s) is in :mod:`tests.studies.ctmle_oat_per_arm_strata_missing_properties`.

Every truth is a quadrature.  An arm mean in stratum ``s`` is a one-dimensional Gauss-Hermite
integral over the normal index ``u_a . (W1, W2)``.  ``ey_obs`` is a three-dimensional
Gauss-Hermite integral of ``sum_a g_0(a | W, s) Qbar_0(a, W, s)``.  The weight-tilted truth
splits ``W1`` at zero and integrates each half line by Gauss-Legendre, as X17 does.  ``S`` is
independent of ``W`` and the weight reads ``W1`` only, so the tilt leaves the stratum masses
unchanged.  A marginal truth is the ``P(S = s)`` mixture of the stratum truths.

The factor projections that the conditions read are functions of the scalar index alone.  For
arm ``a`` and stratum ``s``, ``q_a = Qbar_0(a, W, s)`` is a monotone function of
``t = u_a . (W1, W2)``.  Given ``t``, ``W1`` is normal and ``W3`` is independent, so each
projection is a two-dimensional Gauss-Hermite integral.  :func:`projection_grid` evaluates it on
a grid of ``t`` and :func:`projections` interpolates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from scipy.special import expit, logit
from scipy.stats import binom, nct
from scipy.stats import t as student_t
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from tests.studies.oat_per_arm_laws import (
    G_BOUNDS,
    LOGIT_CLIP,
    _half_line_grid,
    cubic_logit_learner,
    logit_basis,
    main_terms_outcome,
    weight,
)

__all__ = [
    "G_BOUNDS",
    "LOGIT_CLIP",
    "cubic_logit_learner",
    "logit_basis",
    "weight",
]

#: The lower bound of every observation factor, ``nuisance_bound`` of every fit.
NUISANCE_BOUND = 0.01

#: The bounds condition (d) declares: the treatment factor inside ``(0.02, 0.98)`` and each
#: observation factor at least ``0.05``.
TREATMENT_FACTOR_RANGE = (0.02, 0.98)
OBSERVATION_FACTOR_FLOOR = 0.05

#: The thresholds of conditions (a), (b), (e), (f), (g), (r) and (s).
DESIGN_GAP_FLOOR = 0.02
LEARNER_DISTANCE_CEILING = 0.01
STRATUM_ATE_SPACING = 0.03
ARM_MEAN_RANGE = (0.1, 0.9)
SMALLEST_STRATUM_COUNT = 400
SMALLEST_STRATUM_PROBABILITY = 0.99
DESIGN_BIAS = 0.10
PASS_PROBABILITY_FLOOR = 0.90

#: Quadrature sizes.
HERMITE_NODES = 80
JOINT_HERMITE_NODES = 40
#: The index grid of :func:`projection_grid`, in standard deviations of the index.
GRID_HALF_WIDTH = 9.0
GRID_POINTS = 1_801


@dataclass(frozen=True)
class ExtensionLaw:
    """One law of the study.

    Parameters
    ----------
    name : str
        The law's name.
    c : tuple of float
        The outcome intercept of each arm, at ``S = 0``.
    u : tuple of tuple of float
        The ``(W1, W2)`` outcome coefficients of each arm.
    v : tuple of tuple of float
        The ``(W1, W3)`` treatment scores of each arm at ``S = 0``; arm 0 is ``(0, 0)``.
    strata : tuple of float
        ``P(S = s)``.  One entry means the law has no strata and the sample carries no ``S``.
    outcome_slope : tuple of float
        Each arm's outcome logit slope in ``S``; empty for none.
    treatment_slope : tuple of float
        Each arm's treatment score slope in ``S``; empty for none.
    instrument_slope : tuple of float
        Each arm's slope in ``S`` of its ``W3`` treatment coefficient; empty for none.
    observation : tuple of float
        The outcome observation logit ``(W1, W3, S)`` coefficients; empty for a complete
        outcome.
    observation_intercept : tuple of float
        The outcome observation logit intercept of each arm.
    recording : tuple of float
        The ``(W3,)`` coefficient of the treatment recording logit; empty for a recorded
        treatment.
    recording_intercept : tuple of float
        The treatment recording logit intercept of each arm.
    weighted : bool
        Whether the truth is under the weight-tilted law of
        :func:`~tests.studies.oat_per_arm_laws.weight`, and the sample carries ``wt``.
    """

    name: str
    c: tuple[float, ...]
    u: tuple[tuple[float, float], ...]
    v: tuple[tuple[float, float], ...]
    strata: tuple[float, ...] = (1.0,)
    outcome_slope: tuple[float, ...] = ()
    treatment_slope: tuple[float, ...] = ()
    instrument_slope: tuple[float, ...] = ()
    observation: tuple[float, ...] = ()
    observation_intercept: tuple[float, ...] = ()
    recording: tuple[float, ...] = ()
    recording_intercept: tuple[float, ...] = ()
    weighted: bool = False

    @property
    def k(self) -> int:
        """The number of arms."""
        return len(self.c)

    @property
    def stratified(self) -> bool:
        """Whether the law has baseline strata."""
        return len(self.strata) > 1

    @property
    def missing_outcome(self) -> bool:
        """Whether the outcome can be missing."""
        return bool(self.observation)

    @property
    def missing_treatment(self) -> bool:
        """Whether the treatment can be missing."""
        return bool(self.recording)

    @property
    def arm_labels(self) -> tuple[str, ...]:
        """The treatment labels, in arm order: the arm indices as text."""
        return tuple(str(a) for a in range(self.k))

    @property
    def stratum_labels(self) -> tuple[str, ...]:
        """The stratum suffix each estimand carries, ``S=s``; empty without strata."""
        return tuple(f"S={s}" for s in range(len(self.strata))) if self.stratified else ()

    def _slope(self, values: tuple[float, ...], arm: int) -> float:
        return float(values[arm]) if values else 0.0

    def outcome_logit(self, arm: int, w1: Any, w2: Any, s: Any) -> np.ndarray:
        """``logit Qbar_0(arm, W, S)``."""
        u = self.u[arm]
        return np.asarray(
            self.c[arm]
            + u[0] * np.asarray(w1, dtype=float)
            + u[1] * np.asarray(w2, dtype=float)
            + self._slope(self.outcome_slope, arm) * np.asarray(s, dtype=float)
        )

    def outcome_probability(self, arm: int, w1: Any, w2: Any, s: Any = 0.0) -> np.ndarray:
        """``Qbar_0(arm, W, S)``."""
        return np.asarray(expit(self.outcome_logit(arm, w1, w2, s)))

    def mechanism(self, w1: Any, w3: Any, s: Any = 0.0) -> np.ndarray:
        """``g_0(a | W, S)``, shape ``(..., K)``, by a softmax with arm 0 the baseline."""
        w1 = np.asarray(w1, dtype=float)
        w3 = np.asarray(w3, dtype=float)
        s = np.broadcast_to(np.asarray(s, dtype=float), np.broadcast(w1, w3).shape)
        scores = np.stack(
            [
                self.v[a][0] * w1
                + (self.v[a][1] + self._slope(self.instrument_slope, a) * s) * w3
                + self._slope(self.treatment_slope, a) * s
                for a in range(self.k)
            ],
            axis=-1,
        )
        scores = scores - scores.max(axis=-1, keepdims=True)
        expo = np.exp(scores)
        return np.asarray(expo / expo.sum(axis=-1, keepdims=True))

    def observation_probability(self, arm: int, w1: Any, w3: Any, s: Any = 0.0) -> np.ndarray:
        """``pi_0(arm, W, S) = P(Delta = 1 | A = arm, Delta_A = 1, W, S)``; one when complete."""
        w1 = np.asarray(w1, dtype=float)
        if not self.missing_outcome:
            return np.ones(np.broadcast(w1, np.asarray(w3)).shape)
        b1, b3, bs = self.observation
        return np.asarray(
            expit(
                self.observation_intercept[arm]
                + b1 * w1
                + b3 * np.asarray(w3, dtype=float)
                + bs * np.asarray(s, dtype=float)
            )
        )

    def recording_probability(self, arm: int, w3: Any) -> np.ndarray:
        """``P(Delta_A = 1 | A = arm, W)``; one when the treatment is always recorded."""
        w3 = np.asarray(w3, dtype=float)
        if not self.missing_treatment:
            return np.ones(w3.shape)
        return np.asarray(expit(self.recording_intercept[arm] + self.recording[0] * w3))

    def sample(self, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
        """``n`` iid rows and the law's truth.

        Parameters
        ----------
        n : int
            Rows.
        seed : int
            The draw's seed.

        Returns
        -------
        tuple of DataFrame and dict
            Columns ``W1, W2, W3`` (then ``S`` with strata), ``A``, ``Y`` (``NaN`` when
            unobserved), ``Delta`` with a missing outcome, ``DA`` with a missing treatment,
            and ``wt`` on a weighted law; and the truth.
        """
        rng = np.random.default_rng(seed)
        w = rng.standard_normal((n, 3))
        if self.stratified:
            stratum = rng.choice(len(self.strata), size=n, p=np.asarray(self.strata))
        else:
            stratum = np.zeros(n, dtype=int)
        probabilities = self.mechanism(w[:, 0], w[:, 2], stratum)
        cumulative = probabilities.cumsum(axis=1)
        draw = rng.random(n)
        arm = np.minimum((draw[:, None] > cumulative).sum(axis=1), self.k - 1)
        q = np.column_stack(
            [self.outcome_probability(a, w[:, 0], w[:, 1], stratum) for a in range(self.k)]
        )
        y = (rng.random(n) < q[np.arange(n), arm]).astype(float)
        frame = pd.DataFrame({"W1": w[:, 0], "W2": w[:, 1], "W3": w[:, 2]})
        if self.stratified:
            frame["S"] = stratum.astype(int)
        labels: np.ndarray = (
            arm.astype(float) if self.k == 2 else np.asarray(self.arm_labels, dtype=object)[arm]
        )
        if self.missing_outcome:
            pi = np.column_stack(
                [self.observation_probability(a, w[:, 0], w[:, 2], stratum) for a in range(self.k)]
            )
            observed = rng.random(n) < pi[np.arange(n), arm]
            y = np.where(observed, y, np.nan)
        if self.missing_treatment:
            recorded_probability = np.column_stack(
                [self.recording_probability(a, w[:, 2]) for a in range(self.k)]
            )
            recorded = rng.random(n) < recorded_probability[np.arange(n), arm]
            labels = np.where(recorded, labels.astype(object), np.nan)
        frame["A"] = labels
        frame["Y"] = y
        if self.missing_outcome:
            frame["Delta"] = observed.astype(float)
        if self.missing_treatment:
            frame["DA"] = recorded.astype(float)
        if self.weighted:
            frame["wt"] = weight(frame["W1"].to_numpy())
        return frame, self.truth()

    def __call__(self, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
        """The coverage-study form of :meth:`sample`."""
        return self.sample(n, seed)

    def stratum_arm_means(self, s: int) -> np.ndarray:
        """``E[Y(a) | S = s]`` for each arm, under the tilted law when :attr:`weighted`."""
        if self.weighted:
            w1, w2, mass = _half_line_grid()
            share = weight(w1)
            total = float(np.sum(mass * share))
            return np.array(
                [
                    float(np.sum(mass * share * self.outcome_probability(a, w1, w2, s)) / total)
                    for a in range(self.k)
                ]
            )
        nodes, weights = hermegauss(HERMITE_NODES)
        weights = weights / weights.sum()
        means = []
        for a in range(self.k):
            scale = float(np.hypot(*self.u[a]))
            shift = self.c[a] + self._slope(self.outcome_slope, a) * s
            means.append(float(np.sum(weights * expit(shift + scale * nodes))))
        return np.array(means)

    def stratum_observed_mean(self, s: int) -> float:
        """``E[sum_a g_0(a | W, s) Qbar_0(a, W, s) | S = s]``, the natural-course mean."""
        nodes, weights = hermegauss(JOINT_HERMITE_NODES)
        weights = weights / weights.sum()
        if self.weighted:
            h1, h2, half = _half_line_grid()
            g = self.mechanism(h1[..., None], nodes[None, None, :], s)
            q = np.stack([self.outcome_probability(a, h1, h2, s) for a in range(self.k)], axis=-1)
            share = weight(h1)
            joint = np.einsum("ijka,ija->ijk", g, q)
            tilted = half[..., None] * share[..., None] * weights[None, None, :]
            return float(np.sum(tilted * joint) / np.sum(half * share))
        w1, w2, w3 = np.meshgrid(nodes, nodes, nodes, indexing="ij")
        mass = weights[:, None, None] * weights[None, :, None] * weights[None, None, :]
        g = self.mechanism(w1, w3, s)
        q = np.stack([self.outcome_probability(a, w1, w2, s) for a in range(self.k)], axis=-1)
        return float(np.sum(mass[..., None] * g * q))

    def _named(self, means: np.ndarray, observed: float, suffix: str) -> dict[str, float]:
        tail = f"[{suffix}]" if suffix else ""
        out: dict[str, float] = {}
        if self.k == 2:
            out.update(
                {
                    f"ey0{tail}": float(means[0]),
                    f"ey1{tail}": float(means[1]),
                    f"ate{tail}": float(means[1] - means[0]),
                    f"rr{tail}": float(means[1] / means[0]),
                    f"or{tail}": float(_odds(means[1]) / _odds(means[0])),
                    f"ey_obs{tail}": observed,
                    f"par{tail}": float(observed - means[0]),
                    f"paf{tail}": float(1.0 - means[0] / observed),
                }
            )
            return out
        if suffix:
            raise ValueError("the study declares no stratified law with more than two arms")
        labels = self.arm_labels
        reference = labels[0]
        for a, label in enumerate(labels):
            out[f"ey[{label}]"] = float(means[a])
        for a, label in enumerate(labels[1:], start=1):
            pair = f"{label} vs {reference}"
            out[f"ate[{pair}]"] = float(means[a] - means[0])
            out[f"rr[{pair}]"] = float(means[a] / means[0])
            out[f"or[{pair}]"] = float(_odds(means[a]) / _odds(means[0]))
        out["ey_obs"] = observed
        out[f"par[{reference}]"] = float(observed - means[0])
        out[f"paf[{reference}]"] = float(1.0 - means[0] / observed)
        return out

    def truth(self) -> dict[str, float]:
        """Every parameter the study reports, named as the fit names it.

        Returns
        -------
        dict
            The marginal arm means, contrasts, ``ey_obs``, ``par`` and ``paf``, and with
            strata the same names suffixed ``[S=s]`` for each stratum.
        """
        mass = np.asarray(self.strata, dtype=float)
        stratum_means = [self.stratum_arm_means(s) for s in range(len(mass))]
        stratum_observed = [self.stratum_observed_mean(s) for s in range(len(mass))]
        means = np.sum(mass[:, None] * np.vstack(stratum_means), axis=0)
        observed = float(np.sum(mass * np.asarray(stratum_observed)))
        out = self._named(means, observed, "")
        for s, label in enumerate(self.stratum_labels):
            out.update(self._named(stratum_means[s], stratum_observed[s], label))
        return out


def _odds(p: float) -> float:
    return p / (1.0 - p)


BINARY_C = (-0.3, 0.9)
BINARY_U = ((0.8, -0.8), (0.8, 0.8))
BINARY_V = ((0.0, 0.0), (0.5, 1.6))
BINARY_OBSERVATION = (-0.5, 0.7, -0.3)
BINARY_OBSERVATION_INTERCEPT = (1.4, 1.8)

STRATA_MISSING = ExtensionLaw(
    name="strata_missing",
    c=BINARY_C,
    u=BINARY_U,
    v=BINARY_V,
    strata=(0.5, 0.3, 0.2),
    outcome_slope=(0.3, -0.4),
    treatment_slope=(0.0, 0.3),
    instrument_slope=(0.0, -0.6),
    observation=BINARY_OBSERVATION,
    observation_intercept=BINARY_OBSERVATION_INTERCEPT,
)

#: :data:`STRATA_MISSING` under the declared known weight ``w(W) = 0.5 + 1{W1 > 0}``.
STRATA_MISSING_WEIGHTED = ExtensionLaw(
    name="strata_missing_weighted",
    c=BINARY_C,
    u=BINARY_U,
    v=BINARY_V,
    strata=(0.5, 0.3, 0.2),
    outcome_slope=(0.3, -0.4),
    treatment_slope=(0.0, 0.3),
    instrument_slope=(0.0, -0.6),
    observation=BINARY_OBSERVATION,
    observation_intercept=BINARY_OBSERVATION_INTERCEPT,
    weighted=True,
)

STRATA_COMPLETE = ExtensionLaw(
    name="strata_complete",
    c=BINARY_C,
    u=BINARY_U,
    v=BINARY_V,
    strata=(0.5, 0.3, 0.2),
    outcome_slope=(0.3, -0.4),
    treatment_slope=(0.0, 0.3),
    instrument_slope=(0.0, -0.6),
)

MISSING_OUTCOME = ExtensionLaw(
    name="missing_outcome",
    c=BINARY_C,
    u=BINARY_U,
    v=BINARY_V,
    observation=BINARY_OBSERVATION,
    observation_intercept=BINARY_OBSERVATION_INTERCEPT,
)

THREE_ARM_MISSING = ExtensionLaw(
    name="three_arm_missing",
    c=(-0.4, 0.0, 0.4),
    u=((0.7, 0.7), (0.7, -0.7), (-0.8, 0.5)),
    v=((0.0, 0.0), (0.9, 1.0), (-0.3, -1.0)),
    observation=(-0.5, 0.6, 0.0),
    observation_intercept=(1.4, 1.7, 1.1),
)

MISSING_TREATMENT = ExtensionLaw(
    name="missing_treatment",
    c=BINARY_C,
    u=BINARY_U,
    v=BINARY_V,
    observation=BINARY_OBSERVATION,
    observation_intercept=BINARY_OBSERVATION_INTERCEPT,
    recording=(0.5,),
    recording_intercept=(1.6, 1.0),
)

LAWS: Mapping[str, ExtensionLaw] = {
    law.name: law
    for law in (
        STRATA_MISSING,
        STRATA_MISSING_WEIGHTED,
        STRATA_COMPLETE,
        MISSING_OUTCOME,
        THREE_ARM_MISSING,
        MISSING_TREATMENT,
    )
}

#: The laws whose factors the estimator fits, one per sampled configuration.  The weighted
#: twin draws the same rows as :data:`STRATA_MISSING` and differs only in its truth.
DESIGN_LAWS = (
    STRATA_MISSING,
    STRATA_COMPLETE,
    MISSING_OUTCOME,
    THREE_ARM_MISSING,
    MISSING_TREATMENT,
)


# --------------------------------------------------------------------------- the learners


def stratum_dummies(column: np.ndarray, strata: int) -> np.ndarray:
    """Drop-first indicators of a stratum column with ``strata`` levels."""
    return np.column_stack([(column == s).astype(float) for s in range(1, strata)])


def arm_stratum_interactions(design: Any, *, indicators: int, strata: int) -> np.ndarray:
    """``[I, W, D, I x W, I x D]`` from ``[I, W1, W2, W3]`` or ``[I, W1, W2, W3, S]``.

    ``I`` are the ``indicators`` arm columns of
    :meth:`~cleverly.data.CausalData.treatment_design`, ``W`` the three covariates and ``D`` the
    drop-first indicators of ``S`` when ``strata > 1``.

    Parameters
    ----------
    design : array-like
        The outcome design.
    indicators : int
        The number of arm columns, ``K - 1``.
    strata : int
        The number of strata; ``1`` for a design without ``S``.

    Returns
    -------
    ndarray
        The expanded design.
    """
    matrix = np.asarray(design, dtype=float)
    arms = matrix[:, :indicators]
    covariates = matrix[:, indicators : indicators + 3]
    blocks = [covariates]
    if strata > 1:
        blocks.append(stratum_dummies(matrix[:, indicators + 3], strata))
    main = np.column_stack(blocks)
    products = (arms[:, :, None] * main[:, None, :]).reshape(matrix.shape[0], -1)
    return np.column_stack([arms, main, products])


def extension_outcome(k: int, strata: int) -> Pipeline:
    """The correctly specified outcome model: an unpenalized logistic GLM with arm interactions.

    Parameters
    ----------
    k : int
        The number of arms.
    strata : int
        The number of strata; ``1`` for a law without ``S``.

    Returns
    -------
    Pipeline
        The interaction transform, then the logistic regression.
    """
    return Pipeline(
        [
            (
                "interactions",
                FunctionTransformer(
                    arm_stratum_interactions, kw_args={"indicators": k - 1, "strata": strata}
                ),
            ),
            ("model", main_terms_outcome()),
        ]
    )


def confounder_omitted_design(design: Any, *, indicators: int, strata: int) -> np.ndarray:
    """``[I, W2, W3, D]``: the main terms without ``W1``, the confounder."""
    matrix = np.asarray(design, dtype=float)
    blocks = [matrix[:, :indicators], matrix[:, indicators + 1 : indicators + 3]]
    if strata > 1:
        blocks.append(stratum_dummies(matrix[:, indicators + 3], strata))
    return np.column_stack(blocks)


def confounder_omitted_outcome(k: int, strata: int) -> Pipeline:
    """X17's wrong outcome model, main terms without ``W1``, with the stratum indicators.

    Parameters
    ----------
    k : int
        The number of arms.
    strata : int
        The number of strata.

    Returns
    -------
    Pipeline
        The column selection, then an unpenalized logistic regression.
    """
    return Pipeline(
        [
            (
                "drop",
                FunctionTransformer(
                    confounder_omitted_design, kw_args={"indicators": k - 1, "strata": strata}
                ),
            ),
            ("model", main_terms_outcome()),
        ]
    )


def missingness_glm() -> LogisticRegression:
    """The full-covariate observation model the tilt reads: an unpenalized logistic GLM."""
    return LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-10, max_iter=1000)


# ------------------------------------------------------------------- the factor projections

#: The projections :func:`projection_grid` evaluates, by name.  Each is
#: ``E[f(W1, W3) | t, S = s]`` for one integrand ``f``.
PROJECTION_NAMES = ("treated", "recorded_treated", "observed", "recorded")


def _integrands(law: ExtensionLaw, arm: int, s: int) -> dict[str, Callable[..., np.ndarray]]:
    def treated(w1: np.ndarray, w3: np.ndarray) -> np.ndarray:
        return np.asarray(law.mechanism(w1, w3, s)[..., arm])

    def recorded_treated(w1: np.ndarray, w3: np.ndarray) -> np.ndarray:
        return treated(w1, w3) * law.recording_probability(arm, w3)

    def observed(w1: np.ndarray, w3: np.ndarray) -> np.ndarray:
        return recorded_treated(w1, w3) * law.observation_probability(arm, w1, w3, s)

    def recorded(w1: np.ndarray, w3: np.ndarray) -> np.ndarray:
        g = law.mechanism(w1, w3, s)
        return np.asarray(sum(g[..., b] * law.recording_probability(b, w3) for b in range(law.k)))

    return {
        "treated": treated,
        "recorded_treated": recorded_treated,
        "observed": observed,
        "recorded": recorded,
    }


def index_scale(law: ExtensionLaw, arm: int) -> float:
    """The standard deviation of the index ``t = u_a . (W1, W2)``."""
    return float(np.hypot(*law.u[arm]))


@lru_cache(maxsize=64)
def projection_grid(law: ExtensionLaw, arm: int, s: int) -> dict[str, np.ndarray]:
    """Each projection of :data:`PROJECTION_NAMES` on a grid of the index, by Gauss-Hermite.

    Cached: a law is frozen, so its grid never changes, and every condition reads it.

    Parameters
    ----------
    law : ExtensionLaw
        The law.
    arm : int
        The arm.
    s : int
        The stratum.

    Returns
    -------
    dict
        ``"t"`` (the grid) and one array per projection.
    """
    u = np.asarray(law.u[arm], dtype=float)
    scale = index_scale(law, arm)
    t = np.linspace(-GRID_HALF_WIDTH * scale, GRID_HALF_WIDTH * scale, GRID_POINTS)
    variance = float(u @ u)
    mean_w1 = u[0] / variance * t
    sd_w1 = float(np.sqrt(max(1.0 - u[0] ** 2 / variance, 0.0)))
    nodes, weights = hermegauss(JOINT_HERMITE_NODES)
    weights = weights / weights.sum()
    w1, w3 = np.broadcast_arrays(
        mean_w1[:, None, None] + sd_w1 * nodes[None, :, None], nodes[None, None, :]
    )
    mass = weights[:, None] * weights[None, :]
    out: dict[str, np.ndarray] = {"t": t}
    for name, integrand in _integrands(law, arm, s).items():
        out[name] = np.einsum("gij,ij->g", integrand(w1, w3), mass)
    return out


def projections(grid: Mapping[str, np.ndarray], t: np.ndarray) -> dict[str, np.ndarray]:
    """The projections of ``grid`` at the index values ``t``, by linear interpolation."""
    return {name: np.interp(t, grid["t"], grid[name]) for name in PROJECTION_NAMES if name in grid}


def factors(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The three factor projections, and their product, from :func:`projections`.

    Returns
    -------
    dict
        ``treatment`` ``P(A = a | Delta_A = 1, q_a)``, ``outcome_observation``
        ``P(Delta = 1 | A = a, Delta_A = 1, q_a)``, ``treatment_observation``
        ``P(Delta_A = 1 | q_a)`` and ``product`` ``P(C_a = 1 | q_a)``.
    """
    return {
        "treatment": values["recorded_treated"] / values["recorded"],
        "outcome_observation": values["observed"] / values["recorded_treated"],
        "treatment_observation": values["recorded"],
        "product": values["observed"],
    }


@dataclass(frozen=True)
class Draws:
    """A Monte Carlo sample of ``(W, S)`` from one law, with each arm's index.

    Parameters
    ----------
    w1 : ndarray
        ``W1``.
    w2 : ndarray
        ``W2``.
    w3 : ndarray
        ``W3``.
    s : ndarray
        The stratum.
    """

    w1: np.ndarray
    w2: np.ndarray
    w3: np.ndarray
    s: np.ndarray

    def index(self, law: ExtensionLaw, arm: int) -> np.ndarray:
        """``t = u_a . (W1, W2)`` at each draw."""
        u = law.u[arm]
        return np.asarray(u[0] * self.w1 + u[1] * self.w2)


def draw(law: ExtensionLaw, size: int, seed: int) -> Draws:
    """``size`` draws of ``(W, S)`` from ``law``."""
    rng = np.random.default_rng(seed)
    w1, w2, w3 = rng.standard_normal((3, size))
    if law.stratified:
        s = rng.choice(len(law.strata), size=size, p=np.asarray(law.strata))
    else:
        s = np.zeros(size, dtype=int)
    return Draws(w1, w2, w3, s)


def factor_values(law: ExtensionLaw, sample: Draws) -> dict[tuple[int, str], np.ndarray]:
    """Each factor projection of each arm at each draw, inside the draw's own stratum.

    Returns
    -------
    dict
        ``(arm, factor)`` to an array over the draws, for the factors of :func:`factors`.
    """
    out: dict[tuple[int, str], np.ndarray] = {}
    for arm in range(law.k):
        t = sample.index(law, arm)
        columns: dict[str, np.ndarray] = {}
        for s in range(len(law.strata)):
            rows = sample.s == s
            values = factors(projections(projection_grid(law, arm, s), t[rows]))
            for name, value in values.items():
                columns.setdefault(name, np.full(t.size, np.nan))[rows] = value
        for name, value in columns.items():
            out[arm, name] = value
    return out


# ------------------------------------------------------------------------- the conditions


def _l2(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        return float(np.sqrt(np.mean(values**2)))
    return float(np.sqrt(np.sum(weights * values**2) / np.sum(weights)))


def strata_design_gap(law: ExtensionLaw, sample: Draws) -> list[float]:
    """Condition (a): ``L2(P(A = a | q_a, S), P(A = a | q_a))`` per arm.

    ``q_a = Qbar_0(a, W, S)``, so a value of ``q_a`` is reached by a different index in each
    stratum.  The pooled projection mixes the stratum projections at those indices, each
    weighted by ``P(S = s)`` times the normal density of its index.

    Parameters
    ----------
    law : ExtensionLaw
        A stratified law.
    sample : Draws
        Its Monte Carlo sample.

    Returns
    -------
    list of float
        The gap of each arm.
    """
    gaps = []
    for arm in range(law.k):
        scale = index_scale(law, arm)
        grids = [projection_grid(law, arm, s) for s in range(len(law.strata))]
        t = sample.index(law, arm)
        level = law.outcome_logit(arm, sample.w1, sample.w2, sample.s)
        stratified = np.empty(t.size)
        numerator = np.zeros(t.size)
        denominator = np.zeros(t.size)
        for s, grid in enumerate(grids):
            own = sample.s == s
            stratified[own] = factors(projections(grid, t[own]))["treatment"]
            shift = law.c[arm] + law._slope(law.outcome_slope, arm) * s
            index = level - shift
            density = law.strata[s] * np.exp(-0.5 * (index / scale) ** 2)
            numerator += density * factors(projections(grid, index))["treatment"]
            denominator += density
        gaps.append(_l2(stratified - numerator / denominator))
    return gaps


def missing_design_gap(law: ExtensionLaw, sample: Draws) -> list[float]:
    """Condition (b), arm means: ``L2(P(C_a = 1 | q_a), P(A = a | q_a) pi_0(a, W))`` per arm.

    The second is the shipped construction's limit: the per-arm treatment projection times
    the full-covariate observation probability.
    """
    values = factor_values(law, sample)
    gaps = []
    for arm in range(law.k):
        pi = law.observation_probability(arm, sample.w1, sample.w3, sample.s)
        shipped = values[arm, "treatment"] * pi
        gaps.append(_l2(values[arm, "product"] - shipped))
    return gaps


def natural_course_design_gap(law: ExtensionLaw, sample: Draws) -> float:
    """Condition (b), natural course: ``L2(P(Delta = 1 | A, Qbar(A, W)), pi_0(A, W))``.

    The ``L2`` norm is under the law of ``(A, W)``, so each arm's squared gap is weighted by
    ``g_0(a | W)``.
    """
    values = factor_values(law, sample)
    g = law.mechanism(sample.w1, sample.w3, sample.s)
    total = 0.0
    for arm in range(law.k):
        pi = law.observation_probability(arm, sample.w1, sample.w3, sample.s)
        total += float(np.mean(g[:, arm] * (values[arm, "outcome_observation"] - pi) ** 2))
    return float(np.sqrt(total))


def recording_design_gap(law: ExtensionLaw, sample: Draws) -> list[float]:
    """Condition (b), missing treatment: the three-factor product against the two-factor one.

    The two-factor product ignores ``Delta_A``: it is ``P(C_a = 1 | q_a) / P(Delta_A = 1 | q_a)``.
    """
    values = factor_values(law, sample)
    return [
        _l2(values[arm, "product"] - values[arm, "product"] / values[arm, "treatment_observation"])
        for arm in range(law.k)
    ]


def factor_ranges(law: ExtensionLaw, sample: Draws) -> dict[str, tuple[float, float]]:
    """Condition (d): the smallest and largest value of each factor over arms and draws."""
    values = factor_values(law, sample)
    out: dict[str, tuple[float, float]] = {}
    names = ["treatment"]
    if law.missing_outcome:
        names.append("outcome_observation")
    if law.missing_treatment:
        names.append("treatment_observation")
    for name in names:
        stacked = np.concatenate([values[arm, name] for arm in range(law.k)])
        out[name] = (float(np.min(stacked)), float(np.max(stacked)))
    return out


def factors_inside_bounds(ranges: Mapping[str, tuple[float, float]]) -> bool:
    """Whether every factor range meets condition (d)."""
    low, high = TREATMENT_FACTOR_RANGE
    for name, (smallest, largest) in ranges.items():
        if name == "treatment":
            if not low < smallest <= largest < high:
                return False
        elif smallest < OBSERVATION_FACTOR_FLOOR:
            return False
    return True


def _fitting_weight(law: ExtensionLaw, sample: Draws, arm: int, name: str) -> np.ndarray:
    """The share of each draw among the rows a factor is fitted on."""
    if name == "treatment_observation":
        return np.ones(sample.w1.size)
    g = law.mechanism(sample.w1, sample.w3, sample.s)
    if name == "treatment":
        return np.asarray(
            sum(g[:, b] * law.recording_probability(b, sample.w3) for b in range(law.k))
        )
    return np.asarray(g[:, arm] * law.recording_probability(arm, sample.w3))


def learner_limit(q: np.ndarray, target: np.ndarray, share: np.ndarray) -> np.ndarray:
    """The cubic-logit learner's population limit, fitted to ``target`` under ``share``.

    The limit is the unpenalized logistic fit of ``target`` on :func:`logit_basis` of ``q``,
    each draw weighted by ``share``, the density of the rows the factor is fitted on.

    Parameters
    ----------
    q : ndarray
        The design column at each draw.
    target : ndarray
        The factor projection at each draw.
    share : ndarray
        The fitting weight of each draw.

    Returns
    -------
    ndarray
        The limit at each draw.
    """
    basis = logit_basis(q)
    x = np.vstack([basis, basis])
    y = np.r_[np.ones(q.size), np.zeros(q.size)]
    w = np.r_[share * target, share * (1.0 - target)]
    model = LogisticRegression(C=np.inf, solver="newton-cholesky", tol=1e-10, max_iter=1000)
    model.fit(x, y, sample_weight=w)
    return np.asarray(model.predict_proba(basis)[:, 1])


def learner_distances(law: ExtensionLaw, sample: Draws) -> dict[tuple[int, int, str], float]:
    """Condition (e): the learner-limit distance of each arm, stratum and factor.

    The distance is the ``L2`` norm, under the rows the factor is fitted on, between the
    projection and the cubic-logit learner's limit, inside each stratum, as the estimator
    fits it.

    Returns
    -------
    dict
        ``(arm, stratum, factor)`` to the distance.
    """
    values = factor_values(law, sample)
    names = ["treatment"]
    if law.missing_outcome:
        names.append("outcome_observation")
    if law.missing_treatment:
        names.append("treatment_observation")
    out: dict[tuple[int, int, str], float] = {}
    for arm in range(law.k):
        q = law.outcome_probability(arm, sample.w1, sample.w2, sample.s)
        for name in names:
            share = _fitting_weight(law, sample, arm, name)
            for s in range(len(law.strata)):
                rows = sample.s == s
                target = values[arm, name][rows]
                fitted = learner_limit(q[rows], target, share[rows])
                out[arm, s, name] = _l2(fitted - target, share[rows])
    return out


def arm_means_inside(law: ExtensionLaw) -> bool:
    """Condition (f), second part: every marginal and stratum arm mean inside ``[0.1, 0.9]``."""
    truth = law.truth()
    low, high = ARM_MEAN_RANGE
    means = [value for name, value in truth.items() if name.startswith("ey") and "obs" not in name]
    return all(low <= value <= high for value in means)


def stratum_ate_spacing(law: ExtensionLaw) -> float:
    """Condition (f), first part: the smallest distance between two stratum ``ate`` truths."""
    truth = law.truth()
    values = sorted(truth[f"ate[{label}]"] for label in law.stratum_labels)
    return float(min(np.diff(values)))


def smallest_stratum_probability(n: int, share: float) -> float:
    """Condition (g): ``P(Binomial(n, share) >= 400)``."""
    return float(binom.sf(SMALLEST_STRATUM_COUNT - 1, n, share))


def equivalence_pass_probability(replicates: int, standardized_bias: float) -> float:
    """Condition (r): the pass probability of the shipped bias rule at a true bias ``d``.

    ``standardized_bias_verdict`` passes when the 99% Student interval of the mean error lies
    inside ``+-0.25`` empirical standard deviations, that is when
    ``|T| <= 0.25 sqrt(R) - t_{0.995, R - 1}`` with ``T ~ nct(R - 1, d sqrt(R))`` for normal
    errors.

    Parameters
    ----------
    replicates : int
        ``R``.
    standardized_bias : float
        ``d``, the true bias over the true standard deviation.

    Returns
    -------
    float
        The probability.
    """
    bound = 0.25 * np.sqrt(replicates) - float(student_t.ppf(0.995, replicates - 1))
    if bound <= 0.0:
        return 0.0
    ncp = standardized_bias * np.sqrt(replicates)
    return float(nct.cdf(bound, replicates - 1, ncp) - nct.cdf(-bound, replicates - 1, ncp))


def discrimination_pass_probability(replicates: int, standardized_bias: float) -> float:
    """Condition (s): the probability that a control's bias interval clears the margin.

    ``bias_discriminated`` holds when the 99% Student interval lies outside ``+-0.25``
    empirical standard deviations: ``|T| > 0.25 sqrt(R) + t_{0.995, R - 1}``.

    Parameters
    ----------
    replicates : int
        ``R``.
    standardized_bias : float
        ``d``, the control's population bias over its standard deviation.

    Returns
    -------
    float
        The probability.
    """
    bound = 0.25 * np.sqrt(replicates) + float(student_t.ppf(0.995, replicates - 1))
    ncp = abs(standardized_bias) * np.sqrt(replicates)
    return float(nct.sf(bound, replicates - 1, ncp) + nct.cdf(-bound, replicates - 1, ncp))


def logit_q(q: np.ndarray) -> np.ndarray:
    """``logit`` of ``q`` clipped as the learner basis clips it."""
    return np.asarray(logit(np.clip(q, LOGIT_CLIP, 1.0 - LOGIT_CLIP)))
