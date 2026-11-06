r"""Declaration and driver of the ``conditional-projection`` study (P1 plan, section 5.1).

The study measures the projection TMLE of a :class:`cleverly.ConditionalContrast` with
``model=Linear(["W1", "W3"])``: interval calibration of the three coefficients on both
scales, under a known and an estimated mechanism, three arms, weights with the ensemble
learner, clusters, missing outcomes, and the assessment of a fixed external score; a
slow-nuisance ladder for the product remainder; the size and power of the heterogeneity
test; and controls that must fail.  Everything a run reads is fixed here before it starts:

- the laws, as samplers with the plan's coefficients (:func:`sample`);
- the exact truths, by Gauss--Legendre quadrature over ``U(-1, 1)^3`` (Gauss--Hermite over
  the cluster effect), whose error is far below any margin (:func:`quadrature_error`);
- the fit groups (:data:`GROUPS`), each a law, ``n``, ``R``, learner and the property cells
  its fits fill, with the controls that share its fits;
- the design numbers (:func:`design_numbers`): each calibration row's pass probability at
  its ``R``, the family-wise rate, the heterogeneity power and its pass probability, the
  type-I pass probability and each control's detection probability, computed rather than
  copied, and the cost of one run;
- the study record (:data:`STUDY`) with its margins, failure bound, publication policy and
  per-group seed streams keyed by label.

The run is ``python -m tests.canonical.conditional_projection.regenerate`` (the declared
form of :mod:`tests.canonical.declared_run`); its docstring holds the red-cell reading.  The
study is registered in :func:`tests.studies.evidence.registry.registered` with the artifacts
of its declared run, which the fast tier reads.

What the cells can and cannot see (design review B4, B5):

- ``weighted_sl_2k`` uses a weight that moves the estimand by about half a standard deviation,
  and its ``weights_ignored`` control fits the same draws without the weights, read against
  the weighted truth.
- the missing-outcome calibration cells use a consistent outcome learner.  The ``1 / Pi``
  term multiplies the outcome error, so with a consistent outcome its loss barely moves the
  point (the re-review's mutation of the clever covariate moved the ``missing_dr_4k`` rows by
  at most 0.15 SD), but it moves the variance: the SE ratios of ``obs_missing_4k`` fell to
  0.76 to 0.84, so these calibration rows see that defect.  The ``missing_dr_4k`` group gives
  the outcome a declared error that is correlated with ``Pi``, where the TMLE must stay
  unbiased and the one-step without ``1 / Pi`` must not.
- the slow ladder claims bias, with the slope over its rungs reported, and no interval
  calibration (re-review RB2): its nuisances are wrong on purpose, and the plug-in variance of
  a fit with a deliberately wrong estimated treatment model overstates the spread (pilot SE
  ratios up to 1.06 at 1k), which the band would read as a defect.  Coverage is still
  computed and published on its bias rows, ungated.
- the ``iid_se`` control publishes the intercept and ``W1`` only (re-review RB1): ``W3`` varies
  within clusters, so the row-level SE of its slope is close to the clustered one (pilot
  ratio in :data:`PILOT_CSV`) and could not fall below the band.
- ``ep_fixed_obs_4k`` (plan 5.1, finding m7) is not here: the EP estimator refuses until part
  (b), so the cell moves to the ``ep-cate`` study of that part.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss
from scipy import stats
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from cleverly import CausalStudy, ConditionalContrast, PointTreatment, SuperLearner
from cleverly.bases import Linear, Modifiers
from cleverly.effect_curves import _pseudo_outcomes
from cleverly.learners.library import default_library
from tests.parallel import STUDY_JOBS
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed

SLUG = "conditional-projection"

#: The declared run's seed and its resampling seed.
SEED = 20261105
RESAMPLING_SEED = 20261106

#: The modifiers and the basis of every cell.
MODIFIERS = Modifiers(["W1", "W3"])
BASIS = Linear(["W1", "W3"])
TERMS = ("(intercept)", "W1", "W3")

#: The cell-name label of each term: a cell name holds no parenthesis.
TERM_LABELS = {"(intercept)": "intercept", "W1": "W1", "W3": "W3"}

#: Gauss--Legendre nodes per axis of ``W``, and Gauss--Hermite nodes of a cluster effect.
QUADRATURE_NODES = 48
HERMITE_NODES = 40

#: The slow-nuisance ladder's offset ``c n^{-1/3} h(W)``.
SLOW_CONSTANT = 5.0

#: The two-sided 95% critical value every interval of the study uses.
CRITICAL = float(stats.norm.ppf(0.975))


# ----------------------------------------------------------------- the laws


def blip(w: np.ndarray) -> np.ndarray:
    """The arm-1 logit increment ``0.3 + 0.5 W1 + 0.4 sin(pi W3)``."""
    return 0.3 + 0.5 * w[..., 0] + 0.4 * np.sin(np.pi * w[..., 2])


def outcome_logit(w: np.ndarray, a: Any) -> np.ndarray:
    """``logit mu(a, W) = -0.3 + 0.6 W1 - 0.4 W2 + a (0.3 + 0.5 W1 + 0.4 sin(pi W3))``."""
    return -0.3 + 0.6 * w[..., 0] - 0.4 * w[..., 1] + np.asarray(a) * blip(w)


def propensity(w: np.ndarray) -> np.ndarray:
    """``logit pi = 0.4 W1 - 0.5 W2 + 0.3 W3``, in about ``[0.3, 0.7]``."""
    return expit(0.4 * w[..., 0] - 0.5 * w[..., 1] + 0.3 * w[..., 2])


def noweight_propensity(w: np.ndarray) -> np.ndarray:
    """The ``noweight_law`` mechanism ``logit pi = 2.5 W1``, in ``[0.076, 0.924]``."""
    return expit(2.5 * w[..., 0])


def multi_propensity(w: np.ndarray) -> np.ndarray:
    """The three-arm mechanism: a softmax of ``(0, 0.3 W1 - 0.2 W2, 0.4 W1 + 0.3 W3)``."""
    scores = np.stack(
        [
            np.zeros(w.shape[:-1]),
            0.3 * w[..., 0] - 0.2 * w[..., 1],
            0.4 * w[..., 0] + 0.3 * w[..., 2],
        ],
        axis=-1,
    )
    scores = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return scores / scores.sum(axis=-1, keepdims=True)


def observation(w: np.ndarray, a: Any) -> np.ndarray:
    """``logit Pi(a, W) = 1.7 + 0.4 W1 + 0.5 W3 - 0.4 a``, in ``[0.60, 0.93]``.

    ``Pi`` depends on ``W3``, the direction the ``missing_dr`` outcome error loads on, so a
    one-step without ``1 / Pi`` is displaced there (design review B5).
    """
    return expit(1.7 + 0.4 * w[..., 0] + 0.5 * w[..., 2] - 0.4 * np.asarray(a))


def sampling_weight(w: np.ndarray) -> np.ndarray:
    """``w = 0.25 + 1.75 expit(6 W3)``, the ``weighted`` law's sampling weights.

    The ratio of the largest to the smallest weight is about 7.8, and the weighted truth sits
    about half a standard deviation from the unweighted one at ``n = 2,000`` on the intercept
    (design review B4), so a fit that drops the weights is displaced.
    """
    return 0.25 + 1.75 * expit(6.0 * w[..., 2])


def score(w: np.ndarray) -> np.ndarray:
    """The ``assess`` cell's declared, non-optimal score ``s(V) = 0.2 + 0.3 W1``."""
    return 0.2 + 0.3 * w[..., 0]


def hetero_mean(w: np.ndarray, a: Any, slope: float) -> np.ndarray:
    """``mu(a, W) = 0.4 + 0.1 W2 + a (0.2 + s W1)`` on the risk scale."""
    return 0.4 + 0.1 * w[..., 1] + np.asarray(a) * (0.2 + slope * w[..., 0])


def slow_offset(w: np.ndarray) -> np.ndarray:
    """``h(W) = W1^2 - 1/3 + W1 W3``, which breaks the level sets of both true indices."""
    return w[..., 0] ** 2 - 1.0 / 3.0 + w[..., 0] * w[..., 2]


def declared_error(kind: str, w: np.ndarray, a: Any) -> np.ndarray:
    """The logit error of a declared wrong outcome regression, by arm.

    ``noweight``: ``1.5 (W1^2 - 1/3)`` in arm 1 and ``-W1`` in arm 0 (plan 5.1).
    ``missing``: ``0.8 W3 + 0.6 sin(pi W3)`` in arm 1 and ``-0.6 W3`` in arm 0, which loads on
    the direction ``Pi`` moves in.
    """
    a = np.asarray(a, dtype=float)
    if kind == "noweight":
        one, zero = 1.5 * (w[..., 0] ** 2 - 1.0 / 3.0), -w[..., 0]
    elif kind == "missing":
        one = 0.8 * w[..., 2] + 0.6 * np.sin(np.pi * w[..., 2])
        zero = -0.6 * w[..., 2]
    else:
        raise KeyError(kind)
    return a * one + (1.0 - a) * zero


def _frame(w: np.ndarray, a: np.ndarray, y: np.ndarray, **extra: Any) -> pd.DataFrame:
    return pd.DataFrame({"Y": y, "A": a, "W1": w[:, 0], "W2": w[:, 1], "W3": w[:, 2], **extra})


def sample(law: str, n: int, rng: np.random.Generator, *, slope: float = 0.0) -> pd.DataFrame:
    """Draw ``n`` rows of ``law``.

    Parameters
    ----------
    law : str
        A key of :data:`LAWS`.
    n : int
        Rows.
    rng : numpy.random.Generator
        The replication's generator.
    slope : float
        The ``hetero_trial`` slope ``s``.

    Returns
    -------
    pandas.DataFrame
        ``Y``, ``A``, ``W1``, ``W2``, ``W3`` and the law's own columns.
    """
    if law == "clustered":
        clusters = n // 5
        cluster = np.repeat(np.arange(clusters), 5)
        w = rng.uniform(-1.0, 1.0, (clusters * 5, 3))
        w[:, 0] = np.repeat(rng.uniform(-1.0, 1.0, clusters), 5)
        a = np.repeat(rng.binomial(1, expit(0.6 * w[::5, 0])), 5)
        u = np.repeat(rng.normal(0.0, 1.0, clusters), 5)
        y = rng.binomial(1, expit(outcome_logit(w, a) + u))
        return _frame(w, a, y, cluster=cluster)
    w = rng.uniform(-1.0, 1.0, (n, 3))
    if law == "hetero_trial":
        a = rng.binomial(1, 0.5, n)
        y = rng.binomial(1, hetero_mean(w, a, slope))
        return _frame(w, a, y, p0=np.full(n, 0.5), p1=np.full(n, 0.5))
    if law == "multi_arm":
        probabilities = multi_propensity(w)
        a = (rng.uniform(size=(n, 1)) > np.cumsum(probabilities, axis=1)).sum(axis=1)
        y = rng.binomial(1, expit(outcome_logit(w, a / 2.0)))
        return _frame(w, a, y)
    if law == "trial":
        a = rng.binomial(1, 0.5, n)
    elif law == "noweight_law":
        a = rng.binomial(1, noweight_propensity(w))
    else:
        a = rng.binomial(1, propensity(w))
    y = rng.binomial(1, expit(outcome_logit(w, a))).astype(float)
    extra: dict[str, Any] = {}
    if law == "trial":
        extra = {"p0": np.full(n, 0.5), "p1": np.full(n, 0.5)}
    if law == "missing":
        observed = rng.binomial(1, observation(w, a)).astype(float)
        y = np.where(observed == 1.0, y, np.nan)
        extra = {"Delta": observed}
    if law == "weighted":
        extra = {"weight": sampling_weight(w)}
    if law == "assess":
        extra = {"S": score(w)}
    return _frame(w, a, y, **extra)


#: Every law of the study.
LAWS: tuple[str, ...] = (
    "trial",
    "observational",
    "slow_both",
    "noweight_law",
    "crr",
    "multi_arm",
    "weighted",
    "clustered",
    "missing",
    "assess",
    "hetero_trial",
)


# --------------------------------------------------------------- the truths


def _grid(nodes: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``U(-1, 1)^3`` quadrature nodes ``(m, 3)`` and their probability weights ``(m,)``."""
    points, weights = leggauss(QUADRATURE_NODES if nodes is None else nodes)
    mesh = np.stack(np.meshgrid(points, points, points, indexing="ij"), axis=-1).reshape(-1, 3)
    mass = np.einsum("i,j,k->ijk", weights, weights, weights).reshape(-1) / 8.0
    return mesh, mass


def arm_means(law: str, w: np.ndarray, *, slope: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """``(mu_t(W), mu_r(W))`` of the law's contrasted pair, marginal over a cluster effect."""
    if law == "hetero_trial":
        return hetero_mean(w, 1, slope), hetero_mean(w, 0, slope)
    if law == "clustered":
        nodes, weights = hermegauss(HERMITE_NODES)
        weights = weights / weights.sum()

        def marginal(a: int) -> np.ndarray:
            eta = outcome_logit(w, a)[:, None] + nodes[None, :]
            return expit(eta) @ weights

        return marginal(1), marginal(0)
    return expit(outcome_logit(w, 1)), expit(outcome_logit(w, 0))


def design_matrix(w: np.ndarray) -> np.ndarray:
    """The basis ``(1, W1, W3)`` at the rows of ``w``."""
    return np.column_stack([np.ones(len(w)), w[:, 0], w[:, 2]])


def projection(
    phi: np.ndarray, mass: np.ndarray, q_t: np.ndarray, q_r: np.ndarray, scale: str
) -> np.ndarray:
    """The projection of the contrast onto ``phi`` under the measure ``mass``."""
    gram = phi.T @ (mass[:, None] * phi)
    if scale == "difference":
        return np.linalg.solve(gram, phi.T @ (mass * (q_t - q_r)))
    beta = np.zeros(phi.shape[1])
    for _ in range(100):
        e = expit(phi @ beta)
        equation = phi.T @ (mass * ((q_t + q_r) * e - q_t))
        jacobian = phi.T @ ((mass * (q_t + q_r) * e * (1.0 - e))[:, None] * phi)
        step = np.linalg.solve(jacobian, equation)
        beta = beta - step
        if np.max(np.abs(step)) < 1e-15:
            break
    return beta


def law_mass(law: str, w: np.ndarray, mass: np.ndarray) -> np.ndarray:
    """The covariate measure of ``law``'s target: tilted by the sampling weight when weighted."""
    if law != "weighted":
        return mass
    tilted = mass * sampling_weight(w)
    return tilted / tilted.sum()


def truth(
    law: str, scale: str = "difference", *, slope: float = 0.0, nodes: int | None = None
) -> dict[str, float]:
    """The exact coefficients of a cell, keyed by the name the fit reports.

    The ``weighted`` truth is the projection of the weighted population, whose ``W`` law is
    tilted by the sampling weight; its arm regressions are unchanged because the weight is
    a function of ``W``.  The ``assess`` truth is the score's risk difference against the
    constant ATE and its ``(1, s)`` calibration coefficients.

    Parameters
    ----------
    law : str
        A key of :data:`LAWS`.
    scale : {"difference", "log_ratio"}
        The contrast's scale.
    slope : float
        The ``hetero_trial`` slope.
    nodes : int or None
        Quadrature nodes per axis; ``None`` reads :data:`QUADRATURE_NODES`.

    Returns
    -------
    dict of str to float
        The truth of each reported parameter.
    """
    w, mass = _grid(nodes)
    mass = law_mass(law, w, mass)
    q_t, q_r = arm_means(law, w, slope=slope)
    if law == "assess":
        s = score(w)
        tau = q_t - q_r
        ate = float(mass @ tau)
        calibration = projection(np.column_stack([np.ones(len(w)), s]), mass, q_t, q_r, scale)
        return {
            "risk_difference": float(mass @ (s**2 - 2.0 * s * tau) + ate**2),
            "calibration[(intercept)]": float(calibration[0]),
            "calibration[S]": float(calibration[1]),
        }
    stem = "cate_projection" if scale == "difference" else "log_crr_projection"
    beta = projection(design_matrix(w), mass, q_t, q_r, scale)
    return {f"{stem}[{term}]": float(value) for term, value in zip(TERMS, beta, strict=True)}


def quadrature_error(law: str = "observational", scale: str = "difference") -> float:
    """The largest change in a truth when the nodes per axis drop from 48 to 32."""
    fine = truth(law, scale)
    coarse = truth(law, scale, nodes=32)
    return max(abs(fine[name] - coarse[name]) for name in fine)


# ---------------------------------------------------------- the fit groups


CALIBRATION = "interval_calibration"
BIAS = "bias"
TYPE_I = "type_i_error"
POWER = "power"

#: The positive arm and the row-level SE control of a calibration cell.
POSITIVE = "correctly_specified"
ROW_VARIANCE = "row_variance_control"


@dataclass(frozen=True)
class Group:
    """One declared fit group: a law, a size, its replications, and the cells its fits fill.

    Every cell of a group reads the same draws and the same fit, so the controls named here
    share their replicates with the positive cells (design review S2).  A group's draws come
    from the stream :func:`tests.studies.evidence.seeds.stream_seed` keys by its label.

    Parameters
    ----------
    label : str
        The group key, which the seed stream and the cell names read.
    law : str
        A key of :data:`LAWS`.
    n : int
        Rows per replication; the ladder's offset reads this ``n``, not a training fold's.
    replicates : int
        Replications in the declared run.
    learner : str
        A key of :func:`learners`.
    scale : str
        The contrast's scale.
    slope : float
        The ``hetero_trial`` slope.
    calibration : bool
        Whether the group's coefficients are ``interval_calibration`` positives.
    bias_rows : tuple of str
        Terms whose TMLE estimate is a positive ``bias`` row.
    controls : Mapping of str to tuple of str
        Each control the group's fits compute, and the terms it publishes rows for.
    seconds : float
        Declared single-thread seconds per replication, for the cost.
    folds : int or None
        Outer cross-fitting folds; ``None`` keeps the estimator's default.
    """

    label: str
    law: str
    n: int
    replicates: int
    learner: str
    scale: str = "difference"
    slope: float = 0.0
    calibration: bool = True
    bias_rows: tuple[str, ...] = ()
    controls: Mapping[str, tuple[str, ...]] = None  # type: ignore[assignment]
    seconds: float = 0.3
    folds: int | None = None

    def __post_init__(self) -> None:
        if self.controls is None:
            object.__setattr__(self, "controls", {})


#: Replications of every GLM-learner calibration group (design review B6).
R_GLM = 4_800
#: Replications of the ensemble group (design review B6).
R_ENSEMBLE = 3_600

ALL_TERMS = TERMS

#: The groups of section 5.1, revised by the design review, in run order.  A control publishes
#: rows only for the terms its computed displacement moves (:func:`design_numbers`): the
#: T-learner's offset loads on ``W1``, the IPW projection's on the two slopes, the unit-weight
#: one-step's on the intercept and ``W1``, the one-step without ``1 / Pi`` on the intercept and
#: ``W3``, and the dropped weights on the intercept.  ``seconds`` are single-thread
#: measurements: the design review's at ``e87244d0`` where the group is unchanged, the
#: re-review's at ``dd697cdb`` for the two missing-outcome calibration groups, and this
#: declaration's own where the group is new (``tests/unit/test_conditional_projection_design.py``
#: reads them for the cost).
GROUPS: tuple[Group, ...] = (
    Group("trial_2k", "trial", 2_000, R_GLM, "spline", seconds=0.22),
    Group("obs_1k", "observational", 1_000, R_GLM, "spline", seconds=0.21),
    Group("obs_4k", "observational", 4_000, R_GLM, "spline", seconds=0.31),
    Group("crr_1k", "crr", 1_000, R_GLM, "spline", "log_ratio", seconds=0.23),
    Group("crr_4k", "crr", 4_000, R_GLM, "spline", "log_ratio", seconds=0.31),
    Group("multi_4k", "multi_arm", 4_000, R_GLM, "spline", seconds=0.34),
    Group(
        "weighted_sl_2k",
        "weighted",
        2_000,
        R_ENSEMBLE,
        "ensemble",
        controls={"weights_ignored": ("(intercept)",)},
        seconds=43.8,
        folds=5,
    ),
    Group(
        "clustered_4k",
        "clustered",
        4_000,
        R_GLM,
        "spline",
        controls={"iid_se": ("(intercept)", "W1")},
        seconds=0.22,
    ),
    Group("obs_missing_4k", "missing", 4_000, R_GLM, "spline", seconds=0.54),
    Group("obs_missing_crr_4k", "missing", 4_000, R_GLM, "spline", "log_ratio", seconds=0.69),
    Group("assess_score_2k", "assess", 2_000, R_GLM, "spline", seconds=0.30),
    Group(
        "slow_1k",
        "slow_both",
        1_000,
        R_GLM,
        "slow",
        calibration=False,
        bias_rows=ALL_TERMS,
        controls={"ipw": ("W1", "W3")},
        seconds=0.15,
    ),
    Group(
        "slow_4k",
        "slow_both",
        4_000,
        R_GLM,
        "slow",
        calibration=False,
        bias_rows=ALL_TERMS,
        controls={"t_learner": ("W1",), "ipw": ("W1", "W3")},
        seconds=0.25,
    ),
    Group(
        "slow_16k",
        "slow_both",
        16_000,
        R_GLM,
        "slow",
        calibration=False,
        bias_rows=ALL_TERMS,
        controls={"ipw": ("W1", "W3")},
        seconds=0.90,
    ),
    Group("null_2k", "hetero_trial", 2_000, 2_400, "risk_glm", calibration=False, seconds=0.15),
    Group(
        "power_4k",
        "hetero_trial",
        4_000,
        1_000,
        "risk_glm",
        slope=0.10,
        calibration=False,
        seconds=0.30,
    ),
    Group(
        "noweight_dr_4k",
        "noweight_law",
        4_000,
        1_000,
        "declared_noweight",
        calibration=False,
        bias_rows=ALL_TERMS,
        controls={"no_weight": ("(intercept)", "W1")},
        seconds=0.20,
    ),
    Group(
        "missing_dr_4k",
        "missing",
        4_000,
        1_000,
        "declared_missing",
        calibration=False,
        bias_rows=ALL_TERMS,
        controls={"missing_ignored": ("(intercept)", "W3")},
        seconds=0.25,
    ),
)

GROUP_BY_LABEL: Mapping[str, Group] = {group.label: group for group in GROUPS}

#: What each control computes from its group's fit, and the rule it must fail.
CONTROL_CLAIMS: Mapping[str, str] = {
    "iid_se": "the row-level SE on clustered data: its SE-ratio interval must fall below the band",
    "t_learner": "the projection of mu_n(1) - mu_n(0) with no targeting: its bias must lie "
    "wholly outside +-0.25 of its own SD",
    "ipw": "the projection of (2A - 1) Y / pi_n(A, W) with no outcome fit: its bias must lie "
    "wholly outside +-0.25 of its own SD at every rung",
    "no_weight": "the one-step with unit debiasing weights, on the declared wrong mu: its bias "
    "must lie wholly outside the margin",
    "weights_ignored": "the same draws fitted with the spline GLM and without the sampling "
    "weights, read against the weighted truth: its bias must lie wholly outside the margin",
    "missing_ignored": "the one-step without 1 / Pi, on the declared wrong mu: its bias must lie "
    "wholly outside the margin",
}


def calibration_cell(group: str, term: str) -> str:
    """The ``interval_calibration`` positive cell of one term of one group."""
    return f"{group}_{TERM_LABELS[term]}__{POSITIVE}"


def assess_names() -> tuple[str, ...]:
    """The three parameters the ``assess`` group reports."""
    return ("risk_difference", "calibration[(intercept)]", "calibration[S]")


ASSESS_LABELS = {
    "risk_difference": "risk_difference",
    "calibration[(intercept)]": "calibration_intercept",
    "calibration[S]": "calibration_S",
}


def bias_cell(label: str, term: str) -> str:
    """A ``bias`` cell: a group's positive row, or a control's row at a group's size."""
    return f"{label}_{TERM_LABELS[term]}"


def control_label(control: str, group: Group) -> str:
    """The cell prefix of ``control`` on ``group``: the control name and the size."""
    return f"{control}_{group.label.rsplit('_', 1)[-1]}"


def group_cells(group: Group) -> dict[str, tuple[str, ...]]:
    """The property cells ``group``'s fits fill, by family."""
    cells: dict[str, list[str]] = {CALIBRATION: [], BIAS: [], TYPE_I: [], POWER: []}
    if group.law == "assess":
        cells[CALIBRATION] += [
            f"{group.label}_{ASSESS_LABELS[n]}__{POSITIVE}" for n in assess_names()
        ]
    elif group.calibration:
        cells[CALIBRATION] += [calibration_cell(group.label, term) for term in TERMS]
    cells[BIAS] += [bias_cell(group.label, term) for term in group.bias_rows]
    for control, terms in group.controls.items():
        if control == "iid_se":
            cells[CALIBRATION] += [
                f"{group.label}_{TERM_LABELS[term]}__{ROW_VARIANCE}" for term in terms
            ]
        else:
            cells[BIAS] += [bias_cell(control_label(control, group), term) for term in terms]
    if group.label == "null_2k":
        cells[TYPE_I].append("heterogeneity__sharp_null")
    if group.label == "power_4k":
        cells[POWER].append("heterogeneity__slope_0.10")
    return {family: tuple(names) for family, names in cells.items() if names}


def property_cells() -> dict[str, tuple[str, ...]]:
    """Every declared property cell, by family, in group order."""
    out: dict[str, list[str]] = {}
    for group in GROUPS:
        for family, names in group_cells(group).items():
            out.setdefault(family, []).extend(names)
    return {family: tuple(names) for family, names in out.items()}


PROPERTY_CELLS = property_cells()

#: Which role each bias cell plays: a group's own TMLE row is positive, a control's is control.
BIAS_ROLES: Mapping[str, str] = {
    **{bias_cell(group.label, term): "positive" for group in GROUPS for term in group.bias_rows},
    **{
        bias_cell(control_label(control, group), term): "control"
        for group in GROUPS
        for control, terms in group.controls.items()
        if control != "iid_se"
        for term in terms
    },
}


# -------------------------------------------------------------- the learners


def spline_glm() -> Any:
    """The spline-GLM candidate of section 5.0: cubic splines, then a logistic GLM."""
    return make_pipeline(SplineTransformer(n_knots=5, degree=3), LogisticRegression(max_iter=2_000))


def ensemble(clip: tuple[float, float]) -> SuperLearner:
    """The ensemble of section 5.0: spline GLM, shallow regularized HGB and the lasso.

    ``nnloglik`` meta-learner, 5 inner folds, and ``clip`` from the bound the fit applies
    to the nuisance it predicts.  The inner split is seeded once: the draws differ.
    """
    lasso = dict(default_library("classification", random_state=0))["lasso"]
    hgb = HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        l2_regularization=1.0,
        early_stopping=False,
        max_iter=200,
        random_state=0,
    )
    return SuperLearner(
        [("spline", spline_glm()), ("hgb", hgb), ("lasso", lasso)],
        task="classification",
        meta_learner="nnloglik",
        n_folds=5,
        clip=clip,
        random_state=0,
    )


class SlowOffset(ClassifierMixin, BaseEstimator):
    """A declared slow learner: the correct-family GLM, then a logit offset ``c n^{-1/3}``.

    For ``pi`` the GLM is main terms in ``W`` and the offset ``h(W)``.  For the outcome it is
    the pooled GLM on ``[A, W1, W2, W3, A W1, A sin(pi W3)]``, the family the law's logit lies
    in (design review B3), and the offset ``A (h(W) + W1)``: arm 0 carries none.  Both are
    unpenalized.  ``n`` is the replication's ``n``, not a training fold's (design review N1),
    so the offset's size is fixed by the declaration and its error is of order ``n^{-1/3}``.
    """

    def __init__(
        self, target: str = "treatment", n: int = 1_000, columns: tuple[int, ...] = (0, 1, 2)
    ) -> None:
        self.target = target
        self.n = n
        self.columns = columns

    def _features(self, X: np.ndarray) -> np.ndarray:
        if self.target == "treatment":
            return X
        arm = X[:, 0]
        return np.column_stack([X, arm * X[:, 1], arm * np.sin(np.pi * X[:, 3])])

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SlowOffset:
        X = np.asarray(X, dtype=float)
        self.model_ = LogisticRegression(penalty=None, max_iter=5_000).fit(
            self._features(X), y, sample_weight=sample_weight
        )
        self.classes_ = self.model_.classes_
        return self

    def _offset(self, X: np.ndarray) -> np.ndarray:
        w = X[:, list(self.columns)]
        scale = SLOW_CONSTANT * float(self.n) ** (-1.0 / 3.0)
        if self.target == "treatment":
            return scale * slow_offset(w)
        return X[:, 0] * scale * (slow_offset(w) + w[:, 0])

    def predict_proba(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        p = np.clip(self.model_.predict_proba(self._features(X))[:, 1], 1e-9, 1 - 1e-9)
        p = expit(logit(p) + self._offset(X))
        return np.column_stack([1.0 - p, p])


class DeclaredOutcome(ClassifierMixin, BaseEstimator):
    """A declared wrong outcome regression: the true logit plus :func:`declared_error`.

    It reads ``[A, W1, W2, W3]`` and learns nothing, so the error is the declared one in every
    fold, and a control's displacement is computed exactly by quadrature.
    """

    def __init__(self, kind: str = "noweight") -> None:
        self.kind = kind

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> DeclaredOutcome:
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        a, w = X[:, 0], X[:, 1:4]
        p = expit(outcome_logit(w, a) + declared_error(self.kind, w, a))
        return np.column_stack([1.0 - p, p])


class RiskGLM(BaseEstimator):
    """A linear probability model, the correct family of the ``hetero_trial`` risk-scale law.

    Per arm (``outcome_by_arm=True``) on ``W``, clipped into ``[0.01, 0.99]``.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> RiskGLM:
        X = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        weight = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, float)
        self.coef_ = np.linalg.lstsq(X * np.sqrt(weight)[:, None], np.sqrt(weight) * y)[0]
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        X = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        p = np.clip(X @ self.coef_, 0.01, 0.99)
        return np.column_stack([1.0 - p, p])


def learners(group: Group) -> dict[str, Any]:
    """The nuisance learners and fit settings of ``group``."""
    glm = LogisticRegression(max_iter=2_000)
    if group.learner == "slow":
        return {
            "outcome_learner": SlowOffset("outcome", group.n, columns=(1, 2, 3)),
            "treatment_learner": SlowOffset("treatment", group.n),
        }
    if group.learner in ("declared_noweight", "declared_missing"):
        settings: dict[str, Any] = {
            "outcome_learner": DeclaredOutcome(group.learner.removeprefix("declared_")),
            "treatment_learner": glm,
        }
    elif group.learner == "ensemble":
        settings = {
            "outcome_learner": ensemble((0.005, 0.995)),
            "treatment_learner": ensemble((0.01, 0.99)),
            "outcome_by_arm": True,
        }
    elif group.learner == "risk_glm":
        settings = {"outcome_learner": RiskGLM(), "outcome_by_arm": True}
    else:
        settings = {
            "outcome_learner": spline_glm(),
            "treatment_learner": glm,
            "outcome_by_arm": True,
        }
    if group.law == "missing":
        settings["missingness_learner"] = LogisticRegression(max_iter=2_000)
    if group.folds is not None:
        settings["n_folds"] = group.folds
    return settings


def design(group: Group, *, weights: bool = True) -> PointTreatment:
    """The study design of ``group``'s law; ``weights=False`` drops the sampling weights."""
    roles: dict[str, Any] = {}
    if group.law in ("trial", "hetero_trial"):
        roles["treatment_probabilities"] = {0: "p0", 1: "p1"}
    if group.law == "missing":
        roles["missingness"] = "Delta"
    if group.law == "weighted" and weights:
        roles["weights"] = "weight"
    if group.law == "clustered":
        roles["cluster"] = "cluster"
    return PointTreatment(outcome="Y", treatment="A", adjustment=("W1", "W2", "W3"), **roles)


def estimand(group: Group) -> ConditionalContrast:
    """The estimand of ``group``: ``Linear(W1, W3)`` on the group's scale and pair."""
    pair = {"treatment": 2, "reference": 0} if group.law == "multi_arm" else {}
    return ConditionalContrast(MODIFIERS, scale=group.scale, model=BASIS, **pair)  # type: ignore[arg-type]


# ------------------------------------------------------------------ the fits


def _projection_of(phi: np.ndarray, values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """The weighted least-squares projection of ``values`` onto ``phi``."""
    return np.linalg.solve(phi.T @ (weight[:, None] * phi), phi.T @ (weight * values))


def control_estimates(fit: Any, control: str) -> np.ndarray:
    """The coefficients a study-side control computes from ``fit``'s own nuisances.

    ``t_learner`` projects ``Q_1 - Q_0``; ``ipw`` projects ``(2A - 1) Y / g(A, W)``;
    ``no_weight`` projects the one-step pseudo-outcome with unit debiasing weights in place
    of ``1 / g``; ``missing_ignored`` projects the one-step pseudo-outcome with ``1 / g`` and
    without ``1 / Pi``.  Each reads the fit's out-of-fold nuisances at its own truncation.
    """
    data = fit.data
    nuisance = fit.repeats[0].nuisance
    frame = pd.DataFrame(np.asarray(data.covariates, dtype=float), columns=data.covariate_names)
    phi = np.asarray(BASIS.evaluate(frame, MODIFIERS), dtype=float)
    weight = np.ones(len(phi)) if data.weights is None else np.asarray(data.weights, float)
    q_t, q_r, _, _ = _pseudo_outcomes(
        nuisance, data, (1.0, 0.0), g_bounds=tuple(fit.config.g_bounds)
    )
    a = np.asarray(data.treatment, dtype=float)
    observed = np.asarray(data.observed, dtype=bool)
    y = np.where(observed, np.asarray(data.outcome, dtype=float), 0.0)
    g1 = np.asarray(nuisance.propensity.bounded(tuple(fit.config.g_bounds)), dtype=float)
    g1 = g1[:, nuisance.propensity.column_for(1.0)]
    if control == "t_learner":
        values = q_t - q_r
    elif control == "ipw":
        values = a * y / g1 - (1.0 - a) * y / (1.0 - g1)
    elif control == "no_weight":
        values = q_t - q_r + a * (y - q_t) - (1.0 - a) * (y - q_r)
    elif control == "missing_ignored":
        values = (
            q_t
            - q_r
            + observed * a / g1 * (y - q_t)
            - observed * (1.0 - a) / (1.0 - g1) * (y - q_r)
        )
    else:
        raise KeyError(control)
    return _projection_of(phi, values, weight)


def _fit(group: Group, frame: pd.DataFrame, random_state: int, *, weights: bool = True) -> Any:
    study = CausalStudy(frame, design=design(group, weights=weights))
    effect = study.identify(estimand(group))
    settings = {**learners(group), "random_state": random_state}
    if group.law == "weighted" and not weights:
        settings = {
            "outcome_learner": spline_glm(),
            "treatment_learner": LogisticRegression(max_iter=2_000),
            "outcome_by_arm": True,
            "random_state": random_state,
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if group.law == "assess":
            return effect.assess_score("S", computed_from=["W1"], score_kind="external", **settings)
        return effect.estimate(**settings)


def fit_group(group: Group, frame: pd.DataFrame, *, random_state: int) -> dict[str, Any]:
    """Fit one replication of ``group`` and return what each of its cells reads.

    Parameters
    ----------
    group : Group
        The group.
    frame : pandas.DataFrame
        One replication drawn by :func:`sample`.
    random_state : int
        The fold seed.

    Returns
    -------
    dict
        ``"estimates"``: the reported estimate objects by name; ``"controls"``: each
        control's coefficients by term (``iid_se`` holds standard errors); and for a
        heterogeneity group ``"statistic"`` and ``"p_value"``.
    """
    fit = _fit(group, frame, random_state)
    if group.law == "assess":
        return {"estimates": {name: fit[name] for name in assess_names()}, "controls": {}}
    out: dict[str, Any] = {"estimates": dict(fit.estimates), "controls": {}}
    names = list(fit.estimates)
    for control in group.controls:
        if control == "iid_se":
            curves = np.column_stack([np.asarray(fit.estimates[k].influence_curve) for k in names])
            se = np.std(curves, axis=0, ddof=1) / math.sqrt(len(curves))
            out["controls"][control] = dict(zip(TERMS, se, strict=True))
        elif control == "weights_ignored":
            unweighted = _fit(group, frame, random_state, weights=False)
            psis = [unweighted.estimates[k].psi for k in names]
            out["controls"][control] = dict(zip(TERMS, psis, strict=True))
        else:
            values = control_estimates(fit, control)
            out["controls"][control] = dict(zip(TERMS, values, strict=True))
    if group.law == "hetero_trial":
        test = fit.heterogeneity_test()
        out["statistic"], out["p_value"] = float(test.statistic), float(test.p_value)
    return out


def stem(group: Group) -> str:
    """The reported name stem of ``group``'s coefficients."""
    return "cate_projection" if group.scale == "difference" else "log_crr_projection"


# ------------------------------------------------------------ the primary phase

#: The primary phase is the ``obs_4k`` group: the projection with an estimated mechanism.
PRIMARY_GROUP = "obs_4k"
PRIMARY_REPLICATES = GROUP_BY_LABEL[PRIMARY_GROUP].replicates
PRIMARY_N = GROUP_BY_LABEL[PRIMARY_GROUP].n

SCENARIOS = {PRIMARY_GROUP: tuple(f"cate_projection[{term}]" for term in TERMS)}

STUDY = StudyRecord(
    name="Conditional-contrast projection",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "conditional_projection",
    document="docs/technical-reference/method-evidence/conditional-projection.md",
    anchor="conditional-contrast-projection",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation="cleverly-projection-tmle",
    reference=None,
    failure_bound={"cleverly-projection-tmle": 0.01},
    modules=(
        "tests/studies/conditional_projection.py",
        "tests/studies/conditional_projection_properties.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.conditional_projection",
    properties_module="tests.studies.conditional_projection_properties",
    property_cells=PROPERTY_CELLS,
    bias_row_families=frozenset({BIAS}),
    calibration_efficiency_ratio=False,
    publication_policy="gated",
)

CONFIGURATION = {
    group.label: (
        f"{group.law} law, n={group.n}, R={group.replicates}, learner {group.learner}"
        f"{'' if group.folds is None else f', {group.folds} outer folds'}, scale {group.scale}"
    )
    for group in GROUPS
}


def draw_from_seed(group_label: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one sample of a group from an explicit seed, with the group's truths."""
    group = GROUP_BY_LABEL[group_label]
    frame = sample(group.law, n, np.random.default_rng(seed), slope=group.slope)
    return frame, truth(group.law, group.scale, slope=group.slope)


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one primary replication from this study's declared seed."""
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def primary_rows(
    frame: pd.DataFrame, truths: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """One primary row per coefficient of the ``obs_4k`` fit."""
    group = GROUP_BY_LABEL[scenario]
    fitted = fit_group(group, frame, random_state=replicate)["estimates"]
    rows = []
    for name in SCENARIOS[scenario]:
        estimate = fitted[name]
        reference = float(truths[name])
        std_error, low, high = reported_inference(estimate)
        rows.append(
            {
                "implementation": STUDY.implementation,
                "scenario": scenario,
                "replicate": replicate,
                "n": len(frame),
                "estimand": name,
                "truth": reference,
                "estimate": float(estimate.psi),
                "inference_estimate": float(estimate.psi),
                "std_error": std_error,
                "ci_lower": low,
                "ci_upper": high,
                "inference_scale": "identity",
                "covered": int(low <= reference <= high),
                "initial_estimate": math.nan,
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
    frame, truths = drawn
    return primary_rows(frame, truths, scenario, index)


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


# ---------------------------------------------------------- the design numbers


def efficient_covariance(law: str, n: int, *, nodes: int = 32) -> np.ndarray:
    """The efficient covariance of the CATE projection's coefficients at ``n`` rows.

    ``M^{-1} E[phi phi' ((tau - phi'beta)^2 + s_1 / (g Pi_1) + s_0 / ((1 - g) Pi_0))] M^{-1} / n``
    with ``s_a = mu_a (1 - mu_a)``.  A weighted law reads the weighted estimator's curve
    ``w IC / E[w]`` under the sampling law.
    """
    w, mass = _grid(nodes)
    q_t, q_r = arm_means(law, w)
    if law == "noweight_law":
        g = noweight_propensity(w)
    elif law in ("trial", "hetero_trial"):
        g = np.full(len(w), 0.5)
    else:
        g = propensity(w)
    pi_t = observation(w, 1) if law == "missing" else 1.0
    pi_r = observation(w, 0) if law == "missing" else 1.0
    phi = design_matrix(w)
    target = law_mass(law, w, mass)
    beta = projection(phi, target, q_t, q_r, "difference")
    gram = phi.T @ (target[:, None] * phi)
    second = (
        (q_t - q_r - phi @ beta) ** 2
        + q_t * (1 - q_t) / (g * pi_t)
        + q_r * (1 - q_r) / ((1 - g) * pi_r)
    )
    if law == "weighted":
        tilt = sampling_weight(w) / float(mass @ sampling_weight(w))
        middle = phi.T @ ((mass * tilt**2 * second)[:, None] * phi)
    else:
        middle = phi.T @ ((mass * second)[:, None] * phi)
    inverse = np.linalg.inv(gram)
    return inverse @ middle @ inverse / n


def control_moments(
    group: Group, control: str, *, nodes: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    """``(displacement, SD)`` of a bias control's coefficients, by quadrature.

    The displacement is the control's limit minus the group's truth.  The SD is an upper bound
    on the control's own spread where one is cheap (``ipw``: the known-``pi_n`` IPW variance;
    the fitted mechanism lowers it), exact where the control reads no fitted nuisance
    (``no_weight`` and ``missing_ignored`` read the declared ``mu`` and ``pi_n``), and the
    efficient SD of the group's law for ``t_learner`` and ``weights_ignored``, which is at
    least the spread of a plug-in of a correctly specified parametric model.  An upper bound
    on the SD gives a lower bound on the standardized displacement, so a detection
    probability computed from it is a lower bound too.
    """
    w, mass = _grid(nodes)
    phi = design_matrix(w)
    q_t, q_r = arm_means(group.law, w)
    truth_beta = projection(phi, law_mass(group.law, w, mass), q_t, q_r, "difference")
    gram = phi.T @ (mass[:, None] * phi)
    inverse = np.linalg.inv(gram)
    n = group.n
    if control == "weights_ignored":
        limit = projection(phi, mass, q_t, q_r, "difference")
        sd = np.sqrt(np.diag(efficient_covariance("observational", n, nodes=nodes)))
        return limit - truth_beta, sd
    if control == "t_learner":
        scale = SLOW_CONSTANT * n ** (-1.0 / 3.0)
        tilted = expit(outcome_logit(w, 1) + scale * (slow_offset(w) + w[:, 0]))
        limit = projection(phi, mass, tilted, q_r, "difference")
        sd = np.sqrt(np.diag(efficient_covariance("observational", n, nodes=nodes)))
        return limit - truth_beta, sd
    if control == "ipw":
        g = propensity(w)
        fitted = expit(logit(g) + SLOW_CONSTANT * n ** (-1.0 / 3.0) * slow_offset(w))
        mean = g * q_t / fitted - (1 - g) * q_r / (1 - fitted)
        second = g * q_t / fitted**2 + (1 - g) * q_r / (1 - fitted) ** 2
    else:
        kind = "noweight" if control == "no_weight" else "missing"
        g = noweight_propensity(w) if control == "no_weight" else propensity(w)
        m_t = expit(outcome_logit(w, 1) + declared_error(kind, w, 1))
        m_r = expit(outcome_logit(w, 0) + declared_error(kind, w, 0))
        d = m_t - m_r
        if control == "no_weight":
            # A = 1: Y - m_r; A = 0: m_t - Y.
            mean = g * (q_t - m_r) + (1 - g) * (m_t - q_r)
            second = g * (q_t - 2 * m_r * q_t + m_r**2) + (1 - g) * (m_t**2 - 2 * m_t * q_r + q_r)
        else:
            p_t, p_r = observation(w, 1), observation(w, 0)
            mean = d + p_t * (q_t - m_t) - p_r * (q_r - m_r)
            e_t = (q_t - 2 * m_t * q_t + m_t**2) / g**2
            e_r = (q_r - 2 * m_r * q_r + m_r**2) / (1 - g) ** 2
            second = g * ((1 - p_t) * d**2 + p_t * (d**2 + 2 * d * (q_t - m_t) / g + e_t)) + (
                1 - g
            ) * ((1 - p_r) * d**2 + p_r * (d**2 - 2 * d * (q_r - m_r) / (1 - g) + e_r))
    limit = inverse @ (phi.T @ (mass * mean))
    fitted_mean = phi @ limit
    middle = phi.T @ ((mass * (second - 2 * mean * fitted_mean + fitted_mean**2))[:, None] * phi)
    sd = np.sqrt(np.diag(inverse @ middle @ inverse / n))
    return limit - truth_beta, sd


def bias_pass_probability(standardized: float, replicates: int, *, control: bool) -> float:
    """P(the 99% t interval of a standardized bias is inside, or wholly outside, +-0.25).

    The interval is ``b_hat +- t_{0.995} / sqrt(R)``, with ``b_hat`` normal about ``b`` with
    SD ``1 / sqrt(R)`` (the shipped ``bias_equivalent`` and ``bias_discriminated`` endpoints).
    """
    margin = Margins().standardized_bias
    half = float(stats.t.ppf(0.995, replicates - 1)) / math.sqrt(replicates)
    spread = 1.0 / math.sqrt(replicates)
    b = abs(standardized)
    if control:
        return float(stats.norm.sf((margin + half - b) / spread)) + float(
            stats.norm.cdf((-margin - half - b) / spread)
        )
    return float(
        stats.norm.cdf((margin - half - b) / spread) - stats.norm.cdf((-margin + half - b) / spread)
    )


def calibration_row_pass(
    replicates: int, ratio: float = 1.0, *, se_cv: float = 0.1, kurtosis: float = 3.0
) -> float:
    """P(a calibration row passes) at SE ratio ``ratio``, by the design review's approximation.

    The SE ratio's 99% bootstrap interval is normal with SD
    ``ratio sqrt(cv^2 / R + (kurtosis - 1) / (4 R))``, and it must lie inside
    ``[0.93, 1.07]``; the coverage's 99% Clopper--Pearson interval must lie inside
    ``[0.92, 0.98]``, with coverage ``2 Phi(1.96 ratio) - 1``.  The two are treated as
    independent.
    """
    margins = Margins()
    low, high = margins.calibration_se_ratio
    z = float(stats.norm.ppf(0.995))
    sd = ratio * math.sqrt(se_cv**2 / replicates + (kurtosis - 1.0) / (4.0 * replicates))
    half = z * sd
    p_ratio = float(
        stats.norm.cdf((high - half - ratio) / sd) - stats.norm.cdf((low + half - ratio) / sd)
    )
    coverage = float(2.0 * stats.norm.cdf(CRITICAL * ratio) - 1.0)
    k = np.arange(replicates + 1)
    lower = stats.beta.ppf(0.005, np.maximum(k, 1e-12), replicates - k + 1)
    upper = stats.beta.ppf(0.995, k + 1, np.maximum(replicates - k, 1e-12))
    inside = (lower >= margins.calibration_coverage[0]) & (upper <= margins.calibration_coverage[1])
    p_coverage = float(stats.binom.pmf(k, replicates, coverage)[inside].sum())
    return p_ratio * p_coverage


def row_variance_detection(ratio: float, replicates: int) -> float:
    """P(the ``iid_se`` SE-ratio interval lies wholly below 0.93) at SE ratio ``ratio``."""
    z = float(stats.norm.ppf(0.995))
    sd = ratio * math.sqrt(0.01 / replicates + 1.0 / (2.0 * replicates))
    return float(stats.norm.cdf((Margins().calibration_se_ratio[0] - z * sd - ratio) / sd))


def heterogeneity_power(n: int, slope: float, alpha: float = 0.05) -> tuple[float, float]:
    """``(power, noncentrality)`` of the df-2 heterogeneity test on ``hetero_trial``."""
    w, mass = _grid(40)
    mu1, mu0 = hetero_mean(w, 1, slope), hetero_mean(w, 0, slope)
    v = 2.0 * (mu1 * (1 - mu1) + mu0 * (1 - mu0))
    phi = design_matrix(w)
    gram = phi.T @ (mass[:, None] * phi)
    tau = mu1 - mu0
    beta = np.linalg.solve(gram, phi.T @ (mass * tau))
    middle = phi.T @ ((mass * (v + (tau - phi @ beta) ** 2))[:, None] * phi)
    inverse = np.linalg.inv(gram)
    sigma = inverse @ middle @ inverse
    lam = float(n * beta[1:] @ np.linalg.solve(sigma[1:, 1:], beta[1:]))
    critical = float(stats.chi2.ppf(1.0 - alpha, 2))
    return float(stats.ncx2.sf(critical, 2, lam)), lam


def type_i_pass_probability(size: float, replicates: int) -> float:
    """P(the shipped type-I rule passes): the 99% CP upper end at most ``alpha + 0.05``."""
    margins = Margins()
    ceiling = margins.alpha + margins.type_i_margin
    k = np.arange(replicates + 1)
    upper = stats.beta.ppf(0.995, k + 1, np.maximum(replicates - k, 1e-12))
    largest = int(k[upper <= ceiling].max())
    return float(stats.binom.cdf(largest, replicates, size))


#: The pilot's measured standardized bias of the ``slow_1k`` intercept, with the corrected
#: learner (design review S1: 0.161 +- 0.027 over 1,400 replications).  The other rungs read
#: at most 0.12.
SLOW_1K_PILOT_BIAS = 0.161


def design_numbers() -> dict[str, Any]:
    """Every computed number the declaration fixes before the run.

    Returns
    -------
    dict
        ``calibration``: each calibration cell's pass probability at ratio 1, at 0.97 and at
        its piloted ratio (with the pilot's SE spread and kurtosis where they exceed the
        declared 0.1 and 3), and the family-wise rate at 1 and at the piloted ratios; ``slow_1k_bias``: the ladder's marginal row at the pilot's bias
        and at 0.19; ``power`` and ``power_pass``; ``type_i_pass``; ``controls``: each control
        cell's standardized displacement and detection probability; ``cost``: CPU-hours and
        the ideal wall hours on 16 and on 4 cores.
    """
    from tests.studies.evidence.property_verdicts import power_cell_pass_probability

    pilot = pilot_rows()
    calibration = {}
    for group, cell, name in calibration_parameters():
        row = pilot[(group.label, name)]
        calibration[cell] = {
            "replicates": group.replicates,
            "pass_at_1": calibration_row_pass(group.replicates, 1.0),
            "pass_at_0.97": calibration_row_pass(group.replicates, 0.97),
            "pilot_ratio": row["se_ratio"],
            "pilot_replicates": int(row["replicates"]),
            "pass_at_pilot": calibration_row_pass(
                group.replicates,
                row["se_ratio"],
                se_cv=max(row["se_cv"], 0.1),
                kurtosis=max(row["kurtosis"], 3.0),
            ),
        }
    family = float(np.prod([row["pass_at_1"] for row in calibration.values()]))
    family_pilot = float(np.prod([row["pass_at_pilot"] for row in calibration.values()]))
    pooled = pooled_pilot_ratios(pilot)
    for cell, row in calibration.items():
        row["pooled_ratio"] = pooled["ratios"][cell]
        row["pass_at_pooled"] = calibration_row_pass(int(row["replicates"]), row["pooled_ratio"])
    family_pooled = float(np.prod([row["pass_at_pooled"] for row in calibration.values()]))
    controls: dict[str, dict[str, float]] = {}
    for group in GROUPS:
        for control, terms in group.controls.items():
            if control == "iid_se":
                for term in terms:
                    ratio = pilot[(group.label, f"{stem(group)}[{term}]")]["iid_se_ratio"]
                    controls[f"{group.label}_{TERM_LABELS[term]}__{ROW_VARIANCE}"] = {
                        "ratio": ratio,
                        "detection": row_variance_detection(ratio, group.replicates),
                    }
                continue
            shift, sd = control_moments(group, control)
            for term in terms:
                j = TERMS.index(term)
                standardized = float(shift[j] / sd[j])
                controls[bias_cell(control_label(control, group), term)] = {
                    "standardized": standardized,
                    "detection": bias_pass_probability(
                        standardized, group.replicates, control=True
                    ),
                }
    power, lam = heterogeneity_power(GROUP_BY_LABEL["power_4k"].n, 0.10)
    cpu = sum(group.replicates * group.seconds for group in GROUPS) / 3600.0
    cpu += PRIMARY_REPLICATES * GROUP_BY_LABEL[PRIMARY_GROUP].seconds / 3600.0
    return {
        "calibration": calibration,
        "calibration_family": family,
        "calibration_family_pilot": family_pilot,
        "calibration_family_pooled": family_pooled,
        "pilot_homogeneity": pooled["homogeneity"],
        "slow_1k_bias": {
            "pilot": bias_pass_probability(SLOW_1K_PILOT_BIAS, R_GLM, control=False),
            "at_0.19": bias_pass_probability(0.19, R_GLM, control=False),
        },
        "power": power,
        "noncentrality": lam,
        "power_pass": power_cell_pass_probability(power, GROUP_BY_LABEL["power_4k"].replicates),
        "type_i_pass": type_i_pass_probability(0.05, GROUP_BY_LABEL["null_2k"].replicates),
        "controls": controls,
        "cost": {
            "cpu_hours": cpu,
            "wall_hours_16": cpu / 16.0,
            "wall_hours_4": cpu / 4.0,
        },
    }


#: The SE-ratio pilot's committed summary (:mod:`tests.studies.conditional_projection_pilot`):
#: a pilot on its own seed stream, read by :func:`design_numbers`, not a result of the run.
PILOT_CSV = ROOT / "tests" / "studies" / "conditional_projection_pilot.csv"


def pilot_rows() -> dict[tuple[str, str], dict[str, float]]:
    """The pilot's summary row of each ``(group, parameter)``."""
    frame = pd.read_csv(PILOT_CSV)
    return {(str(row["group"]), str(row["name"])): row for row in frame.to_dict("records")}


def pooled_pilot_ratios(pilot: Mapping[tuple[str, str], Mapping[str, float]]) -> dict[str, Any]:
    """Each calibration row's SE ratio, shrunk toward its learner's common ratio.

    A row's piloted ratio is its true ratio plus pilot noise of SD ``se_ratio_se``; read
    alone, that noise moves the plug-in pass probability as much as a real miscalibration
    would.  Within each learner (the spline GLM, the ensemble) the rows are pooled by the
    DerSimonian--Laird random-effects estimate: a precision-weighted common ratio, a
    method-of-moments between-row variance ``tau^2`` (zero when the rows spread no more than
    their noise), and each row shrunk toward the common ratio by ``tau^2 / (tau^2 + se^2)``.

    Returns
    -------
    dict
        ``ratios``: the shrunk ratio of each positive calibration cell; ``homogeneity``: per
        learner, the common ratio, ``tau^2``, Cochran's ``Q``, its degrees of freedom and its
        chi-square p-value.
    """
    by_learner: dict[str, list[tuple[str, float, float]]] = {}
    for group, cell, name in calibration_parameters():
        row = pilot[(group.label, name)]
        by_learner.setdefault(group.learner, []).append(
            (cell, float(row["se_ratio"]), float(row["se_ratio_se"]))
        )
    ratios: dict[str, float] = {}
    homogeneity: dict[str, dict[str, float]] = {}
    for learner, rows in by_learner.items():
        hat = np.array([row[1] for row in rows])
        var = np.array([row[2] for row in rows]) ** 2
        weight = 1.0 / var
        common = float(weight @ hat / weight.sum())
        q = float(weight @ (hat - common) ** 2)
        df = len(rows) - 1
        scale = weight.sum() - (weight**2).sum() / weight.sum()
        tau2 = max(0.0, (q - df) / scale) if df > 0 else 0.0
        shrink = tau2 / (tau2 + var)
        for (cell, _, _), value in zip(rows, common + shrink * (hat - common), strict=True):
            ratios[cell] = float(value)
        homogeneity[learner] = {
            "common_ratio": common,
            "tau2": tau2,
            "q": q,
            "df": float(df),
            "p_value": float(stats.chi2.sf(q, df)) if df > 0 else 1.0,
        }
    return {"ratios": ratios, "homogeneity": homogeneity}


def calibration_parameters() -> list[tuple[Group, str, str]]:
    """Each positive calibration cell with its group and the parameter it reads."""
    out = []
    for group in GROUPS:
        if group.law == "assess":
            out += [
                (group, f"{group.label}_{ASSESS_LABELS[name]}__{POSITIVE}", name)
                for name in assess_names()
            ]
        elif group.calibration:
            out += [
                (group, calibration_cell(group.label, term), f"{stem(group)}[{term}]")
                for term in TERMS
            ]
    return out
