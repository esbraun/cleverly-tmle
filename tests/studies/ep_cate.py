r"""Declaration and driver of the ``ep-cate`` study (P1 plan, section 5.2).

The study measures the EP learner of the conditional average treatment effect
(:class:`cleverly.EPMethod`, ``model=None``) against the oracle DR-learner
:math:`\theta_{n0}` of EPL Theorems 3 to 5: least squares of
:math:`\chi_0 = \tau_0(W) + \{A/\pi_0(W) - (1 - A)/(1 - \pi_0(W))\}\{Y - \mu_0(A, W)\}` on the
same sieve basis.  Every gate reads the paired mean of
:math:`\log \mathrm{ISE}(\text{EP}) - \log \mathrm{ISE}(\theta_{n0})` within a replicate, with
:math:`\mathrm{ISE} = \int (\hat\theta - \theta_0)^2\,dP_V` by Monte Carlo on
:data:`ISE_POINTS` fresh draws of ``W`` (one fixed set per law, its own stream).

Everything a run reads is fixed here before it starts.

**Laws.**  hte3 at ``eac64673`` (``generate_data_CATE.R`` and ``generate_data_CATE_HighDim.R``),
which produced the paper's figures; the plan's minor 3 (rev 2.2) makes the code the law and
keeps the margins.  The declaration commit read the d = 3 generator, and it differs from the
plan's reading of EPL App B.2 in three ways, recorded here:

- the noise SD is 1 in the code, not 2 (variance 4) as App B.2 reads;
- ``positivity=TRUE`` is the *limited* overlap, ``logit pi = W1 + W2 + W3``, and
  ``positivity=FALSE`` the moderate one, ``logit pi = (W1 + W2 + W3) / 3``;
- the simple CATE is ``1 + W1`` under moderate overlap and ``1 + W1 + W2 + W3`` under limited
  overlap; the complex CATE is ``1 + sum W + sum sin(5 W)`` under both.

The baseline is ``0.5 sum W + sum sin(5 W) + sum 1/(W + 1.2)``.  The d = 20 laws are
``sim.CATEHighDim(positivity=FALSE)``: ``W = Z / 2`` with ``Z`` a normal vector with unit
variances and covariances 0.4 truncated to ``[-2, 2]^20`` (drawn here by rejection, which is
exact; hte3 uses a Gibbs sampler), ``logit pi = (W1 + W5 + W9 + W11 + W19) / 1.3``.  The one
declared change from the code is C1: the noise is a standard normal truncated at
:data:`NOISE_TRUNCATION` (which moves the law by ``P(|e| > 4) = 6.3e-5``; symmetric, so the
CATE is unchanged), and ``q_bounds`` covers the outcome range (:func:`q_bounds`).

**Learners.**  The plan's ensemble (section 5.0): ``mu`` per arm by spline GLM, shallow HGB and
the lasso with ``nnls`` (:func:`regression_ensemble`), ``pi`` by the classification ensemble
of :mod:`tests.studies.conditional_projection`.  On ``d3_complex`` (both overlaps) the
declared misspecified learner: per-arm main-terms ``LinearRegression`` with the same
ensemble ``pi`` (the same folds and seeds, so the same fitted ``pi``).  5 outer folds.

**Estimators** per replicate: EP at the declared schedule :func:`schedule`, ``k(n)`` = 3, 4, 5
cosines per modifier at n = 1,000, 2,000, 4,000; the T-learner and the DR-learner on the same
basis and nuisances (descriptive); the oracle :math:`\theta_{n0}`; on the rescue laws the EP
and T-learner at the misspecified learner.  The first :data:`DESCRIPTIVE_REPLICATES`
replicates of ``d3_complex_moderate`` at 1k and 4k also fit a greedy HGB final learner on the
same nuisances (descriptive).  One group fits EP with the size chosen by cross-validation at
n = 2,000 (reporting, the default's experimental row).

**Gates** (plan 5.2, one-sided 99% Student bounds, margins fixed by the plan):

- Thm 5 oracle comparison, every law, ensemble, at n = 4,000: upper bound below
  ``log 1.5``;
- no growth: the upper bound of the change in mean log ratio from 1k to 4k, from an
  independent-samples interval, below ``log 1.25``;
- DR rescue, misspecified learner, ``d3_complex`` both overlaps, at 4,000: upper bound below
  ``log 3``, and its growth bound below ``log 1.25``;
- control that must fail: the T-learner at the misspecified learner, lower bound at 4,000
  above ``log 3``.

**Fixed size** (``ep_fixed_obs_4k``, moved from ``conditional-projection``, its finding m7):
the observational law of :mod:`tests.studies.conditional_projection` at n = 4,000 with its
spline learner, EP with ``model=Linear(["W1", "W3"])`` (the default final stage), each
coefficient's standardized bias inside ``+-0.25`` (the shipped ``bias`` rule).  The same draw
also fits the projection TMLE, and the paired difference is published.

**Primary phase.**  The evidence harness's primary rows carry an interval, which the EP curve
does not have (R1).  The primary scenario is therefore the projection TMLE of ``obs_4k`` on
its own seed: the estimator the fixed-size EP row is compared with.

The run is ``python -m tests.canonical.ep_cate.regenerate``; its docstring holds the red-cell
reading.  ``R`` per law is fixed by :func:`design_numbers` from the committed pilot
(:mod:`tests.studies.ep_cate_pilot`).  The comparator, hte3 ``Lrnr_cate_EP`` with
``targeting_style="dr"``, is declared pending (:data:`STUDY`).  The study is not yet in
:func:`tests.studies.evidence.registry.registered`: registration lands with the artifacts.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

from cleverly import CausalStudy, ConditionalContrast, PointTreatment, SuperLearner
from cleverly.bases import Modifiers, Sieve
from cleverly.effect_curves import _pseudo_outcomes
from cleverly.learners.library import default_library
from tests.parallel import STUDY_JOBS
from tests.studies import conditional_projection as projection
from tests.studies.evidence import replication
from tests.studies.evidence.registry import ROOT, Margins, StudyRecord
from tests.studies.evidence.schema import REPLICATE_COLUMNS, reported_inference
from tests.studies.evidence.seeds import draw_replicate, replicate_seed, stream_seed

SLUG = "ep-cate"

#: The study's seed and its resampling seed; every stream is a label hashed under them.
SEED = 20261121
RESAMPLING_SEED = 20261122

NOISE_SD = 1.0
NOISE_TRUNCATION = 4.0

#: Monte Carlo points of each law's ISE, drawn once per law from stream ``ise_points``.
ISE_POINTS = 20_000

OUTER_FOLDS = 5

#: Cosines per modifier the sieve family holds; the schedule uses at most 5.
SIEVE_MAX = 6

#: Replicates of the descriptive cells: the greedy final learner and the CV-size group.
DESCRIPTIVE_REPLICATES = 100

#: The plan's margins (5.2), on the log ISE ratio.
THM5_MARGIN = math.log(1.5)
GROWTH_MARGIN = math.log(1.25)
RESCUE_MARGIN = math.log(3.0)

#: One-sided confidence of every gate's bound.
CONFIDENCE = 0.99

#: The sizing rule's candidates (plan 5.2, finding N2) and the target probability.
R_CANDIDATES = {3: (300, 450, 600), 20: (200, 300, 400)}
TARGET_PASS = 0.90

ORACLE = "oracle_comparison"
NUISANCE = "nuisance_error"
BIAS = "bias"
DIAGNOSTIC = "diagnostic"


def schedule(n: int) -> int:
    """The declared sieve schedule ``k(n) = 3 + floor(log2(n / 1000))``: 3, 4, 5 at 1k, 2k, 4k."""
    return max(1, 3 + math.floor(math.log2(n / 1000.0) + 1e-9))


# ---------------------------------------------------------------- the laws


@dataclass(frozen=True)
class Law:
    """One data-generating law of the study.

    Parameters
    ----------
    label : str
        The law's name in every cell.
    dimension : int
        3 or 20.
    cate : {"simple", "complex"}
        The CATE.
    overlap : {"moderate", "limited"}
        The propensity (d = 3 only; the d = 20 laws are hte3's ``positivity=FALSE``).
    sizes : tuple of int
        The sample sizes.
    rescue : bool
        Whether the law carries the misspecified-learner cells.
    """

    label: str
    dimension: int
    cate: str
    overlap: str
    sizes: tuple[int, ...]
    rescue: bool = False

    @property
    def columns(self) -> tuple[str, ...]:
        """The covariate columns, which are also the modifiers."""
        return tuple(f"W{j + 1}" for j in range(self.dimension))


LAWS: tuple[Law, ...] = (
    Law("d3_simple_moderate", 3, "simple", "moderate", (1_000, 2_000, 4_000)),
    Law("d3_simple_limited", 3, "simple", "limited", (1_000, 2_000, 4_000)),
    Law("d3_complex_moderate", 3, "complex", "moderate", (1_000, 2_000, 4_000), rescue=True),
    Law("d3_complex_limited", 3, "complex", "limited", (1_000, 2_000, 4_000), rescue=True),
    Law("d20_simple", 20, "simple", "limited", (1_000, 4_000)),
    Law("d20_complex", 20, "complex", "limited", (1_000, 4_000)),
)
LAW_BY_LABEL: Mapping[str, Law] = {law.label: law for law in LAWS}

#: The equicorrelation of the d = 20 normal vector before truncation.
D20_CORRELATION = 0.4


def covariates(law: Law, n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` draws of ``W``: uniform on ``[-1, 1]^3``, or hte3's truncated normal over 2."""
    if law.dimension == 3:
        return rng.uniform(-1.0, 1.0, (n, 3))
    out = np.empty((0, law.dimension))
    while out.shape[0] < n:
        common = rng.standard_normal((2 * n, 1)) * math.sqrt(D20_CORRELATION)
        own = rng.standard_normal((2 * n, law.dimension)) * math.sqrt(1.0 - D20_CORRELATION)
        z = common + own
        out = np.vstack([out, z[np.all(np.abs(z) <= 2.0, axis=1)]])
    return out[:n] / 2.0


def propensity(law: Law, w: np.ndarray) -> np.ndarray:
    """``P(A = 1 | W)``."""
    if law.dimension == 20:
        return expit(w[:, [0, 4, 8, 10, 18]].sum(axis=1) / 1.3)
    total = w.sum(axis=1)
    return expit(total if law.overlap == "limited" else total / 3.0)


def baseline(law: Law, w: np.ndarray) -> np.ndarray:
    """``E[Y | A = 0, W]``."""
    if law.dimension == 3:
        return 0.5 * w.sum(axis=1) + np.sin(5 * w).sum(axis=1) + (1.0 / (w + 1.2)).sum(axis=1)
    w1, w5, w9, w10, w15 = w[:, 0], w[:, 4], w[:, 8], w[:, 9], w[:, 14]
    smooth = (
        np.cos(4 * w1) + np.cos(4 * w5) + np.sin(4 * w9) + 1 / (1.5 + w15) + 1 / (1.5 + w10)
    ) / 5
    return smooth + (np.sin(5 * w1) + np.sin(5 * w5) + np.sin(5 * w9)) / 2


def cate(law: Law, w: np.ndarray) -> np.ndarray:
    """The CATE ``E[Y | A = 1, W] - E[Y | A = 0, W]``, hte3's code."""
    if law.dimension == 3:
        if law.cate == "complex":
            return 1.0 + w.sum(axis=1) + np.sin(5 * w).sum(axis=1)
        return 1.0 + (w[:, 0] if law.overlap == "moderate" else w.sum(axis=1))
    w1, w5, w9, w10, w15 = w[:, 0], w[:, 4], w[:, 8], w[:, 9], w[:, 14]
    if law.cate == "simple":
        return 1.0 + (w1 + w5 + w9 + w15 + w10) / 5
    return 1.0 + (np.sin(4 * w1) + np.sin(4 * w5) + np.cos(4 * w9) + 1.5 * (w15**2 - w10**2)) / 5


def mean_range(law: Law) -> tuple[float, float]:
    """Bounds on ``E[Y | A, W]`` over the support, from the terms' own ranges."""
    if law.dimension == 3:
        base = (-1.5 - 3.0 + 3.0 / 2.2, 1.5 + 3.0 + 3.0 / 0.2)
        effect = (-5.0, 7.0) if law.cate == "complex" else (-2.0, 4.0)
    else:
        base = (-0.6 + 0.8 / 5 - 1.5, 0.6 + 4.0 / 5 + 1.5)
        effect = (0.0, 2.0) if law.cate == "simple" else (1.0 - 4.5 / 5, 1.0 + 4.5 / 5)
    return base[0] + min(0.0, effect[0]), base[1] + max(0.0, effect[1])


def q_bounds(law: Law) -> tuple[float, float]:
    """The declared outcome range: the mean's range widened by the noise truncation."""
    low, high = mean_range(law)
    return (math.floor(low - NOISE_TRUNCATION), math.ceil(high + NOISE_TRUNCATION))


def sample(law: Law, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """One replication of ``law`` at ``n`` rows."""
    w = covariates(law, n, rng)
    a = rng.binomial(1, propensity(law, w))
    noise = stats.truncnorm.rvs(
        -NOISE_TRUNCATION, NOISE_TRUNCATION, scale=NOISE_SD, size=n, random_state=rng
    )
    y = baseline(law, w) + a * cate(law, w) + noise
    frame = pd.DataFrame(w, columns=list(law.columns))
    frame["A"] = a
    frame["Y"] = y
    return frame


def ise_points(law: Law) -> np.ndarray:
    """The fixed Monte Carlo points of ``law``'s ISE."""
    seed = stream_seed(STUDY, "ise_points", law.label)
    return covariates(law, ISE_POINTS, np.random.default_rng(seed))


# ---------------------------------------------------------------- the learners


def regression_ensemble() -> SuperLearner:
    """The ensemble of plan 5.0 for a continuous ``mu``, on the scaled outcome.

    Spline GLM (cubic splines then least squares), the shallow regularized HGB and the
    shipped lasso pipeline, with the ``nnls`` meta-learner and ``clip=(0, 1)``.
    """
    lasso = dict(default_library("regression", random_state=0))["lasso"]
    hgb = HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        l2_regularization=1.0,
        early_stopping=False,
        max_iter=200,
        random_state=0,
    )
    spline = make_pipeline(SplineTransformer(n_knots=5, degree=3), LinearRegression())
    return SuperLearner(
        [("spline", spline), ("hgb", hgb), ("lasso", lasso)],
        task="regression",
        meta_learner="nnls",
        n_folds=5,
        clip=(0.0, 1.0),
        random_state=0,
    )


def greedy_learner() -> HistGradientBoostingRegressor:
    """The descriptive greedy final learner (EPL Sec 7.1): the plan's shallow HGB."""
    return HistGradientBoostingRegressor(
        max_depth=3,
        learning_rate=0.05,
        l2_regularization=1.0,
        early_stopping=False,
        max_iter=200,
        random_state=0,
    )


def learner_settings(law: Law, learner: str, random_state: int) -> dict[str, Any]:
    """The nuisance learners and fit settings of one EP fit."""
    outcome = regression_ensemble() if learner == "ensemble" else LinearRegression()
    return {
        "outcome_learner": outcome,
        "treatment_learner": projection.ensemble((0.01, 0.99)),
        "outcome_by_arm": True,
        "n_folds": OUTER_FOLDS,
        "q_bounds": q_bounds(law),
        "random_state": random_state,
    }


def modifiers(law: Law) -> Modifiers:
    """Every covariate is a modifier, on its declared domain ``[-1, 1]``."""
    return Modifiers(list(law.columns), domains=dict.fromkeys(law.columns, (-1.0, 1.0)))


def effect_of(law: Law, frame: pd.DataFrame) -> Any:
    study = CausalStudy(
        frame, design=PointTreatment(outcome="Y", treatment="A", adjustment=law.columns)
    )
    return study.identify(ConditionalContrast(modifiers(law)))


def fit_ep(law: Law, frame: pd.DataFrame, learner: str, random_state: int, *, size: Any) -> Any:
    """One EP fit of ``law`` through the study layer."""
    sieve = Sieve.default(modifiers(law), max_size=SIEVE_MAX, size=size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect_of(law, frame).estimate(
            "ep", sieve=sieve, **learner_settings(law, learner, random_state)
        )


# ---------------------------------------------------------------- one replicate


def _least_squares(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(design, target, rcond=None)[0]


def ise(prediction: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    """The Monte Carlo ISE and its Monte Carlo standard error."""
    squared = (np.asarray(prediction) - truth) ** 2
    return float(np.mean(squared)), float(np.std(squared, ddof=1) / math.sqrt(squared.size))


def oracle_target(law: Law, frame: pd.DataFrame) -> np.ndarray:
    """``chi_0``, the DR pseudo-outcome at the true nuisances."""
    w = frame.loc[:, list(law.columns)].to_numpy()
    a = frame["A"].to_numpy(dtype=float)
    y = frame["Y"].to_numpy(dtype=float)
    pi = propensity(law, w)
    mu = baseline(law, w) + a * cate(law, w)
    return cate(law, w) + (a / pi - (1.0 - a) / (1.0 - pi)) * (y - mu)


def curves(law: Law, frame: pd.DataFrame, fit: Any) -> dict[str, Any]:
    """Every estimator's ISE at one EP fit's nuisances and basis, and the nuisance errors."""
    points = ise_points(law)
    truth = cate(law, points)
    point_frame = pd.DataFrame(points, columns=list(law.columns))
    rows = frame.loc[:, list(law.columns)]
    curve = fit.model
    design = curve.design(rows)
    at_points = curve.design(point_frame)
    nuisance_fit = fit.fit.nuisance_fit
    nuisance = nuisance_fit.nuisance
    pair = tuple(nuisance.effect_modification.pair)
    q_t, q_r, d_t, d_r = _pseudo_outcomes(
        nuisance,
        fit.fit.data,
        pair,
        g_bounds=tuple(nuisance_fit.config.g_bounds),
        missingness_bound=float(nuisance_fit.config.missingness_bound),
    )
    out: dict[str, Any] = {}
    out["ep"], out["ep_mc_se"] = ise(fit.predict(point_frame), truth)
    out["t_learner"], _ = ise(at_points @ _least_squares(design, q_t - q_r), truth)
    out["dr_learner"], _ = ise(at_points @ _least_squares(design, d_t - d_r), truth)
    out["oracle"], out["oracle_mc_se"] = ise(
        at_points @ _least_squares(design, oracle_target(law, frame)), truth
    )
    w = rows.to_numpy()
    # The arm-wise L2 error, both arms at every row.
    errors = np.concatenate([q_t - baseline(law, w) - cate(law, w), q_r - baseline(law, w)])
    out["mu_l2"] = float(np.sqrt(np.mean(errors**2)))
    g = np.asarray(nuisance.propensity.values, dtype=float)[:, nuisance.propensity.column_for(1.0)]
    out["pi_l2"] = float(np.sqrt(np.mean((g - propensity(law, w)) ** 2)))
    out["size"] = int(fit.sieve.chosen)
    return out


def greedy_ise(law: Law, fit: Any) -> float:
    """The ISE of the greedy HGB final learner on the same debiased regressions."""
    estimator = fit.fit.estimator
    greedy = type(estimator)(
        estimator.tmle_kwargs,
        estimator.effect_modification,
        estimator.size,
        target=estimator.target,
        final_learner=greedy_learner(),
    )
    curve, _ = greedy._curve_at(fit.fit.nuisance_fit, fit.sieve.chosen)
    points = ise_points(law)
    return ise(curve.predict(pd.DataFrame(points, columns=list(law.columns))), cate(law, points))[0]


def fit_replicate(law: Law, n: int, frame: pd.DataFrame, replicate: int) -> dict[str, Any]:
    """Every ISE of one replicate of ``law`` at ``n`` (the gated group)."""
    random_state = replicate
    ensemble = fit_ep(law, frame, "ensemble", random_state, size=schedule)
    out = {"ensemble": curves(law, frame, ensemble)}
    if law.rescue:
        wrong = fit_ep(law, frame, "misspecified", random_state, size=schedule)
        out["misspecified"] = curves(law, frame, wrong)
    greedy = law.label == "d3_complex_moderate" and n in (1_000, 4_000)
    if greedy and replicate < DESCRIPTIVE_REPLICATES:
        out["greedy"] = greedy_ise(law, ensemble)
    return out


def fit_cv_replicate(law: Law, frame: pd.DataFrame, replicate: int) -> dict[str, Any]:
    """The CV-size reporting cell: the default sieve, size chosen by EPL Algorithm 4."""
    fit = fit_ep(law, frame, "ensemble", replicate, size="cv")
    return {"ensemble": curves(law, frame, fit)}


# ---------------------------------------------------------------- the fixed-size cell

FIXED_GROUP = projection.GROUP_BY_LABEL["obs_4k"]
FIXED_LABEL = "ep_fixed_obs_4k"
FIXED_N = 4_000


def fit_fixed(frame: pd.DataFrame, replicate: int) -> dict[str, Any]:
    """EP at a fixed declared basis, and the projection TMLE, on one ``obs_4k`` draw."""
    study = CausalStudy(frame, design=projection.design(FIXED_GROUP))
    effect = study.identify(projection.estimand(FIXED_GROUP))
    settings = {**projection.learners(FIXED_GROUP), "random_state": replicate}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ep_fit = effect.estimate("ep", **settings)
        tmle = effect.estimate(**settings)
    beta = np.asarray(ep_fit.model.beta) * ep_fit.model.width
    return {
        "ep": dict(zip(projection.TERMS, beta, strict=True)),
        "tmle": {term: tmle.estimates[f"cate_projection[{term}]"] for term in projection.TERMS},
    }


def fixed_truth() -> dict[str, float]:
    truths = projection.truth("observational", "difference")
    return {term: float(truths[f"cate_projection[{term}]"]) for term in projection.TERMS}


# ---------------------------------------------------------------- groups and cells


@dataclass(frozen=True)
class Group:
    """One fit group: a law at one ``n`` with its replicate count."""

    label: str
    law: str
    n: int
    replicates: int
    kind: str = "gated"  # "gated", "cv" or "fixed"


#: ``R`` per law, the smallest candidate at which every gate and control of the law reaches
#: :data:`TARGET_PASS` at the pilot's spreads (:func:`design_numbers`, pinned by the design
#: test).  The fixed-size cell keeps the plan's 2,400.
REPLICATES: Mapping[str, int] = {
    "d3_simple_moderate": 300,
    "d3_simple_limited": 300,
    # Not powered: the pilot's rescue ratio (mean 1.46 at 4k) lies above log 3 at every R.
    "d3_complex_moderate": 600,
    "d3_complex_limited": 600,
    "d20_simple": 200,
    "d20_complex": 200,
}
FIXED_REPLICATES = 2_400


def group_label(law: str, n: int) -> str:
    return f"{law}_{n // 1000}k"


GROUPS: tuple[Group, ...] = (
    *(
        Group(group_label(law.label, n), law.label, n, REPLICATES[law.label])
        for law in LAWS
        for n in law.sizes
    ),
    Group("d3_complex_moderate_2k_cv", "d3_complex_moderate", 2_000, DESCRIPTIVE_REPLICATES, "cv"),
    Group(FIXED_LABEL, "observational", FIXED_N, FIXED_REPLICATES, "fixed"),
)
GROUP_BY_LABEL: Mapping[str, Group] = {group.label: group for group in GROUPS}


def gated_n(law: Law) -> tuple[int, int]:
    """The smallest and largest ``n`` of a law: the growth gate's two ends."""
    return law.sizes[0], law.sizes[-1]


def oracle_cells(group: Group) -> dict[str, str]:
    """The ``oracle_comparison`` cells of a group, each with its role."""
    law = LAW_BY_LABEL[group.law]
    largest = group.n == law.sizes[-1]
    if group.kind == "cv":
        return {f"{group.label}__ep": DIAGNOSTIC}
    cells = {
        f"{group.label}__ep": "positive" if largest else DIAGNOSTIC,
        f"{group.label}__t_learner": DIAGNOSTIC,
        f"{group.label}__dr_learner": DIAGNOSTIC,
    }
    if law.rescue:
        cells[f"{group.label}__ep_misspecified"] = "positive" if largest else DIAGNOSTIC
        cells[f"{group.label}__t_misspecified"] = "control" if largest else DIAGNOSTIC
    if group.law == "d3_complex_moderate" and group.n in (1_000, 4_000):
        cells[f"{group.label}__ep_greedy_hgb"] = DIAGNOSTIC
    return cells


def nuisance_cells(group: Group) -> tuple[str, ...]:
    if group.kind != "gated":
        return ()
    learners = ("ensemble", "misspecified") if LAW_BY_LABEL[group.law].rescue else ("ensemble",)
    return tuple(
        f"{group.label}__{learner}_{name}"
        for learner in learners
        for name in ("mu_l2", "pi_l2", "product")
    )


def fixed_cells() -> tuple[str, ...]:
    return tuple(f"{FIXED_LABEL}_{projection.TERM_LABELS[term]}" for term in projection.TERMS)


def group_cells(group: Group) -> dict[str, tuple[str, ...]]:
    """The property cells one group's fits fill, by family."""
    if group.kind == "fixed":
        return {
            BIAS: fixed_cells(),
            DIAGNOSTIC: tuple(f"{cell}__ep_minus_tmle" for cell in fixed_cells()),
        }
    cells = {ORACLE: tuple(oracle_cells(group))}
    nuisance = nuisance_cells(group)
    if nuisance:
        cells[NUISANCE] = nuisance
    return cells


def property_cells() -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for group in GROUPS:
        for family, cells in group_cells(group).items():
            out.setdefault(family, []).extend(cells)
    return {family: tuple(cells) for family, cells in out.items()}


PROPERTY_CELLS = property_cells()


# ---------------------------------------------------------------- the study record

PRIMARY_GROUP = "obs_4k"
PRIMARY_REPLICATES = FIXED_REPLICATES
PRIMARY_N = FIXED_N
SCENARIOS = {PRIMARY_GROUP: tuple(f"cate_projection[{term}]" for term in projection.TERMS)}

IMPLEMENTATION = "cleverly-ep-cate"


@dataclass(frozen=True)
class Comparator:
    """The declared comparator of the gated cells, pending (plan 5.2, finding 24).

    Parameters
    ----------
    reference : str
        The package.
    commit : str
        The pinned commit.
    learner : str
        The learner and its settings.
    kind : str
        The comparison kind of ``docs/roadmap.md``.
    runner : str
        The R script, relative to the study's artifact directory.
    status : str
        ``"pending"`` until the owner's local run adds its ISE rows.
    reason : str
        Why it is pending.
    """

    reference: str
    commit: str
    learner: str
    kind: str
    runner: str
    status: str
    reason: str


#: hte3's EP learner on the same draws, with the declared basis.  Not on :data:`STUDY`: the
#: registry's pending comparator pairs the primary rows, and this comparison is of the
#: ``oracle_comparison`` cells' ISEs.  hte3's CATE debiasing is the linear Method 2 (ridge,
#: lambda 1e-8), cleverly's the logistic Method 1, so the two solve the same score equations
#: with different ``mu*`` and agree only to ``o_p(n^{-1/2})``.
COMPARATOR = Comparator(
    reference="hte3",
    commit="eac64673f6fdf75f130275646657f08d1cb198e9",
    learner=(
        "Lrnr_cate_EP(targeting_style = 'dr', screen_basis_with_lasso = FALSE), additive "
        "cosine sieve of interaction order 1 with basisN = k(n) per modifier"
    ),
    kind="different estimator",
    runner="hte3_ep_runner.R",
    status="pending",
    reason="hte3 runs on the owner's machine (P1 plan 5.2); this session has no R",
)

STUDY = StudyRecord(
    name="EP learner of the CATE",
    slug=SLUG,
    artifacts=ROOT / "tests" / "canonical" / "ep_cate",
    document="docs/technical-reference/conditional-effects.md",
    anchor="status-and-evidence",
    scenarios=SCENARIOS,
    replicates=PRIMARY_REPLICATES,
    n=PRIMARY_N,
    seed=SEED,
    resampling_seed=RESAMPLING_SEED,
    margins=Margins(),
    implementation=IMPLEMENTATION,
    reference=None,
    failure_bound={IMPLEMENTATION: 0.01},
    modules=(
        "tests/studies/ep_cate.py",
        "tests/studies/ep_cate_properties.py",
        "tests/studies/conditional_projection.py",
        "tests/studies/evidence/properties.py",
        "tests/studies/evidence/property_verdicts.py",
        "tests/studies/evidence/schema.py",
        "tests/studies/evidence/seeds.py",
    ),
    runner_module="tests.studies.ep_cate",
    properties_module="tests.studies.ep_cate_properties",
    property_cells=PROPERTY_CELLS,
    bias_row_families=frozenset({BIAS}),
    calibration_efficiency_ratio=False,
    publication_policy="gated",
)

CONFIGURATION = {
    group.label: (
        f"{group.law} law, n={group.n}, R={group.replicates}, {group.kind}"
        + ("" if group.kind == "fixed" else f", {OUTER_FOLDS} outer folds, schedule k(n)")
    )
    for group in GROUPS
}


# ---------------------------------------------------------------- the primary phase


def draw_from_seed(group_label: str, n: int, seed: int) -> tuple[pd.DataFrame, dict[str, float]]:
    """Draw one sample of a group from an explicit seed."""
    rng = np.random.default_rng(seed)
    if group_label in (PRIMARY_GROUP, FIXED_LABEL):
        frame = projection.sample("observational", n, rng)
        return frame, projection.truth("observational", "difference")
    group = GROUP_BY_LABEL[group_label]
    return sample(LAW_BY_LABEL[group.law], n, rng), {}


def draw_scenario(scenario: str, n: int, replicate: int) -> tuple[pd.DataFrame, dict[str, float]]:
    return draw_replicate(STUDY, draw_from_seed, scenario, n, replicate)


def primary_rows(
    frame: pd.DataFrame, truths: Mapping[str, float], scenario: str, replicate: int
) -> list[dict[str, Any]]:
    """One primary row per coefficient of the projection TMLE at ``obs_4k``."""
    fitted = fit_fixed(frame, replicate)["tmle"]
    rows = []
    for term, estimate in fitted.items():
        name = f"cate_projection[{term}]"
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


# ---------------------------------------------------------------- the design numbers


def upper_bound(mean: float, sd: float, replicates: int) -> float:
    """The one-sided :data:`CONFIDENCE` Student upper bound of a mean."""
    return mean + float(stats.t.ppf(CONFIDENCE, replicates - 1)) * sd / math.sqrt(replicates)


def gate_pass(mean: float, sd: float, replicates: int, margin: float, *, below: bool) -> float:
    """P(the one-sided bound of a mean lies below (or above) ``margin``) at ``R`` replicates.

    The sample mean is normal about ``mean`` with SD ``sd / sqrt(R)``; the bound adds
    ``t_{0.99, R-1} sd / sqrt(R)`` (the SD's own sampling error is second order at R >= 200).
    """
    half = float(stats.t.ppf(CONFIDENCE, replicates - 1)) * sd / math.sqrt(replicates)
    spread = sd / math.sqrt(replicates)
    if below:
        return float(stats.norm.cdf((margin - half - mean) / spread))
    return float(stats.norm.sf((margin + half - mean) / spread))


def growth_pass(
    small: tuple[float, float], large: tuple[float, float], replicates: int, margin: float
) -> float:
    """P(the one-sided bound of ``mean(large) - mean(small)`` lies below ``margin``)."""
    spread = math.sqrt((small[1] ** 2 + large[1] ** 2) / replicates)
    half = float(stats.t.ppf(CONFIDENCE, 2 * replicates - 2)) * spread
    return float(stats.norm.cdf((margin - half - (large[0] - small[0])) / spread))


#: The committed pilot summary (:mod:`tests.studies.ep_cate_pilot`), on a stream disjoint
#: from the run's.
PILOT_CSV = ROOT / "tests" / "studies" / "ep_cate_pilot.csv"


def pilot_rows() -> dict[tuple[str, str], dict[str, float]]:
    """``(group, statistic) -> {mean, sd, replicates}`` from the committed pilot."""
    frame = pd.read_csv(PILOT_CSV)
    return {
        (str(row["group"]), str(row["statistic"])): {
            "mean": float(row["mean"]),
            "sd": float(row["sd"]),
            "replicates": int(row["replicates"]),
        }
        for _, row in frame.iterrows()
    }


def law_gates(
    law: Law, pilot: Mapping[tuple[str, str], Mapping[str, float]], replicates: int
) -> dict[str, float]:
    """Every gate's and control's pass probability of ``law`` at ``R`` replicates."""
    small, large = (group_label(law.label, n) for n in gated_n(law))

    def at(group: str, statistic: str) -> tuple[float, float]:
        row = pilot[(group, statistic)]
        return row["mean"], row["sd"]

    out = {
        "thm5": gate_pass(*at(large, "ep"), replicates, THM5_MARGIN, below=True),
        "growth": growth_pass(at(small, "ep"), at(large, "ep"), replicates, GROWTH_MARGIN),
    }
    if law.rescue:
        out["rescue"] = gate_pass(
            *at(large, "ep_misspecified"), replicates, RESCUE_MARGIN, below=True
        )
        out["rescue_growth"] = growth_pass(
            at(small, "ep_misspecified"), at(large, "ep_misspecified"), replicates, GROWTH_MARGIN
        )
        out["control"] = gate_pass(
            *at(large, "t_misspecified"), replicates, RESCUE_MARGIN, below=False
        )
    return out


def sized_replicates(
    law: Law, pilot: Mapping[tuple[str, str], Mapping[str, float]]
) -> tuple[int, bool]:
    """The N2 rule: the smallest candidate ``R`` at which every gate reaches the target.

    Returns the largest candidate and ``False`` when none does (the schedule row then stays
    experimental with the plan's reason, and the gates still run).
    """
    candidates = R_CANDIDATES[law.dimension]
    for replicates in candidates:
        if min(law_gates(law, pilot, replicates).values()) >= TARGET_PASS:
            return replicates, True
    return candidates[-1], False


def fixed_bias_pass(standardized: float, replicates: int = FIXED_REPLICATES) -> float:
    """P(a fixed-size coefficient's bias row passes) at a standardized bias."""
    return projection.bias_pass_probability(standardized, replicates, control=False)


#: CPU seconds of one CV-size replicate (one outer fit and five inner refits), measured once
#: on this declaration's smoke; the other groups' costs are the pilot's ``seconds`` rows.
CV_SECONDS = 200.0


def cost_hours(pilot: Mapping[tuple[str, str], Mapping[str, float]]) -> dict[str, float]:
    """CPU-hours of the declared run, from the pilot's per-replicate seconds."""
    total = 0.0
    for group in GROUPS:
        key = (group.label, "seconds")
        if key in pilot:
            total += pilot[key]["mean"] * group.replicates
        elif group.kind == "cv":
            total += CV_SECONDS * group.replicates
    hours = total / 3600.0
    return {"cpu_hours": hours, "wall_hours_4": hours / 4.0, "wall_hours_16": hours / 16.0}


def design_numbers() -> dict[str, Any]:
    """Every computed number the declaration fixes before the run.

    Returns
    -------
    dict
        ``laws``: per law, the sized ``R``, whether the rule powered it, and every gate's pass
        probability at that ``R``; ``fixed``: the fixed-size bias rows' pass probabilities at
        the projection pilot's bias plus the EP-minus-TMLE shift; ``nuisance``: the slope of
        the product of the nuisance L2 errors per law; ``cost``.
    """
    pilot = pilot_rows()
    laws = {}
    for law in LAWS:
        replicates, powered = sized_replicates(law, pilot)
        laws[law.label] = {
            "replicates": replicates,
            "powered": powered,
            "gates": law_gates(law, pilot, replicates),
        }
    projection_pilot = projection.pilot_rows()
    fixed = {}
    for term in projection.TERMS:
        name = f"cate_projection[{term}]"
        base = float(projection_pilot[("obs_4k", name)]["standardized_bias"])
        shift = pilot.get((FIXED_LABEL, f"shift_{projection.TERM_LABELS[term]}"), {"mean": 0.0})
        bias = base + float(shift["mean"])
        fixed[term] = {"standardized_bias": bias, "pass": fixed_bias_pass(bias)}
    nuisance = {}
    for law in LAWS:
        sizes = np.array(law.sizes, dtype=float)
        products = [pilot[(group_label(law.label, int(n)), "product")]["mean"] for n in law.sizes]
        slope = float(np.polyfit(np.log(sizes), np.log(products), 1)[0])
        nuisance[law.label] = {"product_slope": slope, "faster_than_root_n": slope < -0.5}
    return {"laws": laws, "fixed": fixed, "nuisance": nuisance, "cost": cost_hours(pilot)}
