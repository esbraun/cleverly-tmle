"""The fits, readings and controls of the ``mtp-density-tails`` study.

The laws are those of :mod:`tests.mtp_density_tails_law` (G, T, C, S, B and U), shared with the
unit witnesses of ``tests/unit/test_density_reconstruction.py``.  This module adds what the study
needs beyond them:

- the policy of each case: the law's own (a shift of one; capped at 4.5 on T; held on
  ``[u - 1, u]`` on B and U, the ``mtp-boundaries`` p2 shape with the per-row knee), and three
  variants read only by reported cells: ``Gscale`` (law G under ``Scale(1.25)``), ``Gcap`` (law G
  under ``Shift(1, cap=3.30)``, a cap at the dose's 0.9 quantile, inside the support) and ``Bp2``
  (law B under its p2 policy, read on the classifier route);
- the point fits: the density route with the oracle hazard (exact bin masses, so the grid and
  the reading are what is left), the classifier route the default resolves to (the default
  stack, cross-fitted, with or without the policy steps), and the learned density readings
  (boosting, logistic, the default stacked density) that report a ratio error and no estimate;
- the longitudinal default route on ``longitudinal-mtp``'s continuous law;
- the study-local controls: the pre-fix representation (equal-mass grid read as a histogram),
  the declared support ignored, no row-wise break (the floor at zero, rev 2's reading), the
  pre-fix one-step DR estimate, the density that ignores ``W1``, and the two broken cluster
  helpers;
- the three-mark misfit signature that routes a learned-hazard tail breach.

Every control patches a module attribute inside a context manager for one fit, so it runs in a
worker process without leaking into the next key.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

import cleverly.interventions.policy as policy_module
import cleverly.learners.density as density_module
from cleverly.data import CausalData
from cleverly.estimators import TMLE
from cleverly.interventions import ModifiedPolicy, Piece, Scale, Shift, Support
from cleverly.interventions.policy import PolicySet
from cleverly.learners import SuperLearner
from cleverly.learners.crossfit import make_folds
from cleverly.learners.density import (
    ConditionalDensity,
    bin_occupancy,
    default_density_bins,
    fit_conditional_density,
    isolated_dose_clusters,
)
from cleverly.learners.library import default_library
from cleverly.longitudinal import LTMLE, SDR, LongitudinalData
from tests import mtp_density_tails_law as law
from tests.studies import canonical_longitudinal_mtp as longitudinal_study
from tests.studies import longitudinal_mtp_common as longitudinal
from tests.studies.fractional_glm import QuasiBinomialGLM

INF = np.inf
#: Law G's population 0.9 quantile of ``A``: the cap of the ``Gcap`` case, inside the support.
CAP_G = 3.30
#: The factor of the ``Gscale`` case.
SCALE_G = 1.25
#: Outer folds of every cross-fitted point fit on the classifier route and of every learned
#: density reading.
POINT_FOLDS = 5
#: Outer folds of the longitudinal default-route fits (the declared design).
LONGITUDINAL_FOLDS = 5
TOL = 1e-10
MAX_ITER = 100
#: The declared outcome range of a cross-fitted point fit, which refuses a continuous outcome
#: without one.  No draw of any law reaches it: ``|Y| > 1,000`` needs ``|A|`` near 100, beyond
#: 60 SDs of every dose law.  So the bounds rescale the outcome and clip nothing.
Q_BOUNDS = (-1_000.0, 1_000.0)

#: Every case a fit reads: ``case -> (law, policy kind)``.
CASES: dict[str, tuple[str, str]] = {
    **{name: (name, "own") for name in law.LAWS},
    "Gscale": ("G", "scale"),
    "Gcap": ("G", "cap"),
    "Bp2": ("B", "own"),
}


def law_of(case: str) -> str:
    """The law a case draws from."""
    return CASES[case][0]


# ------------------------------------------------------------------ policies


def _plus_one(a: Any, h: Any) -> np.ndarray:
    return np.asarray(a, dtype=float) + law.DELTA


def _minus_one(b: Any, h: Any) -> np.ndarray:
    return np.asarray(b, dtype=float) - law.DELTA


def _unit(b: Any, h: Any) -> np.ndarray:
    return np.ones_like(np.asarray(b, dtype=float))


class Upper:
    """``u(h) - offset`` of a supported law, as a picklable callable.

    Parameters
    ----------
    name : str
        The law, B or U.
    offset : float
        Subtracted from ``u(h)``; the knee of the p2 policy is ``u(h) - 1``.
    """

    def __init__(self, name: str, offset: float = 0.0) -> None:
        self.name, self.offset = name, offset

    def __call__(self, h: Any) -> np.ndarray:
        w1 = np.asarray(h["W1"], dtype=float)
        w2 = np.asarray(h["W2"], dtype=float)
        return np.asarray(law.upper(self.name, w1, w2) - self.offset, dtype=float)


def policy(case: str) -> Any:
    """The case's policy."""
    name, kind = CASES[case]
    if kind == "scale":
        return Scale(SCALE_G, cap=None)
    if kind == "cap":
        return Shift(law.DELTA, cap=CAP_G)
    if name == "T":
        return Shift(law.DELTA, cap=law.HIGH)
    if name in law.SUPPORTED:
        knee = Upper(name, law.DELTA)
        return ModifiedPolicy(
            "p2",
            pieces=(Piece(-INF, knee, _plus_one, _minus_one, _unit), Piece(knee, INF)),
            policy_kind="known",
        )
    return Shift(law.DELTA, cap=None)


def support(case: str, *, route: str = "density") -> Support | None:
    """The declared support: ``[0, u(W)]`` on B and U, and ``[-2, 4.5]`` on T's classifier fits.

    The density route on T declares none, as the unit witness W1 does: the oracle masses are
    zero beyond the truncation, and the floor breaks each row's segment there.  The classifier
    route on T declares it, as the design probe did, so its steps read the support's top.
    """
    name = law_of(case)
    if name in law.SUPPORTED:
        return Support(0.0, Upper(name))
    if name == "T" and route == "classifier":
        return Support(law.LOW, law.HIGH)
    return None


def assign(case: str, a: Any, w1: Any, w2: Any) -> np.ndarray:
    """``d(a, W)`` of the case."""
    name, kind = CASES[case]
    a = np.asarray(a, dtype=float)
    if kind == "scale":
        return a * SCALE_G
    if kind == "cap":
        return np.where(a + law.DELTA <= CAP_G, a + law.DELTA, a)
    return law.assign(name, a, w1, w2)


def true_ratio(case: str, a: Any, w1: Any, w2: Any) -> np.ndarray:
    """``r0(a, W) = g^d(a | W) / g(a | W)`` of the case."""
    name, kind = CASES[case]
    a = np.asarray(a, dtype=float)

    def g(v: Any) -> np.ndarray:
        return law.pdf(name, v, w1, w2)

    if kind == "scale":
        return np.asarray(g(a / SCALE_G) / (SCALE_G * g(a)), dtype=float)
    if kind == "cap":
        return np.asarray(g(a - law.DELTA) / g(a) * (a <= CAP_G) + (a > CAP_G - law.DELTA))
    return law.r0(name, a, w1, w2)


def eif(case: str, frame: pd.DataFrame) -> np.ndarray:
    """``D0`` at every row, up to its constant: ``r0 (Y - Q0) + Q0(d(A), W)``.

    The influence-curve gap is a standard deviation, so the constant ``psi0`` never enters it.
    """
    w1, w2, a, y = law.columns(frame)
    return np.asarray(
        true_ratio(case, a, w1, w2) * (y - law.q0(a, w1, w2))
        + law.q0(assign(case, a, w1, w2), w1, w2),
        dtype=float,
    )


#: Monte Carlo draws and seed of a variant case's efficiency bound.
BOUND_DRAWS = 1_000_000
BOUND_SEED = 20261114


@cache
def efficiency_sd(case: str) -> float:
    """``SD(D0)`` of the case: the law module's 4,000,000-draw value, or 1,000,000 draws."""
    name, kind = CASES[case]
    if kind == "own":
        return float(law.EFFICIENCY_SD[name])
    frame = law.draw(name, BOUND_DRAWS, BOUND_SEED)
    return float(np.std(eif(case, frame), ddof=1))


@cache
def truth(case: str) -> float:
    """``psi0`` of the case: the law module's value, or ``E[Q0(d(A), W)]`` on 1,000,000 draws."""
    name, kind = CASES[case]
    if kind == "own":
        return float(law.TRUTH[name][0])
    frame = law.draw(name, BOUND_DRAWS, BOUND_SEED)
    w1, w2, a, _ = law.columns(frame)
    return float(np.mean(law.q0(assign(case, a, w1, w2), w1, w2)))


def ratio_error(case: str, ratio: Any, frame: pd.DataFrame) -> float:
    """``||r_n - r0||_n / SD(D0)`` over the sample."""
    w1, w2, a, _ = law.columns(frame)
    gap = np.asarray(ratio, dtype=float) - true_ratio(case, a, w1, w2)
    return float(np.sqrt(np.mean(gap**2)) / efficiency_sd(case))


# ------------------------------------------------------------------ learners


class OracleHazardWithoutW1(law.OracleHazard):
    """The oracle hazard read at ``W1 = 0``: a density that ignores a confounder.

    The double-robustness both-wrong control pairs it with the wrong outcome regression.
    """

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.array(X, dtype=float, copy=True)
        design[:, 0] = 0.0
        return np.asarray(super().predict_proba(design), dtype=float)


class WrongOutcome(LinearRegression):
    """OLS on ``[1, A, W1, W2]``: the wrong outcome regression of the double-robustness cells."""


class WithoutW1(BaseEstimator):
    """A classifier that never sees ``W1``: the design's ``W1`` column is dropped.

    The classifier design is ``[A, atoms, steps, W1, W2]``, so ``W1`` is the second-to-last
    column.

    Parameters
    ----------
    learner : estimator
        The classifier fitted on the remaining columns.
    """

    def __init__(self, learner: Any = None) -> None:
        self.learner = learner

    @staticmethod
    def _keep(X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        return np.delete(design, design.shape[1] - 2, axis=1)

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> WithoutW1:
        self.learner_ = clone(self.learner)
        if sample_weight is None:
            self.learner_.fit(self._keep(X), y)
        else:
            self.learner_.fit(self._keep(X), y, sample_weight=sample_weight)
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return np.asarray(self.learner_.predict_proba(self._keep(X)), dtype=float)


def default_stack() -> SuperLearner:
    """The classifier route's default stack: :class:`SuperLearner` over the default library."""
    return SuperLearner(random_state=0)


def booster() -> HistGradientBoostingClassifier:
    """The default library's gradient-boosting classifier, with its settings."""
    (_, learner), *_ = default_library("classification", random_state=0)
    return clone(learner)


class PerBinLogistic(BaseEstimator):
    """A logistic hazard with a slope on ``W`` per bin: one regression per bin index.

    ``bin_design = "index"`` makes the package pass ``[W1, W2, bin]``.
    """

    bin_design = "index"

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> PerBinLogistic:
        design = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float)
        index = np.rint(design[:, 2]).astype(int)
        self.models_: dict[int, Any] = {}
        self.constant_: dict[int, float] = {}
        for b in np.unique(index):
            rows = index == b
            if np.unique(target[rows]).size < 2:
                self.constant_[int(b)] = float(np.clip(target[rows].mean(), 1e-6, 1 - 1e-6))
                continue
            model = LogisticRegression(C=1e6, max_iter=5000)
            model.fit(design[rows, :2], target[rows])
            self.models_[int(b)] = model
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        index = np.rint(design[:, 2]).astype(int)
        out = np.full(index.size, 0.5)
        for b in np.unique(index):
            rows = index == b
            if int(b) in self.models_:
                out[rows] = self.models_[int(b)].predict_proba(design[rows, :2])[:, 1]
            else:
                out[rows] = self.constant_.get(int(b), 0.5)
        return np.column_stack([1.0 - out, out])


class PooledLogistic(LogisticRegression):
    """A pooled logistic hazard on ``[W1, W2, bin]``: one slope on the bin index."""

    bin_design = "index"

    def __init__(self, C: float = 1e6, max_iter: int = 5000) -> None:
        super().__init__(C=C, max_iter=max_iter)


#: The learned density readings: ``name -> factory``.  ``stacked`` is the default stacked
#: density (``treatment_learner=None`` on the density route).
DENSITY_LEARNERS: dict[str, Callable[[], Any]] = {
    "boosting": booster,
    "pooled_logistic": PooledLogistic,
    "per_bin_logistic": PerBinLogistic,
    "stacked": default_stack,
}


# ------------------------------------------------------------------ controls


def _equal_mass(values: Any, n_bins: int, *, name: str | None = None) -> np.ndarray:
    return np.asarray(
        density_module._equal_mass_edges(np.asarray(values, dtype=float), n_bins, name)
    )


def histogram_read(knots: Any, values: Any) -> np.ndarray:
    """The pre-fix reading: each bin read as its probability over its width, zero outside."""
    e = knots.edges
    v = np.asarray(values, dtype=float).reshape(knots.value.shape[0], -1)
    index = np.clip(np.searchsorted(e, v, side="right") - 1, 0, e.size - 2)
    rows = np.arange(v.shape[0])[:, None]
    out = knots.probabilities[rows, index] / np.diff(e)[index]
    return np.asarray(np.where((v >= e[0]) & (v <= e[-1]), out, 0.0), dtype=float)


@contextlib.contextmanager
def _patched(target: Any, name: str, value: Any) -> Iterator[None]:
    original = getattr(target, name)
    setattr(target, name, value)
    try:
        yield
    finally:
        setattr(target, name, original)


@contextlib.contextmanager
def reading(kind: str) -> Iterator[None]:
    """The representation a density fit reads under: ``shipped`` or a control.

    ``pre_fix`` is the equal-mass grid read as a histogram (the representation before the
    fix); ``no_row_break`` sets the knot floor to zero, so a row's clipped bins beyond its
    support join its segment (rev 2's reading); ``shipped`` and ``support_ignored`` change
    nothing here (the latter is a fit without the declared support).
    """
    with contextlib.ExitStack() as stack:
        if kind == "pre_fix":
            stack.enter_context(_patched(density_module, "bin_edges", _equal_mass))
            stack.enter_context(_patched(density_module, "_read", histogram_read))
        elif kind == "no_row_break":
            knots = density_module._knots
            defaults = {**(knots.__kwdefaults__ or {}), "floor": 0.0}
            stack.enter_context(_patched(knots, "__kwdefaults__", defaults))
        elif kind not in {"shipped", "support_ignored"}:
            raise ValueError(f"unknown reading {kind!r}")
        yield


@contextlib.contextmanager
def steps(enabled: bool) -> Iterator[None]:
    """The classifier route with (the shipped default) or without its policy steps."""
    if enabled:
        yield
        return
    with _patched(policy_module, "_policy_steps", lambda *args, **kwargs: None):
        yield


# ------------------------------------------------------------------ point fits


@dataclass(frozen=True)
class PointFit:
    """One point fit's estimate, its standard error, its influence curve and its ratio.

    Parameters
    ----------
    estimate : float
        The targeted estimate of the case's policy mean.
    std_error : float
        Its reported standard error.
    ci : tuple of float
        Its reported interval.
    influence_curve : numpy.ndarray
        The fit's per-row influence curve.
    ratio : numpy.ndarray
        The fit's policy ratio at the observed doses.
    route : str
        The ratio route the fit used.
    initial : float
        The untargeted plug-in ``E_n[Q(d(A), W)]`` of the outcome learner fitted on the whole
        sample (in sample, as the density route's initial fit is).
    """

    estimate: float
    std_error: float
    ci: tuple[float, float]
    influence_curve: np.ndarray
    ratio: np.ndarray
    route: str
    initial: float = float("nan")


def plug_in(case: str, frame: pd.DataFrame, outcome: str) -> float:
    """``E_n[Q(d(A), W)]`` of the outcome learner fitted on the whole sample."""
    w1, w2, a, y = law.columns(frame)
    learner = law.CorrectOutcome() if outcome == "correct" else WrongOutcome()
    learner.fit(np.column_stack([a, w1, w2]), y)
    shifted = np.column_stack([assign(case, a, w1, w2), w1, w2])
    return float(np.mean(learner.predict(shifted)))


def _point_fit(result: Any, initial: float = float("nan")) -> PointFit:
    single = result.single() if hasattr(result, "single") else result
    (key,) = single.estimates
    estimate = single.estimates[key]
    low, high = estimate.ci
    ratio = np.asarray(single.nuisance.policies.policy_ratio[:, 0], dtype=float)
    record = single.config.ratio_route
    route = record.route if record is not None else "none"
    return PointFit(
        estimate=float(estimate.psi),
        std_error=float(estimate.std_error),
        ci=(float(low), float(high)),
        influence_curve=np.asarray(estimate.influence_curve, dtype=float),
        ratio=ratio,
        route=str(route),
        initial=initial,
    )


def _declared(case: str, route: str, with_support: bool) -> dict[str, Any]:
    declared = support(case, route=route) if with_support else None
    return {} if declared is None else {"treatment_support": declared}


def density_estimator(
    case: str, frame: pd.DataFrame, *, outcome: str = "correct", hazard: str = "oracle"
) -> TMLE:
    """The density-route estimator of :func:`fit_density`, on the grid of ``frame``'s doses.

    Call it inside :func:`reading` for a control, so the grid is the control's.
    """
    name = law_of(case)
    bins = default_density_bins(len(frame))
    edges = tuple(density_module.bin_edges(frame["A"].to_numpy(dtype=float), bins))
    learner = (
        law.OracleHazard(edges, name) if hazard == "oracle" else OracleHazardWithoutW1(edges, name)
    )
    return TMLE(
        policies=(policy(case),),
        outcome_learner=law.CorrectOutcome() if outcome == "correct" else WrongOutcome(),
        treatment_learner=learner,
        ratio="density",
        density_bins=bins,
        cross_fit=False,
        simultaneous=False,
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=0,
    )


def classifier_estimator(case: str, *, outcome: str = "correct", without_w1: bool = False) -> TMLE:
    """The estimator of :func:`fit_classifier`: ``ratio=None``, no treatment learner."""
    learner: Any = WithoutW1(default_stack()) if without_w1 else None
    return TMLE(
        policies=(policy(case),),
        outcome_learner=law.CorrectOutcome() if outcome == "correct" else WrongOutcome(),
        treatment_learner=learner,
        ratio=None,
        cross_fit=True,
        n_folds=POINT_FOLDS,
        q_bounds=Q_BOUNDS,
        simultaneous=False,
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=0,
    )


def point_data(
    case: str, frame: pd.DataFrame, *, route: str, with_support: bool = True
) -> CausalData:
    """``frame`` as the prepared point data a fit of ``case`` reads, with its declared support."""
    data = CausalData.from_frame(
        frame, outcome="Y", treatment="A", covariates=["W1", "W2"], treatment_kind="continuous"
    )
    declared = support(case, route=route) if with_support else None
    return data if declared is None else data.with_dose_declarations(declared)


def fit_density(
    case: str,
    frame: pd.DataFrame,
    *,
    reading_kind: str = "shipped",
    outcome: str = "correct",
    hazard: str = "oracle",
) -> PointFit:
    """The density route with the oracle hazard, no cross-fitting, at the default bins.

    ``reading_kind`` is :func:`reading`'s argument; ``support_ignored`` and ``no_row_break``
    fit without the declared support.  ``outcome`` is ``correct`` or ``wrong`` (OLS on
    ``[1, A, W1, W2]``); ``hazard`` is ``oracle`` or ``without_w1``.
    """
    with reading(reading_kind):
        estimator = density_estimator(case, frame, outcome=outcome, hazard=hazard)
        with_support = reading_kind not in {"support_ignored", "no_row_break"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = estimator.fit(
                frame,
                outcome="Y",
                treatment="A",
                covariates=["W1", "W2"],
                **_declared(case, "density", with_support),
            )
    fitted = _point_fit(result, plug_in(case, frame, outcome))
    if fitted.route != "density":
        raise RuntimeError(f"the density fit recorded the {fitted.route!r} route")
    return fitted


def fit_classifier(
    case: str,
    frame: pd.DataFrame,
    *,
    with_steps: bool = True,
    outcome: str = "correct",
    without_w1: bool = False,
) -> PointFit:
    """The route ``ratio=None`` resolves to on a continuous dose: the classifier, default stack.

    Cross-fitted at :data:`POINT_FOLDS` outer folds.  ``without_w1`` hides ``W1`` from the
    classifier (the both-wrong contrast of the classifier's DR cells).
    """
    estimator = classifier_estimator(case, outcome=outcome, without_w1=without_w1)
    with steps(with_steps), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = estimator.fit(
            frame,
            outcome="Y",
            treatment="A",
            covariates=["W1", "W2"],
            **_declared(case, "classifier", True),
        )
    fitted = _point_fit(result, plug_in(case, frame, outcome))
    if fitted.route != "classifier":
        raise RuntimeError(f"ratio=None resolved to {fitted.route!r}, not the classifier")
    return fitted


def ic_gap(case: str, frame: pd.DataFrame, fitted: PointFit) -> float:
    """``SD(IC - D0) / SD(D0)``: the influence curve's distance from the efficient one."""
    return float(np.std(fitted.influence_curve - eif(case, frame)) / efficiency_sd(case))


def remainder_proxy(case: str, frame: pd.DataFrame, ratio: Any) -> float:
    """``sqrt(n) E_n[(r_n - r0)(Q0 - Q_ols)] / SD(D0)``: the DR remainder with the wrong Q."""
    w1, w2, a, _ = law.columns(frame)
    gap = np.asarray(ratio, dtype=float) - true_ratio(case, a, w1, w2)
    miss = law.q0(a, w1, w2) - law.wrong_q(a, w1, w2, frame)
    return float(np.sqrt(len(frame)) * np.mean(gap * miss) / efficiency_sd(case))


def pre_fix_one_step(case: str, frame: pd.DataFrame) -> tuple[float, float]:
    """The pre-fix DR reading: the one-step estimate with the wrong Q and the histogram ratio.

    Exact bin masses on the equal-mass grid at the default bin count, each bin read as its mass
    over its width (:func:`tests.mtp_density_tails_law.histogram`); the one-step estimate
    ``E_n[Q(d(A), W)] + E_n[r (Y - Q(A, W))]`` and its influence-curve standard error.  Written
    here, independent of the package, as rev 2's study read it.
    """
    name = law_of(case)
    w1, w2, a, y = law.columns(frame)
    edges = law.equal_mass_edges(a, default_density_bins(a.size))
    p = law.bin_probabilities(name, edges, w1, w2)

    def density(values: Any) -> np.ndarray:
        return law.histogram(p, edges, values)

    ratio = law.ratio_from(name, density, a, w1, w2)
    q_obs = law.wrong_q(a, w1, w2, frame)
    q_shift = law.wrong_q(assign(case, a, w1, w2), w1, w2, frame)
    curve = q_shift + ratio * (y - q_obs)
    return float(np.mean(curve)), float(np.std(curve, ddof=1) / np.sqrt(a.size))


# ------------------------------------------------------------------ learned readings


@dataclass(frozen=True)
class DensityReading:
    """A learned density's ratio at every row, with what the misfit signature reads.

    Parameters
    ----------
    ratio : numpy.ndarray
        ``r_n`` at each row.
    error : float
        Its ratio error, ``||r_n - r0||_n / SD(D0)``.
    density : ConditionalDensity
        The fitted density, kept for the signature.
    """

    ratio: np.ndarray
    error: float
    density: ConditionalDensity


def _trimmed(doses: Any) -> np.ndarray:
    a = np.asarray(doses, dtype=float)
    low, high = np.quantile(a, [0.05, 0.95])
    return np.asarray(a[(a >= low) & (a <= high)], dtype=float)


def learned_density(
    case: str,
    frame: pd.DataFrame,
    learner_name: str,
    *,
    reading_kind: str = "shipped",
    with_support: bool = True,
    occupancy_from: pd.DataFrame | None = None,
) -> DensityReading:
    """Cross-fit a learned hazard at :data:`POINT_FOLDS` folds and read its ratio.

    ``reading_kind`` is :func:`reading`'s argument (``pre_fix`` for the representation before
    the fix).  ``occupancy_from`` is the broken reading of the signature's control: the
    occupancy of another sample's doses travels with this sample's grid, the error rev 3's
    ``occupied=`` rule closed.  Every default bin holds a dose of either sample, so the other
    sample is trimmed to its central 90% (:func:`_trimmed`), which empties the outer bins the
    way a training fold read on the wrong rows does.
    """
    name = law_of(case)
    w1, w2, a, _ = law.columns(frame)
    n = a.size
    covariates = np.column_stack([w1, w2])
    bounds = (np.zeros(n), law.upper(name, w1, w2)) if name in law.SUPPORTED else None
    with reading(reading_kind), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bins = default_density_bins(n)
        extra: dict[str, Any] = {}
        if occupancy_from is not None:
            edges = density_module.bin_edges(a, bins)
            extra = {
                "edges": tuple(edges),
                "occupied": bin_occupancy(_trimmed(occupancy_from["A"]), edges),
            }
        else:
            extra = {"n_bins": bins}
        density, _ = fit_conditional_density(
            DENSITY_LEARNERS[learner_name](),
            covariates,
            a,
            np.ones(n),
            make_folds(n, POINT_FOLDS, random_state=0),
            support=bounds if with_support else None,
            **extra,
        )
        # The reading is lazy, so it is evaluated inside the patched representation.
        ratio = law.ratio_from(name, density.density_at, a, w1, w2)
    return DensityReading(ratio, ratio_error(case, ratio, frame), density)


def classifier_ratio(case: str, frame: pd.DataFrame, *, with_steps: bool = True) -> np.ndarray:
    """The default stack's cross-fitted classifier ratio, with or without the policy steps."""
    data = CausalData.from_frame(
        frame, outcome="Y", treatment="A", covariates=["W1", "W2"], treatment_kind="continuous"
    )
    declared = support(case, route="classifier")
    if declared is not None:
        data = data.with_dose_declarations(declared)
    with steps(with_steps), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        evaluated = PolicySet.evaluate_by_classifier(
            (policy(case),),
            data,
            default_stack(),
            make_folds(len(frame), POINT_FOLDS, random_state=0),
        )
    return np.asarray(evaluated.ratio[:, 0], dtype=float)


# ------------------------------------------------------------------ the misfit signature

#: Mark (1): the exact bin probabilities on the same grid read at most this ratio error.
SIGNATURE_EXACT = 0.15
#: Mark (2): the five worst rows carry at least this share of the squared error.
SIGNATURE_WORST_ROWS = 5
SIGNATURE_SHARE = 0.5
#: Mark (3): the log-probability mismatch at the worst row.
SIGNATURE_LOG_GAP = 2.0


@dataclass(frozen=True)
class Signature:
    """The three marks of a learned-hazard misfit on one breaching draw.

    Parameters
    ----------
    exact_error : float
        The ratio error of the exact bin probabilities on the fitted density's grid.
    worst_share : float
        The share of the squared ratio error the five worst rows carry.
    log_gap : float
        At the worst row, the larger of ``log p_exact - log p_fit`` at the bin of ``A`` and
        ``log p_fit - log p_exact`` at the bin of ``A - delta``.
    """

    exact_error: float
    worst_share: float
    log_gap: float

    @property
    def marks(self) -> tuple[bool, bool, bool]:
        """Whether each mark holds."""
        return (
            self.exact_error <= SIGNATURE_EXACT,
            self.worst_share >= SIGNATURE_SHARE,
            self.log_gap >= SIGNATURE_LOG_GAP,
        )

    @property
    def learner_limitation(self) -> bool:
        """All three marks: the breach is the learner's misfit, not a defect suspect."""
        return all(self.marks)


def signature(case: str, frame: pd.DataFrame, fitted: DensityReading) -> Signature:
    """The three marks of :class:`Signature` on a fitted reading, read on its own grid."""
    name = law_of(case)
    w1, w2, a, _ = law.columns(frame)
    density = fitted.density
    edges = np.asarray(density.edges, dtype=float)
    exact = ConditionalDensity(
        law.bin_probabilities(name, edges, w1, w2), edges, density.occupied, density.support
    )
    exact_ratio = law.ratio_from(name, exact.density_at, a, w1, w2)
    squared = (fitted.ratio - true_ratio(case, a, w1, w2)) ** 2
    order = np.argsort(squared)[::-1]
    worst = int(order[0])
    total = float(squared.sum())
    share = float(squared[order[:SIGNATURE_WORST_ROWS]].sum() / total) if total > 0 else 0.0
    p_fit = np.asarray(density.bin_probabilities, dtype=float)[worst]
    p_exact = np.asarray(exact.bin_probabilities, dtype=float)[worst]
    tiny = 1e-300

    def bin_of(value: float) -> int:
        return int(np.clip(np.searchsorted(edges, value, side="right") - 1, 0, edges.size - 2))

    at_dose, below = bin_of(a[worst]), bin_of(a[worst] - law.DELTA)
    low = np.log(p_exact[at_dose] + tiny) - np.log(p_fit[at_dose] + tiny)
    high = np.log(p_fit[below] + tiny) - np.log(p_exact[below] + tiny)
    return Signature(ratio_error(case, exact_ratio, frame), share, float(max(low, high)))


# ------------------------------------------------------------------ cluster helpers

#: The shipped warning's ``min_rows``.
CLUSTER_MIN_ROWS = 3


def warns(doses: Any) -> bool:
    """Whether the shipped helper finds a cluster beyond a gap bin (``min_rows = 3``)."""
    return bool(isolated_dose_clusters(np.asarray(doses, dtype=float), CLUSTER_MIN_ROWS))


def _every_bin(values: Any, edges: Any) -> np.ndarray:
    return np.ones(np.asarray(edges).size - 1, dtype=bool)


def warns_gap_disabled(doses: Any) -> bool:
    """Control (i): the helper with every bin read as occupied, so no gap bin exists."""
    with _patched(density_module, "bin_occupancy", _every_bin):
        return bool(isolated_dose_clusters(np.asarray(doses, dtype=float), CLUSTER_MIN_ROWS))


def warns_eager(doses: Any) -> bool:
    """Control (ii): rev 2's absolute gap test with ``min_rows = 1``.

    Rev 2's grid called any empty run wider than twice the interior core width a gap; rev 3
    also asks the run to be wider than the span of the ``TAIL_ROWS`` doses beside it.  With
    ``TAIL_ROWS = 1`` that span is the distance to the next dose, which every empty run
    exceeds, so the test is rev 2's; and any single dose beyond a gap warns.
    """
    with _patched(density_module, "TAIL_ROWS", 1):
        return bool(isolated_dose_clusters(np.asarray(doses, dtype=float), 1))


# ------------------------------------------------------------------ longitudinal default route

#: The gated estimand of each longitudinal arm, and the regimen it contrasts.
LONGITUDINAL_GATED = "ate_regimen[up vs natural]"
LONGITUDINAL_SCENARIO = longitudinal_study.CONTINUOUS
LONGITUDINAL_ESTIMANDS = longitudinal_study.ESTIMANDS[LONGITUDINAL_SCENARIO]
LONGITUDINAL_TRUTH = dict(longitudinal_study.TRUTH[LONGITUDINAL_SCENARIO])
LONGITUDINAL_ESTIMATORS = {"LTMLE": LTMLE, "SDR": SDR}


def draw_longitudinal(n: int, seed: int) -> pd.DataFrame:
    """``n`` rows of ``longitudinal-mtp``'s continuous law."""
    return longitudinal.sample_continuous(n, seed)


def longitudinal_estimator(estimator: str) -> Any:
    """The longitudinal default-route estimator of :func:`fit_longitudinal`."""
    walk = {"max_iter": MAX_ITER, "tol": TOL} if estimator == "LTMLE" else {}
    return LONGITUDINAL_ESTIMATORS[estimator](
        longitudinal.continuous_regimens(),
        reference="natural",
        outcome_learner=QuasiBinomialGLM(),
        pseudo_learner=LinearRegression(),
        treatment_learner=None,
        n_folds=LONGITUDINAL_FOLDS,
        learner_folds=longitudinal_study.LEARNER_FOLDS,
        g_bounds=longitudinal_study.G_BOUNDS,
        ratio=None,
        simultaneous=False,
        random_state=0,
        **walk,
    )


#: The columns of the longitudinal law, as every longitudinal fit declares them.
LONGITUDINAL_COLUMNS: dict[str, Any] = {
    "outcome": "Y",
    "treatment": ["A1", "A2"],
    "baseline": ["W"],
    "time_varying": [[], ["L2"]],
    "continuous_treatment": ["A1", "A2"],
}


def longitudinal_data(frame: pd.DataFrame) -> LongitudinalData:
    """``frame`` as the prepared container a longitudinal fit reads."""
    return LongitudinalData.from_frame(frame, **LONGITUDINAL_COLUMNS)


def fit_longitudinal(estimator: str, frame: pd.DataFrame) -> Any:
    """The longitudinal default route: ``ratio=None`` at both continuous nodes.

    ``longitudinal-mtp``'s four continuous plans, its primary outcome learners, five outer
    folds, and no treatment learner, so each continuous node resolves to the classifier with
    the default stack and its per-node steps.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = longitudinal_estimator(estimator).fit(frame, **LONGITUDINAL_COLUMNS)
    route = result.config.ratio_route
    if route is None or route.route != "classifier":
        raise RuntimeError(f"the longitudinal default resolved to {route!r}, not the classifier")
    return result
