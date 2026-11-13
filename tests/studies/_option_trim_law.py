r"""Laws, learners, fits and exact-law limits of ``option-ratio-trim`` (X76 part (e)).

``LTMLE(ratio_trim=q)`` and ``SDR(ratio_trim=q)`` replace every per-node density ratio above
the type-7 quantile ``Q_q`` of all ``n x T`` entries, zeros included, by ``Q_q``, and multiply
(:mod:`cleverly.longitudinal.ratio_trim`).  The estimand does not change.  Targeting solves the
trimmed score at every node, so in the limit

.. math::

    \psi(Q^*) - \psi_0 = \sum_t P_0\big[(\tilde R_t - R_t)\,(Q^*_{t+1} - Q^*_t)\big],

with :math:`R_t` the cumulative ratio, :math:`\tilde R_t` the trimmed one and
:math:`Q^*_{T+1} = Y`.  The difference :math:`\tilde R_t - R_t` does not vanish on the trimmed
share at any n, so the remainder is first order in the outcome-regression error: the trimmed
fit is consistent when every outcome regression is, and double robustness is lost on the
trimmed share (Fix-8 plan section 2.5).  :func:`trim_remainder` computes the right side.

Law ``LE`` (plan section 5.6), two nodes, with ``W ~ Uniform(0, 1)``:

* ``A_1 ~ Bernoulli(expit(-3 + 4 W))``, retention ``C_1 ~ Bernoulli(expit(2.5 + 0.5 W))``;
* ``L_2 ~ Bernoulli(0.2 + 0.3 W + 0.2 A_1)``;
* ``A_2 ~ Bernoulli(expit(-2.5 + 3 W + 0.5 L_2 + 0.5 A_1))``, retention
  ``C_2 ~ Bernoulli(expit(2.5 + 0.5 W))``;
* ``Y ~ Bernoulli(0.05 + 0.6 W + 0.15 L_2 + 0.05 A_1 + 0.1 A_2)``.

Under the regimen always-treat ``Q_2 = 0.2 + 0.6 W + 0.15 L_2`` and ``Q_1 = 0.26 + 0.645 W``, so
both iterated regressions are linear and ``LinearRegression`` contains them; the truth is
``0.5825``.  Low-``W`` units have ``g_1 g_2`` near 0.005, so the ratios are heavy-tailed.  The
mechanisms are logistic and fitted by :class:`Logistic`, the study's exact Newton maximum
likelihood.  The wrong outcome learner :class:`DropFirst` drops ``W``, so its node-1 regression
is an intercept; its error is largest at low ``W``, where the trim binds.

**Exact-law limits.**  :func:`quadrature_frame` writes the law as a weighted finite data set,
Gauss-Legendre nodes in ``W`` times every path, each row weighted by its probability.  A fit
with ``weights=`` on it is the estimator applied to the law itself.  The package computes the
trim's quantile from the unweighted entries, which on the quadrature rows is not the law's
quantile, so :func:`exact_limit` replaces the threshold by :func:`population_threshold`, the
quantile of the law's entry distribution, which the sample quantile converges to.
"""

from __future__ import annotations

import contextlib
import importlib
import itertools
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression

T = 2
LAW = "LE"
REGIMEN = "always"
ESTIMAND = "ey_regimen[always]"
#: The declared trim of the gated cells, and the reported one.
TRIM = 0.95
REPORTED_TRIM = 0.99
G_BOUNDS = (1e-8, 1.0)
MAX_ITER = 100
TOL = 1e-10
N_FOLDS = 5
QUADRATURE_NODES = 40
#: Plan 5.0: a control's exact limit must reach this many SDs.
CONTROL_FLOOR_SD = 0.45
TRUTH = 0.5825


# ------------------------------------------------------------------------------- the law
def p_treatment_1(w: Any) -> Any:
    """``P(A_1 = 1 | W)``."""
    return expit(-3.0 + 4.0 * np.asarray(w))


def p_retention(w: Any) -> Any:
    """``P(C_t = 1 | W)``, the probability of staying in follow-up at either node."""
    return expit(2.5 + 0.5 * np.asarray(w))


def p_covariate(w: Any, a1: Any) -> Any:
    """``P(L_2 = 1 | W, A_1)``."""
    return 0.2 + 0.3 * np.asarray(w) + 0.2 * np.asarray(a1)


def p_treatment_2(w: Any, l2: Any, a1: Any) -> Any:
    """``P(A_2 = 1 | W, L_2, A_1)``."""
    return expit(-2.5 + 3.0 * np.asarray(w) + 0.5 * np.asarray(l2) + 0.5 * np.asarray(a1))


def p_outcome(w: Any, l2: Any, a1: Any, a2: Any) -> Any:
    """``P(Y = 1 | W, L_2, A_1, A_2)``."""
    return (
        0.05
        + 0.6 * np.asarray(w)
        + 0.15 * np.asarray(l2)
        + 0.05 * np.asarray(a1)
        + 0.1 * np.asarray(a2)
    )


COLUMNS = ["W", "A1", "C1", "L2", "A2", "C2", "Y"]


def draw(n: int, seed: int) -> pd.DataFrame:
    """One sample of ``LE`` in wide form; a censored unit's later columns are missing."""
    rng = np.random.default_rng(seed)
    w = rng.uniform(size=n)
    a1 = rng.binomial(1, p_treatment_1(w)).astype(float)
    c1 = rng.binomial(1, p_retention(w)).astype(float)
    l2 = rng.binomial(1, p_covariate(w, a1)).astype(float)
    a2 = rng.binomial(1, p_treatment_2(w, l2, a1)).astype(float)
    c2 = rng.binomial(1, p_retention(w)).astype(float)
    y = rng.binomial(1, p_outcome(w, l2, a1, a2)).astype(float)
    frame = pd.DataFrame({"W": w, "A1": a1, "C1": c1, "L2": l2, "A2": a2, "C2": c2, "Y": y})
    frame.loc[frame["C1"] == 0.0, ["L2", "A2", "C2", "Y"]] = np.nan
    frame.loc[frame["C2"] == 0.0, ["Y"]] = np.nan
    return frame[COLUMNS]


def truth() -> float:
    """``E[Y^always]`` by the g-formula: ``Q_1 = 0.26 + 0.645 W`` at ``E[W] = 1/2``."""
    q1 = lambda w: 0.2 + 0.6 * w + 0.15 * float(p_covariate(w, 1.0))  # noqa: E731
    return 0.5 * (q1(0.0) + q1(1.0))


def _ratio_factors() -> list[Any]:
    """Each positive entry's denominator ``g c`` as a function of ``W``, all increasing."""

    def node_1(w: float) -> float:
        return float(p_treatment_1(w) * p_retention(w))

    def node_2(a1: float, l2: float) -> Any:
        return lambda w: float(p_treatment_2(w, l2, a1) * p_retention(w))

    return [node_1, *(node_2(a1, l2) for a1 in (0.0, 1.0) for l2 in (0.0, 1.0))]


def cut_points(threshold: float) -> tuple[float, ...]:
    """The ``W`` at which some entry's ratio equals ``threshold``: the kinks of ``min(r, Q)``."""
    cuts = []
    for factor in _ratio_factors():
        if factor(0.0) * threshold < 1.0 < factor(1.0) * threshold:
            root = brentq(lambda w, f=factor: f(w) * threshold - 1.0, 0.0, 1.0, xtol=1e-15)
            cuts.append(float(root))
    # Two node-2 factors coincide (0.5 L_2 + 0.5 A_1 is the same at (0, 1) and (1, 0)).
    return tuple(sorted(set(cuts)))


def quadrature_frame(nodes: int = QUADRATURE_NODES, cuts: tuple[float, ...] = ()) -> pd.DataFrame:
    """``LE`` as a weighted finite data set: column ``weight`` holds each row's probability.

    ``W`` takes a ``nodes``-point Gauss-Legendre rule on each piece of ``[0, 1]`` between the
    ``cuts``, so a trimmed fit's integrands are smooth on every piece.
    """
    x, weights = np.polynomial.legendre.leggauss(nodes)
    edges = [0.0, *cuts, 1.0]
    points = np.concatenate(
        [lo + (hi - lo) * (x + 1.0) / 2.0 for lo, hi in itertools.pairwise(edges)]
    )
    masses = np.concatenate([(hi - lo) * weights / 2.0 for lo, hi in itertools.pairwise(edges)])
    rows: list[dict[str, float]] = []
    nan = np.nan
    for w, ww in zip(points, masses, strict=True):
        w = float(w)
        for a1 in (0.0, 1.0):
            pa1 = float(p_treatment_1(w)) if a1 else 1.0 - float(p_treatment_1(w))
            pc = float(p_retention(w))
            base = ww * pa1
            rows.append(
                {
                    "W": w,
                    "A1": a1,
                    "C1": 0.0,
                    "L2": nan,
                    "A2": nan,
                    "C2": nan,
                    "Y": nan,
                    "weight": base * (1.0 - pc),
                }
            )
            for l2 in (0.0, 1.0):
                pl = float(p_covariate(w, a1)) if l2 else 1.0 - float(p_covariate(w, a1))
                for a2 in (0.0, 1.0):
                    pa2 = float(p_treatment_2(w, l2, a1))
                    pa2 = pa2 if a2 else 1.0 - pa2
                    path = base * pc * pl * pa2
                    node = {"W": w, "A1": a1, "C1": 1.0, "L2": l2, "A2": a2}
                    rows.append({**node, "C2": 0.0, "Y": nan, "weight": path * (1.0 - pc)})
                    py = float(p_outcome(w, l2, a1, a2))
                    for y, py_ in ((1.0, py), (0.0, 1.0 - py)):
                        rows.append({**node, "C2": 1.0, "Y": y, "weight": path * pc * py_})
    frame = pd.DataFrame(rows)
    return frame[[*COLUMNS, "weight"]]


def true_ratios(frame: pd.DataFrame) -> np.ndarray:
    """``(n, T)`` per-node ratios of always-treat under the true mechanisms, as the trim reads
    them: zero off the plan or off follow-up at the node."""
    w = frame["W"].to_numpy()
    a1 = frame["A1"].to_numpy()
    c1 = frame["C1"].to_numpy()
    l2 = np.nan_to_num(frame["L2"].to_numpy())
    a2 = np.nan_to_num(frame["A2"].to_numpy())
    c2 = np.nan_to_num(frame["C2"].to_numpy())
    r1 = np.where((a1 == 1.0) & (c1 == 1.0), 1.0 / (p_treatment_1(w) * p_retention(w)), 0.0)
    r2 = np.where(
        (c1 == 1.0) & (a2 == 1.0) & (c2 == 1.0),
        1.0 / (p_treatment_2(w, l2, a1) * p_retention(w)),
        0.0,
    )
    return np.column_stack([r1, r2])


def _survival(x: float) -> float:
    """``P(r_1 > x) + P(r_2 > x)`` for ``x > 0`` under ``LE``."""

    def below(density: Any, factor: Any) -> float:
        # ``factor`` increases in W, so ``1 / factor > x`` exactly when W lies below a cut.
        if factor(1.0) * x <= 1.0:
            return float(quad(density, 0.0, 1.0, epsabs=1e-14, epsrel=1e-12, limit=200)[0])
        if factor(0.0) * x >= 1.0:
            return 0.0
        cut = brentq(lambda w: factor(w) * x - 1.0, 0.0, 1.0, xtol=1e-15)
        return float(quad(density, 0.0, cut, epsabs=1e-14, epsrel=1e-12, limit=200)[0])

    node_1 = _ratio_factors()[0]
    total = below(node_1, node_1)
    for a1 in (0.0, 1.0):
        for l2 in (0.0, 1.0):

            def factor(w: float, a1: float = a1, l2: float = l2) -> float:
                return float(p_treatment_2(w, l2, a1) * p_retention(w))

            def density(w: float, a1: float = a1, l2: float = l2) -> float:
                pa1 = p_treatment_1(w) if a1 else 1.0 - p_treatment_1(w)
                pl = p_covariate(w, a1) if l2 else 1.0 - p_covariate(w, a1)
                return float(pa1 * p_retention(w) * pl * factor(w))

            total += below(density, factor)
    return total


def population_threshold(quantile: float) -> float:
    """The ``quantile`` of the law's per-node ratio entries, the limit of the sample ``Q_q``.

    An entry is drawn from either node with probability one half, so ``Q_q`` solves
    ``P(r_1 > x) + P(r_2 > x) = 2 (1 - q)``.
    """
    target = 2.0 * (1.0 - quantile)
    return float(brentq(lambda x: _survival(x) - target, 1.0 + 1e-9, 1e4, xtol=1e-13))


def trimmed_share(quantile: float) -> float:
    """The share of the positive entries above ``Q_q``, the limit of ``trimmed_share``."""
    return 2.0 * (1.0 - quantile) / _survival(1e-12)


# ---------------------------------------------------------------------------- the learners
def newton_logistic(design: np.ndarray, target: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Weighted logistic maximum likelihood with an intercept, by Newton, ridge ``1e-12``."""
    x = np.column_stack([np.ones(len(target)), design])
    beta = np.zeros(x.shape[1])
    beta[0] = logit(np.clip(np.average(target, weights=weight), 1e-6, 1.0 - 1e-6))
    for _ in range(100):
        p = expit(x @ beta)
        hessian = (x * (weight * p * (1.0 - p))[:, None]).T @ x
        step = np.linalg.solve(hessian + 1e-12 * np.eye(len(beta)), x.T @ (weight * (target - p)))
        beta = beta + step
        if np.max(np.abs(step)) < 1e-12:
            return beta
    raise RuntimeError("the logistic fit did not converge")


class Logistic(BaseEstimator, ClassifierMixin):
    """The study's mechanism learner: unpenalized logistic maximum likelihood by Newton."""

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> Logistic:
        y = np.asarray(target, dtype=float)
        weight = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        self.beta_ = newton_logistic(np.asarray(design, dtype=float), y, weight)
        self.classes_ = np.array([0.0, 1.0])
        self.n_features_in_ = np.asarray(design).shape[1]
        return self

    def predict_proba(self, design: Any) -> np.ndarray:
        p = expit(self.beta_[0] + np.asarray(design, dtype=float) @ self.beta_[1:])
        return np.column_stack([1.0 - p, p])


class DropFirst(BaseEstimator, RegressorMixin):
    """The wrong outcome learner: a linear regression without the first column, ``W``.

    At node 1 the design holds ``W`` alone, so the regression is an intercept.
    """

    def fit(self, design: Any, target: Any, sample_weight: Any = None) -> DropFirst:
        x = np.asarray(design, dtype=float)
        y = np.asarray(target, dtype=float)
        if x.shape[1] < 2:
            self.constant_ = float(np.average(y, weights=sample_weight))
            self.model_ = None
            return self
        self.model_ = LinearRegression().fit(x[:, 1:], y, sample_weight=sample_weight)
        return self

    def predict(self, design: Any) -> np.ndarray:
        x = np.asarray(design, dtype=float)
        if self.model_ is None:
            return np.full(len(x), self.constant_)
        return np.asarray(self.model_.predict(x[:, 1:]), dtype=float)


# --------------------------------------------------------------------------- the settings
@dataclass(frozen=True)
class Setting:
    """One fit configuration.

    Parameters
    ----------
    estimator : str
        ``"LTMLE"`` or ``"SDR"``.
    outcome : str
        ``"right"`` (``LinearRegression``) or ``"wrong"`` (:class:`DropFirst`).
    trim : float or None
        The declared ``ratio_trim=``, or ``None``.
    cross_fit : bool
        Five-fold cross-fitting, or in sample.
    """

    estimator: str
    outcome: str
    trim: float | None
    cross_fit: bool = False


SETTINGS: dict[str, Setting] = {
    **{
        f"{est.lower()}_{'stacked' if cross else 'in_sample'}_right_trim95": Setting(
            est, "right", TRIM, cross
        )
        for est in ("LTMLE", "SDR")
        for cross in (False, True)
    },
    **{
        f"{est.lower()}_in_sample_{q}_{label}": Setting(est, q, trim)
        for est in ("LTMLE", "SDR")
        for q in ("right", "wrong")
        for label, trim in (("untrimmed", None), ("trim95", TRIM), ("trim99", REPORTED_TRIM))
    },
}


def longitudinal_data(frame: pd.DataFrame, *, weights: str | None = None) -> Any:
    """The study's ``LongitudinalData``."""
    from cleverly.longitudinal import LongitudinalData

    return LongitudinalData.from_frame(
        frame,
        outcome="Y",
        treatment=["A1", "A2"],
        baseline=["W"],
        time_varying=[[], ["L2"]],
        censoring=["C1", "C2"],
        weights=weights,
    )


def estimator(setting: Setting, **overrides: Any) -> Any:
    """The setting's ``LTMLE`` or ``SDR``."""
    from cleverly.longitudinal import LTMLE, SDR

    outcome = LinearRegression() if setting.outcome == "right" else DropFirst()
    options: dict[str, Any] = {
        "outcome_learner": outcome,
        "pseudo_learner": outcome,
        "treatment_learner": Logistic(),
        "censoring_learner": Logistic(),
        "n_folds": N_FOLDS if setting.cross_fit else 1,
        "simultaneous": False,
        "g_bounds": G_BOUNDS,
        "ratio_trim": setting.trim,
        "random_state": 0,
    }
    if setting.estimator == "LTMLE":
        options.update(max_iter=MAX_ITER, tol=TOL)
    options.update(overrides)
    cls = LTMLE if setting.estimator == "LTMLE" else SDR
    return cls({REGIMEN: 1}, **options)


def fit(frame: pd.DataFrame, name: str, *, weights: str | None = None, **overrides: Any) -> Any:
    """Fit the named setting on one sample (or on a weighted law with ``weights=``)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return estimator(SETTINGS[name], **overrides).fit(longitudinal_data(frame, weights=weights))


# ------------------------------------------------------------------------ exact-law limits
@contextlib.contextmanager
def fixed_threshold(threshold: float) -> Iterator[None]:
    """Make every trim inside use ``threshold`` in place of the sample quantile."""
    estimator_module = importlib.import_module("cleverly.longitudinal.estimator")
    trim_module = importlib.import_module("cleverly.longitudinal.ratio_trim")
    original = estimator_module.trim_ratios

    def trim(data: Any, plans: Any, mechanism: Any, quantile: float) -> dict[str, Any]:
        out = {}
        for plan in plans:
            entries = trim_module.node_ratio_entries(data, plan, mechanism)
            out[plan.label] = trim_module.RatioTrim(
                quantile=float(quantile),
                threshold=threshold,
                entries=entries,
                cumulative=np.cumprod(np.minimum(entries, threshold), axis=1),
                trimmed=entries > threshold,
            )
        return out

    estimator_module.trim_ratios = trim  # type: ignore[attr-defined]
    try:
        yield
    finally:
        estimator_module.trim_ratios = original  # type: ignore[attr-defined]


def limit_fit(name: str, nodes: int = QUADRATURE_NODES) -> tuple[pd.DataFrame, Any]:
    """The setting's weighted fit on the quadrature law, trimmed at the law's quantile."""
    setting = SETTINGS[name]
    if setting.trim is None:
        frame = quadrature_frame(nodes)
        return frame, fit(frame, name, weights="weight")
    threshold = population_threshold(setting.trim)
    frame = quadrature_frame(nodes, cut_points(threshold))
    with fixed_threshold(threshold):
        return frame, fit(frame, name, weights="weight")


def trim_remainder(frame: pd.DataFrame, result: Any, quantile: float | None) -> float:
    r"""``sum_t P_0[(R~_t - R_t)(Q*_{t+1} - Q*_t)]`` of a weighted limit fit.

    ``R_t`` and ``R~_t`` are computed here from the true mechanisms and the law's quantile,
    not read from the fit; ``Q*_{t+1}`` and ``Q*_t`` are the fit's pseudo-outcome and
    targeted prediction at each node.
    """
    ratios = true_ratios(frame)
    cumulative = np.cumprod(ratios, axis=1)
    if quantile is None:
        trimmed = cumulative
    else:
        trimmed = np.cumprod(np.minimum(ratios, population_threshold(quantile)), axis=1)
    weight = frame["weight"].to_numpy()
    total = 0.0
    for step in result.fits[REGIMEN].steps:
        t = step.time - 1
        residual = np.where(step.at_risk, step.pseudo_outcome - step.targeted, 0.0)
        residual = np.nan_to_num(residual)
        total += float(np.sum(weight * (trimmed[:, t] - cumulative[:, t]) * residual))
    return total


@dataclass(frozen=True)
class ExactLimit:
    """The population limit of one setting.

    Parameters
    ----------
    estimate : float
        The limit of the estimate.
    curve_sd : float
        ``sqrt(E[D^2])`` of the limit of the plug-in influence curve.
    """

    estimate: float
    curve_sd: float

    def bias(self) -> float:
        """The limit's bias against the truth."""
        return self.estimate - TRUTH

    def sd(self, n: int) -> float:
        """The plug-in SE's limit at size ``n``."""
        return self.curve_sd / float(np.sqrt(n))


def exact_limit(name: str, nodes: int = QUADRATURE_NODES) -> ExactLimit:
    """A setting's limit, from its weighted fit on the quadrature law."""
    frame, result = limit_fit(name, nodes)
    estimate = result.estimates[ESTIMAND]
    weight = frame["weight"].to_numpy()
    relative = weight / weight.mean()
    curve = np.asarray(estimate.influence_curve, dtype=float) / relative
    return ExactLimit(float(estimate.psi), float(np.sqrt(np.sum(weight * curve**2) / weight.sum())))
