r"""The laws, truths, learners and fits of the adaptive TMLE studies (X90 T3).

A *law* is a trial of ``S = 1`` rows pooled with external ``S = 0`` rows. Every law is given by
its mechanisms as functions of the covariates ``W``: ``s(W) = P(S = 1 | W)``, the treatment
probabilities of the trial and of the external rows, the trial regression
``Q(1, W, a) = E(Y | S = 1, W, a)`` and the external one ``Q(0, W, a)``. Three exact targets
follow by quadrature from those functions alone:

``psi``
    :math:`\Psi = E[Q(1, W, 1) - Q(1, W, 0)]`, the trial effect over the pooled covariates.
``psi_2``
    :math:`\Psi_2 = E[S\{Q(1, W, 1) - Q(1, W, 0)\}] / p`, the trial effect over the trial's.
``projection``
    :math:`\Psi_w(P_0)`, the working-model projection of a *fitted* dictionary: the fold's
    stored transformer is applied to the quadrature nodes (never refitted there) and the
    weighted Gram systems of plan section 2.3 are solved with the true ``g``, ``Pi`` and
    ``tau``. The CV form averages its folds' projections.

The quadrature is a tensor of Gauss-Legendre panels. A panel boundary sits at every knot of
the stored fold transformer and at every indicator threshold of the law, so the hinge and
indicator factors integrate exactly and only the smooth mechanisms (logistic, sine) carry rule
error, which ``tests/unit/test_adaptive_tmle_trial_design.py`` bounds by doubling the nodes. A
normal covariate is truncated at :data:`NORMAL_LIMIT` standard deviations, whose tail mass is
below 1e-11, and carries its density as the panel weight (the plan's Gauss-Hermite rule cannot
break at a knot).

The learners are explicit and fast (plan section 5): additive cubic-spline regressions with
pairwise products and full interaction with the discrete columns (:class:`SplineLearner`),
which are close to correct on every law of the main study. The flexible Super Learner library
of the plan's "Flexible learner" note is :func:`flexible_learners`.
"""

from __future__ import annotations

import contextlib
import math
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit as _expit
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import SplineTransformer, StandardScaler

from cleverly.estimators._working_basis import bias_columns

FloatArray = np.ndarray

G_BOUNDS = (0.01, 0.99)
N_FOLDS = 5
INNER_FOLDS = 5
#: The truncation of a normal covariate, in standard deviations.
NORMAL_LIMIT = 6.0
#: Gauss-Legendre nodes per panel of the truth quadrature.
PANEL_NODES = 5
#: The interior panel boundaries of a normal covariate (beyond its knots and thresholds).
NORMAL_EDGES = (-3.5, -2.0, -1.0, 0.0, 1.0, 2.0, 3.5)


def expit(x: FloatArray) -> FloatArray:
    return np.asarray(_expit(x), dtype=float)


# ------------------------------------------------------------------------------------------
# laws


@dataclass(frozen=True)
class Missing:
    """Outcome missingness (plan 2.8).

    Trial rows: ``P(Delta = 1 | S = 1, W, A) = expit(trial_intercept + trial_a A + trial_w W1)``
    (missing at random given ``(W, A)``). External rows: ``Delta = 1`` with probability
    ``Phi(external_c0 + external_c1 (Y - E(Y | S = 0, W, A)))``, which depends on the outcome
    itself. The recorded external outcomes are then a selected sample, and the engine learns
    the selection into the bias term.
    """

    trial_intercept: float = 1.6
    trial_a: float = -0.5
    trial_w: float = 0.4
    external_c0: float = 0.8
    external_c1: float = 0.3

    def trial_probability(self, w: FloatArray, a: FloatArray | float) -> FloatArray:
        return expit(self.trial_intercept + self.trial_a * a + self.trial_w * w[:, 0])

    @property
    def external_shift(self) -> float:
        """``E(Y - E(Y | S = 0, W, A) | Delta = 1)`` for a unit-variance outcome error."""
        z = self.external_c0 / math.sqrt(1.0 + self.external_c1**2)
        return float(
            self.external_c1 / math.sqrt(1.0 + self.external_c1**2) * norm.pdf(z) / norm.cdf(z)
        )

    @property
    def external_probability(self) -> float:
        return float(norm.cdf(self.external_c0 / math.sqrt(1.0 + self.external_c1**2)))


@dataclass(frozen=True)
class Law:
    """A trial pooled with external rows, by its mechanisms."""

    name: str
    d: int
    kind: str  # "normal" or "uniform"
    lo: float
    hi: float
    s: Callable[[FloatArray], FloatArray]
    a_trial: Callable[[FloatArray], FloatArray]
    a_external: Callable[[FloatArray], FloatArray]
    q1: Callable[[FloatArray, float], FloatArray]
    q0: Callable[[FloatArray, float], FloatArray]
    sd_y: float = 1.0
    family: str = "gaussian"
    #: Indicator thresholds, by covariate position, where a mechanism is not smooth.
    breaks: tuple[tuple[float, ...], ...] = ()
    controls_only: bool = False
    missing: Missing | None = None
    note: str = ""
    #: The covariate positions the treatment, trial-membership and observation mechanisms
    #: depend on: the feature set of the mechanism learners (correct for the law).
    mechanism: tuple[int, ...] = (0,)

    def columns(self) -> list[str]:
        return [f"W{j + 1}" for j in range(self.d)]

    # --- derived mechanisms -------------------------------------------------------------
    def g(self, w: FloatArray) -> FloatArray:
        s = self.s(w)
        return s * self.a_trial(w) + (1.0 - s) * self.a_external(w)

    def pi(self, w: FloatArray, a: float) -> FloatArray:
        """``Pi(1 | W, a) = P(S = 1 | W, A = a)``."""
        s, g = self.s(w), self.g(w)
        at = self.a_trial(w)
        if a == 1.0:
            return np.where(g > 0.0, s * at / np.where(g > 0.0, g, 1.0), 1.0)
        return s * (1.0 - at) / (1.0 - g)

    def q0_observed(self, w: FloatArray, a: float) -> FloatArray:
        """``E(Y | S = 0, W, A = a, Delta = 1)``: the external regression the fit can learn."""
        shift = 0.0 if self.missing is None else self.missing.external_shift * self.sd_y
        return self.q0(w, a) + shift

    def tau_s(self, w: FloatArray, a: float) -> FloatArray:
        """``Q(1, W, a) - Q(0, W, a)`` on the recorded outcomes."""
        return self.q1(w, a) - self.q0_observed(w, a)

    def qt(self, w: FloatArray, a: float) -> FloatArray:
        p = self.pi(w, a)
        return p * self.q1(w, a) + (1.0 - p) * self.q0_observed(w, a)

    def tau_a(self, w: FloatArray) -> FloatArray:
        return self.qt(w, 1.0) - self.qt(w, 0.0)

    def tau_trial(self, w: FloatArray) -> FloatArray:
        """The trial's conditional effect, ``Q(1, W, 1) - Q(1, W, 0)``."""
        return self.q1(w, 1.0) - self.q1(w, 0.0)


# --- the App B laws (plan section 5; the paper omits the S mechanism, set here) -----------

#: The trial share of the simple laws: ``S ~ Bernoulli(0.25)`` independent of ``W``, so that
#: n = 4,000 splits about 1,000 trial and 3,000 external rows (the plan's "3x").
TRIAL_SHARE = 0.25


def _normal_a(b: Callable[[FloatArray, float], FloatArray], name: str, note: str) -> Law:
    """Laws (a) and (b): normal covariates, effect 1.5."""

    def q1(w: FloatArray, a: float) -> FloatArray:
        return 2.5 + 0.9 * w[:, 0] + 1.1 * w[:, 1] + 2.7 * w[:, 2] + 1.5 * a

    return Law(
        name=name,
        d=3,
        kind="normal",
        lo=-NORMAL_LIMIT,
        hi=NORMAL_LIMIT,
        s=lambda w: np.full(w.shape[0], TRIAL_SHARE),
        a_trial=lambda w: np.full(w.shape[0], 0.67),
        a_external=lambda w: expit(0.5 * w[:, 0]),
        q1=q1,
        q0=lambda w, a: q1(w, a) + b(w, a),
        note=note,
    )


def _b_zero(w: FloatArray, a: float) -> FloatArray:
    return np.zeros(w.shape[0])


def _b_a(w: FloatArray, a: float) -> FloatArray:
    return (0.2 + 1.1 * w[:, 0] * (1.0 - a)) * np.ones(w.shape[0])


def _b_b(w: FloatArray, a: float) -> FloatArray:
    return 0.5 + 3.1 * w[:, 0] * (1.0 - a) + 0.8 * w[:, 2]


def _uniform_c(b: Callable[[FloatArray, float], FloatArray], name: str, breaks: Any) -> Law:
    """Laws (c) and (d): uniform covariates, effect 4.2."""

    def q1(w: FloatArray, a: float) -> FloatArray:
        return 1.9 + 4.2 * a + 0.9 * w[:, 0] + 1.4 * w[:, 1] + 2.1 * w[:, 2]

    return Law(
        name=name,
        d=3,
        kind="uniform",
        lo=0.0,
        hi=1.0,
        s=lambda w: np.full(w.shape[0], TRIAL_SHARE),
        a_trial=lambda w: np.full(w.shape[0], 0.67),
        a_external=lambda w: expit(w[:, 0]),
        q1=q1,
        q0=lambda w, a: q1(w, a) + b(w, a),
        breaks=breaks,
    )


def _b_c(w: FloatArray, a: float) -> FloatArray:
    return 0.3 + 0.9 * w[:, 1] * (1.0 - a) + 0.7 * w[:, 2] * (w[:, 1] > 0.5)


def _b_d(w: FloatArray, a: float) -> FloatArray:
    return 0.3 + 1.1 * w[:, 0] * (1.0 - a) + 0.9 * w[:, 1] ** 2 * w[:, 2]


def positivity_law(alpha: float = 0.5) -> Law:
    """The positivity stress law: ``S`` depends on ``W``; constant effect 1.5."""

    def q1(w: FloatArray, a: float) -> FloatArray:
        return 1.9 + 1.5 * a + 0.9 * w[:, 0] + 1.4 * w[:, 1] + 2.1 * w[:, 2]

    return Law(
        name="a3",
        d=3,
        kind="uniform",
        lo=-1.0,
        hi=1.0,
        s=lambda w: expit(
            alpha * (-2.0 + w[:, 0] + w[:, 1] + np.sin(2 * w[:, 0]) + np.sin(2 * w[:, 1]))
        ),
        a_trial=lambda w: np.full(w.shape[0], 0.67),
        a_external=lambda w: expit(-0.5 * w[:, 0]),
        q1=q1,
        q0=lambda w, a: q1(w, a) + (0.2 + 2.1 * w[:, 0] * a),
        sd_y=0.2,
        mechanism=(0, 1),
    )


#: The separating law E1 (plan 5, B1). The parameters come from exact-law computation in the
#: design test only (plan: "never from estimator runs").
E1 = {"a_s": 3.0, "c_s": 0.0, "a_a": -6.0, "b0": -2.0, "b1": -2.0, "b2": 2.0}


def separating_law(**override: float) -> Law:
    p = {**E1, **override}

    def q1(w: FloatArray, a: float) -> FloatArray:
        return 1.0 + 1.5 * a + w[:, 0] + w[:, 1] + w[:, 2]

    def bias(w: FloatArray, a: float) -> FloatArray:
        return p["b0"] + p["b1"] * (w[:, 1] > 0.3) * (1.0 - a) + p["b2"] * np.sign(w[:, 1]) * a

    return Law(
        name="e1",
        d=3,
        kind="uniform",
        lo=-1.0,
        hi=1.0,
        s=lambda w: expit(p["a_s"] * (w[:, 1] - p["c_s"])),
        a_trial=lambda w: np.full(w.shape[0], 0.5),
        a_external=lambda w: expit(p["a_a"] * w[:, 1]),
        q1=q1,
        q0=lambda w, a: q1(w, a) + bias(w, a),
        breaks=((), (0.0, 0.3), ()),
        mechanism=(1,),
    )


#: The heterogeneous-effect law H2 (plan 5, B2); ``gamma`` is set in the design test.
H2_GAMMA = 0.5


def heterogeneous_law(gamma: float = H2_GAMMA) -> Law:
    def q1(w: FloatArray, a: float) -> FloatArray:
        return 2.0 + (1.5 + gamma * w[:, 0]) * a + w[:, 0] + w[:, 1]

    return Law(
        name="h2",
        d=3,
        kind="normal",
        lo=-NORMAL_LIMIT,
        hi=NORMAL_LIMIT,
        s=lambda w: expit(-0.3 + 1.0 * w[:, 0]),
        a_trial=lambda w: np.full(w.shape[0], 0.5),
        a_external=lambda w: expit(0.5 * w[:, 0]),
        q1=q1,
        q0=lambda w, a: q1(w, a) + 0.3 + 0.8 * w[:, 0] * (1.0 - a),
    )


def binomial_law() -> Law:
    """B1y: ``Y`` binary; the external risk equals the trial's under treatment and is a
    shrunken trial risk under control (plan 5)."""

    def q1(w: FloatArray, a: float) -> FloatArray:
        return expit(-0.5 + 0.8 * a + 0.5 * w[:, 0] - 0.3 * w[:, 1])

    def q0(w: FloatArray, a: float) -> FloatArray:
        return q1(w, 1.0) if a == 1.0 else 0.8 * q1(w, 0.0) + 0.1

    return Law(
        name="b1y",
        d=2,
        kind="normal",
        lo=-NORMAL_LIMIT,
        hi=NORMAL_LIMIT,
        s=lambda w: np.full(w.shape[0], 0.4),
        a_trial=lambda w: np.full(w.shape[0], 0.5),
        a_external=lambda w: expit(0.5 * w[:, 0]),
        q1=q1,
        q0=q0,
        family="binomial",
    )


def controls_only_law() -> Law:
    """A4: law (b) with controls-only external data."""
    base = _normal_a(_b_b, "a4", "")
    return Law(
        **{
            **base.__dict__,
            "name": "a4",
            "a_external": lambda w: np.zeros(w.shape[0]),
            "controls_only": True,
        }
    )


def missing_law() -> Law:
    """A5: law (b) with missing outcomes (plan 2.8)."""
    base = _normal_a(_b_b, "a5", "")
    return Law(**{**base.__dict__, "name": "a5", "missing": Missing()})


#: S6 (plan Rev 2, Karim 2026): selective enrollment and a large wiggly bias on the (b) frame.
#: ``B = m s(W, A) / sd_0(s)`` with ``m = 4`` residual SDs; ``sd_0`` is the SD of ``s`` under
#: ``A = 0`` over the covariate law (the design test recomputes it by quadrature).
S6_MAGNITUDE = 4.0
S6_SD0 = 1.224569


def selective_law() -> Law:
    """S ~ Bernoulli(expit(-0.9 + 0.8 W1 + 0.5 W2)): 1.4% of the covariate mass has
    P(S = 1 | W) outside [0.05, 0.95], as in Karim (2026) Sec 4.5."""

    def q1(w: FloatArray, a: float) -> FloatArray:
        return 2.5 + 0.9 * w[:, 0] + 1.1 * w[:, 1] + 2.7 * w[:, 2] + 1.5 * a

    def wiggle(w: FloatArray, a: float) -> FloatArray:
        raw = np.sin(2 * w[:, 0]) + np.cos(3 * w[:, 1]) + (1.0 - a) * np.sin(2 * w[:, 2])
        return S6_MAGNITUDE * raw / S6_SD0

    return Law(
        name="s6",
        d=3,
        kind="normal",
        lo=-NORMAL_LIMIT,
        hi=NORMAL_LIMIT,
        s=lambda w: expit(-0.9 + 0.8 * w[:, 0] + 0.5 * w[:, 1]),
        a_trial=lambda w: np.full(w.shape[0], 0.67),
        a_external=lambda w: expit(0.5 * w[:, 0]),
        q1=q1,
        q0=lambda w, a: q1(w, a) + wiggle(w, a),
        mechanism=(0, 1),
    )


def _all_laws() -> dict[str, Law]:
    laws = [
        _normal_a(_b_zero, "a0", "law (a) with B = 0"),
        _normal_a(_b_a, "a1a", "law (a)"),
        _normal_a(_b_b, "a1b", "law (b)"),
        _uniform_c(_b_c, "a2c", ((), (0.5,), ())),
        _uniform_c(_b_d, "a2d", ()),
        separating_law(),
        positivity_law(),
        controls_only_law(),
        missing_law(),
        binomial_law(),
        heterogeneous_law(),
        selective_law(),
    ]
    return {law.name: law for law in laws}


LAWS: dict[str, Law] = _all_laws()


# ------------------------------------------------------------------------------------------
# sampling


def draw(law: Law, n: int, seed: int) -> pd.DataFrame:
    """One sample of ``n`` rows: columns ``Y, A, S, W1..Wd`` (and ``R`` with missing outcomes)."""
    rng = np.random.default_rng(seed)
    if law.kind == "normal":
        w = rng.normal(size=(n, law.d))
    else:
        w = rng.uniform(law.lo, law.hi, size=(n, law.d))
    s = rng.binomial(1, law.s(w))
    a = np.where(
        s == 1,
        rng.binomial(1, law.a_trial(w)),
        rng.binomial(1, law.a_external(w)),
    ).astype(float)
    mean = np.where(
        s == 1,
        np.where(a == 1.0, law.q1(w, 1.0), law.q1(w, 0.0)),
        np.where(a == 1.0, law.q0(w, 1.0), law.q0(w, 0.0)),
    )
    if law.family == "binomial":
        y = rng.binomial(1, mean).astype(float)
    else:
        noise = rng.normal(size=n)
        y = mean + law.sd_y * noise
    frame = pd.DataFrame(w, columns=law.columns())
    frame.insert(0, "S", s.astype(int))
    frame.insert(0, "A", a.astype(int))
    frame.insert(0, "Y", y)
    if law.missing is not None:
        spec = law.missing
        trial_observed = rng.uniform(size=n) < spec.trial_probability(w, a)
        external_observed = rng.uniform(size=n) < norm.cdf(
            spec.external_c0 + spec.external_c1 * (y - mean) / law.sd_y
        )
        observed = np.where(s == 1, trial_observed, external_observed)
        frame["R"] = observed.astype(int)
        frame.loc[~observed, "Y"] = np.nan
    return frame


# ------------------------------------------------------------------------------------------
# quadrature


def _panels(edges: Sequence[float], m: int) -> tuple[FloatArray, FloatArray]:
    nodes, weights = np.polynomial.legendre.leggauss(m)
    left, right = np.asarray(edges[:-1], float), np.asarray(edges[1:], float)
    half = (right - left) / 2.0
    mid = (right + left) / 2.0
    x = (mid[:, None] + half[:, None] * nodes[None, :]).ravel()
    w = (half[:, None] * weights[None, :]).ravel()
    return x, w


def axis_rule(
    law: Law, j: int, extra: Sequence[float], m: int = PANEL_NODES
) -> tuple[FloatArray, FloatArray]:
    """The panels of covariate ``j``: breaks at the law's thresholds and ``extra`` knots.

    The weights integrate against the covariate's density (standard normal, or uniform on
    ``[lo, hi]``).
    """
    inside = [*extra, *(law.breaks[j] if j < len(law.breaks) else ())]
    if law.kind == "normal":
        inside += list(NORMAL_EDGES)
    edges = sorted({law.lo, law.hi, *(t for t in inside if law.lo < t < law.hi)})
    x, w = _panels(edges, m)
    if law.kind == "normal":
        return x, w * norm.pdf(x)
    return x, w / (law.hi - law.lo)


def grid(
    law: Law, knots: Sequence[Sequence[float]] | None = None, m: int = PANEL_NODES
) -> tuple[FloatArray, FloatArray]:
    """The tensor quadrature nodes ``(N, d)`` and weights ``(N,)`` of ``law``."""
    axes = [axis_rule(law, j, () if knots is None else knots[j], m) for j in range(law.d)]
    mesh = np.meshgrid(*[a[0] for a in axes], indexing="ij")
    weights = np.ones_like(mesh[0])
    for k, (_, wk) in enumerate(axes):
        shape = [1] * law.d
        shape[k] = -1
        weights = weights * wk.reshape(shape)
    return np.column_stack([m_.ravel() for m_ in mesh]), weights.ravel()


def exact_targets(law: Law, m: int = PANEL_NODES) -> dict[str, float]:
    """``psi`` and ``psi_2`` by quadrature; ``p = P(S = 1)``."""
    w, wt = grid(law, None, m)
    tau = law.tau_trial(w)
    s = law.s(w)
    p = float(np.sum(wt * s))
    return {
        "psi": float(np.sum(wt * tau)),
        "psi_2": float(np.sum(wt * s * tau) / p),
        "p": p,
    }


def fold_knots(dictionary: Any, d: int) -> list[list[float]]:
    """The knots of a stored fold transformer, in the covariate's own units."""
    basis = dictionary.basis
    knots: list[list[float]] = [[] for _ in range(d)]
    stored = getattr(basis, "knots_", None)
    if stored:
        for j, z in stored.items():
            knots[j] = list(basis.mean_[j] + basis.scale_[j] * np.asarray(z, float))
    return knots


@dataclass(frozen=True)
class Projection:
    """``Psi_w(P_0)`` of one selection, and the coefficients it solved."""

    value: float
    beta_a: FloatArray
    beta_s: FloatArray


def projection(law: Law, model: Any, population: str, m: int = PANEL_NODES) -> Projection:
    """The working-model projection of one stored selection (plan 2.3).

    ``model`` is an engine ``WorkingModel``: its fitted dictionary and the selected columns are
    applied to quadrature nodes broken at the dictionary's own knots, and the Gram systems are
    solved with the true ``g``, ``Pi`` and ``tau``.
    """
    dictionary = model.dictionary
    w, wt = grid(law, fold_knots(dictionary, law.d), m)
    g = law.g(w)
    pi0, pi1 = law.pi(w, 0.0), law.pi(w, 1.0)
    kept = dictionary.phi(w)
    phi = kept[:, model.effect.selected]
    controls = dictionary.controls_only
    psi0 = bias_columns(kept, 0.0, controls)[:, dictionary.psi_keep][:, model.bias.selected]
    psi1 = bias_columns(kept, 1.0, controls)[:, dictionary.psi_keep][:, model.bias.selected]
    ga = wt * g * (1.0 - g)
    beta_a = np.linalg.lstsq((phi * ga[:, None]).T @ phi, phi.T @ (ga * law.tau_a(w)), rcond=None)[
        0
    ]
    w0 = wt * (1.0 - g) * pi0 * (1.0 - pi0)
    w1 = wt * g * pi1 * (1.0 - pi1)
    gram = (psi0 * w0[:, None]).T @ psi0 + (psi1 * w1[:, None]).T @ psi1
    rhs = psi0.T @ (w0 * law.tau_s(w, 0.0)) + psi1.T @ (w1 * law.tau_s(w, 1.0))
    beta_s = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    tau_hash = (1.0 - pi0) * (psi0 @ beta_s) - (1.0 - pi1) * (psi1 @ beta_s)
    mass = wt if population == "pooled" else wt * law.s(w) / float(np.sum(wt * law.s(w)))
    value = float(np.sum(mass * (phi @ beta_a - tau_hash)))
    return Projection(value, beta_a, beta_s)


def projection_truth(
    law: Law, models: Sequence[Any], population: str, m: int = PANEL_NODES
) -> float:
    """The mean over a fit's selections of their projections."""
    return float(np.mean([projection(law, model, population, m).value for model in models]))


# ------------------------------------------------------------------------------------------
# learners


class _Spline(BaseEstimator):
    """Additive cubic splines of ``W`` with pairwise products and full interaction with the
    discrete columns after ``W`` (the arm, the trial indicator).

    The first ``n_w`` columns are the covariates; every later column is a 0/1 modifier. The
    features are ``B(W)`` and the products ``z_i z_j, z_i^2 z_j, z_i z_j^2`` of standardized
    covariates, then each product of modifier columns times ``[1, features]``.
    """

    def __init__(
        self,
        n_w: int = 3,
        n_knots: int = 5,
        products: bool = True,
        c: float = 10.0,
        use: tuple[int, ...] | None = None,
        alpha: float = 1.0,
    ):
        self.n_w = n_w
        self.n_knots = n_knots
        self.products = products
        self.c = c
        self.use = use
        self.alpha = alpha

    def _covariates(self, x: FloatArray) -> FloatArray:
        w = x[:, : self.n_w]
        return w if self.use is None else w[:, list(self.use)]

    def _base(self, w: FloatArray) -> FloatArray:
        w = np.clip(w, self.low_, self.high_)
        z = (w - self.mean_) / self.sd_
        parts = [self.spline_.transform(w)]
        if self.products:
            for i in range(w.shape[1]):
                for j in range(i + 1, w.shape[1]):
                    parts.append(
                        np.column_stack(
                            [z[:, i] * z[:, j], z[:, i] ** 2 * z[:, j], z[:, i] * z[:, j] ** 2]
                        )
                    )
        return np.hstack(parts)

    def _design(self, x: FloatArray) -> FloatArray:
        x = np.asarray(x, float)
        base = self._base(self._covariates(x))
        mods = x[:, self.n_w :]
        pieces = [base]
        for mask in range(1, 2 ** mods.shape[1]):
            m = np.ones(x.shape[0])
            for k in range(mods.shape[1]):
                if mask >> k & 1:
                    m = m * mods[:, k]
            pieces.append(m[:, None])
            pieces.append(base * m[:, None])
        return np.hstack(pieces)

    def _fit_design(self, x: FloatArray) -> FloatArray:
        x = np.asarray(x, float)
        w = self._covariates(x)
        # Winsorize at the 0.5% and 99.5% quantiles: the cubic products z_i^2 z_j of a
        # covariate far in the tail otherwise extrapolate by orders of magnitude.
        self.low_, self.high_ = np.quantile(w, [0.005, 0.995], axis=0)
        w = np.clip(w, self.low_, self.high_)
        self.mean_ = w.mean(axis=0)
        self.sd_ = np.where(w.std(axis=0) > 0, w.std(axis=0), 1.0)
        self.spline_ = SplineTransformer(
            n_knots=self.n_knots, degree=3, include_bias=False, extrapolation="constant"
        ).fit(w)
        design = self._design(x)
        self.scaler_ = StandardScaler().fit(design)
        return self.scaler_.transform(design)


class SplineRegressor(RegressorMixin, _Spline):
    def fit(self, x: FloatArray, y: FloatArray) -> SplineRegressor:
        self.model_ = Ridge(alpha=self.alpha).fit(self._fit_design(x), y)
        return self

    def predict(self, x: FloatArray) -> FloatArray:
        return self.model_.predict(self.scaler_.transform(self._design(x)))


class SplineClassifier(ClassifierMixin, _Spline):
    def fit(self, x: FloatArray, y: FloatArray) -> SplineClassifier:
        self.model_ = LogisticRegression(C=self.c, max_iter=2000).fit(self._fit_design(x), y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, x: FloatArray) -> FloatArray:
        return self.model_.predict_proba(self.scaler_.transform(self._design(x)))


def spline_learners(law: Law) -> dict[str, Any]:
    """The explicit fast learners of the main study (plan section 5)."""
    cls = SplineClassifier if law.family == "binomial" else SplineRegressor
    learners: dict[str, Any] = {
        "outcome": cls(n_w=law.d),
        "treatment": SplineClassifier(n_w=law.d, products=False, use=law.mechanism),
        "source": SplineClassifier(n_w=law.d, products=False, use=law.mechanism),
    }
    if law.missing is not None:
        learners["missingness"] = SplineClassifier(n_w=law.d, products=False, use=(0,))
    return learners


def main_terms_learners(law: Law) -> dict[str, Any]:
    """Main-terms GLMs (plan K6; Karim 2026's default nuisances)."""
    outcome = LogisticRegression(max_iter=2000) if law.family == "binomial" else LinearRegression()
    learners: dict[str, Any] = {
        "outcome": outcome,
        "treatment": LogisticRegression(max_iter=2000),
        "source": LogisticRegression(max_iter=2000),
    }
    if law.missing is not None:
        learners["missingness"] = LogisticRegression(max_iter=2000)
    return learners


def flexible_learners(law: Law, seed: int = 0) -> dict[str, Any]:
    """The compact Super Learner of the plan's "Flexible learner" note: a spline GLM, a shallow
    regularized gradient booster and the lasso, ``nnloglik`` for probabilities."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.pipeline import make_pipeline

    from cleverly import SuperLearner
    from cleverly.learners import default_library

    def library(task: Any) -> list[tuple[str, Any]]:
        lasso = dict(default_library(task))["lasso"]
        if task == "classification":
            return [
                (
                    "spline_glm",
                    make_pipeline(
                        SplineTransformer(n_knots=5, degree=3),
                        LogisticRegression(max_iter=2000),
                    ),
                ),
                (
                    "hgb",
                    HistGradientBoostingClassifier(
                        max_depth=3,
                        learning_rate=0.05,
                        l2_regularization=1.0,
                        early_stopping=False,
                    ),
                ),
                ("lasso", lasso),
            ]
        return [
            (
                "spline_glm",
                make_pipeline(SplineTransformer(n_knots=5, degree=3), LinearRegression()),
            ),
            (
                "hgb",
                HistGradientBoostingRegressor(
                    max_depth=3, learning_rate=0.05, l2_regularization=1.0, early_stopping=False
                ),
            ),
            ("lasso", lasso),
        ]

    def classifier() -> Any:
        return SuperLearner(
            library("classification"),
            task="classification",
            meta_learner="nnloglik",
            clip=G_BOUNDS,
            random_state=seed,
        )

    outcome = (
        classifier()
        if law.family == "binomial"
        else SuperLearner(library("regression"), task="regression", random_state=seed)
    )
    return {"outcome": outcome, "treatment": classifier(), "source": classifier()}


# ------------------------------------------------------------------------------------------
# fits


@dataclass
class Mutation:
    """A declared option of the study driver (never of the public method): a deliberate
    mutation or a restricted working model that a control fits.

    ``unpenalized``
        K2: the lasso of both working models is replaced by the unpenalized fit of every
        dictionary column (the main-terms dictionary), so the projection is the one the design
        test computes in closed form.
    ``no_source_step``
        K3: the Pi fluctuation is skipped, so the trial-membership probabilities stay at their
        initial fit.
    """

    unpenalized: bool = False
    no_source_step: bool = False


@contextlib.contextmanager
def mutated(mutation: Mutation | None) -> Iterator[None]:
    """Apply ``mutation`` to the engine for the duration of one fit."""
    if mutation is None or not (mutation.unpenalized or mutation.no_source_step):
        yield
        return
    from cleverly.estimators import atmle

    saved = (atmle.select_working_model, atmle.target_source, atmle.MAX_PASSES)
    try:
        if mutation.unpenalized:
            atmle.select_working_model = _unpenalized_selection
        if mutation.no_source_step:
            atmle.target_source = _no_fluctuation
            atmle.MAX_PASSES = 1
        yield
    finally:
        atmle.select_working_model, atmle.target_source, atmle.MAX_PASSES = saved


def _unpenalized_selection(
    basis: FloatArray, multiplier: FloatArray, response: FloatArray, n_unpenalized: int, inner: Any
) -> Any:
    from cleverly.estimators.atmle import Selection

    k = basis.shape[1]
    coef = np.linalg.lstsq(basis * multiplier[:, None], response, rcond=None)[0]
    grid = np.array([1.0, 0.0])
    return Selection(0.0, np.arange(k, dtype=np.int64), coef, grid, np.zeros(2), 0.0)


def _no_fluctuation(*args: Any, **kwargs: Any) -> float:
    """K3: the fluctuation of ``Pi`` is zero, so ``Pi`` keeps its initial fit."""
    return 0.0


def fit_trial(
    frame: pd.DataFrame,
    law: Law,
    *,
    form: str = "cv",
    population: str = "pooled",
    seed: int = 0,
    learners: dict[str, Any] | None = None,
    basis: Any = None,
    mutation: Mutation | None = None,
) -> Any:
    """Fit adaptive TMLE through the public workflow and return the result."""
    from cleverly import (
        AdaptiveTMLEMethod,
        CausalStudy,
        CrossFitting,
        ModelSpec,
        Targeting,
        TrialATE,
        TrialWithExternalData,
    )

    learners = spline_learners(law) if learners is None else learners
    design = TrialWithExternalData(
        outcome="Y",
        treatment="A",
        adjustment=tuple(law.columns()),
        source="S",
        missingness=None if law.missing is None else "R",
        outcome_family=law.family,  # type: ignore[arg-type]
    )
    method = AdaptiveTMLEMethod(
        form=form,  # type: ignore[arg-type]
        models=ModelSpec(
            outcome_learner=learners["outcome"],
            treatment_learner=learners["treatment"],
            missingness_learner=learners.get("missingness"),
        ),
        source_learner=learners["source"],
        working_basis=basis,
        cross_fitting=CrossFitting(n_folds=N_FOLDS, learner_folds=INNER_FOLDS),
        targeting=Targeting(g_bounds=G_BOUNDS),
    ).with_overrides(random_state=seed)
    effect = CausalStudy(frame, design=design).identify(TrialATE(population=population))  # type: ignore[arg-type]
    with mutated(mutation), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return effect.estimate(method=method)


def fit_point(
    frame: pd.DataFrame,
    law: Law,
    *,
    seed: int = 0,
    trial_only: bool = False,
) -> tuple[float, float]:
    """The in-sample point-treatment TMLE of the ATE with the same learners (a cross-fitted
    continuous outcome needs a declared outcome support, which these laws do not have).

    ``trial_only`` fits it on the ``S = 1`` rows: the trial-only TMLE of ``Psi_2`` (plan 2.7),
    the efficiency comparator. Otherwise it fits every row and treats the external rows as
    exchangeable with the trial's: K1, the naive pooled TMLE.
    """
    from cleverly import ATE, CausalStudy, PointTreatment

    rows = frame.loc[frame["S"] == 1] if trial_only else frame
    columns = law.columns()
    study = CausalStudy(
        rows.loc[:, ["Y", "A", *columns]].reset_index(drop=True),
        design=PointTreatment(
            outcome="Y",
            treatment="A",
            adjustment=tuple(columns),
            outcome_family=law.family,  # type: ignore[arg-type]
        ),
    )
    cls = SplineClassifier if law.family == "binomial" else SplineRegressor
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = study.identify(ATE(reference=0)).estimate(
            outcome_learner=cls(n_w=law.d),
            treatment_learner=SplineClassifier(n_w=law.d, products=False),
            cross_fit=False,
            random_state=seed,
        )
    estimate = result["ate"]
    return float(estimate.psi), float(estimate.std_error)


class MainTerms(BaseEstimator):
    """The main-terms dictionary of K2: ``phi(W) = [1, W]`` (the constant is added by the
    engine), unpenalized, so its projection is a closed form of the law."""

    def fit(self, X: Any, y: Any = None) -> MainTerms:
        del y
        x = np.asarray(X, float)
        self.n_features_in_ = x.shape[1]
        self.unscreened_ = np.ones(x.shape[1], dtype=bool)
        return self

    def transform(self, X: Any) -> FloatArray:
        return np.asarray(X, float)


def oracle_parts(
    law: Law, model: Any, population: Any, *, n: int = 400_000, seed: int = 0
) -> tuple[Projection, Any, pd.DataFrame]:
    """The influence function of ``Psi_w`` at ``P_0`` for a stored selection, on a large draw.

    The nuisances are the law's true ones and the coefficients the projection (plan 2.3), so
    ``evaluate`` returns ``D_w(P_0)`` row by row. Exact-law building blocks only: no estimator
    is run. The draw is a Monte Carlo of ``n`` rows, so a standard deviation is exact to about
    ``1 / sqrt(2 n)`` relative.
    """
    from cleverly.estimators import atmle

    frame = draw(Law(**{**law.__dict__, "missing": None}), n, seed)
    w = frame[law.columns()].to_numpy()
    a = frame["A"].to_numpy(float)
    s = frame["S"].to_numpy(float)
    g = law.g(w)
    qt0, qt1 = law.qt(w, 0.0), law.qt(w, 1.0)
    nuisances = atmle.Nuisances(
        g=g,
        theta=g * qt1 + (1.0 - g) * qt0,
        source0=law.pi(w, 0.0),
        source1=law.pi(w, 1.0),
        qt=np.where(a == 1.0, qt1, qt0),
        pibar=law.s(w),
    )
    dictionary = model.dictionary
    kept = dictionary.phi(w)
    controls = dictionary.controls_only
    keep = dictionary.psi_keep

    def psi_at(arm: FloatArray | float) -> FloatArray:
        return bias_columns(kept, arm, controls)[:, keep][:, model.bias.selected]

    unit = atmle.WorkingUnit(
        outcome=frame["Y"].to_numpy(float),
        treatment=a,
        source=s,
        observed=None,
        phi=kept[:, model.effect.selected],
        psi=psi_at(a),
        psi0=psi_at(0.0),
        psi1=psi_at(1.0),
        nuisances=nuisances,
        weights=np.full(n, 1.0 / n),
        controls_only=controls,
    )
    truth = projection(law, model, population)
    _, parts = atmle.evaluate(unit, truth.beta_a, truth.beta_s, population)
    table = pd.DataFrame(
        {
            "covariate": parts.covariate,
            "effect_model": parts.effect_model,
            "bias_model": parts.bias_model,
            "source": parts.source,
            "total": parts.total,
            "S": s,
            "A": a,
        }
    )
    return truth, parts, table


# ------------------------------------------------------------------------------------------
# exact-law quantities of the controls (design test)


def pooled_ate_oracle(law: Law, *, n: int = 400_000, seed: int = 0) -> dict[str, float]:
    """The naive pooled TMLE's limit and standard error at ``P_0`` (control K1).

    Treating the external rows as exchangeable with the trial's targets
    :math:`\\tilde\\Psi = E[\\tilde Q(W, 1) - \\tilde Q(W, 0)]` with
    :math:`\\tilde Q(W, a) = E(Y | W, A = a)` on the pooled rows. Its bias for :math:`\\Psi` is
    :math:`\\Psi^\\#`. The standard deviation is that of the pooled ATE's efficient influence
    function, :math:`(2A - 1) / g(A | W) (Y - \\tilde Q(W, A)) + \\tau_A(W) - \\tilde\\Psi`, by
    Monte Carlo of ``n`` rows (a relative error of about ``1 / sqrt(2 n)``).
    """
    w, wt = grid(law)
    naive = float(np.sum(wt * law.tau_a(w)))
    frame = draw(Law(**{**law.__dict__, "missing": None}), n, seed)
    x = frame[law.columns()].to_numpy()
    a = frame["A"].to_numpy(float)
    g = law.g(x)
    qt = np.where(a == 1.0, law.qt(x, 1.0), law.qt(x, 0.0))
    prob = np.where(a == 1.0, g, 1.0 - g)
    curve = (2.0 * a - 1.0) / prob * (frame["Y"].to_numpy() - qt) + law.tau_a(x) - naive
    return {"naive": naive, "sd": float(np.std(curve))}


def pi_limit_bias(law: Law) -> float:
    """The bias of the untargeted plug-in when ``Pi`` is a main-terms logistic (control K3).

    ``Pi_lim`` is the population maximum-likelihood limit of the main-terms logistic of ``S`` on
    ``(W, A)``, solved by weighted logistic regression on the quadrature nodes. With ``Qt`` and
    ``g`` correct and ``tau_S`` inside the span of the bias dictionary, the untargeted estimate
    converges to :math:`\\Psi + E[(\\Pi_{lim}(W, 0) - \\Pi(W, 0)) \\tau_S(W, 0) -
    (\\Pi_{lim}(W, 1) - \\Pi(W, 1)) \\tau_S(W, 1)]`, which is returned. The targeted fit removes
    it (the clever covariate :math:`C` is the score of this bias).
    """
    w, wt = grid(law)
    g = law.g(w)
    pi0, pi1 = law.pi(w, 0.0), law.pi(w, 1.0)
    features = np.vstack(
        [np.column_stack([w, np.zeros(len(w))]), np.column_stack([w, np.ones(len(w))])]
    )
    mass = np.concatenate([wt * (1.0 - g), wt * g])
    prob = np.concatenate([pi0, pi1])
    fit = LogisticRegression(C=1e8, max_iter=5000).fit(
        np.vstack([features, features]),
        np.concatenate([np.ones(len(features)), np.zeros(len(features))]),
        sample_weight=np.concatenate([mass * prob, mass * (1.0 - prob)]),
    )
    lim0 = fit.predict_proba(np.column_stack([w, np.zeros(len(w))]))[:, 1]
    lim1 = fit.predict_proba(np.column_stack([w, np.ones(len(w))]))[:, 1]
    return float(np.sum(wt * ((lim0 - pi0) * law.tau_s(w, 0.0) - (lim1 - pi1) * law.tau_s(w, 1.0))))
