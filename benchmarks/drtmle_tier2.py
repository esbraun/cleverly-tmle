r"""Tier 2 of the coverage study: a prescribed *rate*, learned rather than handed over.

``docs/drtmle/validation-plan.md`` §5 asks for two tiers and this is the second: *"a series,
spline or histogram regression with a smoothing sequence chosen in advance, so the rate is
analysable and reproducible.  **This is the demonstration.**"*  Tier 1
(``benchmarks/drtmle_injection.py``) hands the estimator a prescribed sequence, which makes
"the intended asymptotic regime was entered" true by definition and is **not** an applied
claim.  Here the good nuisance is *fitted*, which is what the trap the roadmap records
demands: ``tests/e2e/test_double_robustness.py``'s "correct" cell is an oracle, so
:math:`R_2` there is exactly zero and a plain ``TMLE``'s interval is already valid.  The gap
opens only where the good nuisance is estimated.

The two cells are Tier 1's, off the diagonal of the misspecification grid, with both
nuisances now learners:

======================  ==========================================  ==========================
cell                    :math:`\hat Q`                              :math:`\hat g`
======================  ==========================================  ==========================
``q-drift``             oversmoothed kernel regression              a GLM on a subset, wrong
``g-drift``             per-arm GLMs on subsets, wrong              oversmoothed kernel
======================  ==========================================  ==========================

**The smoothing sequence is the deliverable and it is committed here**: a Gaussian product
kernel with bandwidth

.. math::  h_n = c_h\, n^{-\beta}, \qquad \beta = \alpha/2 = 0.125,\ c_h = 1.15

applied **one covariate at a time** (an additive backfit), so that the estimator is
bias-dominated at the sizes this study reaches -- a one-dimensional smoother's variance is
:math:`O(1/(nh))` -- while its bias is :math:`O(h_n^2) = O(n^{-\alpha})`.  That is the same
exponent Tier 1 *injects*, which is the point of matching them: what two tiers have to share
to be about one regime is the rate of the **remainder**, not the rate of a nuisance norm.

**Additive rather than a product kernel, and that is the curse of dimensionality rather than
a modelling assumption.**  A four-dimensional product kernel wide enough to be
bias-dominated at ``n = 600`` smooths over essentially the whole covariate space -- measured
here at an :math:`L_2` error of ``1.81`` against an outcome standard deviation of ``1.75``,
which is not a slow learner but a broken one -- and one narrow enough to be a regression has
a variance of the same order as its bias, so the remainder it produces is sampling noise
rather than the designed drift.  One dimension at a time avoids both.  It does not change
the bias formula the coefficient is committed from, only the variance it has to dominate.

**Why a kernel and not a regressogram, which is what §5 names.**  This is a finding rather
than a preference, and it is §5's own trap arriving through a second door.  A regressogram's
bias **oscillates in sign within every bin**, so while its :math:`L_2` norm is
:math:`O(B^{-1})` its *inner product* with a smooth weight is :math:`O(B^{-2})` -- and the
remainder is an inner product.  Matching a declared remainder rate with a regressogram
therefore needs a bin count so large that the fit is variance-dominated at the sizes this
study can reach: at ``n = 600`` a four-covariate additive regressogram with the bins the rate
would need has more parameters than the bias has room to survive, and its remainder is then
sampling noise rather than a drift.  A local-constant estimator's bias is
:math:`h^2[\tfrac12\nabla^2 m + \nabla m\cdot\nabla\log p]`, which is **smooth and
single-signed against a monotone weight**, so no cancellation is available to it.  §5's list
is illustrative; what it asks for is a sequence chosen in advance, and this is one.

**The drift coefficient is committed by quadrature, exactly as Tier 1's is.**  With the
bias above and :math:`n^{\alpha}h_n^2 = c_h^2`,

.. math::

    c_a = c_h^2 \, P_0\!\left[\frac{\hat g_a - g_{0,a}}{\hat g_a}\; b_a(W)\right]
    \quad\text{or}\quad
    c_a = c_h^2 \, P_0\!\left[\frac{b_a(W)}{g_{0,a}}\; d_a(W)\right]

for the two cells respectively, with :math:`b_a` the analytic bias shape and :math:`d_a` the
fixed outcome error.  :func:`drift_coefficients` is that quadrature, on the **same Sobol
rule** the truth is integrated with, and ``tests/unit/test_drtmle_tier2.py`` asserts every
realised coefficient clears :data:`C_MIN`.  It is a *prediction* rather than an identity --
unlike Tier 1, where :math:`h_a` is normalised to make the coefficient come out at a declared
value -- so what the study reports is the **measured** :math:`n^{\alpha}R_2`
(``benchmarks/drtmle_remainder.plain_remainder``, which is exact at the fitted nuisances)
against it.  §5 permits the design to move at the pilot and only there.

**The wrong nuisance is fitted too, and on this law its bias is *entirely* in the
remainder.**  ``linear_dgp``'s covariates are independent and mean zero, so a subset model's
limit is the truth with the dropped terms deleted and its error has mean zero at every arm --
which means the untargeted plug-in contrast is unbiased and the whole of what a coverage
shortfall can come from is :math:`R_2`.  That is the cleanest separation this design could
have, and it is worth saying because it is the *opposite* of Tier 1's situation, where
:data:`~benchmarks.drtmle_injection.G_DRIFT_ARM0_RATIO` exists to stop an error identical at
both arms making the contrast accidentally right.  Here the arms' errors still differ -- arm
0's model drops a further covariate -- but nothing rests on it.

What the arms' coefficients do rest on is the mechanism bias' sign: :math:`b_0 = -b_1` by
construction, so :math:`c_0 = -c_1` and :math:`c_{ATE}` is a **sum** of magnitudes rather
than a difference.  Cancellation in the contrast is impossible rather than merely unlikely,
which is the property Tier 1 gets by giving its arms opposite signs by hand.

**The base law is Tier 1's**, :func:`~cleverly.datasets.linear_dgp`, and for Tier 1's reason:
its propensity sits near ``0.5``, which is what puts both cells **inside** the supported
contract (``docs/roadmap.md``'s item 25) so that a coverage number is evidence about Theorem
1's estimator rather than about the constrained rendering beside it.  Its independent
standard-normal covariates are also what makes every quadrature below analytic.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cleverly.datasets import DGP, linear_dgp
from cleverly.utils.bounds import OutcomeScaler, expit

__all__ = [
    "ALPHA",
    "BANDWIDTH_C",
    "BANDWIDTH_EXPONENT",
    "CELLS",
    "C_MIN",
    "Q_BOUNDS",
    "KernelMechanism",
    "KernelOutcome",
    "SubsetMechanism",
    "SubsetOutcome",
    "bandwidth",
    "base_law",
    "drift_coefficients",
    "nuisance_error",
    "settings",
    "summary_rows",
    "targeted_coefficients",
]

#: The two off-diagonal cells, named as Tier 1 names them so one harness reads both.
CELLS = ("q-drift", "g-drift")

#: The **remainder** exponent, and deliberately the same value Tier 1 injects: what the two
#: tiers have to share for their numbers to be about one regime is the rate of :math:`R_2`,
#: not the rate of a nuisance norm.
ALPHA = 0.25

#: The bandwidth's exponent.  A local-constant bias is :math:`O(h^2)`, so halving
#: :data:`ALPHA` is what makes the remainder drift at :data:`ALPHA`.  It also means the
#: nuisance's own :math:`L_2` error falls at ``0.125`` -- a genuinely slow learner, which is
#: what Tier 2 is for.
BANDWIDTH_EXPONENT = ALPHA / 2.0

#: The bandwidth's constant, and the one number here chosen to hit a target rather than
#: derived: it is sized so that ``q-drift``'s predicted ``c_ATE`` lands at Tier 1's committed
#: ``0.40``, since :math:`c_a \propto c_h^2` and the two tiers are only comparable if their
#: drifts are.  It comes out at ``0.389`` and ``g-drift``'s at ``0.410``.  ``h(600) = 0.517``
#: and ``h(2400) = 0.435``, at which a one-dimensional smoother's variance is two orders
#: below its bias.
BANDWIDTH_C = 1.15

#: How far the kernel's neighbours are taken, in bandwidths.  Beyond four the Gaussian
#: weight is ``3e-4`` of the centre's and the sum is unchanged to five figures; the cap
#: exists so the kernel matrix can be blocked rather than formed whole.
KERNEL_CUTOFF = 4.0

#: The declared support of the outcome, passed as ``q_bounds=`` to both estimators for Tier
#: 1's reason: a recovered scaler carries an :math:`O(n^{-1/2})` error, and here that is the
#: same order as the *variance* of the slow nuisance.  Tier 1's value, since the law is the
#: same one.
Q_BOUNDS = (-8.0, 14.0)

#: The floor each realised drift coefficient has to clear.  Tier 1's value and Tier 1's
#: reason: it exists so "the drift is nonzero" is an assertion with a number behind it.
C_MIN = 0.02

#: Which covariates each *wrong* model is fitted on.  The omissions are the strongest terms
#: of the truth, so the limits are wrong by a wide margin rather than marginally.  Both arms
#: of the outcome model drop ``W1``, which is what makes the two arms' drift coefficients
#: exact negatives and their contrast a sum of magnitudes; arm 0 drops ``W4`` as well, so the
#: two arms' errors are not the same function.  ``linear_dgp``'s covariates are independent
#: standard normals, so the least-squares limit of a subset model is the truth with the
#: dropped terms deleted, exactly -- which is what makes :func:`outcome_error` analytic.
WRONG_MECHANISM_COLUMNS = (1, 2)
WRONG_OUTCOME_COLUMNS = {1.0: (1, 2, 3), 0.0: (1, 2)}


def base_law() -> DGP:
    """The law both cells are drawn from -- Tier 1's, for Tier 1's reason."""
    return linear_dgp()


def bandwidth(n: int) -> float:
    """The committed smoothing sequence, :math:`h_n = c_h n^{-\\beta}`."""
    return BANDWIDTH_C * float(n) ** -BANDWIDTH_EXPONENT


# ------------------------------------------------------------------ the learners


#: Backfitting passes.  ``linear_dgp``'s covariates are **independent**, so the additive
#: components are orthogonal and one pass already recovers them; three is the cheap margin
#: that makes that a property of the fit rather than of the law.
BACKFIT_PASSES = 3


def _smooth_one(
    column: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    at: np.ndarray,
    h: float,
    block: int = 1024,
) -> np.ndarray:
    """A one-dimensional Nadaraya--Watson smoother, blocked over the prediction rows."""
    out = np.empty(at.size, dtype=float)
    fallback = float(np.average(target, weights=weight))
    for start in range(0, at.size, block):
        piece = at[start : start + block]
        squared = ((piece[:, None] - column[None, :]) / h) ** 2
        kernel = np.where(squared <= KERNEL_CUTOFF**2, np.exp(-0.5 * squared), 0.0) * weight
        mass = kernel.sum(axis=1)
        # A point with no neighbour inside the cutoff takes the marginal mean, which is the
        # coarsest smoother available rather than a `nan` travelling into a clever covariate.
        out[start : start + block] = np.where(
            mass > 0.0, (kernel @ target) / np.maximum(mass, 1e-300), fallback
        )
    return out


class _Additive(BaseEstimator):
    """An **additive** local-constant fit, one Gaussian kernel per covariate.

    Additive rather than a product kernel over all of them, and the reason is the curse of
    dimensionality rather than the shape of the truth.  A four-dimensional product kernel
    wide enough to be bias-dominated at ``n = 600`` smooths over essentially the whole
    covariate space -- measured here at an :math:`L_2` error of ``1.8`` against an outcome
    standard deviation of ``1.75``, which is not a slow learner but a broken one -- while one
    narrow enough to be a regression has a variance of the same order as its bias, and then
    the remainder it produces is sampling noise rather than the designed drift.  One
    dimension at a time has a variance of :math:`O(1/(nh))` and is bias-dominated at a
    bandwidth that still resolves the function.

    Its bias is the sum of the components', :math:`h^2\\sum_j[\\tfrac12 f_j'' + f_j'\\,
    (\\log p_j)']`, which is what :func:`_kernel_bias` writes out -- so the additive
    structure does not change the bias formula the drift coefficient is committed from, only
    the variance it has to dominate.

    The smoothing is a **declared function of the study's sample size**, not of the training
    fold's and not chosen by cross-validation: the whole point of the tier is that the rate
    is known in advance, and a data-driven bandwidth would make it neither identified nor
    reproducible -- which is the objection §5 raises to a Super Learner.
    """

    def __init__(self, n: int) -> None:
        self.n = n

    def _backfit(self, design: np.ndarray, target: np.ndarray, weight: np.ndarray) -> Any:
        h = bandwidth(self.n)
        intercept = float(np.average(target, weights=weight))
        components = [np.zeros(target.size) for _ in range(design.shape[1])]
        for _ in range(BACKFIT_PASSES):
            for index in range(design.shape[1]):
                partial = (
                    target
                    - intercept
                    - sum(values for other, values in enumerate(components) if other != index)
                )
                fitted = _smooth_one(design[:, index], partial, weight, design[:, index], h)
                components[index] = fitted - float(np.average(fitted, weights=weight))
        return {
            "design": design,
            "target": target,
            "weight": weight,
            "h": h,
            "intercept": intercept,
            "components": components,
        }

    def _evaluate(self, state: Any, at: np.ndarray) -> np.ndarray:
        total = np.full(at.shape[0], state["intercept"], dtype=float)
        for index in range(at.shape[1]):
            total += _smooth_one(
                state["design"][:, index],
                state["components"][index],
                state["weight"],
                at[:, index],
                state["h"],
            )
        return total


class KernelOutcome(_Additive):
    """The slow, consistent outcome regression of the ``q-drift`` cell.

    One additive fit **per arm**, rather than one model with the treatment as a smoothed
    covariate: an arm is not a continuous coordinate, and smoothing across it would leak the
    other arm's outcomes into a counterfactual prediction -- which is a misspecification the
    design did not declare and could not compute a coefficient for.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> KernelOutcome:
        design = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        weight = (
            np.ones(target.size)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        arms, covariates = design[:, 0], design[:, 1:]
        self.states_ = {}
        for arm in (1.0, 0.0):
            rows = arms == arm
            if not rows.any():  # pragma: no cover - a fold with one arm cannot happen here
                rows = np.ones(arms.size, dtype=bool)
            self.states_[arm] = self._backfit(covariates[rows], target[rows], weight[rows])
        return self

    def predict(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        arms, covariates = design[:, 0], design[:, 1:]
        out = np.empty(arms.size, dtype=float)
        for arm, state in self.states_.items():
            rows = arms == arm
            if rows.any():
                out[rows] = self._evaluate(state, covariates[rows])
        return np.clip(out, 1e-6, 1.0 - 1e-6)


class KernelMechanism(_Additive):
    """The slow, consistent treatment mechanism of the ``g-drift`` cell.

    An additive smoother of the arm indicator, which **is** a conditional probability rather
    than a margin that has to be squashed into one -- so its bias is the :math:`O(h^2)` above
    with no link in the way, and :func:`_kernel_bias` can write it out.
    """

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> KernelMechanism:
        design = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        weight = (
            np.ones(target.size)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=float).reshape(-1)
        )
        self.state_ = self._backfit(design, target, weight)
        self.classes_ = np.array([0.0, 1.0])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        one = np.clip(self._evaluate(self.state_, np.asarray(X, dtype=float)), 1e-4, 1.0 - 1e-4)
        return np.column_stack([1.0 - one, one])


def SubsetMechanism() -> Any:
    """The ``q-drift`` cell's **wrong** mechanism: a logistic fit on a subset.

    Fitted, not injected, which is what makes this tier an applied claim.  Its limit is the
    population logistic projection onto the retained columns, which is not the truth anywhere
    -- and :func:`wrong_mechanism` is that limit, computed once on a fixed reference draw so
    the drift coefficient below is a number and not a hope.
    """
    return Pipeline(
        [
            ("select", _Columns(WRONG_MECHANISM_COLUMNS)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1e6, max_iter=1000)),
        ]
    )


class _Columns(BaseEstimator):
    """Keep the named columns of a design, dropping the rest."""

    def __init__(self, columns: tuple[int, ...], offset: int = 0) -> None:
        self.columns = columns
        self.offset = offset

    def fit(self, X: Any, y: Any = None) -> _Columns:
        return self

    def transform(self, X: Any) -> np.ndarray:
        matrix = np.asarray(X, dtype=float)
        return matrix[:, [self.offset + index for index in self.columns]]


class SubsetOutcome(BaseEstimator):
    """The ``g-drift`` cell's **wrong** outcome regression: one GLM per arm, each on a subset.

    Per arm rather than one model with a treatment column, and the subsets differ, because an
    error identical at both arms would leave the plug-in contrast exactly right -- Tier 1's
    :data:`~benchmarks.drtmle_injection.G_DRIFT_ARM0_RATIO` warning, which this is the second
    instance of.  Since ``linear_dgp``'s covariates are independent standard normals, the
    least-squares limit is the truth with the dropped terms deleted, so :func:`outcome_error`
    is exact rather than approximate.
    """

    def __init__(self) -> None:
        self.models_: dict[float, Any] = {}

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> SubsetOutcome:
        design = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=float).reshape(-1)
        arms, covariates = design[:, 0], design[:, 1:]
        for arm, columns in WRONG_OUTCOME_COLUMNS.items():
            rows = arms == arm
            if not rows.any():  # pragma: no cover - a fold with one arm cannot happen here
                rows = np.ones(arms.size, dtype=bool)
            model = LinearRegression().fit(
                covariates[rows][:, list(columns)],
                target[rows],
                sample_weight=None if sample_weight is None else np.asarray(sample_weight)[rows],
            )
            self.models_[arm] = model
        return self

    def predict(self, X: Any) -> np.ndarray:
        design = np.asarray(X, dtype=float)
        arms, covariates = design[:, 0], design[:, 1:]
        out = np.empty(arms.size, dtype=float)
        for arm, columns in WRONG_OUTCOME_COLUMNS.items():
            rows = arms == arm
            if rows.any():
                out[rows] = self.models_[arm].predict(covariates[rows][:, list(columns)])
        return np.clip(out, 1e-6, 1.0 - 1e-6)


# ------------------------------------------------------------------ what the law says

#: Rows of the fixed reference draw the wrong mechanism's *limit* is computed on.  Large and
#: seeded, so the number is deterministic and committed with the design rather than
#: re-estimated per replicate -- there is no closed form for a logistic projection, and a
#: coefficient that moved between replicates would not be a design constant at all.
REFERENCE_N = 200_000
REFERENCE_SEED = 20260101


@cache
def wrong_mechanism_coefficients() -> tuple[np.ndarray, float]:
    """``(theta, intercept)`` of the population logistic projection onto the kept columns."""
    dgp = base_law()
    frame, _ = dgp.sample(REFERENCE_N, seed=REFERENCE_SEED)
    covariates = np.column_stack([frame[name].to_numpy() for name in dgp.covariate_names])
    model = LogisticRegression(C=1e6, max_iter=1000).fit(
        covariates[:, list(WRONG_MECHANISM_COLUMNS)], frame["A"].to_numpy()
    )
    return np.asarray(model.coef_, dtype=float).reshape(-1), float(model.intercept_[0])


def wrong_mechanism(w: Any) -> np.ndarray:
    r""":math:`g_1(1|W) \neq g_0(1|W)`: the ``q-drift`` cell's mechanism limit."""
    theta, intercept = wrong_mechanism_coefficients()
    latent = np.asarray(w, dtype=float)
    return np.asarray(
        expit(intercept + latent[:, list(WRONG_MECHANISM_COLUMNS)] @ theta), dtype=float
    )


def outcome_error(w: Any, arm: float) -> np.ndarray:
    r""":math:`\hat Q(a, W) - \bar Q_0(a, W)`: the ``g-drift`` cell's fixed outcome error.

    Exact rather than approximate, because the covariates are independent standard normals:
    the least-squares limit of a model fitted on a subset is the truth with the dropped
    terms deleted, so the error is minus the sum of those terms.
    """
    latent = np.asarray(w, dtype=float)
    coefficients = {0: 1.0, 1: 0.5, 2: -0.8, 3: 0.4}
    dropped = [index for index in coefficients if index not in WRONG_OUTCOME_COLUMNS[arm]]
    return -sum(coefficients[index] * latent[:, index] for index in dropped)


def _kernel_bias(dgp: DGP, w: Any, arm: float, *, mechanism: bool) -> np.ndarray:
    r"""The local-constant bias shape, with :math:`h^2` divided out.

    .. math::

        E[\hat m(w)] - m(w) \;\approx\; h^2\Bigl[\tfrac12\nabla^2 m(w)
                                          + \nabla m(w)\cdot\nabla\log p(w)\Bigr]

    and :math:`p` is a standard normal product density, so
    :math:`\nabla\log p(w) = -w`.  For the **outcome** regression the truth is affine, the
    Laplacian vanishes and the bias is exactly :math:`-\beta\cdot w`; for the **mechanism**
    it is a logistic of a linear index, so both terms survive and are written out.
    """
    latent = np.asarray(w, dtype=float)
    if not mechanism:
        beta = np.array([1.0, 0.5, -0.8, 0.4])
        return -(latent @ beta)
    gamma = np.array([0.3, -0.2, 0.1, 0.0])
    g = np.asarray(dgp.propensity(latent), dtype=float)
    curvature = 0.5 * g * (1.0 - g) * (1.0 - 2.0 * g) * float(gamma @ gamma)
    gradient = (g * (1.0 - g))[:, None] * gamma[None, :]
    shape = curvature - np.sum(gradient * latent, axis=1)
    return shape if arm == 1.0 else -shape


def _arm(values: Any, arm: float) -> Any:
    """``P(A = arm | W)`` from the arm-1 column, by complement -- the binary path's rule."""
    return values if arm == 1.0 else 1.0 - values


# ------------------------------------------------- what targeting does to the predicted drift
#
# Tier 1's finding applies here unchanged, and the arithmetic below is deliberately the same
# arithmetic: the fluctuation's population score removes the part of *any* outcome-regression
# error it can reach, whether that error was injected or is a smoother's bias.  What differs is
# that Tier 1 can *choose* its shape to survive and this tier cannot -- a kernel's bias is what
# it is -- so here the targeted coefficient is a prediction to be read, not a target to be hit.


def _limit_mechanism(cell: str, w: Any, arm: float) -> np.ndarray:
    r""":math:`\lim \hat g_a`: the wrong projection in ``q-drift``, the truth in ``g-drift``."""
    dgp = base_law()
    values = (
        wrong_mechanism(w)
        if cell == "q-drift"
        else np.asarray(dgp.propensity(np.asarray(w, dtype=float)), dtype=float)
    )
    return np.asarray(_arm(values, arm), dtype=float)


def _limit_outcome(cell: str, w: Any, arm: float) -> np.ndarray:
    r""":math:`\lim \hat Q_a`, on the outcome's own scale.

    The truth in ``q-drift``, where the kernel is consistent and only its *bias* drifts; the
    **wrong** subset regression in ``g-drift``, whose error does not shrink.  Tier 1's
    convention and for its reason: evaluating the fluctuation's direction at a regression no
    fit holds would answer for a different sequence.
    """
    dgp = base_law()
    truth = np.asarray(dgp.outcome_mean(np.asarray(w, dtype=float), arm, None), dtype=float)
    return truth if cell == "q-drift" else truth + outcome_error(w, arm)


def _fluctuation_direction(cell: str, w: Any, arm: float) -> np.ndarray:
    r""":math:`S_a`, the raw-scale direction the single :math:`\varepsilon_a` moves the
    regression in -- ``benchmarks/drtmle_injection._fluctuation_direction``'s twin."""
    scaler = OutcomeScaler(*Q_BOUNDS)
    scaled = scaler.scale(_limit_outcome(cell, w, arm))
    return np.asarray(
        (Q_BOUNDS[1] - Q_BOUNDS[0]) * scaled * (1.0 - scaled) / _limit_mechanism(cell, w, arm),
        dtype=float,
    )


def _outcome_offset(cell: str, w: Any, arm: float) -> np.ndarray:
    r""":math:`\hat Q_a - \bar Q_{0,a}` with any :math:`h_n^2` divided out.

    The kernel's analytic bias shape in ``q-drift``; the fixed subset error in ``g-drift``.
    """
    if cell == "q-drift":
        return BANDWIDTH_C**2 * _kernel_bias(base_law(), w, arm, mechanism=False)
    return np.asarray(outcome_error(w, arm), dtype=float)


@cache
def _absorbed(cell: str) -> tuple[float, ...]:
    r""":math:`\varepsilon_a` at the population score, per arm, in the order ``(1.0, 0.0)``.

    Solved from the offset the design predicts rather than chosen, since nothing here is free:
    :math:`\varepsilon_a = -P_0[w_a \Delta_a]/P_0[w_a S_a]` with
    :math:`\Delta_a = \hat Q_a - \bar Q_{0,a}`.  In ``q-drift`` that offset carries the
    :math:`h_n^2` this divides out, so the ratio is scale-free and the constant is a limit --
    which is why the cache key is the cell alone.
    """
    dgp = base_law()
    if cell not in CELLS:
        raise ValueError(f"cell must be one of {list(CELLS)}; got {cell!r}")
    out = []
    for arm in (1.0, 0.0):

        def weight(w: Any, a: float = arm) -> np.ndarray:
            truth = _arm(np.asarray(dgp.propensity(np.asarray(w, float)), float), a)
            return np.asarray(truth / _limit_mechanism(cell, w, a), dtype=float)

        numerator = dgp.expectation(lambda w, a=arm: weight(w, a) * _outcome_offset(cell, w, a))
        denominator = dgp.expectation(
            lambda w, a=arm: weight(w, a) * _fluctuation_direction(cell, w, a)
        )
        out.append(-numerator / denominator)
    return tuple(out)


def targeted_coefficients(cell: str) -> dict[str, float]:
    r"""The predicted :math:`b_1`, :math:`b_0` and :math:`b_{ATE}`: the **estimator's bias**.

    :math:`b_a = \lim n^{\alpha} P_0[u_a(\bar Q^*_a - \bar Q_{0,a})]`, the same limit
    :func:`drift_coefficients` takes at the *initial* regression.  With
    :math:`\bar Q^*_a - \bar Q_{0,a} = \Delta_a + \varepsilon_a S_a` and :math:`\varepsilon_a`
    from :func:`_absorbed`, it is one further quadrature over shapes this module already has.

    A **prediction** rather than a target, which is the difference from Tier 1: there the shape
    is free and is solved to make this come out at a declared number, and a kernel's bias is
    what it is.  So what a run reports is the measured column against this, and
    ``benchmarks/drtmle_remainder.targeted_remainder`` is the measurement.

    **And on this law it comes out at the plug-in coefficient**, ``0.3895`` against ``0.3886``
    in ``q-drift`` and equal to five figures in ``g-drift``, where Tier 1's two differ by a
    factor of 436.  That is not luck and it is the same fact this module's own docstring
    already states for a different purpose: ``linear_dgp``'s covariates are independent
    standard normals and both error shapes here are **linear** in them, so each has population
    mean zero, and the fluctuation's step is driven by exactly that mean.  In ``g-drift`` the
    score weight is identically one at the limit, so :math:`\varepsilon_a` is the mean of the
    outcome error and is zero exactly.

    So Tier 2 never had Tier 1's problem, and this column is what says so rather than what
    fixes anything -- which is also why Tier 2 produced a coverage gap in C3a's pilot while
    Tier 1 could not.  Absorption is a property of whether the nuisance's error is aligned with
    the fluctuation's direction, and a smoother's bias on a symmetric law is not.
    """
    if cell not in CELLS:
        raise ValueError(f"cell must be one of {list(CELLS)}; got {cell!r}")
    dgp = base_law()
    epsilons = dict(zip((1.0, 0.0), _absorbed(cell), strict=True))
    per_arm = {}
    for arm in (1.0, 0.0):

        def integrand(w: Any, a: float = arm) -> np.ndarray:
            truth = np.asarray(_arm(np.asarray(dgp.propensity(np.asarray(w, float)), float), a))
            weight = (_limit_mechanism(cell, w, a) - truth) / _limit_mechanism(cell, w, a)
            offset = _outcome_offset(cell, w, a) + epsilons[a] * _fluctuation_direction(cell, w, a)
            if cell == "q-drift":
                return np.asarray(weight * offset, dtype=float)
            # In `g-drift` the *mechanism* drifts, so the n^-alpha sits in the weight rather
            # than in the offset and the limit is taken of the weight's shape instead.
            bias = BANDWIDTH_C**2 * _kernel_bias(dgp, w, a, mechanism=True)
            return np.asarray(bias / truth * offset, dtype=float)

        per_arm[arm] = dgp.expectation(integrand)
    return {"b1": per_arm[1.0], "b0": per_arm[0.0], "b_ate": per_arm[1.0] - per_arm[0.0]}


def drift_coefficients(cell: str) -> dict[str, float]:
    r"""The predicted :math:`c_1`, :math:`c_0` and :math:`c_{ATE}` for a cell, by quadrature.

    :math:`c_a = \lim n^{\alpha} R_{2,a}`, with the local-constant bias above in place of the
    injected shape Tier 1 normalises.  A **prediction** rather than an identity, which is the
    honest difference between the tiers: Tier 1's :math:`h_a` is scaled so the coefficient
    comes out at a declared number, and here the estimator's bias is what it is.  What the
    study reports beside it is the measured :math:`n^{\alpha}R_2`
    (:func:`benchmarks.drtmle_remainder.plain_remainder`, exact at the fitted nuisances).

    Integrated through :meth:`~cleverly.datasets.DGP.expectation`, the **same Sobol rule**
    the truth uses, for Tier 1's reason: a second quadrature would put a Monte Carlo error of
    its own between a coefficient and the coverage it explains.
    """
    dgp = base_law()
    if cell not in CELLS:
        raise ValueError(f"cell must be one of {list(CELLS)}; got {cell!r}")
    per_arm = {}
    for arm in (1.0, 0.0):
        if cell == "q-drift":

            def integrand(w: Any, arm: float = arm) -> Any:
                wrong = _arm(wrong_mechanism(w), arm)
                truth = _arm(np.asarray(dgp.propensity(np.asarray(w, dtype=float))), arm)
                bias = _kernel_bias(dgp, w, arm, mechanism=False)
                return BANDWIDTH_C**2 * (wrong - truth) / wrong * bias
        else:

            def integrand(w: Any, arm: float = arm) -> Any:
                truth = _arm(np.asarray(dgp.propensity(np.asarray(w, dtype=float))), arm)
                bias = _kernel_bias(dgp, w, arm, mechanism=True)
                return BANDWIDTH_C**2 * bias / truth * outcome_error(w, arm)

        per_arm[arm] = dgp.expectation(integrand)
    return {"c1": per_arm[1.0], "c0": per_arm[0.0], "c_ate": per_arm[1.0] - per_arm[0.0]}


def nuisance_error(cell: str, n: int) -> dict[str, float]:
    r"""``L2(P_0)`` distance from each nuisance's *leading term* to the truth.

    The columns §5 calls "verifying the regime was entered", predicted analytically here and
    measured against the fit by the study: the drifting nuisance's norm has to fall at
    :math:`h_n^2` and the misspecified one's has to stay bounded away from zero.  Only the
    **bias** is here, since the design is bias-dominated by construction -- and if that turns
    out not to hold at a study size, it shows up as the measured norm exceeding this one.
    """
    dgp = base_law()
    h2 = bandwidth(n) ** 2
    if cell == "q-drift":
        drifting = {
            arm: np.sqrt(
                dgp.expectation(
                    lambda w, a=arm: (h2 * _kernel_bias(dgp, w, a, mechanism=False)) ** 2
                )
            )
            for arm in (1.0, 0.0)
        }
        wrong = np.sqrt(
            dgp.expectation(
                lambda w: (
                    (wrong_mechanism(w) - np.asarray(dgp.propensity(np.asarray(w, dtype=float))))
                    ** 2
                )
            )
        )
        return {
            "q_error_1": float(drifting[1.0]),
            "q_error_0": float(drifting[0.0]),
            "g_error": float(wrong),
        }
    fixed = {
        arm: np.sqrt(dgp.expectation(lambda w, a=arm: outcome_error(w, a) ** 2))
        for arm in (1.0, 0.0)
    }
    drifting_g = np.sqrt(
        dgp.expectation(lambda w: (h2 * _kernel_bias(dgp, w, 1.0, mechanism=True)) ** 2)
    )
    return {
        "q_error_1": float(fixed[1.0]),
        "q_error_0": float(fixed[0.0]),
        "g_error": float(drifting_g),
    }


def settings(cell: str, n: int) -> dict[str, Any]:
    """The estimator keywords a cell's fits share, learners included.

    ``reduced_outcome_learner`` and ``reduced_treatment_learner`` are named explicitly for
    Tier 1's reason: ``DRTMLE``'s reductions default to the primary *specification*, so
    leaving them off would hand this cell's kernel to :math:`Q_r`, :math:`g_{r,1}` and
    :math:`g_{r,2}` -- making the reductions a smoother of a smoother rather than the
    univariate regressions the derivation is about.
    """
    if cell == "q-drift":
        outcome: Any = KernelOutcome(n)
        treatment: Any = SubsetMechanism()
    elif cell == "g-drift":
        outcome = SubsetOutcome()
        treatment = KernelMechanism(n)
    else:
        raise ValueError(f"cell must be one of {list(CELLS)}; got {cell!r}")
    return {
        "outcome_learner": outcome,
        "treatment_learner": treatment,
        "q_bounds": Q_BOUNDS,
        "n_folds": 5,
        "learner_folds": 3,
        "simultaneous": False,
        "estimands": ("ate", "ey1", "ey0"),
    }


def summary_rows() -> list[list[str]]:
    """One row per cell: what the design committed to, printed before any fit."""
    rows = []
    for cell in CELLS:
        predicted = drift_coefficients(cell)
        targeted = targeted_coefficients(cell)
        rows.append(
            [
                cell,
                f"{ALPHA:.2f}",
                f"{predicted['c1']:+.4f}",
                f"{predicted['c0']:+.4f}",
                f"{predicted['c_ate']:+.4f}",
                f"{targeted['b1']:+.4f}",
                f"{targeted['b0']:+.4f}",
                f"{targeted['b_ate']:+.4f}",
                f"{min(abs(targeted[key]) for key in ('b1', 'b0', 'b_ate')):.4f}",
            ]
        )
    return rows


SUMMARY_HEADERS: tuple[str, ...] = (
    "cell",
    "alpha",
    "c1 (pred)",
    "c0 (pred)",
    "c_ate (pred)",
    "b1 (pred)",
    "b0 (pred)",
    "b_ate (pred)",
    "min |b|",
)


def exact_remainder(cell: str, n: int) -> dict[str, float]:
    """The *predicted* remainder at this size, :math:`n^{-\\alpha}c_a`.

    Named as Tier 1's is so one harness reads both, and **not** the same kind of number: Tier
    1's is an exact quadrature over a prescribed sequence, and this is the design's
    prediction for a fitted one.  What the study compares it against is
    :func:`benchmarks.drtmle_remainder.plain_remainder`, which is exact at the nuisances the
    fit actually produced.
    """
    coefficients = drift_coefficients(cell)
    factor = float(n) ** -ALPHA
    return {
        "r2_1": factor * coefficients["c1"],
        "r2_0": factor * coefficients["c0"],
        "r2_ate": factor * coefficients["c_ate"],
    }


def exact_targeted_remainder(cell: str, n: int) -> dict[str, float]:
    """The *predicted* bias at this size, :math:`n^{-\\alpha}b_a`.

    Named as Tier 1's is so one harness reads both, and **not** the same kind of number for
    that tier's reason: Tier 1 solves its population score exactly at each size, and this is
    the design's leading-order prediction for a fitted sequence.  What the study compares it
    against is :func:`benchmarks.drtmle_remainder.targeted_remainder`.
    """
    coefficients = targeted_coefficients(cell)
    factor = float(n) ** -ALPHA
    return {
        "r2_1": factor * coefficients["b1"],
        "r2_0": factor * coefficients["b0"],
        "r2_ate": factor * coefficients["b_ate"],
    }


#: Kept beside the functions so a caller cannot reach for a shape the design does not have.
SCALER = OutcomeScaler(*Q_BOUNDS)
