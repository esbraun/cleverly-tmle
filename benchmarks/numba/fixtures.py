"""Deterministic cached-nuisance inputs, built outside every timed block.

What a kernel here consumes is what a *fitted* estimator would hand it: an out-of-fold
propensity, an outcome regression at the observed treatment and at every arm, masks,
folds, weights, clusters.  Fitting learners to produce those would put scikit-learn inside
the measurement, which is the one thing this whole package exists to keep out -- so they
are generated directly, from a law chosen to look like a real fit's output rather than to
be convenient.

**Why generated rather than saved.**  A committed fixture at ``n = 1,000,000`` is 100 MB
per array; a seeded generator is four lines and gives every implementation byte-identical
input by construction, since they are all handed the same arrays.  :func:`digest` is what
records *which* input a row was measured on, so two runs can be compared without either
of them shipping the data.

**The regimes matter more than the size.**  A clever covariate under good overlap is
``O(1)`` and every kernel looks alike; under weak overlap it reaches the hundreds, the
logistic fluctuation has to travel, and the one-step walk takes thousands of steps instead
of tens.  A benchmark run only at ``overlap="good"`` measures the easy case and reports it
as the answer, so :class:`Regime` is a required argument rather than a default.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = [
    "ClusterFixture",
    "InfluenceFixture",
    "LongitudinalFixture",
    "Regime",
    "SurvivalFixture",
    "TargetingFixture",
    "digest",
    "make_cluster",
    "make_influence",
    "make_longitudinal",
    "make_survival",
    "make_targeting",
]

Regime = Literal["good", "moderate", "severe", "rare_outcome", "collinear"]

#: The propensity floor each regime is drawn to sit near.  ``severe`` is deliberately
#: *finite* -- a genuine zero makes the clever covariate infinite and measures the
#: exception path rather than the kernel -- but far enough down that ``1/g`` reaches the
#: hundreds, which is where the fluctuation stops converging in two Newton steps.
_FLOOR = {
    "good": 0.25,
    "moderate": 0.06,
    "severe": 0.004,
    "rare_outcome": 0.15,
    "collinear": 0.10,
}


def _linear_predictor(rng: np.random.Generator, n: int, spread: float) -> np.ndarray:
    covariates = rng.standard_normal((n, 4))
    beta = np.array([0.8, -0.5, 0.3, 0.6])
    return spread * (covariates @ beta) / np.sqrt(beta @ beta)


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700.0, 700.0)))


@dataclass(frozen=True)
class TargetingFixture:
    """Everything the targeting step and the estimand layer read, at one configuration.

    Every array is on the ``[0, 1]`` outcome scale, as the estimator's are: the fluctuation
    is a logistic one and a continuous outcome reaches it already scaled.
    """

    outcome: np.ndarray
    """``(n,)`` observed outcome, scaled."""
    initial_observed: np.ndarray
    """``(n,)`` initial regression at the treatment each unit received."""
    initial_arms: np.ndarray
    """``(n, K)`` initial regression at each counterfactual arm."""
    propensity: np.ndarray
    """``(n, K)`` bounded mechanism."""
    treatment_indicator: np.ndarray
    """``(n, K)`` one-hot of the arm each unit received."""
    covariate_observed: np.ndarray
    """``(n, p)`` clever covariate at the observed treatment -- what the score reads."""
    covariate_arms: np.ndarray
    """``(n, K, p)`` clever covariate at each arm -- what the update is applied at."""
    weights: np.ndarray
    """``(n,)`` observation weights, mean one."""
    observed: np.ndarray
    """``(n,)`` boolean: the outcome was recorded."""
    folds: np.ndarray
    """``(n,)`` fold index."""
    n_folds: int
    arms: tuple[float, ...]
    regime: Regime

    @property
    def n(self) -> int:
        return int(self.outcome.shape[0])

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    @property
    def dim(self) -> int:
        return int(self.covariate_observed.shape[1])


def make_targeting(
    n: int,
    *,
    n_arms: int = 2,
    n_folds: int = 10,
    regime: Regime = "good",
    seed: int = 20260803,
    missingness: float = 0.0,
) -> TargetingFixture:
    """Cached nuisances for one point-treatment fit.

    The mechanism is a softmax of covariate-driven linear predictors floored at the
    regime's bound, so the arms sum to one before truncation and the covariate's
    ``1/g`` has the tail the regime asks for.  The outcome regression is a smooth
    function of the same covariates with an arm effect, so the residual is neither
    degenerate nor independent of the mechanism -- which is the case a fused kernel has
    to get right and a fixture of independent noise would not exercise.
    """
    rng = np.random.default_rng(seed)
    floor = _FLOOR[regime]
    spread = {"good": 0.7, "moderate": 1.8, "severe": 3.4}.get(regime, 1.2)

    scores = np.column_stack(
        [_linear_predictor(rng, n, spread) * (1.0 if k else 0.0) for k in range(n_arms)]
    )
    if regime == "collinear":
        # Two arms whose mechanisms are nearly the same function of W: the clever
        # covariate's columns are then close to parallel and the Hessian near-singular,
        # which is where a solver's line search actually runs.
        scores[:, -1] = scores[:, -1] * 0.02 + scores[:, 0] * 0.98
    exponentials = np.exp(scores - scores.max(axis=1, keepdims=True))
    propensity = exponentials / exponentials.sum(axis=1, keepdims=True)
    propensity = np.clip(propensity, floor, 1.0 - floor)
    propensity /= propensity.sum(axis=1, keepdims=True)
    propensity = np.clip(propensity, floor, 1.0)

    arm_codes = _sample_arms(rng, propensity)
    indicator = np.zeros((n, n_arms))
    indicator[np.arange(n), arm_codes] = 1.0

    base = _linear_predictor(rng, n, 1.1)
    effects = np.linspace(0.0, 0.9, n_arms)
    initial_arms = _expit(base[:, None] + effects[None, :])
    if regime == "rare_outcome":
        initial_arms = _expit(base[:, None] + effects[None, :] - 3.2)
    initial_arms = np.clip(initial_arms, 5e-4, 1.0 - 5e-4)
    initial_observed = initial_arms[np.arange(n), arm_codes]

    outcome = rng.binomial(1, initial_observed).astype(float)

    # The ATE covariate: 1{A = a}/g_a at the observed treatment, 1/g_a at arm a, one
    # column per arm.  This is the ``mean`` group's covariate and the shape every other
    # group's is a variation on, so a kernel benchmarked here is benchmarked on the
    # real thing rather than on an invented matrix.
    covariate_arms = np.zeros((n, n_arms, n_arms))
    for k in range(n_arms):
        covariate_arms[:, k, k] = 1.0 / propensity[:, k]
    covariate_observed = indicator / propensity

    weights = np.ones(n)
    observed = (
        np.ones(n, dtype=bool)
        if missingness <= 0.0
        else rng.random(n) > np.clip(missingness, 0.0, 0.9)
    )
    folds = rng.permutation(np.arange(n) % max(1, n_folds))

    return TargetingFixture(
        outcome=outcome,
        initial_observed=initial_observed,
        initial_arms=initial_arms,
        propensity=propensity,
        treatment_indicator=indicator,
        covariate_observed=covariate_observed,
        covariate_arms=covariate_arms,
        weights=weights,
        observed=observed,
        folds=folds,
        n_folds=max(1, n_folds),
        arms=tuple(float(k) for k in range(n_arms)),
        regime=regime,
    )


def _sample_arms(rng: np.random.Generator, propensity: np.ndarray) -> np.ndarray:
    """Draw one arm per row from its own multinomial, vectorised.

    ``rng.choice`` cannot take a per-row ``p``, and a Python loop over a million rows in a
    *fixture builder* is minutes of setup before any measurement starts.  The inverse-CDF
    construction is the same draw.
    """
    cumulative = np.cumsum(propensity / propensity.sum(axis=1, keepdims=True), axis=1)
    draws = rng.random((propensity.shape[0], 1))
    return np.asarray((draws > cumulative).sum(axis=1), dtype=int)


@dataclass(frozen=True)
class InfluenceFixture:
    """A targeted fit, ready for the estimand and influence-curve layer."""

    outcome: np.ndarray
    targeted_observed: np.ndarray
    targeted_arms: np.ndarray
    """``(n, K)``."""
    propensity: np.ndarray
    treatment_indicator: np.ndarray
    weights: np.ndarray
    observed: np.ndarray
    arm_fractions: np.ndarray
    """``(K,)`` share of the sample in each arm -- the ATT/ATC denominator."""

    @property
    def n(self) -> int:
        return int(self.outcome.shape[0])

    @property
    def n_arms(self) -> int:
        return int(self.targeted_arms.shape[1])


def make_influence(
    n: int, *, n_arms: int = 2, regime: Regime = "good", seed: int = 20260803
) -> InfluenceFixture:
    """A fixture for the multi-estimand influence-curve kernels.

    Derived from :func:`make_targeting` and then fluctuated by a plausible ``epsilon``, so
    the targeted predictions are a fluctuation of an initial fit rather than an
    independent draw -- which is what makes the residual ``Y - Q*`` have the size a real
    one has.
    """
    fixture = make_targeting(n, n_arms=n_arms, regime=regime, seed=seed)
    epsilon = np.full(n_arms, 0.05)
    logit = np.log(fixture.initial_arms / (1.0 - fixture.initial_arms))
    targeted_arms = _expit(logit + fixture.covariate_arms @ epsilon)
    targeted_arms = np.clip(targeted_arms, 5e-4, 1.0 - 5e-4)
    codes = np.argmax(fixture.treatment_indicator, axis=1)
    return InfluenceFixture(
        outcome=fixture.outcome,
        targeted_observed=targeted_arms[np.arange(n), codes],
        targeted_arms=targeted_arms,
        propensity=fixture.propensity,
        treatment_indicator=fixture.treatment_indicator,
        weights=fixture.weights,
        observed=fixture.observed,
        arm_fractions=fixture.treatment_indicator.mean(axis=0),
    )


@dataclass(frozen=True)
class ClusterFixture:
    """Influence curves with a cluster structure, for the aggregation kernels."""

    influence: np.ndarray
    """``(n, m)``."""
    cluster: np.ndarray
    """``(n,)`` integer codes, *not* assumed sorted or dense."""
    n_clusters: int
    shape: str

    @property
    def n(self) -> int:
        return int(self.influence.shape[0])

    @property
    def n_estimands(self) -> int:
        return int(self.influence.shape[1])


def make_cluster(
    n: int,
    *,
    n_clusters: int = 500,
    n_estimands: int = 5,
    shape: Literal["balanced", "poisson", "skewed"] = "balanced",
    seed: int = 20260803,
) -> ClusterFixture:
    """Influence curves whose rows belong to clusters of the requested size profile.

    ``shape`` is a first-class dimension because the cost of an indexed accumulation is
    not a function of the row count alone: a balanced design has every cluster the same
    size and perfect load balance across threads, while a skewed one puts a large share
    of the rows in a handful of clusters, which is where a per-cluster parallelisation
    stalls and a per-row one does not.

    The codes are deliberately **shuffled and sparse** -- gaps in the integer range, in no
    particular order -- because the production implementation calls ``np.unique`` to
    densify them and a fixture of ``0..C-1`` in order would hide that sort's cost.
    """
    rng = np.random.default_rng(seed)
    if shape == "balanced":
        codes = np.arange(n) % n_clusters
    elif shape == "poisson":
        sizes = rng.poisson(max(1.0, n / n_clusters), n_clusters) + 1
        codes = np.repeat(np.arange(n_clusters), sizes)[:n]
        if codes.size < n:
            codes = np.concatenate([codes, rng.integers(0, n_clusters, n - codes.size)])
    else:  # skewed: a Zipf-like profile, a few clusters holding most of the rows
        weights = 1.0 / (1.0 + np.arange(n_clusters))
        codes = rng.choice(n_clusters, size=n, p=weights / weights.sum())
    codes = np.asarray(codes, dtype=np.int64)
    # Sparse, unsorted labels: what a real `id` column looks like.
    relabel = rng.permutation(n_clusters) * 7 + 1000
    codes = relabel[codes]
    influence = rng.standard_normal((n, n_estimands))
    influence[:, 0] *= 3.0  # estimands do not share a scale
    return ClusterFixture(
        influence=np.ascontiguousarray(influence),
        cluster=codes,
        n_clusters=n_clusters,
        shape=shape,
    )


@dataclass(frozen=True)
class LongitudinalFixture:
    """Cached node-level predictions for the sequential backward recursion.

    The learner is gone: ``initial[t]`` is what ``cross_fit_predictions`` would have
    returned at node ``t``.  What remains is exactly the package-owned half -- the masks,
    the cumulative mechanism, the clever covariate, the fluctuation, the carry-back.
    """

    treatment_probability: np.ndarray
    """``(n, T)`` ``P(A_t = 1 | H_t)``."""
    censoring_probability: np.ndarray
    """``(n, T)`` ``P(C_t = 1 | H_t)``."""
    treated: np.ndarray
    """``(n, T)`` observed arm, 0/1."""
    uncensored: np.ndarray
    """``(n, T)`` 0/1."""
    outcome: np.ndarray
    """``(n,)`` end-of-study outcome, scaled."""
    initial: np.ndarray
    """``(T, n)`` node regressions, in ``[0, 1]``."""
    assignment: np.ndarray
    """``(R, n, T)`` the arm each regimen assigns each unit at each node."""
    weights: np.ndarray
    labels: tuple[str, ...]

    @property
    def n(self) -> int:
        return int(self.outcome.shape[0])

    @property
    def n_times(self) -> int:
        return int(self.treated.shape[1])

    @property
    def n_regimens(self) -> int:
        return int(self.assignment.shape[0])


def make_longitudinal(
    n: int,
    *,
    n_times: int = 5,
    n_regimens: int = 2,
    regime: Regime = "good",
    dynamic: bool = False,
    seed: int = 20260803,
) -> LongitudinalFixture:
    """Cached nuisances for a longitudinal fit with ``n_times`` nodes.

    Positivity is made to *degrade over time* -- the mechanism's spread grows with the
    node index -- because that is what a real longitudinal fit does: the cumulative
    product of ``2T`` factors is what the clever covariate divides by, and a fixture whose
    factors are all ``0.5`` gives a covariate of ``2^T`` regardless of the data and tells
    a benchmark nothing about where the work goes.
    """
    rng = np.random.default_rng(seed)
    floor = _FLOOR[regime]
    spread = {"good": 1.2, "moderate": 2.2, "severe": 3.2}.get(regime, 1.6)
    growth = np.linspace(0.8, spread, n_times)

    treatment_probability = np.empty((n, n_times))
    censoring_probability = np.empty((n, n_times))
    for t in range(n_times):
        treatment_probability[:, t] = np.clip(
            _expit(_linear_predictor(rng, n, growth[t])), floor, 1.0 - floor
        )
        censoring_probability[:, t] = np.clip(
            _expit(2.2 + _linear_predictor(rng, n, 0.4)), max(floor, 0.5), 1.0 - 1e-3
        )
    treated = rng.binomial(1, treatment_probability).astype(float)
    uncensored = rng.binomial(1, censoring_probability).astype(float)
    # Censoring is monotone: once out, out.
    uncensored = np.minimum.accumulate(uncensored, axis=1)

    outcome = rng.random(n)
    initial = np.clip(rng.random((n_times, n)) * 0.6 + 0.2, 1e-3, 1.0 - 1e-3)

    if dynamic:
        # A rule reading a time-varying covariate: a different arm per unit per node,
        # which is the shape that makes the assignment a matrix rather than a vector.
        thresholds = np.linspace(-0.3, 0.3, n_regimens)
        signal = rng.standard_normal((n, n_times))
        assignment = np.stack([(signal > threshold).astype(float) for threshold in thresholds])
    else:
        plans = [
            np.array([(r >> t) & 1 for t in range(n_times)], dtype=float) for r in range(n_regimens)
        ]
        assignment = np.stack([np.broadcast_to(p, (n, n_times)).copy() for p in plans])

    return LongitudinalFixture(
        treatment_probability=treatment_probability,
        censoring_probability=censoring_probability,
        treated=treated,
        uncensored=uncensored,
        outcome=outcome,
        initial=initial,
        assignment=np.ascontiguousarray(assignment),
        weights=np.ones(n),
        labels=tuple(f"r{i}" for i in range(n_regimens)),
    )


@dataclass(frozen=True)
class SurvivalFixture:
    """A longitudinal fixture plus per-node, per-cause event indicators."""

    base: LongitudinalFixture
    event: np.ndarray
    """``(n, T)`` all-cause: 1 where the unit had *any* absorbing event at that node."""
    cause_event: np.ndarray
    """``(J, n, T)`` per cause."""
    causes: tuple[str, ...]

    @property
    def n(self) -> int:
        return self.base.n

    @property
    def n_times(self) -> int:
        return self.base.n_times

    @property
    def n_causes(self) -> int:
        return len(self.causes)


def make_survival(
    n: int,
    *,
    n_times: int = 20,
    n_regimens: int = 2,
    n_causes: int = 1,
    incidence: float = 0.06,
    regime: Regime = "good",
    seed: int = 20260803,
) -> SurvivalFixture:
    """Cached nuisances for a discrete-time survival or competing-risks fit.

    The event is absorbing and drawn per node at ``incidence``, split across causes, so the
    risk set shrinks down the nodes exactly as it does in a real fit -- which is the whole
    point of the representation benchmark: the work at node ``t`` is proportional to the
    survivors, not to ``n``, and an implementation that ignores that does ``T`` full-length
    passes where it needs a shrinking one.
    """
    base = make_longitudinal(n, n_times=n_times, n_regimens=n_regimens, regime=regime, seed=seed)
    rng = np.random.default_rng(seed + 1)
    hazard = np.clip(incidence * (0.6 + rng.random((n, n_times))), 1e-4, 0.9)
    draws = rng.random((n, n_times)) < hazard
    # Absorbing, and recorded the way ``LongitudinalData.event_by`` reads it: one where
    # the event had happened *at or before* the node.  So take the first draw and carry it
    # forward, which is the cumulative maximum of the one-hot rather than its cumulative
    # sum -- the two agree here only because the one-hot has a single one per row, and
    # writing the maximum says which of the two the array means.
    first = np.argmax(draws, axis=1)
    ever = draws.any(axis=1)
    onset = np.zeros((n, n_times))
    onset[np.arange(n)[ever], first[ever]] = 1.0
    event = np.maximum.accumulate(onset, axis=1)

    which = rng.integers(0, n_causes, n)
    cause_event = np.zeros((n_causes, n, n_times))
    for j in range(n_causes):
        rows = which == j
        cause_event[j][rows] = event[rows]
    return SurvivalFixture(
        base=base,
        event=np.ascontiguousarray(event),
        cause_event=np.ascontiguousarray(cause_event),
        causes=tuple(f"cause{j}" for j in range(n_causes)),
    )


def digest(*arrays: np.ndarray) -> str:
    """A short content hash of the arrays a measurement was taken on.

    Written into every result row so that two rows claiming to be the same configuration
    can be checked to have been, rather than assumed to have been.
    """
    hasher = hashlib.blake2b(digest_size=8)
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        hasher.update(str(contiguous.shape).encode())
        hasher.update(str(contiguous.dtype).encode())
        hasher.update(contiguous.tobytes())
    return hasher.hexdigest()
