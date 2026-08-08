"""Synthetic data-generating processes with known truth.

Validating a causal estimator needs data where the answer is known, and knowing
the answer *precisely* matters: if the reference value carries its own Monte Carlo
error of 0.02, a bias test cannot resolve a bias of 0.01.

Each process here is defined by explicit structural functions -- a propensity
:math:`g(W)`, an outcome mean :math:`\\bar Q(a, W)`, and optionally a missingness or
intermediate mechanism.  The population estimands then follow from those functions
by integration over the covariate distribution, which
:meth:`DGP.truth` evaluates with a *quasi*-Monte Carlo (Sobol) rule rather than
plain sampling.  Sobol points fill the space far more evenly than random draws, so
the same budget buys several extra digits: the reference values are accurate to
roughly 1e-5 and, being deterministic, identical across runs and platforms.

The ATT and ATC are integrated against :math:`g(W)` and :math:`1 - g(W)` directly
instead of simulating treatment assignment, which removes another source of error:

.. math::

    \\mathrm{ATT} = \\frac{E[g(W)\\,(\\bar Q(1, W) - \\bar Q(0, W))]}{E[g(W)]}.

Every generator returns ``(frame, truth)`` where ``frame`` is a pandas or polars
dataframe (``backend=``) and ``truth`` maps estimand names to their population
values.  ``truth["sample_*"]`` additionally reports the realised-sample estimands,
which is the right reference when checking a single fit rather than averaging over
replications.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import stats

from .._typing import Backend, FloatArray
from ..utils.bounds import expit
from ..utils.frames import as_frame, backend_of, column_array, frame_from_dict

__all__ = [
    "DGP",
    "MultiArmDGP",
    "ShiftDGP",
    "make_biased_sample",
    "make_binary_outcome",
    "make_cde",
    "make_clustered",
    "make_heterogeneous",
    "make_instrument",
    "make_linear_ate",
    "make_missing_outcome",
    "make_missing_outcome_binary",
    "make_multi_arm",
    "make_nonlinear_ate",
    "make_shift_dose",
    "make_weak_overlap",
]

#: Sobol points used for the population estimands: 2**18 = 262144, which puts the
#: quasi-Monte Carlo error well below any tolerance the tests apply.
_TRUTH_POINTS = 2**18

#: The scramble every reference value is taken at.  A *default* rather than a constant of
#: the rule -- :meth:`DGP.quadrature` takes any scramble, and two of them are independent
#: randomisations of one integration rule rather than two rules.  This one is fixed so that
#: :meth:`DGP.truth` is reproducible across runs and platforms, which is what a reference
#: value has to be; a caller measuring the rule's *own* error varies it instead.
_SCRAMBLE_SEED = 20240101


@dataclass(frozen=True)
class DGP:
    """A structural causal model whose population estimands can be computed exactly.

    Attributes
    ----------
    name:
        Identifier, used in reports and as part of the truth cache key.
    n_latent:
        Number of independent standard-normal latent variables.  The first
        ``len(covariate_names)`` of them are observed covariates; any extras are
        unobserved (a cluster random effect, or a confounder deliberately withheld).
    covariate_names:
        Names of the observed covariates.
    propensity:
        ``g(W) = P(A = 1 | latent)``.
    outcome_mean:
        ``E[Y | A = a, Z = z, latent]``.  ``z`` is ``None`` for processes without an
        intermediate variable.
    family:
        ``"binomial"`` when ``outcome_mean`` returns a probability, else ``"gaussian"``.
    noise_scale:
        Standard deviation of the additive error, for a gaussian outcome.
    missingness:
        ``P(Delta = 1 | A = a, latent)``, or ``None`` when outcomes are always observed.
    intermediate:
        ``P(Z = 1 | A = a, latent)``, or ``None``.
    cluster_size:
        When set, observations are grouped into clusters of this size and the last
        latent variable is shared within a cluster -- producing genuine
        within-cluster dependence rather than merely a grouping label.
    """

    name: str
    n_latent: int
    covariate_names: tuple[str, ...]
    propensity: Callable[[FloatArray], FloatArray]
    outcome_mean: Callable[[FloatArray, float, float | None], FloatArray]
    family: str = "gaussian"
    noise_scale: float = 1.0
    missingness: Callable[[FloatArray, float], FloatArray] | None = None
    intermediate: Callable[[FloatArray, float], FloatArray] | None = None
    cluster_size: int | None = None
    hidden_names: tuple[str, ...] = field(default=())

    # ------------------------------------------------------------------ truth

    def truth(self, intermediate_value: float | None = None) -> dict[str, float]:
        """Population estimands, by quasi-Monte Carlo integration.

        Cached per ``(name, intermediate_value)``, so repeated calls inside a
        simulation study cost nothing after the first.
        """
        return dict(_cached_truth(self, intermediate_value))

    def quadrature(
        self, points: int = _TRUTH_POINTS, *, scramble: int = _SCRAMBLE_SEED
    ) -> FloatArray:
        """The ``(points, n_latent)`` latent matrix :meth:`expectation` integrates on.

        :meth:`expectation` is the escape hatch for a population *scalar*; this is the one
        for a population quantity that is not a scalar -- a whole evaluation frame, say, whose
        rows a fitted model has to be predicted at before anything can be averaged.
        A remainder diagnostic is the caller: :math:`P_0\\hat D` is an integral of
        a *fitted* curve, so it cannot be written as a closure over the law and handed to
        :meth:`expectation` -- the integrand has to be materialised, predicted at, and only
        then averaged.

        **The grids are nested within a scramble, and that is the property to preserve.**  The
        underlying sequence is scrambled Sobol, so ``quadrature(2**12)`` is the first 4,096
        rows of ``quadrature(2**14)`` *at the same* ``scramble``, and at the default both are
        prefixes of the rule :meth:`truth` uses.  A ladder of grids is therefore a sequence of
        *refinements* of one rule rather than a set of unrelated draws.  Only powers of two
        keep that -- and Sobol's balance properties hold only at powers of two in any case --
        so anything else is refused rather than silently rebalanced.

        **``scramble`` is what makes the rule randomised rather than merely deterministic**,
        and the distinction is the one thing here that is easy to get backwards.  Two scrambles
        are not two grids to be compared for a discretisation: they are **independent
        randomisations of one rule**, each unbiased for the integral at every point count --
        the randomisation is over the scramble and not over how many points are taken.  So the
        spread of an integral across scrambles is a standard error, assuming nothing about a
        convergence rate, where the movement between two *rungs* assumes one and is a
        stability diagnostic only.  Owen's analysis of scrambled nets is where that unbiasedness
        comes from, and ``docs/roadmap.md``'s E1b is where reading a ladder as the first cost
        two withdrawn claims.  A caller that changes ``scramble`` between two quantities meant
        to cancel -- :math:`\\psi_0` and :math:`P_0\\hat D`, say -- has differenced two rules and
        put the :math:`O(1)` error back; pass one scramble to both.

        >>> dgp = linear_dgp()
        >>> coarse, fine = dgp.quadrature(2**10), dgp.quadrature(2**12)
        >>> bool(np.array_equal(coarse, fine[: 2**10]))
        True
        >>> bool(np.array_equal(dgp.quadrature(2**10), dgp.quadrature(2**10, scramble=7)))
        False
        """
        if points <= 0 or points & (points - 1):
            raise ValueError(
                f"quadrature points must be a positive power of two, so that a coarser grid "
                f"is a prefix of a finer one; got {points}"
            )
        return _sobol_normal(self.n_latent, points, scramble)

    def expectation(self, integrand: Callable[[FloatArray], FloatArray]) -> float:
        r"""``E_0[integrand(latent)]``, by the rule :meth:`truth` integrates with.

        The escape hatch for a population quantity that is not an estimand: a remainder
        term, a drift coefficient, the :math:`L_2` distance between a prescribed nuisance
        and the true one.  ``integrand`` is handed the ``(count, n_latent)`` latent matrix
        -- the same argument :attr:`propensity` and :attr:`outcome_mean` take, hidden
        columns included -- and returns one value per row.

        It exists so that such a quantity is integrated on **the same rule as the truth it
        is compared against**.  A coverage study with injected nuisances is the caller: its
        prescribed nuisance sequence has a drift coefficient
        :math:`c_a = P_0[(g_{1,a} - g_{0,a})/g_{1,a} \cdot h_a]` that has to be verified
        against the same law the coverage is measured at, and a second quadrature -- a large
        random draw, or Sobol points with a different seed -- would put a Monte Carlo error
        of its own between the two and leave a disagreement unattributable.

        Not cached, unlike :meth:`truth`: the key would have to be the callable, and two
        closures over different constants are not the same integrand however they compare.

        >>> dgp = linear_dgp()
        >>> mean_g = dgp.expectation(dgp.propensity)
        >>> bool(0.4 < mean_g < 0.6)
        True
        """
        latent = self.quadrature()
        values = np.asarray(integrand(latent), dtype=float).reshape(-1)
        if values.size != _TRUTH_POINTS:
            raise ValueError(
                f"integrand must return one value per integration point; got {values.size} "
                f"for {_TRUTH_POINTS} points"
            )
        return float(np.mean(values))

    def _integrate(self, intermediate_value: float | None) -> dict[str, float]:
        latent = self.quadrature()
        g = np.clip(np.asarray(self.propensity(latent), dtype=float), 0.0, 1.0)
        q1 = np.asarray(self.outcome_mean(latent, 1.0, intermediate_value), dtype=float)
        q0 = np.asarray(self.outcome_mean(latent, 0.0, intermediate_value), dtype=float)
        return _estimands_from(q1, q0, g, self.family)

    def incremental_truth(self, deltas: Sequence[float]) -> dict[str, float]:
        r"""``E[Y^{q_delta}]`` per tilt, and the contrast against the first.

        No new class and no new integration scheme: an incremental intervention's mean is
        a closed-form function of the three pieces :meth:`_integrate` already builds, so
        it is a method rather than a ``DGP`` of its own -- unlike a dose, where the
        mechanism itself changes shape.

        .. math::

            \Psi(\delta) = E_W\!\left[\frac{\delta g \bar Q_1 + (1 - g)\bar Q_0}
                                             {\delta g + 1 - g}\right]

        The integrand is bounded by :math:`\max(\bar Q_1, \bar Q_0)` whatever the
        mechanism does, which is worth noting because every other truth here has to worry
        about ``1 / g``: this one needs no clipping and its quadrature error does not blow
        up as overlap fails.  The first delta is the reference, matching the estimator's
        rule that contrasts are taken against the first declared.
        """
        latent = self.quadrature()
        g = np.clip(np.asarray(self.propensity(latent), dtype=float), 0.0, 1.0)
        q1 = np.asarray(self.outcome_mean(latent, 1.0, None), dtype=float)
        q0 = np.asarray(self.outcome_mean(latent, 0.0, None), dtype=float)
        names = ["natural course" if delta == 1.0 else f"odds x{delta:g}" for delta in deltas]
        means = {
            name: float(np.mean((delta * g * q1 + (1.0 - g) * q0) / (delta * g + 1.0 - g)))
            for name, delta in zip(names, deltas, strict=True)
        }
        out = {f"ey_ipsi[{name}]": value for name, value in means.items()}
        reference = names[0]
        for name in names[1:]:
            out[f"ate_ipsi[{name} vs {reference}]"] = means[name] - means[reference]
        return out

    def sample_truth(
        self, latent: FloatArray, intermediate_value: float | None = None
    ) -> dict[str, float]:
        """Estimands for one realised sample, given its latent variables."""
        g = np.clip(np.asarray(self.propensity(latent), dtype=float), 0.0, 1.0)
        q1 = np.asarray(self.outcome_mean(latent, 1.0, intermediate_value), dtype=float)
        q0 = np.asarray(self.outcome_mean(latent, 0.0, intermediate_value), dtype=float)
        return _estimands_from(q1, q0, g, self.family)

    # ----------------------------------------------------------------- sample

    def sample(
        self,
        n: int,
        *,
        seed: int | np.random.Generator | None = None,
        intermediate_value: float | None = None,
        backend: Backend | str | None = None,
    ) -> tuple[Any, dict[str, float]]:
        """Draw ``n`` observations and report the truth.

        Returns
        -------
        ``(frame, truth)``.  ``truth`` holds the population estimands plus
        ``sample_*`` entries for the realised sample.
        """
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        if self.intermediate is not None and intermediate_value is None:
            # A controlled direct effect is only defined at a specific value of the
            # intermediate. Default to 0 so callers that do not know or care about the
            # intermediate -- CoverageStudy, for one -- still get a coherent truth.
            intermediate_value = 0.0
        latent, cluster = self._latent(n, rng)

        g = np.clip(np.asarray(self.propensity(latent), dtype=float), 1e-9, 1.0 - 1e-9)
        a = rng.binomial(1, g).astype(float)

        z = None
        if self.intermediate is not None:
            pz = np.clip(np.asarray(self.intermediate(latent, 1.0), dtype=float), 1e-9, 1 - 1e-9)
            pz_zero = np.clip(
                np.asarray(self.intermediate(latent, 0.0), dtype=float), 1e-9, 1 - 1e-9
            )
            probability = np.where(a == 1.0, pz, pz_zero)
            z = rng.binomial(1, probability).astype(float)

        mean = self._observed_mean(latent, a, z)
        if self.family == "binomial":
            y = rng.binomial(1, np.clip(mean, 0.0, 1.0)).astype(float)
        else:
            y = mean + rng.normal(scale=self.noise_scale, size=n)

        observed = None
        if self.missingness is not None:
            pi = np.clip(
                np.where(
                    a == 1.0,
                    np.asarray(self.missingness(latent, 1.0), dtype=float),
                    np.asarray(self.missingness(latent, 0.0), dtype=float),
                ),
                1e-9,
                1.0 - 1e-9,
            )
            observed = rng.binomial(1, pi).astype(float)
            y = np.where(observed == 1.0, y, np.nan)

        payload: dict[str, Any] = {"Y": y, "A": a}
        for index, name in enumerate(self.covariate_names):
            payload[name] = latent[:, index]
        if z is not None:
            payload["Z"] = z
        if observed is not None:
            payload["Delta"] = observed
        if cluster is not None:
            payload["cluster"] = cluster

        truth = self.truth(intermediate_value)
        for key, value in self.sample_truth(latent, intermediate_value).items():
            truth[f"sample_{key}"] = value
        return frame_from_dict(payload, backend=backend), truth

    def _latent(self, n: int, rng: np.random.Generator) -> tuple[FloatArray, FloatArray | None]:
        latent = rng.normal(size=(n, self.n_latent))
        if self.cluster_size is None:
            return latent, None
        size = int(self.cluster_size)
        if size < 2:
            raise ValueError(f"cluster_size must be at least 2; got {size}")
        cluster = np.repeat(np.arange((n + size - 1) // size), size)[:n].astype(float)
        # The final latent variable is shared within a cluster, which is what creates
        # the within-cluster correlation the cluster-robust variance must absorb.
        shared = rng.normal(size=int(cluster.max()) + 1)
        latent[:, -1] = shared[cluster.astype(int)]
        return latent, cluster

    def _observed_mean(self, latent: FloatArray, a: FloatArray, z: FloatArray | None) -> FloatArray:
        if z is None:
            one = self.outcome_mean(latent, 1.0, None)
            zero = self.outcome_mean(latent, 0.0, None)
            return np.asarray(np.where(a == 1.0, one, zero), dtype=float)
        out = np.empty(latent.shape[0], dtype=float)
        for a_value in (0.0, 1.0):
            for z_value in (0.0, 1.0):
                mask = (a == a_value) & (z == z_value)
                if mask.any():
                    out[mask] = np.asarray(
                        self.outcome_mean(latent, a_value, z_value), dtype=float
                    )[mask]
        return out


def _estimands_from(q1: FloatArray, q0: FloatArray, g: FloatArray, family: str) -> dict[str, float]:
    """Population (or sample) estimands from counterfactual means and propensities."""
    contrast = q1 - q0
    truth: dict[str, float] = {
        "ey1": float(np.mean(q1)),
        "ey0": float(np.mean(q0)),
        "ate": float(np.mean(contrast)),
        "att": float(np.sum(g * contrast) / np.sum(g)),
        "atc": float(np.sum((1.0 - g) * contrast) / np.sum(1.0 - g)),
    }
    if family == "binomial":
        risk_one, risk_zero = truth["ey1"], truth["ey0"]
        truth["rr"] = float(risk_one / risk_zero)
        truth["or"] = float((risk_one / (1.0 - risk_one)) / (risk_zero / (1.0 - risk_zero)))
    return truth


@lru_cache(maxsize=64)
def _cached_truth(dgp: DGP, intermediate_value: float | None) -> tuple[tuple[str, float], ...]:
    return tuple(dgp._integrate(intermediate_value).items())


# 32 rather than 8 because a caller measuring the rule's own error walks a *ladder* of point
# counts across a *set* of scrambles, and the product of the two evicted the whole cache at 8.
# What it costs to miss is one Sobol generation and one `norm.ppf` -- milliseconds at these
# sizes -- so this is about not repeating that per fold and per estimand, not about the cost
# of the sequence.
@lru_cache(maxsize=32)
def _sobol_normal(dimension: int, count: int, seed: int = _SCRAMBLE_SEED) -> FloatArray:
    """``count`` standard-normal quasi-random points in ``dimension`` dimensions.

    A scrambled Sobol sequence mapped through the normal quantile function.  Scrambling
    avoids the pathological low-dimensional projections of an unscrambled sequence, and
    the seed selects *which* scrambling: at the default the reference values are
    reproducible across runs and platforms, and two other seeds give two independent
    randomisations of the same rule.  See :meth:`DGP.quadrature` on why that difference
    matters more than it looks.
    """
    engine = stats.qmc.Sobol(d=dimension, scramble=True, seed=seed)
    uniform = engine.random(count)
    clipped = np.clip(uniform, 1e-12, 1.0 - 1e-12)
    return np.asarray(stats.norm.ppf(clipped), dtype=float)


# ---------------------------------------------------------------- generators


def _make(
    dgp: DGP,
    n: int,
    seed: int | np.random.Generator | None,
    backend: Backend | str | None,
    intermediate_value: float | None = None,
) -> tuple[Any, dict[str, float]]:
    return dgp.sample(n, seed=seed, intermediate_value=intermediate_value, backend=backend)


def linear_dgp(effect: float = 1.5) -> DGP:
    """Everything linear: both nuisance models are correctly specified by a GLM."""

    def propensity(w: FloatArray) -> FloatArray:
        return expit(0.3 * w[:, 0] - 0.2 * w[:, 1] + 0.1 * w[:, 2])

    def outcome_mean(w: FloatArray, a: float, z: float | None) -> FloatArray:
        del z
        return 2.0 + effect * a + w[:, 0] + 0.5 * w[:, 1] - 0.8 * w[:, 2] + 0.4 * w[:, 3]

    return DGP(
        name=f"linear_ate(effect={effect})",
        n_latent=4,
        covariate_names=("W1", "W2", "W3", "W4"),
        propensity=propensity,
        outcome_mean=outcome_mean,
    )


def make_linear_ate(
    n: int = 1000,
    *,
    seed: int | np.random.Generator | None = None,
    effect: float = 1.5,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """A homogeneous treatment effect with linear nuisance functions.

    The baseline case: a correctly specified GLM should be unbiased here, so any bias
    a test detects points at the estimator rather than at model misspecification.
    Because the effect is constant, ``ate == att == atc == effect`` exactly.
    """
    return _make(linear_dgp(effect), n, seed, backend)


def nonlinear_dgp() -> DGP:
    """Nonlinear, heterogeneous, and interacted -- a GLM is misspecified for both."""

    def propensity(w: FloatArray) -> FloatArray:
        return expit(
            0.6 * w[:, 0] - 0.4 * w[:, 1] ** 2 + 0.5 * w[:, 1] * w[:, 2] + 0.3 * (w[:, 3] > 0)
        )

    def outcome_mean(w: FloatArray, a: float, z: float | None) -> FloatArray:
        del z
        baseline = (
            1.0
            + 0.8 * np.sin(1.5 * w[:, 0])
            + 0.6 * w[:, 1] ** 2
            - 0.5 * w[:, 2] * w[:, 3]
            + 0.4 * np.abs(w[:, 3])
        )
        effect = 2.0 + 0.7 * w[:, 0] - 0.5 * (w[:, 1] > 0)
        return baseline + effect * a

    return DGP(
        name="nonlinear_ate",
        n_latent=4,
        covariate_names=("W1", "W2", "W3", "W4"),
        propensity=propensity,
        outcome_mean=outcome_mean,
    )


def make_nonlinear_ate(
    n: int = 1000,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """Nonlinear nuisance functions and a heterogeneous effect.

    This is the process that exercises double robustness: a linear model is wrong for
    both ``g`` and ``Qbar``, the treatment effect varies with ``W``, and so ``ate``,
    ``att`` and ``atc`` all differ.  An estimator that quietly assumes a constant effect
    will fail here.
    """
    return _make(nonlinear_dgp(), n, seed, backend)


def heterogeneous_dgp(slope: float = 1.5) -> DGP:
    r"""Effect modification aligned with the propensity, so ``att > ate > atc`` strictly.

    Every other process here leaves the *order* of the three contrasts to chance.
    :func:`linear_dgp` has a constant effect, so all three coincide and no arrangement of
    them can be wrong; :func:`nonlinear_dgp` separates them, but which of ``att`` and
    ``atc`` comes out larger is an accident of its coefficients rather than something a
    test can state in advance.  Neither can catch an estimator that conditions on the
    wrong arm or inverts the propensity odds :math:`g_1/g_0`, because both mistakes return
    a plausible number in the plausible range.

    Here the conditional effect

    .. math:: \tau(W) = \bar Q(1, W) - \bar Q(0, W) = 1 + \text{slope} \cdot W_1

    is increasing in :math:`W_1` and so is the propensity, so the treated are drawn
    disproportionately from the covariate values where the effect is large:

    .. math::

        \mathrm{ATT} = \frac{E[\tau(W) g(W)]}{E[g(W)]}
              \;>\; E[\tau(W)] = \mathrm{ATE}
              \;>\; \frac{E[\tau(W)(1 - g(W))]}{E[1 - g(W)]} = \mathrm{ATC},

    the inequalities being strict exactly because :math:`\operatorname{Cov}(\tau, g) > 0`.
    At the default slope the three sit near ``1.68``, ``1.00`` and ``0.32``: far enough
    apart that the ordering survives sampling error at moderate ``n``, which is what makes
    it assertable rather than merely true.

    Both nuisance functions are GLM-correct in their own right -- the propensity is exactly
    logistic and the outcome mean is linear in ``[A, W1, W2, A*W1]`` -- so a failure here
    points at the estimator rather than at misspecification.  A GLM given only main effects
    cannot represent the ``A * W1`` interaction, which is deliberate: it is what makes this
    process discriminate between an estimator that assumes a constant effect and one that
    does not.
    """

    def propensity(w: FloatArray) -> FloatArray:
        return expit(1.2 * w[:, 0] - 0.3 * w[:, 1])

    def outcome_mean(w: FloatArray, a: float, z: float | None) -> FloatArray:
        del z
        baseline = 0.5 + 0.7 * w[:, 0] + 0.4 * w[:, 1]
        effect = 1.0 + slope * w[:, 0]
        return baseline + effect * a

    return DGP(
        name=f"heterogeneous(slope={slope})",
        n_latent=2,
        covariate_names=("W1", "W2"),
        propensity=propensity,
        outcome_mean=outcome_mean,
    )


def make_heterogeneous(
    n: int = 1000,
    *,
    seed: int | np.random.Generator | None = None,
    slope: float = 1.5,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """A sample whose treated and control populations have genuinely different effects.

    Use this wherever a test needs ``att``, ``ate`` and ``atc`` to be distinguishable and
    ordered in a known direction -- ``att > ate > atc`` -- rather than merely unequal.
    """
    return _make(heterogeneous_dgp(slope=slope), n, seed, backend)


def weak_overlap_dgp(strength: float = 3.0) -> DGP:
    """Strong confounding, so propensity scores crowd against 0 and 1."""

    def propensity(w: FloatArray) -> FloatArray:
        return expit(strength * w[:, 0] + 0.7 * strength * w[:, 1])

    def outcome_mean(w: FloatArray, a: float, z: float | None) -> FloatArray:
        del z
        return 1.0 + 1.0 * a + 1.5 * w[:, 0] + 1.0 * w[:, 1] + 0.5 * w[:, 2]

    return DGP(
        name=f"weak_overlap(strength={strength})",
        n_latent=3,
        covariate_names=("W1", "W2", "W3"),
        propensity=propensity,
        outcome_mean=outcome_mean,
    )


def make_weak_overlap(
    n: int = 1000,
    *,
    seed: int | np.random.Generator | None = None,
    strength: float = 3.0,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """A practical positivity violation.

    Propensity scores reach into the tails, so a handful of units carry enormous
    inverse-probability weight.  Used to check that truncation diagnostics fire and
    that the estimate degrades gracefully rather than catastrophically.
    """
    return _make(weak_overlap_dgp(strength), n, seed, backend)


def instrument_dgp(instrument_strength: float = 1.5) -> DGP:
    """Three covariates with cleanly separated roles: confounder, instrument, predictor.

    ``W1`` drives both treatment and outcome and so must be adjusted for.  ``W2`` is an
    *instrument*: it drives treatment strongly and is absent from the outcome model.
    ``W3`` predicts only the outcome.
    """

    def propensity(w: FloatArray) -> FloatArray:
        return expit(0.8 * w[:, 0] + instrument_strength * w[:, 1])

    def outcome_mean(w: FloatArray, a: float, z: float | None) -> FloatArray:
        del z  # W2 is deliberately absent: it is an instrument, not a confounder
        return 1.0 + 1.0 * a + 1.5 * w[:, 0] + 0.8 * w[:, 2]

    return DGP(
        name=f"instrument(strength={instrument_strength})",
        n_latent=3,
        covariate_names=("W1", "W2", "W3"),
        propensity=propensity,
        outcome_mean=outcome_mean,
    )


def make_instrument(
    n: int = 1000,
    *,
    seed: int | np.random.Generator | None = None,
    instrument_strength: float = 1.5,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """Instrument-inflation: a covariate that predicts treatment but not the outcome.

    Adjusting for ``W2`` is not merely unnecessary, it is harmful.  Because ``W2`` does
    not confound, including it in ``g`` leaves the bias where it was while pushing
    propensity scores towards 0 and 1, which inflates the variance of the clever
    covariate and hence of the estimate.  Omitting ``W1`` on the other hand *is* biasing.
    A propensity model chosen by prediction accuracy takes both; one chosen
    collaboratively -- against the loss of the targeted outcome model -- should take
    ``W1`` and leave ``W2``, which is the result :class:`~cleverly.CTMLE` is built to
    deliver.  The effect is constant, so ``ate == att == atc == 1``.
    """
    return _make(instrument_dgp(instrument_strength), n, seed, backend)


def missing_outcome_dgp(strength: float = 1.0) -> DGP:
    """Outcomes missing at random given ``(A, W)``.

    ``strength`` scales how hard the process is for a complete-case analysis, and it is
    worth being precise about what makes it hard.  Missingness at random given ``(A, W)``
    does *not* on its own break a complete-case fit: a correctly specified regression of
    ``Y`` on ``(A, W)`` among the observed rows already identifies the estimand, whatever
    the mechanism.  What breaks it is missingness plus an outcome model that cannot fit
    ``Qbar``.  So the two move together here.

    At ``strength = 1`` the outcome mean is linear and a GLM is correctly specified for
    it, and roughly three quarters of the outcomes are observed -- a process on which a
    complete-case analysis is consistent, and modelling the mechanism is a matter of
    efficiency rather than bias.  Above 1 the outcome mean picks up curvature and an
    ``A``-by-``W1`` interaction that no main-effects model can reach, while the mechanism
    sharpens on the same covariate: at ``strength = 2`` the first percentile of
    ``P(Delta = 1 | A, W)`` is about 0.13.  Because the complete cases then carry a
    shifted ``W1`` distribution, the linear approximation fitted to them is the wrong one
    to extrapolate over the full marginal -- which is what makes a complete-case analysis
    biased rather than merely inefficient.  ``strength = 1`` reproduces the original
    process exactly, so existing truths and seeds are unaffected.
    """
    if strength < 1.0:
        raise ValueError(f"strength must be at least 1.0; got {strength}")
    extra = strength - 1.0

    def propensity(w: FloatArray) -> FloatArray:
        return expit(0.4 * w[:, 0] - 0.3 * w[:, 1])

    def outcome_mean(w: FloatArray, a: float, z: float | None) -> FloatArray:
        del z
        linear = 1.0 + 1.2 * a + 0.9 * w[:, 0] + 0.6 * w[:, 1] - 0.4 * w[:, 2]
        if extra == 0.0:
            return linear
        # Curvature and an interaction: neither is reachable by a main-effects GLM, so
        # the complete-case half of double robustness stops being available.
        return linear + extra * (
            1.1 * np.tanh(1.5 * w[:, 0]) + 0.8 * w[:, 1] ** 2 - 0.9 * a * w[:, 0]
        )

    def missingness(w: FloatArray, a: float) -> FloatArray:
        return expit(1.2 + 0.6 * a - (0.8 + 0.6 * extra) * w[:, 0] + 0.3 * w[:, 2])

    return DGP(
        name="missing_outcome" if extra == 0.0 else f"missing_outcome_x{strength:g}",
        n_latent=3,
        covariate_names=("W1", "W2", "W3"),
        propensity=propensity,
        outcome_mean=outcome_mean,
        missingness=missingness,
    )


def make_missing_outcome(
    n: int = 1000,
    *,
    seed: int | np.random.Generator | None = None,
    strength: float = 1.0,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """Missing outcomes, with a missingness mechanism that depends on treatment.

    ``P(Delta = 1 | A, W)`` depends on both ``A`` and ``W``, so the complete cases carry a
    different covariate distribution from the sample -- which is what the ``1 / pi``
    factor in the clever covariate corrects for.  Note that this alone does not make a
    complete-case analysis *biased*: under missingness at random a correctly specified
    outcome regression identifies the estimand without any missingness model at all.  The
    mechanism is what supplies the other half of double robustness, for when the outcome
    model is wrong.  Raise ``strength`` above 1 for a process where it is.
    """
    return _make(missing_outcome_dgp(strength), n, seed, backend)


def missing_outcome_binary_dgp() -> DGP:
    """A binary outcome *and* missing outcomes, so ``rr`` and ``or`` have a truth.

    The two features are bundled nowhere else: every other process varies one thing at a
    time.  Combining them is what gives the ratio estimands under ``delta=`` a population
    value to be checked against, and it puts the outcome scaler on its identity branch so
    the true conditional mean can be handed straight to an oracle learner.
    """

    def propensity(w: FloatArray) -> FloatArray:
        return expit(0.4 * w[:, 0] - 0.3 * w[:, 1] + 0.2 * w[:, 2])

    def outcome_mean(w: FloatArray, a: float, z: float | None) -> FloatArray:
        del z
        return expit(-0.6 + 0.9 * a + 0.7 * w[:, 0] - 0.5 * w[:, 1] + 0.4 * w[:, 2])

    def missingness(w: FloatArray, a: float) -> FloatArray:
        return expit(1.0 + 0.5 * a - 0.9 * w[:, 0] + 0.4 * w[:, 1])

    return DGP(
        name="missing_outcome_binary",
        n_latent=3,
        covariate_names=("W1", "W2", "W3"),
        propensity=propensity,
        outcome_mean=outcome_mean,
        family="binomial",
        missingness=missingness,
    )


def make_missing_outcome_binary(
    n: int = 1000,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """A binary outcome with outcomes missing at random given ``(A, W)``."""
    return _make(missing_outcome_binary_dgp(), n, seed, backend)


def cde_dgp() -> DGP:
    """A binary intermediate variable on the pathway from ``A`` to ``Y``."""

    def propensity(w: FloatArray) -> FloatArray:
        return expit(0.3 * w[:, 0] + 0.2 * w[:, 1])

    def intermediate(w: FloatArray, a: float) -> FloatArray:
        return expit(-0.3 + 1.1 * a + 0.5 * w[:, 0] - 0.4 * w[:, 2])

    def outcome_mean(w: FloatArray, a: float, z: float | None) -> FloatArray:
        if z is None:
            raise ValueError("this process requires an intermediate value")
        return 0.5 + 0.9 * a + 1.4 * z + 0.6 * a * z + 0.8 * w[:, 0] - 0.5 * w[:, 1] + 0.3 * w[:, 2]

    return DGP(
        name="controlled_direct_effect",
        n_latent=3,
        covariate_names=("W1", "W2", "W3"),
        propensity=propensity,
        outcome_mean=outcome_mean,
        intermediate=intermediate,
    )


def make_cde(
    n: int = 1000,
    *,
    seed: int | np.random.Generator | None = None,
    intermediate_value: float = 0.0,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """Controlled direct effect: the effect of ``A`` with ``Z`` held fixed.

    The truth reported is the CDE at ``intermediate_value``.  With the interaction in
    this process the CDE differs between ``z = 0`` (0.9) and ``z = 1`` (1.5), so a fit
    that ignores ``Z`` cannot match either.
    """
    return _make(cde_dgp(), n, seed, backend, intermediate_value=intermediate_value)


def clustered_dgp(cluster_size: int = 10) -> DGP:
    """Within-cluster dependence through a shared latent variable."""

    def propensity(w: FloatArray) -> FloatArray:
        return expit(0.3 * w[:, 0] + 0.6 * w[:, 2])

    def outcome_mean(w: FloatArray, a: float, z: float | None) -> FloatArray:
        del z
        return 1.0 + 1.0 * a + 0.8 * w[:, 0] + 0.5 * w[:, 1] + 1.5 * w[:, 2]

    return DGP(
        name=f"clustered(size={cluster_size})",
        n_latent=3,
        covariate_names=("W1", "W2"),
        propensity=propensity,
        outcome_mean=outcome_mean,
        cluster_size=cluster_size,
        hidden_names=("cluster_effect",),
    )


def make_clustered(
    n: int = 1000,
    *,
    seed: int | np.random.Generator | None = None,
    cluster_size: int = 10,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """Clustered data with an *unobserved* shared effect driving both ``A`` and ``Y``.

    The third latent variable is shared within a cluster and is deliberately not
    included among the covariates, so the influence curves are correlated within
    clusters.  Ignoring ``id=`` here understates the standard error -- which is exactly
    what the cluster-variance test checks.
    """
    return _make(clustered_dgp(cluster_size), n, seed, backend)


def binary_outcome_dgp() -> DGP:
    """A binary outcome, so the risk ratio and odds ratio are defined."""

    def propensity(w: FloatArray) -> FloatArray:
        return expit(0.4 * w[:, 0] - 0.3 * w[:, 1] + 0.2 * w[:, 2])

    def outcome_mean(w: FloatArray, a: float, z: float | None) -> FloatArray:
        del z
        return expit(-0.6 + 0.9 * a + 0.7 * w[:, 0] - 0.5 * w[:, 1] + 0.4 * w[:, 2])

    return DGP(
        name="binary_outcome",
        n_latent=3,
        covariate_names=("W1", "W2", "W3"),
        propensity=propensity,
        outcome_mean=outcome_mean,
        family="binomial",
    )


def make_binary_outcome(
    n: int = 1000,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """A binary outcome with known risks, risk ratio and odds ratio.

    Note that the marginal risk ratio reported in ``truth`` is the *marginal* one --
    a ratio of standardised risks -- not the conditional odds ratio in the structural
    logistic model.  They differ, and conflating them is a common source of apparent
    "bias" that is really a different estimand.
    """
    return _make(binary_outcome_dgp(), n, seed, backend)


def biased_sampling_dgp(heterogeneity: float = 1.2) -> DGP:
    """The population behind :func:`make_biased_sample`.

    The treatment effect varies with ``W1``, which is also what drives selection into
    the sample.  That combination is the whole point: with a homogeneous effect the
    selected and full populations share an ATE and weighting could not be seen to do
    anything, so a test built on such a process would pass with the weights ignored.
    """

    def propensity(w: FloatArray) -> FloatArray:
        return expit(0.4 * w[:, 0] - 0.3 * w[:, 1])

    def outcome_mean(w: FloatArray, a: float, z: float | None) -> FloatArray:
        del z
        return 1.0 + w[:, 0] + 0.5 * w[:, 1] + a * (1.0 + heterogeneity * w[:, 0])

    return DGP(
        name=f"biased_sampling(heterogeneity={heterogeneity})",
        n_latent=2,
        covariate_names=("W1", "W2"),
        propensity=propensity,
        outcome_mean=outcome_mean,
    )


def sampling_probability(w1: FloatArray) -> FloatArray:
    """``P(S = 1 | W1)`` for :func:`make_biased_sample`, bounded away from zero.

    Bounded because inverse-probability-of-sampling weights inherit the positivity
    problem of any inverse-probability estimator: a selection probability near zero
    produces a weight that a handful of rows carry the whole estimate on.
    """
    return np.clip(expit(0.9 * np.asarray(w1, dtype=float) - 0.2), 0.15, 0.9)


def make_biased_sample(
    n_population: int = 4000,
    *,
    seed: int | np.random.Generator | None = None,
    heterogeneity: float = 1.2,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """A sample selected with known, unequal probabilities from a known population.

    Draws ``n_population`` rows, keeps each with probability ``P(S = 1 | W1)``, and
    returns *only the kept rows* -- so the frame is smaller than ``n_population``, by
    design.  It carries two extra columns: ``sampling_prob``, the known selection
    probability, and ``sampling_weight``, its reciprocal.

    This is the survey/selection-bias case with everything known, which makes it the
    process that can actually test the claim in :mod:`cleverly.data.weighting`: because
    selection depends only on observed data and ``w = 1 / P(S = 1 | W1)``, the tilted law
    ``dP_w`` *is* the population law, so a weighted fit estimates the population ATE.  An
    unweighted fit estimates the ATE among the selected, which is a different number
    here -- reported as ``truth["ate_selected"]`` -- and that gap is what a test asserts
    the weighting closes.

    Returns
    -------
    ``(frame, truth)``, where ``truth`` holds the population estimands plus
    ``ate_selected`` (the estimand an unweighted analysis targets), ``n_population`` and
    ``n_selected``.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    dgp = biased_sampling_dgp(heterogeneity)
    native, truth = dgp.sample(n_population, seed=rng, backend=backend)
    population = as_frame(native)

    pi = sampling_probability(column_array(population, "W1"))
    selected = rng.binomial(1, pi).astype(bool)
    if int(selected.sum()) < 10:  # pragma: no cover - only reachable for a tiny population
        raise ValueError(f"only {int(selected.sum())} rows were selected; raise n_population")

    payload: dict[str, Any] = {
        name: column_array(population, name)[selected] for name in population.columns
    }
    payload["sampling_prob"] = pi[selected]
    payload["sampling_weight"] = 1.0 / pi[selected]

    # The estimand an unweighted analysis of the selected rows targets: the ATE in the
    # selected population, computed from the same structural equations that generated it.
    latent = np.column_stack(
        [np.asarray(payload[name], dtype=float) for name in dgp.covariate_names]
    )
    truth["ate_selected"] = float(
        np.mean(dgp.outcome_mean(latent, 1.0, None) - dgp.outcome_mean(latent, 0.0, None))
    )
    truth["n_population"] = float(n_population)
    truth["n_selected"] = float(selected.sum())
    return frame_from_dict(payload, backend=backend_of(population)), truth


@dataclass(frozen=True)
class MultiArmDGP:
    """A process with ``K >= 2`` treatment arms and known counterfactual means.

    Deliberately *not* a mode of :class:`DGP`.  That class is binary all the way down --
    ``propensity`` returns one margin, ``outcome_mean`` takes ``a`` as a scalar 0 or 1,
    and its truth is assembled from a ``(q1, q0, g)`` triple -- and widening it would
    have meant reworking every process and every test that consumes one, to no benefit
    for the binary cases.  The two live side by side instead, and the shared quasi-Monte
    Carlo integration is the only thing that needs to be common.

    Attributes
    ----------
    arm_logits:
        ``(n, K)`` scores whose softmax is ``g(a | W)``.  A softmax rather than a list of
        margins so the mechanism is a proper distribution over the arms by construction,
        with no chance of a process whose probabilities fail to sum to one.
    outcome_mean:
        ``E[Y | A = a, latent]`` for an arm *index*.
    labels:
        What the treatment column holds.  Strings by default, because that is what a
        real multi-arm treatment looks like and it exercises the label/code distinction.
    """

    name: str
    n_latent: int
    covariate_names: tuple[str, ...]
    arm_logits: Callable[[FloatArray], FloatArray]
    outcome_mean: Callable[[FloatArray, int], FloatArray]
    labels: tuple[str, ...]
    family: str = "gaussian"
    noise_scale: float = 1.0

    @property
    def n_arms(self) -> int:
        return len(self.labels)

    def probabilities(self, latent: FloatArray) -> FloatArray:
        """``g(a | W)`` as an ``(n, K)`` array whose rows sum to one."""
        logits = np.asarray(self.arm_logits(latent), dtype=float)
        shifted = logits - logits.max(axis=1, keepdims=True)
        weights = np.exp(shifted)
        return np.asarray(weights / weights.sum(axis=1, keepdims=True), dtype=float)

    def truth(self, reference: str | None = None, *, conditional: bool = False) -> dict[str, float]:
        """Population ``E[Y(a)]`` per arm and ``E[Y(a)] - E[Y(ref)]`` per contrast.

        Keyed by the names the estimator reports, so a test can compare without
        translating.  ``reference`` defaults to the arm the estimator would choose --
        the lowest *sorted* label, which is not generally the first one written down.

        ``conditional=True`` adds the conditional effects, ``att[a vs ref]`` and
        ``atc[a vs ref]``.  Off by default because they are off by default in the report
        too: a caller looping this mapping against a fit's estimates would otherwise be
        asking for parameters the fit was never told to produce.

        A conditional effect weights the contrast by the arm's own mechanism --
        ``E[g_c(W)(Q_a - Q_ref)] / E[g_c(W)]`` is ``E[Q_a - Q_ref | A = c]`` by Bayes,
        with no positivity needed because the process supplies ``g`` exactly -- so it
        integrates over the same quasi-Monte Carlo points as the means.
        """
        latent = _sobol_normal(self.n_latent, _TRUTH_POINTS)
        arms = [
            np.asarray(self.outcome_mean(latent, index), dtype=float)
            for index in range(self.n_arms)
        ]
        means = {label: float(np.mean(arms[index])) for index, label in enumerate(self.labels)}
        base = sorted(self.labels)[0] if reference is None else reference
        truth = {f"ey[{label}]": value for label, value in means.items()}
        for label, value in means.items():
            if label != base:
                truth[f"ate[{label} vs {base}]"] = value - means[base]
        if not conditional:
            return truth

        mechanism = self.probabilities(latent)
        reference_index = self.labels.index(base)
        for index, label in enumerate(self.labels):
            if label == base:
                continue
            contrast = arms[index] - arms[reference_index]
            for stem, conditioning in (("att", index), ("atc", reference_index)):
                weight = mechanism[:, conditioning]
                truth[f"{stem}[{label} vs {base}]"] = float(
                    np.mean(weight * contrast) / np.mean(weight)
                )
        return truth

    def sample(
        self,
        n: int,
        *,
        seed: int | np.random.Generator | None = None,
        backend: Backend | str | None = None,
    ) -> tuple[Any, dict[str, float]]:
        """Draw ``n`` observations and report the truth."""
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        latent = rng.normal(size=(n, self.n_latent))
        probabilities = self.probabilities(latent)
        # One multinomial draw per row. Vectorised through the CDF rather than looped,
        # so a large simulation does not pay a Python-level draw per observation.
        thresholds = np.cumsum(probabilities, axis=1)
        arm = (rng.random((n, 1)) > thresholds).sum(axis=1)
        arm = np.clip(arm, 0, self.n_arms - 1)

        mean = np.zeros(n)
        for index in range(self.n_arms):
            rows = arm == index
            if rows.any():
                mean[rows] = np.asarray(self.outcome_mean(latent[rows], index), dtype=float)
        if self.family == "binomial":
            y = rng.binomial(1, np.clip(mean, 0.0, 1.0)).astype(float)
        else:
            y = mean + rng.normal(scale=self.noise_scale, size=n)

        payload: dict[str, Any] = {"Y": y, "A": np.array(self.labels)[arm]}
        for index, name in enumerate(self.covariate_names):
            payload[name] = latent[:, index]
        return frame_from_dict(payload, backend=backend), self.truth()


def multi_arm_dgp(*, effect: float = 0.6, confounding: float = 0.8) -> MultiArmDGP:
    """Three arms, confounded, with an effect that is not linear in the arm index.

    The middle arm is not halfway between the other two, so an estimator that treated
    the treatment as a single numeric column would be biased rather than merely
    inefficient -- which is the property that makes this process worth having.

    The outcome mean is otherwise additive in the covariates and carries no arm-covariate
    interaction, so the ``K - 1`` indicator design represents it *exactly*.  That is
    deliberate: it makes recovery of the truth a genuine consistency check rather than a
    measurement of how much bias a misspecified working model happens to leave, which is
    a different question and needs a different test to answer honestly.
    """
    step = np.array([0.0, 1.0, 2.4])

    def arm_logits(w: FloatArray) -> FloatArray:
        return np.column_stack(
            [
                np.zeros(w.shape[0]),
                confounding * w[:, 0] - 0.4 * w[:, 1],
                -0.5 * w[:, 0] + confounding * w[:, 1],
            ]
        )

    def outcome_mean(w: FloatArray, arm: int) -> FloatArray:
        return effect * step[arm] + w[:, 0] - 0.5 * w[:, 1] + 0.2 * w[:, 2]

    return MultiArmDGP(
        name="multi_arm",
        n_latent=3,
        covariate_names=("W1", "W2", "W3"),
        arm_logits=arm_logits,
        outcome_mean=outcome_mean,
        labels=("low", "medium", "high"),
        noise_scale=0.8,
    )


def make_multi_arm(
    n: int = 1000,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """A three-armed confounded process with known counterfactual means."""
    return multi_arm_dgp().sample(n, seed=seed, backend=backend)


@dataclass(frozen=True)
class ShiftDGP:
    r"""A continuous dose with a known mean under each modified treatment policy.

    A third process class beside :class:`DGP` and :class:`MultiArmDGP`, for the reason
    :class:`MultiArmDGP` gives: ``DGP`` is binary all the way down, ``MultiArmDGP`` is
    categorical, and a dose is neither.  Here the mechanism is a conditional *density*
    :math:`A \mid W \sim N(\mu(W), \sigma^2)` and the parameter is
    :math:`E[Y^{d}] = E[\bar Q(d(A, W), W)]` for :math:`d(a, w) = \min(a + \delta, u)`.

    **The dose response has to be non-linear, and that is the whole design.**  If
    :math:`\bar Q` were linear in ``a`` -- say :math:`\beta a + h(w)` -- then an uncapped
    shift would give :math:`E[\bar Q(A + \delta, W)] - E[\bar Q(A, W)] = \beta\delta`
    whatever the density of ``A`` is.  The density would cancel out of the point estimate
    entirely, so an oracle test built on such a process would pass with the conditional
    density estimator returning arbitrary numbers, and would be checking the outcome
    regression alone.  The quadratic term makes the truth depend on
    :math:`E[A \mid W]`, so the density has to be right for the estimate to be, and the
    cap makes it depend on the density's *spread* as well as its mean.

    Attributes
    ----------
    dose_mean, dose_scale:
        :math:`\mu(W)` and :math:`\sigma`.  Normal rather than something heavier-tailed
        so the truth is a clean two-dimensional integral, and confounded through
        ``dose_mean`` so the estimand is not the marginal one.
    outcome_mean:
        :math:`E[Y \mid A = a, W]` given ``(latent, dose)`` as arrays of the same length,
        so the caller can write the interaction between them.
    """

    name: str
    n_latent: int
    covariate_names: tuple[str, ...]
    dose_mean: Callable[[FloatArray], FloatArray]
    outcome_mean: Callable[[FloatArray, FloatArray], FloatArray]
    dose_scale: float = 1.0
    family: str = "gaussian"
    noise_scale: float = 1.0

    def shifted(self, dose: FloatArray, delta: float, cap: float | None) -> FloatArray:
        """``d(a, w) = min(a + delta, cap)``, holding a unit at its own dose past the cap.

        Written here as well as in :class:`~cleverly.interventions.Shift` on purpose: an
        oracle that imported the estimator's own policy would agree with it by
        construction, including where both are wrong.
        """
        moved = np.asarray(dose, dtype=float) + float(delta)
        if cap is None:
            return moved
        return np.where(moved > float(cap), np.asarray(dose, dtype=float), moved)

    def truth(self, shifts: Sequence[tuple[float, float | None, str]]) -> dict[str, float]:
        r"""``E[Y^{d}]`` per shift and ``E[Y^{d}] - E[Y^{d_ref}]`` per contrast.

        Keyed by the names the estimator reports, so a test can compare without
        translating.  ``shifts`` is ``(delta, cap, name)`` per policy and the *first* is
        the reference, which is the rule the estimator follows.

        Integrated over a Sobol sequence in ``n_latent + 1`` dimensions: the covariates
        take the first columns and the last supplies :math:`A`'s conditional noise, so
        the joint law of ``(W, A)`` is covered rather than ``W`` alone.  A shift's
        parameter reads the dose the unit *received*, so integrating over ``W`` and
        plugging in :math:`\mu(W)` would compute a different functional.
        """
        draws = _sobol_normal(self.n_latent + 1, _TRUTH_POINTS)
        latent, noise = draws[:, : self.n_latent], draws[:, self.n_latent]
        dose = np.asarray(self.dose_mean(latent), dtype=float) + self.dose_scale * noise
        means = {
            name: float(
                np.mean(
                    np.asarray(
                        self.outcome_mean(latent, self.shifted(dose, delta, cap)), dtype=float
                    )
                )
            )
            for delta, cap, name in shifts
        }
        base = shifts[0][2]
        truth = {f"ey_shift[{name}]": value for name, value in means.items()}
        for name, value in means.items():
            if name != base:
                truth[f"ate_shift[{name} vs {base}]"] = value - means[base]
        return truth

    def sample(
        self,
        n: int,
        *,
        shifts: Sequence[tuple[float, float | None, str]] = ((0.0, None, "natural course"),),
        seed: int | np.random.Generator | None = None,
        backend: Backend | str | None = None,
    ) -> tuple[Any, dict[str, float]]:
        """Draw ``n`` observations and report the truth for the given policies."""
        rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        latent = rng.normal(size=(n, self.n_latent))
        dose = np.asarray(self.dose_mean(latent), dtype=float) + self.dose_scale * rng.normal(
            size=n
        )
        mean = np.asarray(self.outcome_mean(latent, dose), dtype=float)
        if self.family == "binomial":
            y = rng.binomial(1, np.clip(mean, 0.0, 1.0)).astype(float)
        else:
            y = mean + rng.normal(scale=self.noise_scale, size=n)

        payload: dict[str, Any] = {"Y": y, "A": dose}
        for index, name in enumerate(self.covariate_names):
            payload[name] = latent[:, index]
        return frame_from_dict(payload, backend=backend), self.truth(shifts)


def shift_dgp(*, curvature: float = 0.25, confounding: float = 0.7) -> ShiftDGP:
    """A confounded dose with a quadratic response, so the density is load-bearing.

    ``curvature`` is the coefficient on :math:`a^2`.  At zero the process is still a
    valid one, but the shift effect collapses to a constant that no density can move --
    see :class:`ShiftDGP`.  It is a parameter rather than a constant so a test can
    *demonstrate* that, rather than the docstring merely asserting it.

    The outcome mean is otherwise additive in the covariates and carries no
    dose-covariate interaction, so a design holding ``[a, a^2, W]`` represents it
    exactly.  Deliberate, and for the reason :func:`multi_arm_dgp` gives: recovery of the
    truth is then a consistency check rather than a measurement of how much bias a
    misspecified working model happens to leave.
    """

    def dose_mean(w: FloatArray) -> FloatArray:
        return 2.0 + confounding * w[:, 0] - 0.3 * w[:, 1]

    def outcome_mean(w: FloatArray, a: FloatArray) -> FloatArray:
        return 1.0 + 0.5 * a + curvature * a**2 + w[:, 0] - 0.5 * w[:, 1] + 0.2 * w[:, 2]

    return ShiftDGP(
        name="shift_dose",
        n_latent=3,
        covariate_names=("W1", "W2", "W3"),
        dose_mean=dose_mean,
        outcome_mean=outcome_mean,
        dose_scale=1.0,
        noise_scale=0.8,
    )


def make_shift_dose(
    n: int = 1000,
    *,
    shifts: Sequence[tuple[float, float | None, str]] = (
        (0.0, None, "natural course"),
        (0.5, 5.0, "+0.5"),
    ),
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    """A confounded continuous dose with known means under each declared shift.

    The default policies are the natural course and a ``+0.5`` shift capped at ``5.0``,
    which is what :class:`~cleverly.interventions.Shift` would be given for the same
    analysis -- and the names match, so the truth is keyed exactly as the estimator
    reports.
    """
    return shift_dgp().sample(n, shifts=shifts, seed=seed, backend=backend)


#: Every generator, for parametrised tests and the simulation harness.
#:
#: Multi-arm, continuous-dose and longitudinal processes are deliberately absent: every
#: consumer here -- the parametrised structural tests, the coverage study, the simulation
#: harness -- reads a truth keyed ``ate``/``ey1``/``ey0`` and a binary ``A`` column, and a
#: three-armed, dose-valued or node-indexed entry would report neither.  Reach for
#: :func:`make_multi_arm`, :func:`make_shift_dose` and
#: :func:`~cleverly.datasets.longitudinal.make_longitudinal` directly.  The longitudinal
#: process does follow the ``(n, seed) -> (frame, truth)`` convention and its truth is
#: keyed by the names a fit reports, so :class:`~cleverly.validation.CoverageStudy` takes
#: it; it is only this dict's ``ate``-keyed readers that it cannot serve.
GENERATORS: dict[str, Callable[..., tuple[Any, dict[str, float]]]] = {
    "linear_ate": make_linear_ate,
    "nonlinear_ate": make_nonlinear_ate,
    "weak_overlap": make_weak_overlap,
    "instrument": make_instrument,
    "missing_outcome": make_missing_outcome,
    "cde": make_cde,
    "heterogeneous": make_heterogeneous,
    "clustered": make_clustered,
    "binary_outcome": make_binary_outcome,
    "missing_outcome_binary": make_missing_outcome_binary,
}


def available() -> Sequence[str]:
    """Names of the bundled data-generating processes."""
    return tuple(GENERATORS)
