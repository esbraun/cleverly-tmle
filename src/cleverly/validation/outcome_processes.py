"""The generated-outcome processes that refutation draws from (X82 plan 5.5 and 5.6).

Each declaration names a law with a known truth. The Gaussian processes replace an outcome
measured once. The hazard processes replace a survival or competing-risk outcome node by
node. Parts C and E of X82 share this catalog.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def standardized_sum(
    covariates: Any, adjustment_scale: float, *, centre: Any = None, scale: Any = None
) -> np.ndarray:
    """``adjustment_scale`` times the root-count normalized sum of standardized covariates.

    Parameters
    ----------
    covariates : array-like
        Covariate matrix with one row per observation.
    adjustment_scale : float
        Multiplier on the normalized sum.
    centre : array-like or None
        Column centres to standardize with. ``None`` uses the rows' own means.
    scale : array-like or None
        Column scales to standardize with. ``None`` uses the rows' own standard deviations.

    Returns
    -------
    ndarray
        One value per row, 0 where no column varies.
    """
    values = np.asarray(covariates, dtype=float)
    if values.ndim != 2 or not values.shape[1]:
        return np.zeros(values.shape[0], dtype=float)
    center = np.mean(values, axis=0) if centre is None else np.asarray(centre, dtype=float)
    spread = np.std(values, axis=0) if scale is None else np.asarray(scale, dtype=float)
    active = spread > 0.0
    if not np.any(active):
        return np.zeros(values.shape[0], dtype=float)
    standardized = (values[:, active] - center[active]) / spread[active]
    # Sum over root count rather than mean: a mean of p standardized columns has standard
    # deviation about 1/sqrt(p), so the confounding signal would vanish against the fixed
    # noise scale on a wide adjustment set.
    divisor = np.sqrt(float(np.count_nonzero(active)))
    return np.asarray(adjustment_scale * np.sum(standardized, axis=1) / divisor, dtype=float)


@dataclass(frozen=True)
class GaussianNoise:
    """Declare one Gaussian noise law for a generated outcome.

    Parameters
    ----------
    mean : float
        Mean of each independent noise draw.
    standard_deviation : float
        Positive standard deviation of each independent noise draw.
    """

    mean: float = 0.0
    standard_deviation: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.mean):
            raise ValueError("Gaussian noise mean must be finite")
        if not np.isfinite(self.standard_deviation) or self.standard_deviation <= 0.0:
            raise ValueError("Gaussian noise standard_deviation must be finite and positive")

    def draw(self, rng: np.random.Generator, size: int) -> np.ndarray:
        """Draw independent values from the declared law.

        Parameters
        ----------
        rng : numpy.random.Generator
            Generator that supplies the draw.
        size : int
            Number of values to draw.

        Returns
        -------
        ndarray
            Independent Gaussian values.
        """
        return np.asarray(rng.normal(self.mean, self.standard_deviation, size=size), dtype=float)


@dataclass(frozen=True)
class GaussianIndependentOutcome:
    """Declare an outcome independent of treatment and adjustment variables.

    Parameters
    ----------
    noise : GaussianNoise
        Gaussian law that generates every outcome value.
    """

    noise: GaussianNoise = GaussianNoise()
    name: str = field(default="gaussian_independent_noise", init=False)
    adjustment_function: str = field(default="none", init=False)
    treatment_term: str = field(default="none", init=False)
    family: str = field(default="gaussian", init=False)
    known_effect: float = field(default=0.0, init=False)

    @property
    def noise_law(self) -> GaussianNoise:
        """Return the recorded Gaussian noise declaration."""
        return self.noise

    def draw(self, data: Any, *, seed: int) -> np.ndarray:
        """Draw one independent outcome vector.

        Parameters
        ----------
        data : CausalData
            Analysis rows that determine the output length.
        seed : int
            Child seed for this draw.

        Returns
        -------
        ndarray
            One generated outcome per analysis row.
        """
        return self.noise.draw(np.random.default_rng(seed), data.n)


@dataclass(frozen=True)
class GaussianAdjustmentOutcome:
    r"""Declare ``f(W) + effect * A + epsilon`` with Gaussian noise.

    The adjustment function standardizes each nonconstant covariate, sums the standardized
    columns, divides by the square root of how many columns it summed, and multiplies the
    result by ``adjustment_scale``. The root-count divisor holds the adjustment standard
    deviation at about ``adjustment_scale`` for any number of approximately independent
    covariates, so the confounding signal does not shrink against the fixed noise scale as
    the adjustment set widens. Correlated columns raise that standard deviation instead,
    which only makes a failure to adjust easier to detect. The known binary arm contrast is
    ``effect`` because the treatment term is additive.

    Parameters
    ----------
    effect : float
        Declared effect of treatment code one against code zero.
    adjustment_scale : float
        Multiplier on the root-count normalized standardized covariate sum.
    noise : GaussianNoise
        Gaussian law for ``epsilon``.
    """

    effect: float = 1.0
    adjustment_scale: float = 1.0
    noise: GaussianNoise = GaussianNoise()
    name: str = field(default="gaussian_adjustment_dependent", init=False)
    adjustment_function: str = field(
        default="scaled_standardized_covariate_sum_over_root_count", init=False
    )
    treatment_term: str = field(default="effect_times_treatment_code", init=False)
    family: str = field(default="gaussian", init=False)

    def __post_init__(self) -> None:
        if not np.isfinite(self.effect):
            raise ValueError("Gaussian adjustment outcome effect must be finite")
        if not np.isfinite(self.adjustment_scale):
            raise ValueError("Gaussian adjustment outcome adjustment_scale must be finite")

    @property
    def known_effect(self) -> float:
        """Return the derived code-one against code-zero mean contrast."""
        return float(self.effect)

    @property
    def noise_law(self) -> GaussianNoise:
        """Return the recorded Gaussian noise declaration."""
        return self.noise

    def adjustment(self, covariates: Any, *, centre: Any = None, scale: Any = None) -> np.ndarray:
        """Evaluate the declared adjustment function.

        Parameters
        ----------
        covariates : array-like
            Covariate matrix with one row per observation.
        centre : array-like or None
            Column centres to standardize with. ``None`` uses the rows' own means.
        scale : array-like or None
            Column scales to standardize with. ``None`` uses the rows' own standard
            deviations. The plasmode passes the original data's centre and scale, so the
            function is the same on every resample.

        Returns
        -------
        ndarray
            Adjustment value for each row.
        """
        return standardized_sum(covariates, self.adjustment_scale, centre=centre, scale=scale)

    def draw(self, data: Any, *, seed: int) -> np.ndarray:
        """Draw one adjustment-dependent outcome vector.

        Parameters
        ----------
        data : CausalData
            Analysis rows that supply covariates and treatment.
        seed : int
            Child seed for the Gaussian noise.

        Returns
        -------
        ndarray
            One generated outcome per analysis row.
        """
        noise = self.noise.draw(np.random.default_rng(seed), data.n)
        # A row with an unrecorded treatment has no code to add an effect to.  Its
        # generated outcome is never read, because the composite indicator is zero there,
        # so the arm-0 value keeps the vector finite and leaves the known effect intact.
        treatment = np.where(np.isfinite(data.treatment), data.treatment, 0.0)
        return np.asarray(
            self.adjustment(data.covariates) + self.effect * treatment + noise,
            dtype=float,
        )


def _expit(values: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-values)), dtype=float)


def _finite(value: float, label: str) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite")


@dataclass(frozen=True)
class LogisticHazardOutcome:
    r"""Declare a discrete-time event hazard for a survival fit.

    At node ``t`` a unit still at risk has the event with probability
    ``expit(intercept + f(W) + effect * d_t)``. ``f`` is the standardized covariate sum of
    :class:`GaussianAdjustmentOutcome`, and ``d_t`` is the number of treated decision nodes up
    to ``t``. With ``effect=0`` and ``adjustment_scale=0`` the hazard reads nothing, and that
    declaration is the independent process of ``dummy_outcome``.

    Parameters
    ----------
    intercept : float
        Log-odds of the hazard at ``f(W) = 0`` and ``d_t = 0``.
    effect : float
        Log-odds change per treated decision node.
    adjustment_scale : float
        Multiplier on the standardized covariate sum.
    """

    intercept: float = -2.0
    effect: float = 0.5
    adjustment_scale: float = 1.0
    name: str = field(default="logistic_hazard", init=False)
    family: str = field(default="binomial", init=False)

    def __post_init__(self) -> None:
        _finite(self.intercept, "LogisticHazardOutcome intercept")
        _finite(self.effect, "LogisticHazardOutcome effect")
        _finite(self.adjustment_scale, "LogisticHazardOutcome adjustment_scale")

    @property
    def causes(self) -> int:
        """Return the number of absorbing causes: one."""
        return 1

    @property
    def independent(self) -> bool:
        """Return whether the hazard reads neither the covariates nor the treatment."""
        return self.effect == 0.0 and self.adjustment_scale == 0.0

    def probabilities(self, adjustment: Any, dose: Any) -> np.ndarray:
        """Return the event probability of each unit at one node.

        Parameters
        ----------
        adjustment : array-like
            ``f(W)`` for each unit.
        dose : array-like
            Treated decision nodes up to this node, for each unit.

        Returns
        -------
        ndarray
            ``(n, 1)`` probabilities of the one cause.
        """
        linear = self.intercept + np.asarray(adjustment, dtype=float)
        linear = linear + self.effect * np.asarray(dose, dtype=float)
        return _expit(linear)[:, None]


@dataclass(frozen=True)
class MultinomialHazardOutcome:
    r"""Declare cause-specific discrete-time hazards for a competing-risk fit.

    At node ``t`` a unit still at risk has the event of cause ``j`` with probability
    ``exp(eta_j) / (1 + sum_k exp(eta_k))``, where ``eta_j = intercepts[j] + f(W) +
    effects[j] * d_t``. The probabilities of no event and of each cause sum to 1 at every
    node. Independent per-cause logistic hazards could sum above 1, so this catalog has none.

    Parameters
    ----------
    intercepts : tuple of float
        One log-odds against no event per cause, in the fit's cause order.
    effects : tuple of float
        One log-odds change per treated decision node per cause.
    adjustment_scale : float
        Multiplier on the standardized covariate sum.
    """

    intercepts: tuple[float, ...] = (-2.5, -2.5)
    effects: tuple[float, ...] = (0.5, 0.0)
    adjustment_scale: float = 1.0
    name: str = field(default="multinomial_hazard", init=False)
    family: str = field(default="binomial", init=False)

    def __post_init__(self) -> None:
        intercepts = tuple(float(value) for value in self.intercepts)
        effects = tuple(float(value) for value in self.effects)
        object.__setattr__(self, "intercepts", intercepts)
        object.__setattr__(self, "effects", effects)
        if not intercepts or len(intercepts) != len(effects):
            raise ValueError(
                "MultinomialHazardOutcome needs one intercept and one effect per cause"
            )
        for value in (*intercepts, *effects):
            _finite(value, "MultinomialHazardOutcome intercepts and effects")
        _finite(self.adjustment_scale, "MultinomialHazardOutcome adjustment_scale")

    @property
    def causes(self) -> int:
        """Return the number of absorbing causes."""
        return len(self.intercepts)

    @property
    def independent(self) -> bool:
        """Return whether the hazards read neither the covariates nor the treatment."""
        return all(effect == 0.0 for effect in self.effects) and self.adjustment_scale == 0.0

    def probabilities(self, adjustment: Any, dose: Any) -> np.ndarray:
        """Return the probability of each cause for each unit at one node.

        Parameters
        ----------
        adjustment : array-like
            ``f(W)`` for each unit.
        dose : array-like
            Treated decision nodes up to this node, for each unit.

        Returns
        -------
        ndarray
            ``(n, J)`` cause probabilities. Each row sums to at most 1, and the rest is the
            probability of no event.
        """
        shift = np.asarray(adjustment, dtype=float)[:, None]
        exposure = np.asarray(dose, dtype=float)[:, None]
        linear = (
            np.asarray(self.intercepts)[None, :]
            + shift
            + np.asarray(self.effects)[None, :] * exposure
        )
        # Stabilize against overflow: subtract the row maximum, with no event at 0.
        top = np.maximum(np.max(linear, axis=1, keepdims=True), 0.0)
        weights = np.exp(linear - top)
        return np.asarray(weights / (np.exp(-top) + weights.sum(axis=1, keepdims=True)))
