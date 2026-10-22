"""The generated-outcome processes that refutation draws from (X82 plan 5.5 and 5.6).

Each declaration names a law with a known truth. The Gaussian, Bernoulli and bounded processes
replace an outcome measured once. The hazard processes replace a survival or competing-risk
outcome node by node. Parts C and E of X82 share this catalog.

Every point process has one mean model ``m(a, W)`` and one arm term ``c(a)``. A scalar
``effect`` gives ``c(a) = effect * code(a)``. A declared ``arm_effects`` mapping gives each arm
its own term. The truth of every point target is then a finite sum of ``m`` over the rows
(X82 part E plan, section 3).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

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


def _arm_effects(value: Any, label: str) -> tuple[tuple[Any, float], ...] | None:
    """Normalize an ``arm_effects`` declaration to a hashable tuple of pairs."""
    if value is None:
        return None
    pairs = tuple(value.items()) if isinstance(value, dict) else tuple(value)
    normalized: list[tuple[Any, float]] = []
    for pair in pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"{label} arm_effects maps each arm label to one effect")
        arm, effect = pair
        effect = float(effect)
        if not np.isfinite(effect):
            raise ValueError(f"{label} arm_effects must be finite")
        normalized.append((arm, effect))
    arms = [arm for arm, _ in normalized]
    if not arms or len(set(map(repr, arms))) != len(arms):
        raise ValueError(f"{label} arm_effects names each arm once")
    return tuple(normalized)


def arm_term(
    process: Any, codes: Any, *, arm_codes: Sequence[float], levels: Sequence[Any]
) -> np.ndarray:
    """Evaluate the arm term ``c(a)`` of a point process at each treatment code.

    Parameters
    ----------
    process : object
        A point process with ``effect`` and ``arm_effects``.
    codes : array-like
        One internal treatment code per row. A non-finite code reads as the first arm.
    arm_codes : sequence of float
        The internal codes of the declared arms, in order.
    levels : sequence
        The labels of the declared arms, in the order of ``arm_codes``.

    Returns
    -------
    ndarray
        ``c(a)`` for each row.

    Raises
    ------
    ValueError
        If ``arm_effects`` misses a declared arm, or names an arm that is not declared.
    """
    values = np.asarray(codes, dtype=float)
    # A row with an unrecorded treatment has no code. Its generated outcome is never read,
    # because the composite indicator is zero there, so the first arm keeps it finite.
    first = float(arm_codes[0]) if len(arm_codes) else 0.0
    values = np.where(np.isfinite(values), values, first)
    declared = getattr(process, "arm_effects", None)
    if declared is None:
        return np.asarray(float(process.effect) * values, dtype=float)
    lookup: dict[float, float] = {}
    for arm, effect in declared:
        matches = [
            float(code)
            for code, level in zip(arm_codes, levels, strict=True)
            if level == arm or code == arm
        ]
        if not matches:
            raise ValueError(
                f"arm_effects names {arm!r}, which is not a declared arm of {list(levels)}"
            )
        lookup[matches[0]] = effect
    missing = [level for code, level in zip(arm_codes, levels, strict=True) if code not in lookup]
    if missing:
        raise ValueError(f"arm_effects gives no effect for the declared arm(s) {missing}")
    out = np.empty(values.shape, dtype=float)
    for code, effect in lookup.items():
        out[values == code] = effect
    unknown = ~np.isin(values, list(lookup))
    if np.any(unknown):
        raise ValueError("a treatment code is not a declared arm")
    return out


def _independent_arms(process: Any) -> bool:
    declared = getattr(process, "arm_effects", None)
    if declared is None:
        return float(process.effect) == 0.0
    return len({effect for _, effect in declared}) == 1


def treatment_term(
    process: Any,
    codes: Any,
    *,
    arm_codes: Sequence[float],
    levels: Sequence[Any],
    intermediate: Any = None,
) -> np.ndarray:
    """Evaluate the additive treatment term ``c(a) + gamma z`` of a point process.

    The term is the arm term of :func:`arm_term` plus ``intermediate_effect`` times the
    intermediate value. The process reads the term inside its link, so the intermediate
    enters without an interaction with the arm.

    Parameters
    ----------
    process : object
        A point process with ``effect``, ``arm_effects`` and ``intermediate_effect``.
    codes : array-like
        One internal treatment code per row.
    arm_codes : sequence of float
        The internal codes of the declared arms, in order.
    levels : sequence
        The labels of the declared arms, in the order of ``arm_codes``.
    intermediate : array-like, float or None
        The intermediate value of each row, one value for every row, or ``None`` when the
        fit has no intermediate.

    Returns
    -------
    ndarray
        The treatment term of each row.
    """
    terms = arm_term(process, codes, arm_codes=arm_codes, levels=levels)
    gamma = float(getattr(process, "intermediate_effect", 0.0))
    if intermediate is None or gamma == 0.0:
        return terms
    return np.asarray(terms + gamma * np.asarray(intermediate, dtype=float), dtype=float)


def _validate_extras(process: Any, label: str) -> None:
    _finite(getattr(process, "intermediate_effect", 0.0), f"{label} intermediate_effect")
    cluster_sd = getattr(process, "cluster_sd", 0.0)
    if not np.isfinite(cluster_sd) or cluster_sd < 0.0:
        raise ValueError(f"{label} cluster_sd must be finite and nonnegative")


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
    cluster_sd : float
        Standard deviation of a cluster random intercept that the clustered plasmode adds
        to each drawn cluster. The default 0 adds none. The intercept has mean 0, so no
        truth changes.
    """

    noise: GaussianNoise = GaussianNoise()
    cluster_sd: float = 0.0
    name: str = field(default="gaussian_independent_noise", init=False)
    adjustment_function: str = field(default="none", init=False)
    treatment_term: str = field(default="none", init=False)
    family: str = field(default="gaussian", init=False)
    known_effect: float = field(default=0.0, init=False)
    effect: float = field(default=0.0, init=False)
    arm_effects: None = field(default=None, init=False)
    intermediate_effect: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        _validate_extras(self, "GaussianIndependentOutcome")

    @property
    def noise_law(self) -> GaussianNoise:
        """Return the recorded Gaussian noise declaration."""
        return self.noise

    @property
    def independent(self) -> bool:
        """Return whether the outcome reads neither the covariates nor the treatment."""
        return True

    def mean(self, covariates: Any, arms: Any, **_: Any) -> np.ndarray:
        """Return the outcome mean of each row: the noise mean.

        Parameters
        ----------
        covariates : array-like
            Covariate matrix with one row per observation.
        arms : array-like
            The arm term ``c(a)`` of each row; unused.
        **_ : Any
            Standardization and bound keywords of the other processes; unused.

        Returns
        -------
        ndarray
            The noise mean for each row.
        """
        del arms
        return np.full(np.asarray(covariates).shape[0], self.noise.mean, dtype=float)

    def sample(self, rng: np.random.Generator, mean: Any, **_: Any) -> np.ndarray:
        """Draw one outcome per row around ``mean``.

        Parameters
        ----------
        rng : numpy.random.Generator
            Generator that supplies the draw.
        mean : array-like
            The outcome mean of each row.
        **_ : Any
            Bound keywords of the bounded process; unused.

        Returns
        -------
        ndarray
            Independent Gaussian values.
        """
        centre = np.asarray(mean, dtype=float)
        return np.asarray(centre - self.noise.mean + self.noise.draw(rng, centre.size))

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
    r"""Declare ``f(W) + c(a) + epsilon`` with Gaussian noise.

    The adjustment function standardizes each nonconstant covariate, sums the standardized
    columns, divides by the square root of how many columns it summed, and multiplies the
    result by ``adjustment_scale``. The root-count divisor holds the adjustment standard
    deviation at about ``adjustment_scale`` for any number of approximately independent
    covariates, so the confounding signal does not shrink against the fixed noise scale as
    the adjustment set widens. Correlated columns raise that standard deviation instead,
    which only makes a failure to adjust easier to detect. The arm term is
    ``c(a) = effect * code(a)``, or ``arm_effects[a]`` when that mapping is declared. The
    arm contrast is ``c(v) - c(r)`` because the arm term is additive.

    Parameters
    ----------
    effect : float
        Declared effect of one unit of the treatment code.
    adjustment_scale : float
        Multiplier on the root-count normalized standardized covariate sum.
    noise : GaussianNoise
        Gaussian law for ``epsilon``.
    arm_effects : dict or tuple of tuple or None
        One effect per declared arm label, in place of ``effect * code``. Point fits only.
    intermediate_effect : float
        Effect ``gamma`` of the binary intermediate ``Z`` of a controlled direct effect.
        The mean adds ``gamma z`` with no arm interaction. Point fits only.
    cluster_sd : float
        Standard deviation of a cluster random intercept that the clustered plasmode adds
        to each drawn cluster. The default 0 adds none. The intercept has mean 0, so no
        truth changes.
    """

    effect: float = 1.0
    adjustment_scale: float = 1.0
    noise: GaussianNoise = GaussianNoise()
    arm_effects: Any = None
    intermediate_effect: float = 0.0
    cluster_sd: float = 0.0
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
        _validate_extras(self, "GaussianAdjustmentOutcome")
        effects = _arm_effects(self.arm_effects, "GaussianAdjustmentOutcome")
        object.__setattr__(self, "arm_effects", effects)
        if effects is not None:
            object.__setattr__(self, "treatment_term", "declared_effect_per_arm")

    @property
    def known_effect(self) -> float:
        """Return the effect of one unit of the treatment code."""
        return float(self.effect)

    @property
    def noise_law(self) -> GaussianNoise:
        """Return the recorded Gaussian noise declaration."""
        return self.noise

    @property
    def independent(self) -> bool:
        """Return whether the outcome reads neither the covariates nor the treatment."""
        return (
            self.adjustment_scale == 0.0
            and _independent_arms(self)
            and self.intermediate_effect == 0.0
        )

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

    def mean(
        self, covariates: Any, arms: Any, *, centre: Any = None, scale: Any = None, **_: Any
    ) -> np.ndarray:
        """Return ``f(W) + c(a) + mu`` for each row.

        Parameters
        ----------
        covariates : array-like
            Covariate matrix with one row per observation.
        arms : array-like
            The arm term ``c(a)`` of each row.
        centre : array-like or None
            Column centres of the adjustment function.
        scale : array-like or None
            Column scales of the adjustment function.
        **_ : Any
            Bound keywords of the bounded process; unused.

        Returns
        -------
        ndarray
            The outcome mean of each row.
        """
        shift = self.adjustment(covariates, centre=centre, scale=scale)
        return np.asarray(shift + np.asarray(arms, dtype=float) + self.noise.mean, dtype=float)

    def sample(self, rng: np.random.Generator, mean: Any, **_: Any) -> np.ndarray:
        """Draw one outcome per row around ``mean``.

        Parameters
        ----------
        rng : numpy.random.Generator
            Generator that supplies the draw.
        mean : array-like
            The outcome mean of each row.
        **_ : Any
            Bound keywords of the bounded process; unused.

        Returns
        -------
        ndarray
            One Gaussian value per row.
        """
        centre = np.asarray(mean, dtype=float)
        return np.asarray(centre - self.noise.mean + self.noise.draw(rng, centre.size))

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
        arms = arm_term(
            self, data.treatment, arm_codes=data.arm_codes, levels=data.treatment_levels
        )
        mean = self.mean(data.covariates, arms)
        return self.sample(np.random.default_rng(seed), mean)


#: The link functions of the Bernoulli and bounded processes.
Link = Literal["logit", "identity"]


def _unit_mean(process: Any, covariates: Any, arms: Any, centre: Any, scale: Any) -> np.ndarray:
    """``m(a, W)`` on the unit interval for the Bernoulli and bounded processes."""
    terms = np.asarray(arms, dtype=float)
    if process.link == "logit":
        shift = standardized_sum(covariates, process.adjustment_scale, centre=centre, scale=scale)
        return _expit(process.intercept + shift + terms)
    standard = standardized_sum(covariates, 1.0, centre=centre, scale=scale)
    return np.asarray(
        process.intercept + process.adjustment_scale * (_expit(standard) - 0.5) + terms,
        dtype=float,
    )


def _validate_link(process: Any, label: str) -> None:
    _finite(process.intercept, f"{label} intercept")
    _finite(process.effect, f"{label} effect")
    _finite(process.adjustment_scale, f"{label} adjustment_scale")
    if process.link not in ("logit", "identity"):
        raise ValueError(f"{label} link must be 'logit' or 'identity'")
    _validate_extras(process, label)
    object.__setattr__(process, "arm_effects", _arm_effects(process.arm_effects, label))


@dataclass(frozen=True)
class BernoulliAdjustmentOutcome:
    r"""Declare a binary outcome ``Y ~ Bern(m(a, W))``.

    Under ``link="logit"`` the mean is ``expit(intercept + f(W) + c(a))``, where ``f`` is the
    standardized sum of :class:`GaussianAdjustmentOutcome`. Under ``link="identity"`` it is
    ``intercept + adjustment_scale * (expit(s(W)) - 1/2) + c(a)``, where ``s`` is that sum
    with scale 1. The identity link gives an arm contrast that does not vary with ``W``, so
    the held design reads the standard-error ratio and the coverage on a binary outcome. A
    refutation refuses an identity-link declaration whose mean leaves ``[0, 1]`` on a row.

    Parameters
    ----------
    intercept : float
        Log-odds (logit link) or risk (identity link) at ``f(W) = 0`` and ``c(a) = 0``.
    effect : float
        Arm term per unit of the treatment code.
    adjustment_scale : float
        Multiplier on the covariate term.
    link : {"logit", "identity"}
        The link of the mean model.
    arm_effects : dict or tuple of tuple or None
        One arm term per declared arm label, in place of ``effect * code``.
    intermediate_effect : float
        Effect ``gamma`` of the binary intermediate ``Z`` of a controlled direct effect,
        added inside the link with no arm interaction. Point fits only.
    """

    intercept: float = -0.5
    effect: float = 1.0
    adjustment_scale: float = 1.0
    link: Link = "logit"
    arm_effects: Any = None
    intermediate_effect: float = 0.0
    name: str = field(default="bernoulli_adjustment_dependent", init=False)
    family: str = field(default="binomial", init=False)

    def __post_init__(self) -> None:
        _validate_link(self, "BernoulliAdjustmentOutcome")

    @property
    def independent(self) -> bool:
        """Return whether the outcome reads neither the covariates nor the treatment."""
        return (
            self.adjustment_scale == 0.0
            and _independent_arms(self)
            and self.intermediate_effect == 0.0
        )

    def mean(
        self, covariates: Any, arms: Any, *, centre: Any = None, scale: Any = None, **_: Any
    ) -> np.ndarray:
        """Return the risk ``m(a, W)`` of each row.

        Parameters
        ----------
        covariates : array-like
            Covariate matrix with one row per observation.
        arms : array-like
            The arm term ``c(a)`` of each row.
        centre : array-like or None
            Column centres of the covariate term.
        scale : array-like or None
            Column scales of the covariate term.
        **_ : Any
            Bound keywords of the bounded process; unused.

        Returns
        -------
        ndarray
            The risk of each row. The identity link can leave ``[0, 1]``; the refutation
            refuses such a declaration before it draws.
        """
        return _unit_mean(self, covariates, arms, centre, scale)

    def sample(self, rng: np.random.Generator, mean: Any, **_: Any) -> np.ndarray:
        """Draw one binary outcome per row with risk ``mean``.

        Parameters
        ----------
        rng : numpy.random.Generator
            Generator that supplies the draw.
        mean : array-like
            The risk of each row, in ``[0, 1]``.
        **_ : Any
            Bound keywords of the bounded process; unused.

        Returns
        -------
        ndarray
            Zero-one outcomes.
        """
        risk = np.asarray(mean, dtype=float)
        return np.asarray(rng.random(risk.size) < risk, dtype=float)

    def admissible(self, mean: Any, **_: Any) -> bool:
        """Return whether every risk lies in ``[0, 1]``.

        Parameters
        ----------
        mean : array-like
            Risks to check.
        **_ : Any
            Bound keywords of the bounded process; unused.

        Returns
        -------
        bool
            ``True`` when every risk is a probability.
        """
        risk = np.asarray(mean, dtype=float)
        return bool(np.all(np.isfinite(risk)) and np.all((risk >= 0.0) & (risk <= 1.0)))


@dataclass(frozen=True)
class BoundedAdjustmentOutcome:
    r"""Declare a bounded continuous outcome on the fit's ``q_bounds``.

    The draw is ``lo + (hi - lo) * B`` with ``B ~ Beta(m phi, (1 - m) phi)``, where
    ``(lo, hi)`` are the fit's ``q_bounds``, ``phi`` is ``precision``, and ``m = m(a, W)`` is
    the mean model of :class:`BernoulliAdjustmentOutcome`. The Beta law has mean ``m``
    exactly, so the outcome mean is ``lo + (hi - lo) m(a, W)`` and every truth is a finite
    sum. A refutation refuses a declaration whose ``m`` leaves the open interval ``(0, 1)``
    on a row.

    Parameters
    ----------
    intercept : float
        Log-odds (logit link) or mean (identity link) of ``m`` at ``f(W) = 0`` and
        ``c(a) = 0``.
    effect : float
        Arm term per unit of the treatment code, on the scale of ``m``'s link.
    adjustment_scale : float
        Multiplier on the covariate term.
    link : {"logit", "identity"}
        The link of the mean model.
    precision : float
        The Beta precision ``phi``. A larger value gives less noise around the mean.
    arm_effects : dict or tuple of tuple or None
        One arm term per declared arm label, in place of ``effect * code``.
    intermediate_effect : float
        Effect ``gamma`` of the binary intermediate ``Z`` of a controlled direct effect,
        added inside the link with no arm interaction. Point fits only.
    """

    intercept: float = 0.0
    effect: float = 0.5
    adjustment_scale: float = 1.0
    link: Link = "logit"
    precision: float = 20.0
    arm_effects: Any = None
    intermediate_effect: float = 0.0
    name: str = field(default="bounded_beta_adjustment_dependent", init=False)
    family: str = field(default="gaussian", init=False)

    def __post_init__(self) -> None:
        _validate_link(self, "BoundedAdjustmentOutcome")
        if not np.isfinite(self.precision) or self.precision <= 0.0:
            raise ValueError("BoundedAdjustmentOutcome precision must be finite and positive")

    @property
    def independent(self) -> bool:
        """Return whether the outcome reads neither the covariates nor the treatment."""
        return (
            self.adjustment_scale == 0.0
            and _independent_arms(self)
            and self.intermediate_effect == 0.0
        )

    def mean(
        self,
        covariates: Any,
        arms: Any,
        *,
        centre: Any = None,
        scale: Any = None,
        bounds: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Return the outcome mean ``lo + (hi - lo) m(a, W)`` of each row.

        Parameters
        ----------
        covariates : array-like
            Covariate matrix with one row per observation.
        arms : array-like
            The arm term ``c(a)`` of each row.
        centre : array-like or None
            Column centres of the covariate term.
        scale : array-like or None
            Column scales of the covariate term.
        bounds : tuple of float
            The fit's ``q_bounds``.

        Returns
        -------
        ndarray
            The outcome mean of each row.

        Raises
        ------
        ValueError
            If ``bounds`` is missing.
        """
        lower, upper = _bounds(bounds)
        unit = _unit_mean(self, covariates, arms, centre, scale)
        return np.asarray(lower + (upper - lower) * unit, dtype=float)

    def sample(
        self,
        rng: np.random.Generator,
        mean: Any,
        *,
        bounds: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Draw one bounded outcome per row with mean ``mean``.

        Parameters
        ----------
        rng : numpy.random.Generator
            Generator that supplies the draw.
        mean : array-like
            The outcome mean of each row, inside ``bounds``.
        bounds : tuple of float
            The fit's ``q_bounds``.

        Returns
        -------
        ndarray
            One value per row in ``bounds``.
        """
        lower, upper = _bounds(bounds)
        unit = (np.asarray(mean, dtype=float) - lower) / (upper - lower)
        draws = rng.beta(unit * self.precision, (1.0 - unit) * self.precision)
        return np.asarray(lower + (upper - lower) * draws, dtype=float)

    def admissible(self, mean: Any, *, bounds: tuple[float, float] | None = None) -> bool:
        """Return whether every mean lies strictly inside ``bounds``.

        Parameters
        ----------
        mean : array-like
            Outcome means to check.
        bounds : tuple of float
            The fit's ``q_bounds``.

        Returns
        -------
        bool
            ``True`` when each Beta law is proper.
        """
        lower, upper = _bounds(bounds)
        values = np.asarray(mean, dtype=float)
        return bool(np.all(np.isfinite(values)) and np.all((values > lower) & (values < upper)))


def _bounds(bounds: tuple[float, float] | None) -> tuple[float, float]:
    if bounds is None:
        raise ValueError("the bounded process draws on the fit's q_bounds; none were given")
    lower, upper = float(bounds[0]), float(bounds[1])
    if not (np.isfinite(lower) and np.isfinite(upper) and lower < upper):
        raise ValueError("q_bounds must be two finite values with lower < upper")
    return lower, upper


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
