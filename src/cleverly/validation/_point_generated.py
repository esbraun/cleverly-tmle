"""Point generated-outcome targets and the point plasmode law (X82 plan 5.5.1, 5.6; part E1).

A point target names an arm-indexed functional: a difference contrast, a level, a risk ratio or
an odds ratio, each over a population (every row, the value arm, or the reference arm). Its
truth on a generated law is a finite sum of the process mean ``m(a, W)`` over rows, so the
held design and the plasmode both have an exact reference.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from .outcome_processes import arm_term

#: The clip of a fitted plasmode treatment model on a fit that records no ``g`` bounds.
PLASMODE_G_BOUNDS = (0.01, 0.99)

#: The relative range below which an arm contrast counts as constant over the rows.
CONSTANT_TOLERANCE = 1e-12

TargetKind = Literal["difference", "level", "rr", "or"]
Population = Literal["all", "value", "reference"]


@dataclass(frozen=True)
class PointTarget:
    """One arm-indexed point target and how its truth is summed.

    Parameters
    ----------
    kind : {"difference", "level", "rr", "or"}
        The functional of the arm levels.
    population : {"all", "value", "reference"}
        Rows the levels average over: every row (``ate``, levels, ratios), the rows of the
        value arm (``att``), or the rows of the reference arm (``atc``).
    value_code : float
        Internal code of the value arm, or of the one arm of a level.
    reference_code : float or None
        Internal code of the reference arm. ``None`` for a level.
    outcome_bounds : tuple of float or None
        The fit's ``q_bounds``, which a bounded process draws on.
    """

    kind: TargetKind
    population: Population
    value_code: float
    reference_code: float | None
    outcome_bounds: tuple[float, float] | None = None

    @property
    def ratio(self) -> bool:
        """Return whether the readings use the log scale."""
        return self.kind in ("rr", "or")

    @property
    def codes(self) -> tuple[float, ...]:
        """Return the arm codes the truth reads."""
        if self.reference_code is None:
            return (self.value_code,)
        return (self.value_code, self.reference_code)

    def population_code(self) -> float | None:
        """Return the arm that defines the population, or ``None`` for every row."""
        if self.population == "value":
            return self.value_code
        if self.population == "reference":
            return self.reference_code
        return None

    def truth(self, means: dict[float, np.ndarray], weights: np.ndarray) -> float:
        """Return the target's value given the arm means and the population weights.

        Parameters
        ----------
        means : dict of float to ndarray
            ``m(a, W_i)`` for each arm code the target reads, one value per row.
        weights : ndarray
            Nonnegative population weight of each row.

        Returns
        -------
        float
            The exact value of the target on the generated law.

        Raises
        ------
        ValueError
            If the population has no weight.
        """
        total = float(np.sum(weights))
        if not total > 0.0:
            raise ValueError("the target population holds no row")

        def level(code: float) -> float:
            return float(np.sum(weights * means[code]) / total)

        value = level(self.value_code)
        if self.reference_code is None:
            return value
        reference = level(self.reference_code)
        if self.kind == "difference":
            return value - reference
        if self.kind == "rr":
            return value / reference
        return (value / (1.0 - value)) / (reference / (1.0 - reference))

    def levels(self, means: dict[float, np.ndarray], weights: np.ndarray) -> tuple[float, ...]:
        """Return the population level of each arm the target reads.

        Parameters
        ----------
        means : dict of float to ndarray
            ``m(a, W_i)`` for each arm code the target reads.
        weights : ndarray
            Nonnegative population weight of each row.

        Returns
        -------
        tuple of float
            One level per code of :attr:`codes`.
        """
        total = float(np.sum(weights))
        return tuple(float(np.sum(weights * means[code]) / total) for code in self.codes)

    def covariate_part_vanishes(self, means: dict[float, np.ndarray], weights: np.ndarray) -> bool:
        """Return whether the covariate part of the influence curve is identically zero.

        The covariate part of a difference contrast is ``m(v, W) - m(r, W) - psi``, of a
        level ``m(a, W) - psi``, and of a log ratio a combination of the two levels' parts.
        Each vanishes when the functions it reads are constant over the population rows (X82
        plan 5.5.1). The check reads the generated law, not the target name.

        Parameters
        ----------
        means : dict of float to ndarray
            ``m(a, W_i)`` for each arm code the target reads.
        weights : ndarray
            Nonnegative population weight of each row.

        Returns
        -------
        bool
            ``True`` when the held-design standard-error ratio and coverage have an exact
            reference.
        """
        rows = weights > 0.0
        if not np.any(rows):
            return False
        if self.kind == "difference" and self.reference_code is not None:
            vectors = [means[self.value_code][rows] - means[self.reference_code][rows]]
        else:
            vectors = [means[code][rows] for code in self.codes]
        return all(_constant(vector) for vector in vectors)


def _constant(vector: np.ndarray) -> bool:
    spread = float(np.max(vector) - np.min(vector))
    return spread <= CONSTANT_TOLERANCE * max(1.0, float(np.max(np.abs(vector))))


def arm_means(
    process: Any,
    covariates: np.ndarray,
    codes: tuple[float, ...],
    data: Any,
    *,
    centre: Any = None,
    scale: Any = None,
    bounds: tuple[float, float] | None = None,
) -> dict[float, np.ndarray]:
    """Evaluate ``m(a, W_i)`` for each arm code in ``codes`` at every row of ``covariates``.

    Parameters
    ----------
    process : object
        A point process of the catalog.
    covariates : ndarray
        Covariate matrix with one row per unit.
    codes : tuple of float
        Arm codes to evaluate.
    data : CausalData
        Declares the arm codes and labels the arm term reads.
    centre : array-like or None
        Column centres of the covariate term.
    scale : array-like or None
        Column scales of the covariate term.
    bounds : tuple of float or None
        The fit's ``q_bounds``, for a bounded process.

    Returns
    -------
    dict of float to ndarray
        One mean vector per code.
    """
    n = covariates.shape[0]
    out: dict[float, np.ndarray] = {}
    for code in codes:
        terms = arm_term(
            process,
            np.full(n, float(code)),
            arm_codes=data.arm_codes,
            levels=data.treatment_levels,
        )
        out[float(code)] = np.asarray(
            process.mean(covariates, terms, centre=centre, scale=scale, bounds=bounds),
            dtype=float,
        )
    return out


def held_weights(target: PointTarget, data: Any) -> np.ndarray:
    """Return the population weight of each held row: the fit weight within the population.

    Parameters
    ----------
    target : PointTarget
        The target whose population the weights select.
    data : CausalData
        The held rows.

    Returns
    -------
    ndarray
        One nonnegative weight per row.
    """
    weights = np.asarray(data.weights, dtype=float)
    code = target.population_code()
    if code is None:
        return weights
    return np.where(np.asarray(data.treatment, dtype=float) == code, weights, 0.0)


@dataclass(frozen=True)
class PlasmodeLaw:
    """The point plasmode law of X82 plan 5.6, frozen once on the fitted data.

    Each draw resamples ``n`` units from the empirical law of ``W``, draws ``A*`` from a
    frozen treatment model, and draws ``Y*`` from the declared process. The covariate term
    standardizes with the original data's centre and scale, so it is the same function on
    every draw. The units are i.i.d. draws from a discrete law, so the refit draws its own
    folds.

    The treatment model is the declared known mechanism when the fit has one. Otherwise it
    is an unpenalized logistic regression of ``A`` on ``W``, multinomial for more than two
    arms, fitted once and always clipped: to the fit's ``g`` bounds, or to
    :data:`PLASMODE_G_BOUNDS` when the fit records none. With two arms the clip bounds the
    value-arm probability. With more arms each probability is floored at the lower bound and
    the row renormalized, and :attr:`floor` records the smallest probability after that. An
    unclipped model can reach 0 or 1 under separation, and the law would then cause a
    positivity failure that the report blames on the estimator.

    Parameters
    ----------
    covariates : ndarray
        The support points ``W_1, ..., W_n``.
    probabilities : ndarray
        ``(n, K)`` frozen probability of each arm at each support point, in the order of
        ``arm_codes``.
    known : ndarray or None
        The declared known mechanism, which each resample carries.
    centre : ndarray
        Column means of the original covariates.
    scale : ndarray
        Column standard deviations of the original covariates.
    arm_codes : tuple of float
        Internal codes of the arms.
    bounds : tuple of float or None
        The clip of a fitted model; ``None`` for a known mechanism.
    floor : float
        The smallest frozen arm probability.
    """

    covariates: np.ndarray
    probabilities: np.ndarray
    known: np.ndarray | None
    centre: np.ndarray
    scale: np.ndarray
    arm_codes: tuple[float, ...]
    bounds: tuple[float, float] | None
    floor: float

    @property
    def treated(self) -> np.ndarray:
        """Return the frozen probability of the second arm, the treated arm of a binary fit."""
        return np.asarray(self.probabilities[:, 1], dtype=float)

    @classmethod
    def fit(cls, result: Any) -> PlasmodeLaw:
        """Freeze the law on a fitted result.

        Parameters
        ----------
        result : TMLEResult
            The fit whose data and ``g`` bounds the law reads.

        Returns
        -------
        PlasmodeLaw
            The frozen law.
        """
        data = result.data
        covariates = np.asarray(data.covariates, dtype=float)
        codes = tuple(float(code) for code in data.arm_codes)
        known = None
        bounds: tuple[float, float] | None = None
        if data.known_treatment is not None:
            known = np.asarray(data.known_treatment.values, dtype=float)
            # A known mechanism holds one column per arm in the order of ``arm_codes``.
            if known.shape[1] != len(codes):
                raise RuntimeError("the known mechanism does not hold one column per arm")
            probabilities = known
        else:
            from sklearn.linear_model import LogisticRegression

            recorded = getattr(getattr(result, "config", None), "g_bounds", None)
            bounds = PLASMODE_G_BOUNDS
            if isinstance(recorded, tuple) and len(recorded) == 2:
                bounds = (float(recorded[0]), float(recorded[1]))
            model = LogisticRegression(C=np.inf, max_iter=5000)
            if len(codes) == 2:
                model.fit(covariates, np.asarray(data.treatment == codes[1], dtype=int))
                treated = np.clip(model.predict_proba(covariates)[:, 1], *bounds)
                probabilities = np.column_stack([1.0 - treated, treated])
            else:
                labels = np.asarray(data.treatment, dtype=float)
                model.fit(covariates, labels)
                fitted = model.predict_proba(covariates)
                order = [list(model.classes_).index(code) for code in codes]
                floored = np.maximum(fitted[:, order], bounds[0])
                probabilities = floored / floored.sum(axis=1, keepdims=True)
        return cls(
            covariates=covariates,
            probabilities=np.asarray(probabilities, dtype=float),
            known=known,
            centre=np.mean(covariates, axis=0),
            scale=np.std(covariates, axis=0),
            arm_codes=codes,
            bounds=bounds,
            floor=float(np.min(probabilities)),
        )

    def adjustment(self, process: Any, rows: np.ndarray) -> np.ndarray:
        """Return the declared Gaussian adjustment function, standardized on the original data.

        Parameters
        ----------
        process : GaussianAdjustmentOutcome
            The declared process.
        rows : ndarray
            Support rows to evaluate.

        Returns
        -------
        ndarray
            ``f(W)`` at each row.
        """
        return np.asarray(
            process.adjustment(self.covariates[rows], centre=self.centre, scale=self.scale)
        )

    def means(
        self, process: Any, data: Any, target: PointTarget
    ) -> tuple[dict[float, np.ndarray], np.ndarray]:
        """Return the arm means over the support and the population weights of the truth.

        Parameters
        ----------
        process : object
            The declared point process.
        data : CausalData
            Declares the arm codes and labels.
        target : PointTarget
            The target whose truth is summed.

        Returns
        -------
        tuple
            ``m(a, W_i)`` per arm code over the ``n`` support points, then the population
            weights: 1 for every point, or the frozen probability of the population arm.
        """
        means = arm_means(
            process,
            self.covariates,
            target.codes,
            data,
            centre=self.centre,
            scale=self.scale,
            bounds=target.outcome_bounds,
        )
        code = target.population_code()
        if code is None:
            weights = np.ones(self.covariates.shape[0], dtype=float)
        else:
            weights = np.asarray(self.probabilities[:, self.arm_codes.index(code)], dtype=float)
        return means, weights

    def draw(self, data: Any, process: Any, *, seed: int, target: PointTarget) -> Any:
        """Return one plasmode data set.

        Parameters
        ----------
        data : CausalData
            The fitted data, whose rows the draw resamples.
        process : object
            The declared point process.
        seed : int
            Child seed of the draw.
        target : PointTarget
            Supplies the outcome bounds of a bounded process.

        Returns
        -------
        CausalData
            The resampled units with a drawn treatment and a generated outcome.
        """
        rng = np.random.default_rng(seed)
        n = self.covariates.shape[0]
        arms = len(self.arm_codes)
        # Cumulate from the last arm, so a binary draw is arm 1 exactly when u < g(1 | W).
        cumulative = np.cumsum(self.probabilities[:, ::-1], axis=1)
        for _ in range(1000):
            rows = rng.integers(0, n, size=n)
            uniform = np.asarray(rng.random(n), dtype=float)
            above = np.sum(uniform[:, None] >= cumulative[rows], axis=1)
            index = arms - 1 - np.minimum(above, arms - 1)
            if np.unique(index).size == arms:
                break
        else:  # pragma: no cover - a 1000-fold failure needs an arm share near 0
            raise RuntimeError("the plasmode draw left an arm empty 1000 times")
        treatment = np.asarray(self.arm_codes, dtype=float)[index]
        terms = arm_term(process, treatment, arm_codes=data.arm_codes, levels=data.treatment_levels)
        mean = process.mean(
            self.covariates[rows],
            terms,
            centre=self.centre,
            scale=self.scale,
            bounds=target.outcome_bounds,
        )
        outcome = process.sample(rng, mean, bounds=target.outcome_bounds)
        sample = data.subset(rows).with_treatment(treatment)
        if self.known is not None:
            from ..data.known_mechanism import KnownMechanism

            sample = replace(sample, known_treatment=KnownMechanism(self.known[rows], "array"))
        return sample.with_outcome(np.asarray(outcome, dtype=float), family=process.family)
