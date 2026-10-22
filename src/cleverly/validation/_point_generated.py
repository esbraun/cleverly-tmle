"""Point generated-outcome targets and the point plasmode law (X82 plan 5.5.1, 5.6; part E).

A point target names an arm-indexed functional: a difference contrast, a level, a risk ratio,
an odds ratio, or a natural-course functional (``ey_obs``, ``par``, ``paf``). It can also
name an intervention functional: the mean under a regime or a tilt of the treatment
mechanism, a contrast of two of them, or one coefficient of a marginal structural model
(part E3). Each averages over a population (every row, the value arm, or the reference arm)
and, on a stratified fit, over one stratum. Its truth on a generated law is a finite sum of
the process mean ``m(a, W)`` over rows, so the held design and the plasmode both have an
exact reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np

from .outcome_processes import treatment_term

#: The clip of a fitted plasmode model on a fit that records no ``g`` bounds.
PLASMODE_G_BOUNDS = (0.01, 0.99)

#: The relative range below which an arm contrast counts as constant over the rows.
CONSTANT_TOLERANCE = 1e-12

TargetKind = Literal[
    "difference",
    "level",
    "rr",
    "or",
    "natural",
    "par",
    "paf",
    "intervention",
    "intervention_difference",
    "msm",
]
Population = Literal["all", "value", "reference"]

#: The natural-course kinds, which average the mean at each row's own arm.
NATURAL_KINDS = ("natural", "par", "paf")

#: The intervention kinds, which average the mean under a declared law of the arms.
INTERVENTION_KINDS = ("intervention", "intervention_difference")

#: The links of a working model whose projection :class:`Projection` solves.
PROJECTION_LINKS = ("identity", "log", "logit")


@dataclass(frozen=True, eq=False)
class InterventionLaw:
    """The law of the arms that one intervention assigns at each row (part E3).

    A regime assigns a fixed law: a static arm, a rule of ``W``, or a known stochastic
    density. An incremental intervention tilts the treatment mechanism: it multiplies the
    odds of each arm by a factor, so ``q(a | W) = f_a g(a | W) / sum_b f_b g(b | W)``
    (Kennedy 2019). Its law therefore needs the true ``g``, which only the plasmode knows.

    Parameters
    ----------
    label : str
        The name of the intervention in the reported parameter.
    values : ndarray or None
        ``(n, K)`` fixed probability of each arm at each row, in the order of the arm
        codes, or ``None`` for a tilt.
    factors : ndarray or None
        ``(K,)`` odds factor of each arm, or ``None`` for a fixed law.
    """

    label: str
    values: np.ndarray | None = None
    factors: np.ndarray | None = None

    @property
    def reads_mechanism(self) -> bool:
        """Return whether the law is a tilt of the treatment mechanism."""
        return self.factors is not None

    def density(self, mechanism: np.ndarray | None) -> np.ndarray:
        """Return the ``(n, K)`` probability of each arm at each row.

        Parameters
        ----------
        mechanism : ndarray or None
            ``(n, K)`` true treatment mechanism, which a tilt reads.

        Returns
        -------
        ndarray
            The law of the arms at each row.

        Raises
        ------
        ValueError
            If a tilt has no mechanism to read.
        """
        if self.factors is None:
            assert self.values is not None
            return self.values
        if mechanism is None:
            raise ValueError("a tilt of the treatment mechanism needs the true mechanism")
        tilted = np.asarray(mechanism, dtype=float) * self.factors[None, :]
        return np.asarray(tilted / tilted.sum(axis=1, keepdims=True), dtype=float)


@dataclass(frozen=True, eq=False)
class Projection:
    """One coefficient of a marginal structural model's projection (part E3).

    The coefficient vector ``beta`` solves
    ``sum_i w_i sum_a h(a, V_i) m'(eta) phi(a, V_i) (Q(a, W_i) - m(a, V_i; beta)) = 0``,
    the working-model normal equations of :mod:`cleverly.msm`, with ``Q`` the process mean.
    :meth:`coefficient` solves them by Gauss-Newton iterations written here, so the truth
    does not share the package's projection solver.

    Parameters
    ----------
    design : ndarray
        ``(n, K, p)`` working-model design ``phi(a, V)``.
    weights : ndarray
        ``(n, K)`` working-model weights ``h(a, V)``.
    link : {"identity", "log", "logit"}
        The working model's link.
    term : int
        The index of the reported coefficient.
    """

    design: np.ndarray
    weights: np.ndarray
    link: str
    term: int

    def _inverse(self, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean, slope, _ = self._derivatives(eta)
        return mean, slope

    def _derivatives(self, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the mean ``m``, ``dm/deta`` and ``d2m/deta2`` of the link at ``eta``."""
        if self.link == "identity":
            return eta, np.ones_like(eta), np.zeros_like(eta)
        if self.link == "log":
            mean = np.exp(eta)
            return mean, mean, mean
        if self.link == "logit":
            mean = 1.0 / (1.0 + np.exp(-eta))
            slope = mean * (1.0 - mean)
            return mean, slope, slope * (1.0 - 2.0 * mean)
        raise ValueError(f"no projection for the link {self.link!r}")

    def covariate_part(self, means: np.ndarray, population: np.ndarray) -> np.ndarray:
        """Return the covariate part of the reported coefficient's curve at each row.

        The part is ``e_j' M^{-1} sum_a h m'(eta) phi (Q - m)`` at each row, with
        ``M = P[sum_a h (m'^2 - (Q - m) m'') phi phi']`` the derivative of the normal
        equations (:func:`cleverly.msm.solve_projection` states the same matrix).

        Parameters
        ----------
        means : ndarray
            ``(n, K)`` process mean at each arm.
        population : ndarray
            ``(n,)`` population weight of each row.

        Returns
        -------
        ndarray
            ``(n,)`` covariate part of the coefficient :attr:`term` names.
        """
        phi = np.asarray(self.design, dtype=float)
        h = np.asarray(self.weights, dtype=float)
        w = np.asarray(population, dtype=float)
        beta = self.coefficients(means, w)
        fitted, slope, curvature = self._derivatives(phi @ beta)
        residual = means - fitted
        bend = h * (slope**2 - residual * curvature)
        matrix = np.einsum("i,ik,ikp,ikq->pq", w, bend, phi, phi) / float(np.sum(w))
        rows = np.einsum("ik,ikp->ip", h * slope * residual, phi)
        return np.asarray(np.linalg.solve(matrix, rows.T)[self.term], dtype=float)

    def coefficients(self, means: np.ndarray, population: np.ndarray) -> np.ndarray:
        """Return the projection of ``means`` onto the working model.

        Parameters
        ----------
        means : ndarray
            ``(n, K)`` process mean ``Q(a, W_i)`` at each arm.
        population : ndarray
            ``(n,)`` population weight of each row.

        Returns
        -------
        ndarray
            ``(p,)`` working-model coefficients.

        Raises
        ------
        ValueError
            If the iterations do not converge.
        """
        phi = np.asarray(self.design, dtype=float)
        mass = np.asarray(population, dtype=float)[:, None] * np.asarray(self.weights, dtype=float)
        beta = np.zeros(phi.shape[2])
        if self.link == "log":
            beta[0] = np.log(np.sum(mass * means) / np.sum(mass))
        elif self.link == "logit":
            level = np.sum(mass * means) / np.sum(mass)
            beta[0] = np.log(level / (1.0 - level))

        def loss(candidate: np.ndarray) -> float:
            fitted, _ = self._inverse(phi @ candidate)
            return float(np.sum(mass * (means - fitted) ** 2))

        for _ in range(500):
            fitted, slope = self._inverse(phi @ beta)
            gradient = phi * (slope * np.sqrt(mass))[:, :, None]
            residual = (means - fitted) * np.sqrt(mass)
            jacobian = gradient.reshape(-1, phi.shape[2])
            step = np.linalg.lstsq(jacobian, residual.reshape(-1), rcond=None)[0]
            size, current = 1.0, loss(beta)
            while loss(beta + size * step) > current and size > 1e-8:
                size /= 2.0
            beta = beta + size * step
            if float(np.max(np.abs(size * step))) <= 1e-12 * max(1.0, float(np.max(np.abs(beta)))):
                return beta
        raise ValueError("the working-model projection did not converge")

    def coefficient(self, means: np.ndarray, population: np.ndarray) -> float:
        """Return the reported coefficient of the projection.

        Parameters
        ----------
        means : ndarray
            ``(n, K)`` process mean at each arm.
        population : ndarray
            ``(n,)`` population weight of each row.

        Returns
        -------
        float
            The coefficient :attr:`term` names.
        """
        return float(self.coefficients(means, population)[self.term])


@dataclass(frozen=True)
class PointTarget:
    """One arm-indexed point target and how its truth is summed.

    Parameters
    ----------
    kind : str
        The functional of the arm levels: ``"difference"``, ``"level"``, ``"rr"``,
        ``"or"``, ``"natural"``, ``"par"``, ``"paf"``, ``"intervention"``,
        ``"intervention_difference"`` or ``"msm"``. ``natural`` is ``ey_obs``, the mean
        at each row's own arm. ``par`` is ``ey_obs - ey[r]`` and ``paf`` is
        ``par / ey_obs``. ``intervention`` is the mean under :attr:`value_law`, and
        ``intervention_difference`` subtracts the mean under :attr:`reference_law`.
        ``msm`` is one coefficient of :attr:`projection`.
    population : {"all", "value", "reference"}
        Rows the levels average over: every row (``ate``, levels, ratios, natural
        course), the rows of the value arm (``att``), or the rows of the reference arm
        (``atc``).
    value_code : float or None
        Internal code of the value arm, of the one arm of a level, or of the reference arm
        ``r`` of ``par`` and ``paf``. ``None`` for ``ey_obs``.
    reference_code : float or None
        Internal code of the reference arm of a contrast. ``None`` otherwise.
    outcome_bounds : tuple of float or None
        The fit's ``q_bounds``, which a bounded process draws on.
    stratum : int or None
        Code of the stratum the target averages over, or ``None`` for every row.
    intermediate : float or None
        The value ``z`` at which a controlled direct effect fixes the intermediate, or
        ``None`` on a fit with no intermediate.
    arms : tuple of float
        Every arm code, which the natural-course, intervention and MSM kinds read.
    value_law : InterventionLaw or None
        The intervention of an intervention kind.
    reference_law : InterventionLaw or None
        The reference intervention of ``intervention_difference``.
    projection : Projection or None
        The working model of ``msm``.
    """

    kind: TargetKind
    population: Population
    value_code: float | None
    reference_code: float | None
    outcome_bounds: tuple[float, float] | None = None
    stratum: int | None = None
    intermediate: float | None = None
    arms: tuple[float, ...] = ()
    value_law: InterventionLaw | None = field(default=None, compare=False)
    reference_law: InterventionLaw | None = field(default=None, compare=False)
    projection: Projection | None = field(default=None, compare=False)

    @property
    def ratio(self) -> bool:
        """Return whether the readings use the log scale."""
        return self.kind in ("rr", "or")

    @property
    def natural(self) -> bool:
        """Return whether the target reads the mean at each row's own arm."""
        return self.kind in NATURAL_KINDS

    @property
    def intervention(self) -> bool:
        """Return whether the target averages the mean under a declared law of the arms."""
        return self.kind in INTERVENTION_KINDS

    @property
    def reads_mechanism(self) -> bool:
        """Return whether the truth reads the true treatment mechanism."""
        laws = (self.value_law, self.reference_law)
        return self.natural or any(law is not None and law.reads_mechanism for law in laws)

    @property
    def codes(self) -> tuple[float, ...]:
        """Return the arm codes the truth reads."""
        if self.natural or self.intervention or self.kind == "msm":
            return self.arms
        assert self.value_code is not None
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

    def _natural(self, means: dict[float, np.ndarray], mixture: Any) -> np.ndarray:
        if mixture is None:
            raise ValueError("a natural-course target needs the arm probabilities of each row")
        return np.asarray(sum(mixture[code] * means[code] for code in self.arms), dtype=float)

    def _under(
        self, law: InterventionLaw | None, means: dict[float, np.ndarray], mixture: Any
    ) -> np.ndarray:
        """Return ``sum_a q(a | W_i) m(a, W_i)`` at each row for the law ``q``."""
        assert law is not None
        mechanism = None
        if mixture is not None:
            mechanism = np.column_stack([mixture[code] for code in self.arms])
        density = law.density(mechanism)
        return np.asarray(
            sum(density[:, index] * means[code] for index, code in enumerate(self.arms)),
            dtype=float,
        )

    def _intervention_vector(self, means: dict[float, np.ndarray], mixture: Any) -> np.ndarray:
        value = self._under(self.value_law, means, mixture)
        if self.kind == "intervention":
            return value
        return value - self._under(self.reference_law, means, mixture)

    def truth(
        self,
        means: dict[float, np.ndarray],
        weights: np.ndarray,
        mixture: dict[float, np.ndarray] | None = None,
    ) -> float:
        """Return the target's value given the arm means and the population weights.

        Parameters
        ----------
        means : dict of float to ndarray
            ``m(a, W_i)`` for each arm code the target reads, one value per row.
        weights : ndarray
            Nonnegative population weight of each row.
        mixture : dict of float to ndarray or None
            The probability of each arm at each row, which the natural-course kinds read:
            the indicator of the held arm, or the frozen treatment model of the plasmode.

        Returns
        -------
        float
            The exact value of the target on the generated law.

        Raises
        ------
        ValueError
            If the population has no weight, or a natural-course target has no mixture.
        """
        total = float(np.sum(weights))
        if not total > 0.0:
            raise ValueError("the target population holds no row")

        def average(values: np.ndarray) -> float:
            return float(np.sum(weights * values) / total)

        if self.intervention:
            return average(self._intervention_vector(means, mixture))
        if self.kind == "msm":
            assert self.projection is not None
            table = np.column_stack([means[code] for code in self.arms])
            return self.projection.coefficient(table, weights)
        if self.natural:
            observed = average(self._natural(means, mixture))
            if self.kind == "natural":
                return observed
            assert self.value_code is not None
            attributable = observed - average(means[self.value_code])
            return attributable if self.kind == "par" else attributable / observed
        assert self.value_code is not None
        value = average(means[self.value_code])
        if self.reference_code is None:
            return value
        reference = average(means[self.reference_code])
        if self.kind == "difference":
            return value - reference
        if self.kind == "rr":
            return value / reference
        return (value / (1.0 - value)) / (reference / (1.0 - reference))

    def levels(
        self,
        means: dict[float, np.ndarray],
        weights: np.ndarray,
        mixture: dict[float, np.ndarray] | None = None,
    ) -> tuple[float, ...]:
        """Return the population levels the target divides by or takes the odds of.

        Parameters
        ----------
        means : dict of float to ndarray
            ``m(a, W_i)`` for each arm code the target reads.
        weights : ndarray
            Nonnegative population weight of each row.
        mixture : dict of float to ndarray or None
            The probability of each arm at each row, for the natural-course kinds.

        Returns
        -------
        tuple of float
            One level per code of :attr:`codes`. A natural-course target returns its
            natural-course level alone.
        """
        total = float(np.sum(weights))
        if self.natural:
            return (float(np.sum(weights * self._natural(means, mixture)) / total),)
        return tuple(float(np.sum(weights * means[code]) / total) for code in self.codes)

    def covariate_part_vanishes(
        self,
        means: dict[float, np.ndarray],
        weights: np.ndarray,
        mixture: dict[float, np.ndarray] | None = None,
    ) -> bool:
        """Return whether the covariate part of the influence curve is identically zero.

        The covariate part of a difference contrast is ``m(v, W) - m(r, W) - psi``, of a
        level ``m(a, W) - psi``, and of a log ratio a combination of the two levels' parts.
        The natural-course mean reads ``m(A, W) - psi``, ``par`` reads
        ``m(A, W) - m(r, W) - psi``, and ``paf`` combines ``m(A, W)`` and ``m(r, W)``. An
        intervention mean reads ``sum_a q(a | W) m(a, W) - psi``, and a contrast of two
        interventions the difference of two such sums. An MSM coefficient reads
        :meth:`Projection.covariate_part`. Each vanishes when the functions it reads are
        constant over the population rows (X82 plan 5.5.1). The check reads the generated
        law, not the target name.

        Parameters
        ----------
        means : dict of float to ndarray
            ``m(a, W_i)`` for each arm code the target reads.
        weights : ndarray
            Nonnegative population weight of each row.
        mixture : dict of float to ndarray or None
            The probability of each arm at each row, for the natural-course kinds.

        Returns
        -------
        bool
            ``True`` when the held-design standard-error ratio and coverage have an exact
            reference.
        """
        rows = weights > 0.0
        if not np.any(rows):
            return False
        if self.kind == "msm":
            assert self.projection is not None
            table = np.column_stack([means[code] for code in self.arms])
            vectors = [self.projection.covariate_part(table, weights)[rows]]
        elif self.intervention:
            vectors = [self._intervention_vector(means, mixture)[rows]]
        elif self.natural:
            observed = self._natural(means, mixture)[rows]
            if self.kind == "natural":
                vectors = [observed]
            else:
                assert self.value_code is not None
                reference = means[self.value_code][rows]
                vectors = [observed - reference] if self.kind == "par" else [observed, reference]
        elif self.kind == "difference" and self.reference_code is not None:
            assert self.value_code is not None
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
    intermediate: float | None = None,
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
    intermediate : float or None
        The intermediate value every row is set to, for a controlled direct effect.

    Returns
    -------
    dict of float to ndarray
        One mean vector per code.
    """
    n = covariates.shape[0]
    out: dict[float, np.ndarray] = {}
    for code in codes:
        terms = treatment_term(
            process,
            np.full(n, float(code)),
            arm_codes=data.arm_codes,
            levels=data.treatment_levels,
            intermediate=intermediate,
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
        The target whose population and stratum the weights select.
    data : CausalData
        The held rows.

    Returns
    -------
    ndarray
        One nonnegative weight per row.
    """
    weights = np.asarray(data.weights, dtype=float)
    if target.stratum is not None:
        weights = np.where(np.asarray(data.strata) == target.stratum, weights, 0.0)
    code = target.population_code()
    if code is None:
        return weights
    return np.where(np.asarray(data.treatment, dtype=float) == code, weights, 0.0)


def held_mixture(target: PointTarget, data: Any) -> dict[float, np.ndarray] | None:
    """Return the indicator of each held row's arm, which a natural-course truth reads.

    Parameters
    ----------
    target : PointTarget
        The target. Only the natural-course kinds read a mixture.
    data : CausalData
        The held rows.

    Returns
    -------
    dict of float to ndarray or None
        ``1{A_i = a}`` for each arm code, or ``None`` for an arm-indexed target.
    """
    if not target.natural:
        return None
    treatment = np.asarray(data.treatment, dtype=float)
    return {code: np.asarray(treatment == code, dtype=float) for code in target.arms}


def _frozen_logistic(features: np.ndarray, labels: np.ndarray) -> Any:
    """Fit an unpenalized logistic model once, for a frozen plasmode observation model."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=np.inf, max_iter=5000)
    model.fit(features, np.asarray(labels, dtype=int))
    return model


def _arm_design(covariates: np.ndarray, codes: tuple[float, ...], arm: Any) -> np.ndarray:
    """Return ``(W, 1{A = a_2}, ..., 1{A = a_K})`` for an arm code or a vector of them."""
    arms = np.broadcast_to(np.asarray(arm, dtype=float), (covariates.shape[0],))
    indicators = [np.asarray(arms == code, dtype=float) for code in codes[1:]]
    return np.column_stack([covariates, *indicators])


def _per_arm(
    model: Any, covariates: np.ndarray, codes: tuple[float, ...], bounds: tuple[float, float]
) -> np.ndarray:
    """Return the clipped probability of the event at each support point under each arm."""
    return np.column_stack(
        [
            np.clip(model.predict_proba(_arm_design(covariates, codes, code))[:, 1], *bounds)
            for code in codes
        ]
    )


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
    arms, fitted once on the rows with a recorded treatment and always clipped: to the fit's
    ``g`` bounds, or to :data:`PLASMODE_G_BOUNDS` when the fit records none. With two arms
    the clip bounds the value-arm probability. With more arms each probability is floored at
    the lower bound and the row renormalized, and :attr:`floor` records the smallest
    probability after that. An unclipped model can reach 0 or 1 under separation, and the
    law would then cause a positivity failure that the report blames on the estimator.

    A fit with an observation process draws it from a frozen logistic model with the same
    clip (part E2). A missing outcome draws ``Delta*`` given ``(W, A*)``. A missing
    treatment draws the recording indicator given ``W``. An intermediate draws ``Z*`` given
    ``(W, A*)``. A clustered fit resamples whole clusters and gives each drawn copy its own
    cluster id. A Gaussian process with ``cluster_sd > 0`` adds one mean-zero intercept per
    drawn cluster, so no truth changes.

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
    strata : ndarray or None
        The stratum code of each support point.
    response : ndarray or None
        ``(n, K)`` frozen probability that the outcome is observed, per arm.
    recording : ndarray or None
        Frozen probability that the treatment is recorded at each support point.
    mediator : ndarray or None
        ``(n, K)`` frozen probability that the intermediate is 1, per arm.
    clusters : ndarray or None
        The cluster code of each support point.
    """

    covariates: np.ndarray
    probabilities: np.ndarray
    known: np.ndarray | None
    centre: np.ndarray
    scale: np.ndarray
    arm_codes: tuple[float, ...]
    bounds: tuple[float, float] | None
    floor: float
    strata: np.ndarray | None = None
    response: np.ndarray | None = None
    recording: np.ndarray | None = None
    mediator: np.ndarray | None = None
    clusters: np.ndarray | None = None

    @property
    def treated(self) -> np.ndarray:
        """Return the frozen probability of the second arm, the treated arm of a binary fit."""
        return np.asarray(self.probabilities[:, 1], dtype=float)

    @property
    def observation_models(self) -> tuple[str, ...]:
        """Return the names of the frozen observation models this law draws."""
        named = (
            ("outcome response", self.response),
            ("treatment recording", self.recording),
            ("intermediate", self.mediator),
        )
        return tuple(label for label, model in named if model is not None)

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
        recorded = getattr(getattr(result, "config", None), "g_bounds", None)
        clip = PLASMODE_G_BOUNDS
        if isinstance(recorded, tuple) and len(recorded) == 2:
            clip = (float(recorded[0]), float(recorded[1]))
        rows = np.asarray(data.treatment_recorded, dtype=bool)
        treatment = np.asarray(data.treatment, dtype=float)
        if data.known_treatment is not None:
            known = np.asarray(data.known_treatment.values, dtype=float)
            # A known mechanism holds one column per arm in the order of ``arm_codes``.
            if known.shape[1] != len(codes):
                raise RuntimeError("the known mechanism does not hold one column per arm")
            probabilities = known
        else:
            from sklearn.linear_model import LogisticRegression

            bounds = clip
            model = LogisticRegression(C=np.inf, max_iter=5000)
            if len(codes) == 2:
                model.fit(covariates[rows], np.asarray(treatment[rows] == codes[1], dtype=int))
                treated = np.clip(model.predict_proba(covariates)[:, 1], *bounds)
                probabilities = np.column_stack([1.0 - treated, treated])
            else:
                model.fit(covariates[rows], treatment[rows])
                fitted = model.predict_proba(covariates)
                order = [list(model.classes_).index(code) for code in codes]
                floored = np.maximum(fitted[:, order], bounds[0])
                probabilities = floored / floored.sum(axis=1, keepdims=True)
        response = recording = mediator = None
        if data.has_missing_outcome:
            design = _arm_design(covariates, codes, treatment)
            model = _frozen_logistic(design, np.asarray(data.observed))
            response = _per_arm(model, covariates, codes, clip)
        if data.has_missing_treatment:
            model = _frozen_logistic(covariates, rows)
            recording = np.clip(model.predict_proba(covariates)[:, 1], *clip)
        if data.has_intermediate:
            design = _arm_design(covariates, codes, treatment)
            model = _frozen_logistic(design, np.asarray(data.intermediate))
            mediator = _per_arm(model, covariates, codes, clip)
        return cls(
            covariates=covariates,
            probabilities=np.asarray(probabilities, dtype=float),
            known=known,
            centre=np.mean(covariates, axis=0),
            scale=np.std(covariates, axis=0),
            arm_codes=codes,
            bounds=bounds,
            floor=float(np.min(probabilities)),
            strata=None if data.strata is None else np.asarray(data.strata),
            response=response,
            recording=recording,
            mediator=mediator,
            clusters=None if data.cluster is None else np.asarray(data.cluster),
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
    ) -> tuple[dict[float, np.ndarray], np.ndarray, dict[float, np.ndarray] | None]:
        """Return the arm means over the support, the population weights, and the mixture.

        A clustered draw resamples clusters with equal probability, so each support point
        has expected count 1 in a draw, and the truth weights every point equally.

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
            ``m(a, W_i)`` per arm code over the ``n`` support points; the population
            weights (1 for every point, or the frozen probability of the population arm,
            times the stratum indicator); and the frozen arm probabilities that a
            natural-course target or a tilt reads, or ``None``.
        """
        means = arm_means(
            process,
            self.covariates,
            target.codes,
            data,
            centre=self.centre,
            scale=self.scale,
            bounds=target.outcome_bounds,
            intermediate=target.intermediate,
        )
        code = target.population_code()
        if code is None:
            weights = np.ones(self.covariates.shape[0], dtype=float)
        else:
            weights = np.asarray(self.probabilities[:, self.arm_codes.index(code)], dtype=float)
        if target.stratum is not None:
            assert self.strata is not None
            weights = np.where(self.strata == target.stratum, weights, 0.0)
        mixture = None
        if target.reads_mechanism:
            mixture = {
                code: np.asarray(self.probabilities[:, index], dtype=float)
                for index, code in enumerate(self.arm_codes)
            }
        return means, weights, mixture

    def _rows(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray | None]:
        """Draw the support rows of one data set, and the new cluster ids of a clustered law."""
        n = self.covariates.shape[0]
        if self.clusters is None:
            return np.asarray(rng.integers(0, n, size=n)), None
        labels = np.unique(self.clusters)
        members = [np.flatnonzero(self.clusters == label) for label in labels]
        chosen = rng.integers(0, labels.size, size=labels.size)
        rows = np.concatenate([members[index] for index in chosen])
        ids = np.concatenate(
            [np.full(members[index].size, copy) for copy, index in enumerate(chosen)]
        )
        return rows, np.asarray(ids, dtype=np.int64)

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
        arms = len(self.arm_codes)
        # Cumulate from the last arm, so a binary draw is arm 1 exactly when u < g(1 | W).
        cumulative = np.cumsum(self.probabilities[:, ::-1], axis=1)
        for _ in range(1000):
            rows, ids = self._rows(rng)
            uniform = np.asarray(rng.random(rows.size), dtype=float)
            above = np.sum(uniform[:, None] >= cumulative[rows], axis=1)
            index = arms - 1 - np.minimum(above, arms - 1)
            recorded = np.ones(rows.size, dtype=bool)
            if self.recording is not None:
                recorded = rng.random(rows.size) < self.recording[rows]
            # Every arm must appear among the recorded rows, or the refit has fewer arms. A
            # retry conditions the draw on that event, which tilts the law away from the
            # frozen truth. At the floors of the clip the event has probability near 1 for
            # n >= 100 (test_the_arm_retry_is_rare), so the tilt is negligible there.
            if np.unique(index[recorded]).size == arms:
                break
        else:  # pragma: no cover - a 1000-fold failure needs an arm share near 0
            raise RuntimeError("the plasmode draw left an arm empty 1000 times")
        treatment = np.asarray(self.arm_codes, dtype=float)[index]
        intermediate = None
        if self.mediator is not None:
            chance = self.mediator[rows, index]
            intermediate = np.asarray(rng.random(rows.size) < chance, dtype=float)
        terms = treatment_term(
            process,
            treatment,
            arm_codes=data.arm_codes,
            levels=data.treatment_levels,
            intermediate=intermediate,
        )
        mean = process.mean(
            self.covariates[rows],
            terms,
            centre=self.centre,
            scale=self.scale,
            bounds=target.outcome_bounds,
        )
        cluster_sd = float(getattr(process, "cluster_sd", 0.0))
        if ids is not None and cluster_sd > 0.0:
            intercepts = rng.normal(0.0, cluster_sd, size=int(ids.max()) + 1)
            mean = np.asarray(mean, dtype=float) + intercepts[ids]
        outcome = np.asarray(process.sample(rng, mean, bounds=target.outcome_bounds), dtype=float)
        sample = data.subset(rows)
        if ids is not None:
            sample = replace(sample, cluster=ids)
        if intermediate is not None:
            sample = replace(sample, intermediate=intermediate)
        if self.recording is not None:
            sample = replace(sample, treatment_observed=recorded)
        sample = sample.with_treatment(treatment)
        if self.known is not None:
            from ..data.known_mechanism import KnownMechanism

            sample = replace(sample, known_treatment=KnownMechanism(self.known[rows], "array"))
        if self.response is not None:
            observed = rng.random(rows.size) < self.response[rows, index]
            outcome = np.where(observed, outcome, np.nan)
            sample = replace(sample, observed=np.asarray(observed, dtype=bool))
        return sample.with_outcome(outcome, family=process.family)
