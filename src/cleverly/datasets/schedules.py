r"""Two laws of the navigation program with exact truths: contact schedules and service doses.

The tutorial ``docs/examples/schedules-and-intensity.ipynb`` asks two policy questions of the
program, and each question has its own law.

**Contact schedules.** :func:`make_contact_schedules` draws one discharge per row with three
covariates, a contact schedule ``cadence`` with three levels, and a transition score in
:math:`[0, 1]`.

======================  =================================================================
node                    law
======================  =================================================================
:math:`W_1, W_2, W_3`   ``discharge_risk``, ``age``, ``baseline_support_need``, independent
                        standard normal
:math:`A`               ``cadence`` in ``("low", "medium", "high")``, with
                        :math:`P(A = a \mid W) \propto \exp\{\lambda_a(W)\}`, where
                        :math:`\lambda_{low} = 0`, :math:`\lambda_{medium} = 0.5 W_1 - 0.4 W_2`
                        and :math:`\lambda_{high} = W_1 + 0.4 W_3`
:math:`Y`               ``transition_score``, Beta with mean
                        :math:`m(a, W) = \operatorname{expit}\{-0.1 + s_a - 0.6 W_1
                        - 0.3 W_2 - 0.3 W_3\}`, :math:`s = (0, 0.3, 0.8)`, and precision 8
======================  =================================================================

An odds tilt multiplies the odds of each level :math:`a` by a factor :math:`\delta_a`, and a
level the tilt does not name keeps the factor 1.  The tilted mean is
:math:`E[\sum_a \delta_a g(a \mid W) m(a, W) / \sum_b \delta_b g(b \mid W)]`, a
three-dimensional integral that :func:`contact_schedules_truth` takes with 40 Gauss-Hermite
nodes in each covariate.

**Service doses.** :func:`make_navigation_doses` draws one discharge per row with three
covariates, two service doses and a binary transition success.

======================  =================================================================
node                    law
======================  =================================================================
:math:`W_1, W_2, W_3`   ``discharge_risk``, ``age``, ``medication_burden``, independent
                        standard normal
:math:`H`               ``navigator_hours``, normal with mean
                        :math:`4 + 0.5 W_1 - 0.3 W_2` and standard deviation 1
:math:`M`               ``pharmacist_minutes``, normal with mean
                        :math:`40 + 4 W_1 + 3 W_3` and standard deviation 10, independent
                        of :math:`H` given :math:`W`
:math:`Y`               ``transition_success``, Bernoulli with
                        :math:`\operatorname{expit}\{-0.2 - 0.6 W_1 + 0.3 W_2 - 0.3 W_3
                        + 0.3 h + 0.25 m + 0.35 h m - 0.1 h^2\}`, where :math:`h = H - 4`
                        and :math:`m = (M - 40) / 10`
======================  =================================================================

The product term :math:`0.35 h m` makes the two doses interact, so the effect of raising both
differs from the sum of the effects of raising each.  A plan of :data:`DOSE_PLANS` adds a
fixed amount to a dose unless the result would pass the dose's cap, in which case the patient
keeps the observed dose, as :class:`~cleverly.interventions.Shift` does.  Given :math:`W`, each
plan mean is an integral over the two dose residuals, which :func:`navigation_doses_truth` takes
with Gauss-Legendre nodes on each side of the cap's kink, and the covariates are integrated with
Gauss-Hermite nodes.

``tests/unit/test_datasets_schedules.py`` compares each truth with a Monte Carlo draw and
rejects deliberately wrong truths.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from typing import Any

import numpy as np

from .._typing import Backend, FloatArray
from ..utils.bounds import expit
from ..utils.frames import frame_from_dict

__all__ = [
    "DOSE_PLANS",
    "SCHEDULE_LEVELS",
    "contact_schedules_truth",
    "make_contact_schedules",
    "make_navigation_doses",
    "navigation_doses_truth",
]

#: The three contact schedules, from the least to the most contact.
SCHEDULE_LEVELS: tuple[str, ...] = ("low", "medium", "high")

#: The score step of each schedule on the logit scale, in the order of :data:`SCHEDULE_LEVELS`.
_SCHEDULE_STEP = np.array([0.0, 0.3, 0.8])

#: The precision of the Beta draw of the transition score.
_SCORE_PRECISION = 8.0

#: The dose plans, by name: for each dose, the amount added and the cap that no added dose
#: may pass.  A dose a plan leaves out keeps its observed value.
DOSE_PLANS: dict[str, dict[str, tuple[float, float]]] = {
    "current practice": {},
    "more navigator time": {"navigator_hours": (0.5, 7.0)},
    "more pharmacist time": {"pharmacist_minutes": (5.0, 70.0)},
    "both": {"navigator_hours": (0.5, 7.0), "pharmacist_minutes": (5.0, 70.0)},
}

#: The standard deviation of each dose given the covariates.
_HOURS_SD = 1.0
_MINUTES_SD = 10.0


def _schedule_logits(w: FloatArray) -> FloatArray:
    """The ``(n, 3)`` scores whose softmax is ``P(cadence | W)``."""
    return np.stack(
        [
            np.zeros_like(w[..., 0]),
            0.5 * w[..., 0] - 0.4 * w[..., 1],
            1.0 * w[..., 0] + 0.4 * w[..., 2],
        ],
        axis=-1,
    )


def _schedule_probabilities(w: FloatArray) -> FloatArray:
    """``P(cadence = a | W)`` for each level, along the last axis."""
    logits = _schedule_logits(w)
    weights = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return np.asarray(weights / weights.sum(axis=-1, keepdims=True), dtype=float)


def _score_mean(w: FloatArray) -> FloatArray:
    """``E[transition_score | cadence = a, W]`` for each level, along the last axis."""
    base = -0.1 - 0.6 * w[..., 0] - 0.3 * w[..., 1] - 0.3 * w[..., 2]
    return np.asarray(expit(base[..., None] + _SCHEDULE_STEP), dtype=float)


def _tilt_factors(factors: Mapping[str, float]) -> FloatArray:
    """The factor of each level, in the order of :data:`SCHEDULE_LEVELS`; 1 where unnamed."""
    unknown = set(factors) - set(SCHEDULE_LEVELS)
    if unknown:
        raise ValueError(f"unknown schedule levels {sorted(unknown)}; use {SCHEDULE_LEVELS}")
    values = np.array([float(factors.get(level, 1.0)) for level in SCHEDULE_LEVELS])
    if np.any(values <= 0.0):
        raise ValueError("every odds factor must be positive")
    return values


@cache
def _hermite_grid(nodes: int, dimensions: int) -> tuple[FloatArray, FloatArray]:
    """Points and weights of a product Gauss-Hermite rule for independent standard normals."""
    points, weights = np.polynomial.hermite_e.hermegauss(nodes)
    weights = weights / weights.sum()
    grids = np.meshgrid(*([points] * dimensions), indexing="ij")
    weight_grids = np.meshgrid(*([weights] * dimensions), indexing="ij")
    stacked = np.stack([grid.reshape(-1) for grid in grids], axis=-1)
    product = np.prod(np.stack([grid.reshape(-1) for grid in weight_grids], axis=-1), axis=-1)
    return stacked, product


def _tilted_mean(factors: FloatArray) -> float:
    """The mean transition score when the odds of each schedule are multiplied by ``factors``."""
    points, weights = _hermite_grid(40, 3)
    tilted = _schedule_probabilities(points) * factors
    tilted = tilted / tilted.sum(axis=-1, keepdims=True)
    return float(weights @ np.sum(tilted * _score_mean(points), axis=-1))


def contact_schedules_truth(
    tilts: Mapping[str, Mapping[str, float]] | None = None, reference: str | None = None
) -> dict[str, float]:
    r"""Return the exact mean score under each schedule and each odds tilt, and each contrast.

    Parameters
    ----------
    tilts : mapping or None
        Each tilt's name mapped to its odds factors, a mapping from a level of
        :data:`SCHEDULE_LEVELS` to a positive factor.  A level a tilt leaves out keeps the
        factor 1, so an empty mapping is current practice.  ``None`` asks for no tilt.
    reference : str or None
        The tilt each tilt contrast is taken against.  ``None`` takes the first tilt.

    Returns
    -------
    dict of str to float
        ``ey[<level>]``, the mean transition score if every patient got that schedule;
        ``ey_ipsi[<name>]``, the mean under each tilt; and ``ate_ipsi[<name> vs <reference>]``
        for each other tilt.  The keys are the names that an :class:`~cleverly.ATE` or an
        :class:`~cleverly.IncrementalEffect` fit reports.

    See Also
    --------
    cleverly.datasets.make_contact_schedules : Draws data from this law.

    Notes
    -----
    The module docstring gives the law.  The covariates enter through a smooth integrand, so a
    40-node Gauss-Hermite rule in each of the three covariates is exact to about ``1e-12``.

    Examples
    --------
    >>> from cleverly.datasets import contact_schedules_truth
    >>> truth = contact_schedules_truth({"now": {}, "more": {"medium": 1.5, "high": 2.0}})
    >>> round(truth["ate_ipsi[more vs now]"], 4)
    0.0174
    """
    points, weights = _hermite_grid(40, 3)
    arms = weights @ _score_mean(points)
    truth = {f"ey[{level}]": float(arms[index]) for index, level in enumerate(SCHEDULE_LEVELS)}
    tilts = {} if tilts is None else tilts
    names = list(tilts)
    if not names:
        return truth
    base = names[0] if reference is None else reference
    if base not in tilts:
        raise ValueError(f"reference {base!r} is not one of the tilts {names}")
    means = {name: _tilted_mean(_tilt_factors(tilts[name])) for name in names}
    truth.update({f"ey_ipsi[{name}]": value for name, value in means.items()})
    for name, value in means.items():
        if name != base:
            truth[f"ate_ipsi[{name} vs {base}]"] = value - means[base]
    return truth


def make_contact_schedules(
    n: int = 1000,
    *,
    seed: int | np.random.Generator | None = None,
    tilts: Mapping[str, Mapping[str, float]] | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    r"""Draw discharges with a three-level contact schedule and a bounded transition score.

    Parameters
    ----------
    n : int
        Number of discharges.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    tilts : mapping or None
        The odds tilts whose exact means the truth reports, as in
        :func:`contact_schedules_truth`.  The first tilt is the reference.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend. ``None`` uses pandas when installed, then the first available backend.

    Returns
    -------
    dataframe
        One row per discharge: ``transition_score`` (in 0 to 1), ``cadence`` (``"low"``,
        ``"medium"`` or ``"high"``), ``discharge_risk``, ``age`` and
        ``baseline_support_need``.
    truth : dict of str to float
        The exact values of :func:`contact_schedules_truth` for ``tilts``.

    See Also
    --------
    cleverly.datasets.contact_schedules_truth : The exact mean score under an odds tilt.
    cleverly.datasets.navigation_protocol : The protocol of the program these data describe.

    Notes
    -----
    Patients with a higher ``discharge_risk`` get the more intensive schedules more often and
    score lower, so a plain comparison of the schedules is confounded.  The module docstring gives
    the law.

    Examples
    --------
    >>> from cleverly.datasets import make_contact_schedules
    >>> frame, truth = make_contact_schedules(n=500, seed=3)
    >>> frame.shape
    (500, 5)
    >>> sorted(frame["cadence"].unique())
    ['high', 'low', 'medium']
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, 3))
    probabilities = _schedule_probabilities(w)
    level = (rng.random(n)[:, None] > np.cumsum(probabilities, axis=1)).sum(axis=1)
    level = np.minimum(level, len(SCHEDULE_LEVELS) - 1)
    mean = _score_mean(w)[np.arange(n), level]
    score = rng.beta(mean * _SCORE_PRECISION, (1.0 - mean) * _SCORE_PRECISION)
    columns = {
        "transition_score": score,
        "cadence": np.array(SCHEDULE_LEVELS, dtype=object)[level],
        "discharge_risk": w[:, 0],
        "age": w[:, 1],
        "baseline_support_need": w[:, 2],
    }
    return frame_from_dict(columns, backend=backend), contact_schedules_truth(tilts)


def _dose_means(w: FloatArray) -> tuple[FloatArray, FloatArray]:
    """The mean navigator hours and pharmacist minutes given the covariates."""
    hours = 4.0 + 0.5 * w[..., 0] - 0.3 * w[..., 1]
    minutes = 40.0 + 4.0 * w[..., 0] + 3.0 * w[..., 2]
    return hours, minutes


def _success_probability(w: FloatArray, hours: FloatArray, minutes: FloatArray) -> FloatArray:
    """``P(transition_success = 1 | H, M, W)`` of the dose law."""
    h = hours - 4.0
    m = (minutes - 40.0) / 10.0
    eta = (
        -0.2
        - 0.6 * w[..., 0]
        + 0.3 * w[..., 1]
        - 0.3 * w[..., 2]
        + 0.3 * h
        + 0.25 * m
        + 0.35 * h * m
        - 0.1 * h**2
    )
    return np.asarray(expit(eta), dtype=float)


def _shifted(dose: FloatArray, step: tuple[float, float] | None) -> FloatArray:
    """The dose a plan assigns: ``dose + delta`` unless that passes the cap, else ``dose``."""
    if step is None:
        return dose
    delta, cap = step
    raised = dose + delta
    return np.where(raised <= cap, raised, dose)


@cache
def _legendre(nodes: int) -> tuple[FloatArray, FloatArray]:
    return np.polynomial.legendre.leggauss(nodes)


def _residual_rule(kink: FloatArray, nodes: int = 32) -> tuple[FloatArray, FloatArray]:
    """Nodes and weights over a standard normal residual, split at a kink for each row.

    Each row gets ``2 * nodes`` Gauss-Legendre points: ``nodes`` on ``[-9, kink]`` and ``nodes``
    on ``[kink, 9]``, weighted by the standard normal density.  The mass beyond 9 is below
    ``1e-18``.
    """
    x, v = _legendre(nodes)
    k = np.clip(kink, -9.0, 9.0)[:, None]
    lower_half, upper_half = (k + 9.0) / 2.0, (9.0 - k) / 2.0
    lower = -9.0 + lower_half * (x + 1.0)
    upper = k + upper_half * (x + 1.0)
    points = np.concatenate([lower, upper], axis=1)
    widths = np.concatenate(
        [
            np.broadcast_to(lower_half * v, lower.shape),
            np.broadcast_to(upper_half * v, upper.shape),
        ],
        axis=1,
    )
    density = np.exp(-0.5 * points**2) / np.sqrt(2.0 * np.pi)
    return points, widths * density


def _plan_mean(plan: Mapping[str, tuple[float, float]]) -> float:
    """The success share when every patient's doses follow ``plan``."""
    w, weights = _hermite_grid(12, 3)
    hours_mean, minutes_mean = _dose_means(w)
    hours_step = plan.get("navigator_hours")
    minutes_step = plan.get("pharmacist_minutes")

    def kink(mean: FloatArray, sd: float, step: tuple[float, float] | None) -> FloatArray:
        # The residual at which the raised dose reaches the cap; any value without a cap.
        if step is None:
            return np.zeros_like(mean)
        delta, cap = step
        return np.asarray((cap - delta - mean) / sd, dtype=float)

    e_hours, v_hours = _residual_rule(kink(hours_mean, _HOURS_SD, hours_step))
    e_minutes, v_minutes = _residual_rule(kink(minutes_mean, _MINUTES_SD, minutes_step))
    hours = _shifted(hours_mean[:, None] + _HOURS_SD * e_hours, hours_step)
    minutes = _shifted(minutes_mean[:, None] + _MINUTES_SD * e_minutes, minutes_step)
    probability = _success_probability(w[:, None, None, :], hours[:, :, None], minutes[:, None, :])
    inner = np.einsum("rij,ri,rj->r", probability, v_hours, v_minutes)
    return float(weights @ inner)


@cache
def _plan_means() -> tuple[float, ...]:
    """The success share under each plan of :data:`DOSE_PLANS`, in its order."""
    return tuple(_plan_mean(plan) for plan in DOSE_PLANS.values())


def navigation_doses_truth() -> dict[str, float]:
    r"""Return the exact success share under each dose plan, and each plan's effect.

    Returns
    -------
    dict of str to float
        ``ey_policy[<plan>]`` for each plan of :data:`DOSE_PLANS`, and
        ``ate_policy[<plan> vs current practice]`` for each other plan.  The keys are the names
        that a :class:`~cleverly.ModifiedTreatmentPolicyEffect` fit reports when its policies
        carry the plan names.

    See Also
    --------
    cleverly.datasets.make_navigation_doses : Draws data from this law.

    Notes
    -----
    Given the covariates, the success probability is smooth in each dose residual except at the
    residual where the raised dose reaches its cap.  Each residual is integrated with 32
    Gauss-Legendre nodes on each side of that point, and the covariates with a 12-node
    Gauss-Hermite rule in each dimension.  Doubling every node count moves no value by more
    than ``1e-9``.

    Examples
    --------
    >>> from cleverly.datasets import navigation_doses_truth
    >>> truth = navigation_doses_truth()
    >>> round(truth["ate_policy[both vs current practice]"], 4)
    0.0731
    """
    means = dict(zip(DOSE_PLANS, _plan_means(), strict=True))
    truth = {f"ey_policy[{name}]": value for name, value in means.items()}
    for name, value in means.items():
        if name != "current practice":
            truth[f"ate_policy[{name} vs current practice]"] = value - means["current practice"]
    return truth


def make_navigation_doses(
    n: int = 1000,
    *,
    seed: int | np.random.Generator | None = None,
    backend: Backend | str | None = None,
) -> tuple[Any, dict[str, float]]:
    r"""Draw discharges with two service doses and a binary transition success.

    Parameters
    ----------
    n : int
        Number of discharges.
    seed : int, Generator, or None
        Seed or NumPy random generator.
    backend : {"pandas", "polars", "pyarrow"} or None, default=None
        Dataframe backend. ``None`` uses pandas when installed, then the first available backend.

    Returns
    -------
    dataframe
        One row per discharge: ``transition_success`` (1 or 0), ``navigator_hours``,
        ``pharmacist_minutes``, ``discharge_risk``, ``age`` and ``medication_burden``.
    truth : dict of str to float
        The exact values of :func:`navigation_doses_truth`.

    See Also
    --------
    cleverly.datasets.navigation_doses_truth : The exact success share under each dose plan.
    cleverly.datasets.navigation_protocol : The protocol of the program these data describe.

    Notes
    -----
    Patients with a higher ``discharge_risk`` get more of both services and succeed less
    often, so a plain regression of success on the doses is confounded.  The module docstring
    gives the law, and :data:`DOSE_PLANS` the plans.

    Examples
    --------
    >>> from cleverly.datasets import make_navigation_doses
    >>> frame, truth = make_navigation_doses(n=500, seed=3)
    >>> frame.shape
    (500, 6)
    >>> sorted(truth)[:2]
    ['ate_policy[both vs current practice]', 'ate_policy[more navigator time vs current practice]']
    """
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((n, 3))
    hours_mean, minutes_mean = _dose_means(w)
    hours = hours_mean + _HOURS_SD * rng.standard_normal(n)
    minutes = minutes_mean + _MINUTES_SD * rng.standard_normal(n)
    success = (rng.random(n) < _success_probability(w, hours, minutes)).astype(float)
    columns = {
        "transition_success": success,
        "navigator_hours": hours,
        "pharmacist_minutes": minutes,
        "discharge_risk": w[:, 0],
        "age": w[:, 1],
        "medication_burden": w[:, 2],
    }
    return frame_from_dict(columns, backend=backend), navigation_doses_truth()
