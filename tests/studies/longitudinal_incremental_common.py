r"""The design both registered IPSI studies share: laws, plans, fits, truths and bounds.

``longitudinal-incremental`` (S1, :mod:`tests.studies.canonical_longitudinal_incremental`) and
``longitudinal-incremental-dropout`` (S2,
:mod:`tests.studies.canonical_longitudinal_incremental_dropout`) fit
:class:`~cleverly.longitudinal.IPSI` on the exact laws of
:mod:`tests.discrete_law_longitudinal_incremental`.  Every nuisance is the saturated
:class:`~tests.discrete_law_longitudinal.CellMeans`, which is consistent on these laws
(``X19-plan.md`` 3.5).

**The fallback-read bound** (``X19-plan.md`` 6.2).  ``CellMeans`` answers a design row it never
saw with the training mean of its target.  A cross-fitted IPSI fit reads such a prediction in
three ways:

=====================================  =========================================================
read                                   when the fallback enters with a nonzero weight
=====================================  =========================================================
the carried value at an unseen arm     the history :math:`h` has training rows at arm
                                       :math:`a`, but none of them stays observed (dropout);
                                       or :math:`h` has no training row at all, so
                                       :math:`\hat\pi(h)` is the node mean too
the residual at the observed arm       the held-out row's own cell :math:`(h, a)` has no
                                       observed training row
an arm the history never took          never: the cell mean of :math:`A_t` in :math:`h` is 0
                                       or 1, so :math:`\hat q_t(a \mid h) = 0` multiplies it
=====================================  =========================================================

:func:`fallback_read_mass` sums the expected mass of these reads over the nodes.
:func:`fallback_bound` turns it into a bias bound in standard-error units:
mass :math:`\times \prod_t r_t \times \omega_{\min}^{-T} \times R / (\sigma / \sqrt n)`.
Here :math:`r_t` is the largest per-node weight of the plan, :math:`R` the range of the
outcome mean, and :math:`\sigma` the efficient influence function standard deviation.  The
design test asserts that the bound is below :data:`FALLBACK_BOUND` in every declared positive
cell before the declaration commit.  A one-fold fit reads no fallback, since every held-out
cell is a training cell.
"""

from __future__ import annotations

import itertools
import re
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from cleverly.longitudinal import IPSI
from tests import discrete_law_longitudinal_incremental as law
from tests.discrete_law_longitudinal import CellMeans
from tests.longitudinal_ipsi import node

#: The largest fallback-read bias a declared positive cell may carry, in standard errors.
FALLBACK_BOUND = 0.1
#: The bounds of every fit: no partial product of a non-tilt factor of these laws reaches
#: them, so no bound binds and the fits are the untruncated functional.
G_BOUNDS = (1e-8, 1.0)
#: The declared support of a continuous outcome: every outcome mean of these laws lies in
#: [1, 8] and the noise is standard normal, so a draw leaves the support with probability
#: below 1e-18.  A cross-fitted fit of a continuous outcome needs it (the shipped scale rule).
Q_BOUNDS = (-8.0, 17.0)
#: The causes of a competing-risk fit, in the oracle's level order.
CAUSES = ("event", "competing")

Plan = tuple[tuple[str, Any], ...]


def constant_plans(deltas: Mapping[str, float], n_times: int = 3) -> dict[str, Plan]:
    """One tilt plan per ``delta``, each tilting every node."""
    return {label: law.constant(delta, n_times) for label, delta in deltas.items()}


def regimens(plans: Mapping[str, Plan]) -> dict[str, Any]:
    """``regimens=`` of an :class:`~cleverly.longitudinal.IPSI` fit of the oracle's ``plans``."""
    return {
        label: tuple(node(entry, position, label) for position, entry in enumerate(plan, 1))
        for label, plan in plans.items()
    }


# ------------------------------------------------------------------------------- names


def names(
    plans: Mapping[str, Plan],
    reference: str | None,
    *,
    horizons: Sequence[int] | None = None,
    competing: bool = False,
) -> tuple[str, ...]:
    """Every parameter a fit of ``plans`` reports, in the fit's order."""
    labels = list(plans)
    others = [label for label in labels if label != reference] if reference else []
    if horizons is None:
        return (
            *(f"ey_regimen[{label}]" for label in labels),
            *(f"ate_regimen[{label} vs {reference}]" for label in others),
        )
    if not competing:
        return (
            *(f"risk_regimen[{label} @ t={h}]" for label in labels for h in horizons),
            *(f"ate_regimen[{label} vs {reference} @ t={h}]" for label in others for h in horizons),
        )
    return (
        *(
            f"cif_regimen[{label}, {cause} @ t={h}]"
            for label in labels
            for cause in CAUSES
            for h in horizons
        ),
        *(
            f"ate_regimen[{label} vs {reference}, {cause} @ t={h}]"
            for label in others
            for cause in CAUSES
            for h in horizons
        ),
    )


_NAME = re.compile(
    r"^(?P<kind>ey|risk|cif|ate)_regimen\[(?P<left>.+?)(?: vs (?P<right>.+?))?"
    r"(?:, (?P<cause>[a-z_]+))?(?: @ t=(?P<horizon>\d+))?\]$"
)


def parse(name: str) -> tuple[str, str | None, int, int | None]:
    """``(left, right, cause code, horizon)`` of a reported parameter name."""
    match = _NAME.match(name)
    if match is None:
        raise ValueError(f"not an IPSI parameter name: {name!r}")
    cause = match["cause"]
    horizon = match["horizon"]
    return (
        match["left"],
        match["right"],
        1 if cause is None else CAUSES.index(cause) + 1,
        None if horizon is None else int(horizon),
    )


def _options(target: Any, cause: int, horizon: int | None, extra: Mapping[str, Any]) -> dict:
    options = dict(extra)
    if horizon is not None:
        options.update(horizon=horizon, cause=cause)
    return options


def truth(target: Any, plans: Mapping[str, Plan], name: str, **extra: Any) -> float:
    """The exact value of ``name`` on law ``target``."""
    left, right, cause, horizon = parse(name)
    options = _options(target, cause, horizon, extra)
    value = law.truth(target, plans[left], **options)
    if right is not None:
        value -= law.truth(target, plans[right], **options)
    return float(value)


def eif_sd(target: Any, plans: Mapping[str, Plan], name: str, **extra: Any) -> float:
    """The efficient influence function standard deviation of ``name`` on law ``target``."""
    left, right, cause, horizon = parse(name)
    options = _options(target, cause, horizon, extra)
    if right is not None:
        options["other"] = plans[right]
    return law.eif_sd(target, plans[left], **options)


def truths(
    target: Any, plans: Mapping[str, Plan], wanted: Sequence[str], **extra: Any
) -> dict[str, float]:
    return {name: truth(target, plans, name, **extra) for name in wanted}


def combined_sd(target: Any, plans: Mapping[str, Plan], weights: Mapping[str, float]) -> float:
    r"""The influence function standard deviation of :math:`\sum_k c_k \psi(\text{plan}_k)`.

    The efficient influence function is linear in the functional, so a linear combination
    of plan means (a working-model coefficient over a fixed plan grid) has the same
    combination of the plans' cell parts and outcome coefficients.
    """
    probs = target.probs
    cells = np.zeros(probs.shape)
    coefficient = np.zeros(probs.shape)
    for label, weight in weights.items():
        part, slope = law.eif(target, plans[label])
        cells = cells + weight * np.nan_to_num(part)
        coefficient = coefficient + weight * slope
    keep = probs > 0
    variance = float(np.sum(probs[keep] * cells[keep] ** 2))
    variance += law.SIGMA**2 * float(np.sum(probs[keep] * coefficient[keep] ** 2))
    return float(np.sqrt(variance))


# --------------------------------------------------------------------------------- fits


def fit(
    frame: pd.DataFrame,
    target: Any,
    plans: Mapping[str, Plan],
    *,
    n_folds: int,
    reference: str | None,
    horizons: Sequence[int] | None = None,
    simultaneous: bool = False,
    outcome_learner: Any = None,
    treatment_learner: Any = None,
    censoring_learner: Any = None,
    **declared: Any,
) -> Any:
    """One :class:`~cleverly.longitudinal.IPSI` fit with saturated cell means at every slot.

    ``declared`` holds further ``fit`` keywords: ``id``, ``weights`` or
    ``treatment_probabilities``.  A ``msm`` entry goes to the estimator instead.
    """
    msm = declared.pop("msm", None)
    censored = any(axis.kind == "censoring" for axis in target.axes)
    settings: dict[str, Any] = {
        "reference": reference,
        "horizons": None if horizons is None else list(horizons),
        "msm": msm,
        "outcome_learner": CellMeans() if outcome_learner is None else outcome_learner,
        "pseudo_learner": CellMeans() if outcome_learner is None else outcome_learner,
        "treatment_learner": CellMeans() if treatment_learner is None else treatment_learner,
        "censoring_learner": (
            (CellMeans() if censoring_learner is None else censoring_learner) if censored else None
        ),
        "n_folds": n_folds,
        "learner_folds": 2,
        "g_bounds": G_BOUNDS,
        "q_bounds": None if target.survival else Q_BOUNDS,
        "simultaneous": simultaneous,
        "random_state": 0,
    }
    if "treatment_probabilities" in declared:
        settings["treatment_learner"] = None
    with warnings.catch_warnings():
        # A cell-mean probability of exactly 0 or 1 is the law's own (law Ax); the bound
        # warning it raises says nothing about these fits.
        warnings.simplefilter("ignore")
        return IPSI(regimens(plans), **settings).fit(frame, **law.columns(target), **declared)


def plug_in(result: Any, name: str) -> float:
    """The mean of the first-node regression's tilted value: the estimate before correction.

    The recursion runs on the declared outcome scale, so the mean is mapped back through the
    result's scaler.
    """
    left, right, _, _ = parse(name)

    def first(label: str) -> float:
        return float(np.mean(result.fits[label].steps[0].marginal))

    if right is None:
        return float(result.scaler.unscale_level(first(left)))
    return float(result.scaler.unscale_difference(first(left) - first(right)))


def known_treatment(frame: pd.DataFrame, target: Any) -> np.ndarray:
    r"""The law's own :math:`\pi_t(h)` at each row and node, ``nan`` once a row is gone."""
    probs = target.probs
    k = len(target.axes)
    out = np.full((len(frame), target.n_times), np.nan)
    for t in range(1, target.n_times + 1):
        j = target.axis_of("treatment", t)
        upper = probs.sum(axis=tuple(range(j + 1, k)))
        lower = upper.sum(axis=j)
        treated = np.take(upper, 1, axis=j)
        pi = np.where(lower > 0, treated / np.where(lower > 0, lower, 1.0), np.nan)
        prefix = [axis.name for axis in target.axes[:j]]
        values = frame[prefix].to_numpy(dtype=float)
        present = ~np.isnan(values).any(axis=1)
        index = tuple(np.nan_to_num(values[:, i]).astype(int) for i in range(j))
        out[:, t - 1] = np.where(present, pi[index], np.nan)
    return out


# ------------------------------------------------------------------- the fallback bound


def _at_risk(target: Any, values: Mapping[str, int], position: int) -> bool:
    return not target._structural(values, target.axes[position])


def fallback_read_mass(target: Any, n_train: int, plan: Plan | None = None) -> float:
    """The expected mass of fallback reads with a nonzero weight, summed over the nodes.

    ``n_train`` is the training size of one fold.  A label node reads the arm it sets
    whatever :math:`\\hat\\pi` is, so ``plan`` adds that read where the arm's cell has no
    observed training row.
    """
    probs = target.probs
    axes = target.axes
    k = len(axes)
    total = 0.0
    for t in range(1, target.n_times + 1):
        j = target.axis_of("treatment", t)
        c = target.axis_of("censoring", t)
        event = target.axis_of("event", t)
        history = probs.sum(axis=tuple(range(j, k)))
        joint = probs.sum(axis=tuple(range(j + 1, k)))
        last = c if c is not None else j
        kept = probs.sum(axis=tuple(range(last + 1, k)))
        if c is not None:
            kept = np.take(kept, 1, axis=c)
        del event
        label = None if plan is None or plan[t - 1][0] != "arm" else int(plan[t - 1][1])
        for cell in itertools.product(*(range(axis.levels) for axis in axes[:j])):
            values = dict(zip([axis.name for axis in axes[:j]], cell, strict=True))
            if not _at_risk(target, values, j):
                continue
            mass = float(history[cell])
            if mass <= 0.0:
                continue
            unseen = (1.0 - mass) ** n_train
            for arm in (0, 1):
                taken = float(joint[(*cell, arm)])
                observed = float(kept[(*cell, arm)])
                lost = (1.0 - observed) ** n_train
                if label is not None:
                    if arm == label:
                        total += mass * lost
                    continue
                if taken <= 0.0:
                    continue
                total += mass * (lost - (1.0 - taken) ** n_train) + mass * unseen
                total += taken * lost
    return total


def node_ratio(target: Any, entry: tuple[str, Any], t: int) -> float:
    """The largest per-node weight of one plan node at node ``t``."""
    kind, value = entry
    if kind == "tilt":
        return float(max(value, 1.0 / value))
    probs = target.probs
    k = len(target.axes)
    j = target.axis_of("treatment", t)
    upper = probs.sum(axis=tuple(range(j + 1, k)))
    lower = upper.sum(axis=j)
    taken = np.take(upper, int(value), axis=j)
    share = taken[lower > 0] / lower[lower > 0]
    return float(1.0 / share[share > 0].min())


def outcome_range(target: Any) -> float:
    if target.survival:
        return 1.0
    support = target.mu[(target.probs > 0) & (target.mu != 0)]
    return float(support.max() - support.min())


def minimum_retention(target: Any) -> float:
    """The smallest probability of staying observed at a node, one on a law without dropout."""
    probs = target.probs
    k = len(target.axes)
    smallest = 1.0
    for t in range(1, target.n_times + 1):
        c = target.axis_of("censoring", t)
        if c is None:
            continue
        upper = probs.sum(axis=tuple(range(c + 1, k)))
        lower = upper.sum(axis=c)
        kept = np.take(upper, 1, axis=c)
        share = kept[lower > 0] / lower[lower > 0]
        smallest = min(smallest, float(share.min()))
    return smallest


def fallback_bound(target: Any, plans: Sequence[Plan], *, n: int, n_folds: int, sd: float) -> float:
    """The fallback-read bias bound of a cell in standard errors (module docstring)."""
    if n_folds == 1:
        return 0.0
    n_train = n * (n_folds - 1) // n_folds
    worst = 0.0
    for plan in plans:
        ratio = float(np.prod([node_ratio(target, entry, t) for t, entry in enumerate(plan, 1)]))
        mass = fallback_read_mass(target, n_train, plan)
        worst = max(worst, mass * ratio)
    scale = outcome_range(target) / minimum_retention(target) ** target.n_times
    return worst * scale / (sd / np.sqrt(n))


# --------------------------------------------------------------- a constant mechanism


def constant_mechanism_limit(target: Any, plan: Plan) -> float:
    r"""The limit of an IPSI fit of ``plan`` with a constant :math:`\hat\pi_t` and right regressions.

    The constant is the limit of ``DummyClassifier(strategy="prior")`` at node :math:`t`, the
    marginal :math:`\pi^*_t = P(A_t = 1)`.  "Right regressions" are the limits the regressions
    take when the carried values use :math:`\pi^*` (``X19-plan.md`` 6.9):
    :math:`m^*_t(h, a) = E\{\bar m^*_{t+1}(H_{t+1}) \mid h, a\}` with
    :math:`\bar m^*_t = \sum_a q^*_t(a) m^*_t(h, a)`.  Every residual term then has mean zero,
    so the limit is :math:`E\{\bar m^*_1\}` plus the means of the mechanism terms
    :math:`W^*_{t-1}\delta_t(m^*_t(H_t, 1) - m^*_t(H_t, 0))(A_t - \pi^*_t)/D^{*2}_t`.  It is
    stated for an uncensored end-of-study law with tilt nodes only.
    """
    if target.survival or any(axis.kind == "censoring" for axis in target.axes):
        raise ValueError("the limit is stated for an uncensored end-of-study law")
    probs = target.probs
    axes = target.axes
    k = len(axes)
    columns = [target.axis_of("treatment", t) for t in range(1, target.n_times + 1)]
    pi_star = []
    for j in columns:
        upper = probs.sum(axis=tuple(range(j + 1, k)))
        pi_star.append(float(np.take(upper, 1, axis=j).sum()))
    deltas = [value for kind, value in plan]
    if any(kind != "tilt" for kind, _ in plan):
        raise ValueError("the limit is stated for tilt nodes only")

    def q1(t: int) -> float:
        pi, delta = pi_star[t - 1], deltas[t - 1]
        return delta * pi / (delta * pi + 1.0 - pi)

    # m[t][prefix through A_t] and the carried bar-m[t][prefix through H_t], backwards.
    m: dict[int, np.ndarray] = {}
    carried: dict[int, np.ndarray] = {}
    value = target.mu.copy()  # shape of the full law; law A ends at A_T
    for t in range(target.n_times, 0, -1):
        j = columns[t - 1]
        # Reduce every axis after A_t by its conditional mass.
        joint = probs
        reduced = value
        for axis_index in range(k - 1, j, -1):
            parent = joint.sum(axis=axis_index)
            weight = np.where(
                np.expand_dims(parent, axis_index) > 0,
                joint
                / np.where(
                    np.expand_dims(parent, axis_index) > 0, np.expand_dims(parent, axis_index), 1.0
                ),
                0.0,
            )
            reduced = (weight * reduced).sum(axis=axis_index)
            joint = parent
        m[t] = reduced  # shape: axes through A_t
        bar = q1(t) * np.take(reduced, 1, axis=j) + (1.0 - q1(t)) * np.take(reduced, 0, axis=j)
        carried[t] = bar  # shape: axes before A_t
        value = np.broadcast_to(np.expand_dims(bar, tuple(range(j, k))), probs.shape).copy()
    first = columns[0]
    marginal = probs.sum(axis=tuple(range(first, k)))
    limit = float(np.sum(marginal * carried[1]))
    for t in range(1, target.n_times + 1):
        j = columns[t - 1]
        joint = probs.sum(axis=tuple(range(j + 1, k)))  # through A_t
        pi, delta = pi_star[t - 1], deltas[t - 1]
        blip = np.take(m[t], 1, axis=j) - np.take(m[t], 0, axis=j)
        weight = np.ones(joint.shape)
        for s in range(1, t):
            arm = np.arange(2).reshape(
                [-1 if index == columns[s - 1] else 1 for index in range(j + 1)]
            )
            factor = (deltas[s - 1] * arm + 1.0 - arm) / (
                deltas[s - 1] * pi_star[s - 1] + 1.0 - pi_star[s - 1]
            )
            weight = weight * factor
        arm_t = np.arange(2).reshape([-1 if index == j else 1 for index in range(j + 1)])
        derivative = delta * np.expand_dims(blip, j) * (arm_t - pi) / (delta * pi + 1.0 - pi) ** 2
        limit += float(np.sum(joint * weight * derivative))
    return limit
