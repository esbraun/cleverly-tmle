r"""Clever covariates and the parametric submodels they index.

The targeting step moves the initial outcome regression :math:`\bar Q^0` along a
one-dimensional (or here, low-dimensional) parametric submodel whose score at
:math:`\epsilon = 0` equals the efficient influence function of the target
parameter.  Solving for :math:`\epsilon` therefore makes the estimator solve the
efficient score equation, which is what buys double robustness and asymptotic
efficiency.

For a logistic fluctuation the submodel is

.. math::

    \operatorname{logit} \bar Q^*_\epsilon(a, W)
      = \operatorname{logit} \bar Q^0(a, W) + \epsilon^\top h(a, W)

where :math:`h` is the *clever covariate*.  Its form depends on the target:

``mean`` (used for ``EY1``, ``EY0``, ``ATE``, ``RR``, ``OR``)
    Two columns, :math:`h_1(a, W) = \mathbb 1\{a = 1\} / (g_1(W)\,\pi_1(W))` and
    :math:`h_0(a, W) = \mathbb 1\{a = 0\} / (g_0(W)\,\pi_0(W))`.  Fitting both
    coefficients solves the score equation for each counterfactual mean
    separately, which is what makes the risk ratio and odds ratio available in
    addition to the difference.

``att`` / ``atc``
    A single column contrasting the arms, with the control arm reweighted by the
    propensity odds -- see :func:`att_submodel`.

Here :math:`\pi_a(W) = P(\Delta = 1 \mid A = a, W)` is the probability that the
outcome is observed; with no missingness it is one and drops out.  Note the
:math:`\Delta` indicator itself does *not* appear in :math:`h`: it enters by
restricting the fluctuation regression to rows with an observed outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .._typing import BoolArray, FloatArray

__all__ = [
    "Submodel",
    "TargetGroup",
    "atc_submodel",
    "att_submodel",
    "mean_submodel",
    "submodel_for",
]

TargetGroup = Literal["mean", "att", "atc"]


@dataclass(frozen=True)
class Submodel:
    """Clever covariates evaluated at the observed and both counterfactual arms.

    Attributes
    ----------
    observed:
        ``(n, k)`` covariate at the treatment each unit actually received; this is
        what the fluctuation regression uses.
    at_one, at_zero:
        ``(n, k)`` covariate with the treatment set to 1 and 0.  Applying the
        fitted ``epsilon`` to these gives the targeted counterfactual predictions.
    names:
        Column labels, for reporting ``epsilon`` back to the user.
    group:
        Which estimand family this submodel targets.
    """

    observed: FloatArray
    at_one: FloatArray
    at_zero: FloatArray
    names: tuple[str, ...]
    group: TargetGroup

    def __post_init__(self) -> None:
        shapes = {self.observed.shape, self.at_one.shape, self.at_zero.shape}
        if len(shapes) != 1:
            raise ValueError(f"submodel covariates have mismatched shapes: {shapes}")
        if self.observed.shape[1] != len(self.names):
            raise ValueError(
                f"{self.observed.shape[1]} covariate column(s) but {len(self.names)} name(s)"
            )

    @property
    def n(self) -> int:
        return int(self.observed.shape[0])

    @property
    def dim(self) -> int:
        return int(self.observed.shape[1])

    @property
    def max_abs(self) -> float:
        """Largest absolute clever-covariate value.

        A large value is the signature of a practical positivity violation: a
        single unit whose inverse propensity weight dominates the estimating
        equation.  :mod:`cleverly.sensitivity.positivity` reports it.
        """
        if self.observed.size == 0:
            return 0.0
        return float(np.max(np.abs(self.observed)))


def _arm_columns(
    n: int, probabilities: FloatArray | None, label: str
) -> tuple[FloatArray, FloatArray]:
    """Split an ``(n, 2)`` arm-indexed probability array into its two columns."""
    if probabilities is None:
        return np.ones(n), np.ones(n)
    probs = np.asarray(probabilities, dtype=float)
    if probs.shape != (n, 2):
        raise ValueError(f"{label} must have shape ({n}, 2); got {probs.shape}")
    if np.any(probs <= 0):
        raise ValueError(f"{label} must be strictly positive after bounding")
    return probs[:, 0], probs[:, 1]


def _selection_indicator(n: int, selection: FloatArray | None) -> FloatArray:
    """``1`` when the unit's realised intermediate equals the targeted value."""
    if selection is None:
        return np.ones(n)
    indicator = np.asarray(selection, dtype=float).reshape(-1)
    if indicator.shape[0] != n:
        raise ValueError(f"selection has length {indicator.shape[0]}, expected {n}")
    return indicator


def mean_submodel(
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
) -> Submodel:
    """Two-column submodel targeting both counterfactual means.

    Parameters
    ----------
    treatment:
        Binary treatment indicator, length ``n``.
    propensity:
        ``g(W) = P(A = 1 | W)``, already truncated away from 0 and 1.
    missingness:
        Optional ``(n, 2)`` array of ``P(Delta = 1 | A = a, W)`` for ``a = 0, 1``.
    intermediate_density, selection:
        For a controlled direct effect at intermediate value ``z``:
        ``P(Z = z | A = a, W)`` per arm, and the indicator ``1{Z_i = z}``.  The
        indicator multiplies only the *observed* covariate -- the counterfactual
        columns are already evaluated at ``Z = z`` by construction.
    """
    a = np.asarray(treatment, dtype=float).reshape(-1)
    g1 = np.asarray(propensity, dtype=float).reshape(-1)
    n = a.shape[0]
    if g1.shape[0] != n:
        raise ValueError(f"propensity has length {g1.shape[0]}, expected {n}")
    if np.any(g1 <= 0) or np.any(g1 >= 1):
        raise ValueError("propensity scores must lie strictly inside (0, 1) after truncation")
    g0 = 1.0 - g1
    pi0, pi1 = _arm_columns(n, missingness, "missingness probabilities")
    pz0, pz1 = _arm_columns(n, intermediate_density, "intermediate probabilities")
    keep = _selection_indicator(n, selection)

    inv_one = 1.0 / (g1 * pi1 * pz1)
    inv_zero = 1.0 / (g0 * pi0 * pz0)

    observed = np.column_stack([(1.0 - a) * keep * inv_zero, a * keep * inv_one])
    at_one = np.column_stack([np.zeros(n), inv_one])
    at_zero = np.column_stack([inv_zero, np.zeros(n)])
    return Submodel(observed, at_one, at_zero, ("h0", "h1"), "mean")


def att_submodel(
    treatment: FloatArray,
    propensity: FloatArray,
    treated_fraction: float,
    *,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
) -> Submodel:
    r"""One-column submodel targeting the ATT.

    .. math::

        h(a, W) = \frac{1}{P(A = 1)}
                  \left( \frac{\mathbb 1\{a = 1\}}{\pi_1(W)}
                       - \frac{\mathbb 1\{a = 0\}}{\pi_0(W)}
                         \frac{g_1(W)}{g_0(W)} \right)

    The treated arm needs no reweighting -- the ATT conditions on ``A = 1``, so the
    treated outcomes are already drawn from the target population.  Control units
    are reweighted by the propensity odds ``g_1 / g_0`` to make them resemble the
    treated, which is why the ATT is far more sensitive than the ATE to small
    ``g_0(W)``; it is also why ``g_bounds="auto"`` uses a more conservative bound
    for this estimand.
    """
    a = np.asarray(treatment, dtype=float).reshape(-1)
    g1 = np.asarray(propensity, dtype=float).reshape(-1)
    n = a.shape[0]
    if not 0.0 < treated_fraction < 1.0:
        raise ValueError(f"treated_fraction must lie in (0, 1); got {treated_fraction}")
    if np.any(g1 <= 0) or np.any(g1 >= 1):
        raise ValueError("propensity scores must lie strictly inside (0, 1) after truncation")
    g0 = 1.0 - g1
    pi0, pi1 = _arm_columns(n, missingness, "missingness probabilities")
    pz0, pz1 = _arm_columns(n, intermediate_density, "intermediate probabilities")
    keep = _selection_indicator(n, selection)

    treated_term = 1.0 / (treated_fraction * pi1 * pz1)
    control_term = (g1 / g0) / (treated_fraction * pi0 * pz0)

    observed = (keep * (a * treated_term - (1.0 - a) * control_term)).reshape(-1, 1)
    at_one = treated_term.reshape(-1, 1)
    at_zero = (-control_term).reshape(-1, 1)
    return Submodel(observed, at_one, at_zero, ("h_att",), "att")


def atc_submodel(
    treatment: FloatArray,
    propensity: FloatArray,
    treated_fraction: float,
    *,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
) -> Submodel:
    """One-column submodel targeting the ATC -- the mirror image of the ATT."""
    a = np.asarray(treatment, dtype=float).reshape(-1)
    g1 = np.asarray(propensity, dtype=float).reshape(-1)
    n = a.shape[0]
    control_fraction = 1.0 - treated_fraction
    if not 0.0 < control_fraction < 1.0:
        raise ValueError(f"treated_fraction must lie in (0, 1); got {treated_fraction}")
    if np.any(g1 <= 0) or np.any(g1 >= 1):
        raise ValueError("propensity scores must lie strictly inside (0, 1) after truncation")
    g0 = 1.0 - g1
    pi0, pi1 = _arm_columns(n, missingness, "missingness probabilities")
    pz0, pz1 = _arm_columns(n, intermediate_density, "intermediate probabilities")
    keep = _selection_indicator(n, selection)

    control_term = 1.0 / (control_fraction * pi0 * pz0)
    treated_term = (g0 / g1) / (control_fraction * pi1 * pz1)

    observed = (keep * (a * treated_term - (1.0 - a) * control_term)).reshape(-1, 1)
    at_one = treated_term.reshape(-1, 1)
    at_zero = (-control_term).reshape(-1, 1)
    return Submodel(observed, at_one, at_zero, ("h_atc",), "atc")


def submodel_for(
    group: TargetGroup,
    treatment: FloatArray,
    propensity: FloatArray,
    *,
    treated_fraction: float | None = None,
    missingness: FloatArray | None = None,
    intermediate_density: FloatArray | None = None,
    selection: FloatArray | None = None,
) -> Submodel:
    """Dispatch to the submodel builder for an estimand family."""
    if group not in ("mean", "att", "atc"):
        raise ValueError(f"unknown target group {group!r}; expected 'mean', 'att' or 'atc'")
    extras = {
        "missingness": missingness,
        "intermediate_density": intermediate_density,
        "selection": selection,
    }
    if group == "mean":
        return mean_submodel(treatment, propensity, **extras)
    if treated_fraction is None:
        raise ValueError(f"the {group!r} submodel needs treated_fraction")
    if group == "att":
        return att_submodel(treatment, propensity, treated_fraction, **extras)
    return atc_submodel(treatment, propensity, treated_fraction, **extras)


def weighted_form(submodel: Submodel, weights: FloatArray) -> tuple[Submodel, FloatArray]:
    r"""Recast a fluctuation in *weighted* form.

    R's ``tmle`` calls this ``target.gwt``: instead of using :math:`h` as a
    regression covariate, move its magnitude into the observation weights and keep
    only its sign.  The submodel becomes
    :math:`\operatorname{logit}\bar Q^*_\epsilon = \operatorname{logit}\bar Q^0
    + \epsilon^\top \operatorname{sign}(h)`, and because

    .. math:: \sum_i (w_i |h_i|)\, \operatorname{sign}(h_i)\,(Y_i - \bar Q^*_i)
              = \sum_i w_i\, h_i\, (Y_i - \bar Q^*_i)

    the *estimating equation being solved is identical*.  What changes is the
    design matrix, which no longer contains the extreme values that make the fit
    numerically fragile under weak overlap.  For the two-column ``mean`` submodel
    the columns have disjoint support (one per arm), so a single weight vector
    serves both.
    """
    magnitude = np.abs(submodel.observed).sum(axis=1)
    signed = Submodel(
        np.sign(submodel.observed),
        np.sign(submodel.at_one),
        np.sign(submodel.at_zero),
        submodel.names,
        submodel.group,
    )
    return signed, np.asarray(weights, dtype=float) * magnitude


def restrict(submodel: Submodel, mask: BoolArray) -> Submodel:
    """Row-subset a submodel (used to drop rows with an unobserved outcome)."""
    index = np.asarray(mask)
    return Submodel(
        submodel.observed[index],
        submodel.at_one[index],
        submodel.at_zero[index],
        submodel.names,
        submodel.group,
    )
