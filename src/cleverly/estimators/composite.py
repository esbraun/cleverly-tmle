r"""The composite-indicator construction for an observational missing outcome or treatment.

This module is the only definition of the composite indicator and its mechanism.  The fit,
a retarget, the truncation curve, the refits, the validators and the post-fit readers all
call it, and no composite array is stored on a result.  A retarget at new bounds therefore
cannot read a stale composite.

For arm :math:`a` the construction sets

.. math::

    C_a = \Delta_A\,\Delta\,1\{A = a\}, \qquad
    g_{c,a}(W) = P(C_a = 1 \mid W)
             = \pi_A(W)\,g(a \mid \Delta_A = 1, W)\,\pi(a, W),

with :math:`\pi_A(W) = P(\Delta_A = 1 \mid W)` and
:math:`\pi(a, W) = P(\Delta = 1 \mid A = a, \Delta_A = 1, W)`.  The product is the law of
total probability, so it holds for every law and no factor needs a causal reading.  On
:math:`O'_a = (W, C_a, C_a Y)` the indicator :math:`C_a` is a binary treatment, and
Benkeser, Carone, van der Laan and Gilbert (2017), Theorem 1, applies to it as stated.  The
fit stacks the :math:`K` per-arm estimators, with no fluctuation parameter shared across
arms.  ``docs/technical-reference/dr-tmle/theorem.md`` states the contract and the
conditions.

**The view** (:func:`composite_view`) is a copy of the data whose treatment holds the arm
code where :math:`\Delta_A \Delta = 1` and ``NaN`` elsewhere, and whose observation
indicator is :math:`\Delta_A \Delta`.  Every indicator downstream is
``view.treatment == arm``, which equals :math:`C_a` on every row.  The view is for
targeting, influence, validation and the readers of a result.  It never enters a nuisance
fit, and :func:`~cleverly.estimators._nuisance.fit_nuisances` refuses it.

**The carrier.**  A composite mechanism is never a distribution over the arms, because
:math:`\sum_a g_{c,a} < 1` whenever a row can be unrecorded.  It always travels as an
``(n, K)`` :class:`~cleverly.estimators._nuisance.Propensity` with ``simplex=False``, so
every site that takes the two-arm complement form reads it column by column instead.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np

from .._typing import BoolArray, FloatArray
from ..data.causal_data import CausalData
from ..data.known_mechanism import KNOWN_TREATMENT_DELTA_REFUSAL
from ..exceptions import CapabilityError
from ..utils.bounds import bound
from ._nuisance import NuisanceEstimates, Propensity

__all__ = [
    "COMPOSITE_NESTED_REFUSAL",
    "CompositeState",
    "MissingDataRoute",
    "complement_form",
    "composite_bounds",
    "composite_clipping_mask",
    "composite_mechanism",
    "composite_state",
    "composite_view",
    "missing_data_route",
    "missing_treatment_design_refusal",
    "missing_treatment_refusal",
    "targeting_inputs",
    "unidentified_targets",
]

#: Which construction a point-treatment fit with missing data runs.  ``"complete"`` has
#: no missing data.  ``"missing_outcome"`` is the shipped missing-outcome TMLE, which also
#: serves ``DRTMLE(guard=())``.  ``"randomized_missing_outcome"`` is Díaz and van der Laan
#: (2017) for a randomized trial.  ``"composite"`` is this module's construction.
MissingDataRoute = Literal["complete", "missing_outcome", "randomized_missing_outcome", "composite"]

#: The refusal of ``evaluation=`` and ``reduced_crossfit="nested"`` on a composite
#: DR-TMLE fit, with or without a missing treatment.
COMPOSITE_NESTED_REFUSAL = (
    "The nested construction and the evaluation companion carry one fold-free treatment "
    "mechanism. The composite mechanism has up to three factors, and neither construction "
    "carries them. Use the default pooled reductions and no evaluation=."
)


def missing_data_route(estimator: Any, data: CausalData) -> MissingDataRoute:
    """The construction a fit of ``estimator`` on ``data`` runs.

    One function decides the route.  The shared preflight reads it, the fit records it
    under ``result.extra["missing_data"]``, and every reader of a fitted result branches on
    the recorded value.  The rule, with ``declared`` meaning ``randomized=True`` on a
    :class:`~cleverly.DRTMLE`, or a :class:`~cleverly.DRTMLE` fit on data that declares a
    known mechanism (:attr:`~cleverly.data.CausalData.known_treatment`):

    ======  =========  ========  ==============  ================================
    delta   missing A  declared  guard           route
    ======  =========  ========  ==============  ================================
    no      no         any       any             ``"complete"``
    yes     no         yes       non-empty       ``"randomized_missing_outcome"``
    yes     no         no        non-empty       ``"composite"``
    yes     no         any       empty, TMLE     ``"missing_outcome"``
    any     yes        no        any             ``"composite"``
    any     yes        yes       any             ``"randomized_missing_outcome"``
    ======  =========  ========  ==============  ================================

    The last row is refused by :func:`missing_treatment_refusal`: the declaration selects
    the randomized construction, which observes the treatment on every row.  A
    :class:`~cleverly.TMLE` with ``delta=`` on data that declares a known mechanism takes
    ``"missing_outcome"``, the ordinary missing-outcome TMLE, which divides by the
    declaration.

    Parameters
    ----------
    estimator : TMLE
        The estimator.  Its ``guard`` and ``randomized`` are read where it has them.
    data : CausalData
        The prepared data.  Its known treatment mechanism is read.

    Returns
    -------
    str
        One of :data:`MissingDataRoute`.
    """
    guard = tuple(getattr(estimator, "guard", ()) or ())
    declared = bool(getattr(estimator, "randomized", False)) or (
        data.known_treatment is not None
        and getattr(estimator, "_assessment_method", None) == "drtmle"
    )
    if data.has_missing_treatment:
        return "randomized_missing_outcome" if declared else "composite"
    if not data.has_missing_outcome:
        return "complete"
    if guard:
        return "randomized_missing_outcome" if declared else "composite"
    return "missing_outcome"


def complement_form(propensity: Propensity) -> bool:
    """Whether a mechanism takes the two-arm complement form.

    A targeted mechanism is carried as one column exactly when this holds.  A composite
    mechanism is never on the simplex, so it never takes the form, at any arm count.

    Parameters
    ----------
    propensity : Propensity
        The mechanism.

    Returns
    -------
    bool
        ``True`` for a two-arm mechanism on the simplex.
    """
    return propensity.n_arms == 2 and propensity.simplex


def composite_view(data: CausalData) -> CausalData:
    r"""The data seen through the composite indicator :math:`C_a`.

    The treatment holds the arm code where :math:`\Delta_A \Delta = 1` and ``NaN``
    elsewhere.  The observation indicator is :math:`\Delta_A \Delta`, and the outcome is
    zero where it is 0.  The view masks by :math:`\Delta_A` directly, so a code that the
    container kept on an unrecorded row cannot reach an indicator.

    Parameters
    ----------
    data : CausalData
        The prepared data, with or without a missing outcome or treatment.

    Returns
    -------
    CausalData
        The view.  It is marked, and a nuisance fit refuses it.
    """
    recorded = data.treatment_recorded & np.asarray(data.observed, dtype=bool)
    codes = np.where(recorded, np.asarray(data.treatment, dtype=float), np.nan)
    outcome = np.where(recorded, np.asarray(data.outcome, dtype=float), 0.0)
    return replace(
        data,
        treatment=codes,
        observed=recorded,
        outcome=outcome,
        treatment_observed=None,
        composite_view=True,
    )


def composite_bounds(
    bounds: tuple[float, float], nuisance_bound: float, factors: int
) -> tuple[float, float]:
    """The bounds the composite mechanism is tilted and clipped inside.

    The floor is the smallest product of the separately bounded factors, so the initial
    clip moves no value.  At ``g_bounds=(0.01, 0.99)`` and ``nuisance_bound=0.01`` with both
    observation factors, it is ``1e-6``.  The default ``g_bounds="auto"`` floors the treatment
    factor at ``5 / (sqrt(n) ln n)`` instead.  The ceiling is the treatment factor's.

    Parameters
    ----------
    bounds : tuple of float
        The resolved ``g_bounds`` of the treatment factor.
    nuisance_bound : float
        The floor of each observation factor.
    factors : int
        How many observation factors the composite carries: 0, 1 or 2.

    Returns
    -------
    tuple of float
        ``(lower, upper)``.
    """
    lower = float(bounds[0])
    for _ in range(int(factors)):
        lower = lower * float(nuisance_bound)
    return lower, float(bounds[1])


def composite_mechanism(
    nuisance: NuisanceEstimates,
    bounds: tuple[float, float],
    nuisance_bound: float,
    *,
    missingness: FloatArray | None = None,
) -> Propensity:
    r"""The composite mechanism :math:`g_{c,a}(W)`, ``(n, K)``, off the simplex.

    Each factor is bounded as the shipped missing-outcome TMLE bounds it: the treatment
    factor by ``bounds`` through :meth:`~cleverly.estimators._nuisance.Propensity.bounded`,
    and each observation factor below by ``nuisance_bound``.  An absent factor is not
    multiplied.  The order is the treatment factor, then the outcome observation factor,
    then the treatment observation factor.  Without a missing treatment the product is the
    shipped denominator :math:`g\pi` bit for bit.

    Parameters
    ----------
    nuisance : NuisanceEstimates
        The fitted factors.  ``propensity`` is the treatment factor, ``missingness`` the
        outcome observation factor and ``treatment_observation`` the treatment observation
        factor.
    bounds : tuple of float
        The ``g_bounds`` of the treatment factor.
    nuisance_bound : float
        The floor of each observation factor.
    missingness : ndarray or None
        A replacement ``(n, K)`` outcome observation factor, as the MNAR tilt supplies.
        ``None`` reads the fitted one.

    Returns
    -------
    Propensity
        The composite, with ``simplex=False``.
    """
    lower = float(nuisance_bound)
    values = np.asarray(nuisance.propensity.bounded(bounds), dtype=float)
    if missingness is not None:
        values = values * np.clip(np.asarray(missingness, dtype=float), lower, 1.0)
    elif nuisance.missingness is not None:
        values = values * bound(np.asarray(nuisance.missingness, dtype=float), lower, 1.0)
    if nuisance.treatment_observation is not None:
        recorded = bound(np.asarray(nuisance.treatment_observation, dtype=float), lower, 1.0)
        values = values * recorded.reshape(-1, 1)
    return Propensity(values, nuisance.arms, simplex=False)


def composite_clipping_mask(
    nuisance: NuisanceEstimates, bounds: tuple[float, float], nuisance_bound: float
) -> BoolArray:
    """Which initial composite cells change when any fitted factor is bounded.

    Read the raw factors before :func:`composite_state` removes them. Comparing its
    already-bounded product with the product bounds cannot detect factor clipping.
    The mask counts a cell once when several of its factors are clipped.

    Parameters
    ----------
    nuisance : NuisanceEstimates
        The fitted factors before the composite transformation.
    bounds : tuple of float
        The treatment factor's bounds.
    nuisance_bound : float
        The floor of each observation factor.

    Returns
    -------
    BoolArray
        One clipping flag per row and arm.
    """
    clipped = np.asarray(nuisance.propensity.truncate(bounds).clipped, dtype=bool).copy()
    for factor in (nuisance.missingness, nuisance.treatment_observation):
        if factor is None:
            continue
        values = np.asarray(factor, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        clipped |= (values < nuisance_bound) | (values > 1.0)
    return clipped


def _factors(nuisance: NuisanceEstimates, missingness: FloatArray | None) -> int:
    present = (missingness is not None or nuisance.missingness is not None) + (
        nuisance.treatment_observation is not None
    )
    return int(present)


@dataclass(frozen=True)
class CompositeState:
    """The view, the nuisances it is targeted with, and the bounds of its mechanism.

    Parameters
    ----------
    data : CausalData
        The composite view of the data.
    nuisance : NuisanceEstimates
        The fitted nuisances with the composite as the mechanism and no separate
        observation factor.
    bounds : tuple of float
        The bounds the composite is tilted and clipped inside.
    """

    data: CausalData
    nuisance: NuisanceEstimates
    bounds: tuple[float, float]


def composite_state(
    data: CausalData,
    nuisance: NuisanceEstimates,
    *,
    g_bounds: tuple[float, float],
    nuisance_bound: float,
    missingness: FloatArray | None = None,
) -> CompositeState:
    """The targeting inputs of the composite construction at one pair of bounds.

    Parameters
    ----------
    data : CausalData
        The prepared data, not a view.
    nuisance : NuisanceEstimates
        The fitted factors.
    g_bounds : tuple of float
        The ``g_bounds`` of the treatment factor.
    nuisance_bound : float
        The floor of each observation factor.
    missingness : ndarray or None
        A replacement outcome observation factor, or ``None``.

    Returns
    -------
    CompositeState
        The view, the composite nuisances and the composite bounds.
    """
    if data.composite_view:
        raise ValueError("composite_state takes the prepared data, and this is already a view")
    mechanism = composite_mechanism(nuisance, g_bounds, nuisance_bound, missingness=missingness)
    targeted = replace(
        nuisance,
        propensity=mechanism,
        missingness=None,
        treatment_observation=None,
    )
    return CompositeState(
        data=composite_view(data),
        nuisance=targeted,
        bounds=composite_bounds(g_bounds, nuisance_bound, _factors(nuisance, missingness)),
    )


def targeting_inputs(
    result: Any, nuisance: NuisanceEstimates | None = None
) -> tuple[CausalData, NuisanceEstimates]:
    """The data and nuisances a reader of ``result`` reads the treatment and mechanism from.

    The one accessor for the treatment, the observation indicator and the mechanism of a
    fitted point-treatment result.  On a composite fit it returns the view and the
    composite nuisances at the fit's own bounds; on every other fit it returns
    ``result.data`` and the nuisances unchanged.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.
    nuisance : NuisanceEstimates or None
        The nuisances of one repeat.  ``None`` reads ``result.nuisance``.

    Returns
    -------
    tuple
        ``(data, nuisance)``.
    """
    fitted = result.nuisance if nuisance is None else nuisance
    if result.extra.get("missing_data") != "composite":
        return result.data, fitted
    state = composite_state(
        result.data,
        fitted,
        g_bounds=result.config.g_bounds,
        nuisance_bound=result.config.missingness_bound,
    )
    return state.data, state.nuisance


_IDENTIFICATION = (
    "When the missingness of {A} may depend on {A}, P({A} = a | W) is not identified; only "
    "P({A} = a | Delta_A = 1, W) is."
)
_MEANS_REMEDY = "Request arm means or contrasts."
_DROP_REMEDY = (
    "Fit arm means, or drop the rows with a missing treatment if they are missing "
    "completely at random."
)
#: Why a learned rule stays refused: its value is identified under the composite conditions,
#: and no composite derivation is written for it.
_LEARNED_RULE_REASON = (
    "A learned-rule value is identified under the composite conditions, but no composite "
    "derivation for a rule learned from the fit is written here, and the value is "
    "cross-fitted (roadmap F27)."
)
#: Targets that read the treatment law of every row.
_UNIDENTIFIED_TARGETS = ("att", "atc", "ey_obs", "par", "paf")


def _message(data: CausalData, reason: str, remedy: str) -> CapabilityError:
    unrecorded = int(np.count_nonzero(~data.treatment_recorded))
    return CapabilityError(
        f"{data.treatment_name} is missing on {unrecorded} row(s). A missing treatment is "
        "supported for arm means and their contrasts (ey, ate, rr, or) on in-sample TMLE "
        "and DRTMLE fits, and for regime means and arm-indexed MSM coefficients on in-sample "
        "TMLE fits, through the composite indicator Delta_A * Delta * 1{A = a}. "
        f"{reason} {remedy}".rstrip()
    )


def missing_treatment_refusal(
    estimator: Any, data: CausalData, estimands: tuple[str, ...] | None
) -> CapabilityError | None:
    """The refusal of a fit with a declared missing treatment, or ``None`` if it is admitted.

    The shared preflight raises what this returns, before any learner and before any
    check that reads the arms.  It names each composition it refuses.  An estimator that
    is neither a :class:`~cleverly.TMLE` nor a :class:`~cleverly.DRTMLE` gets the generic
    reason, so the default is a refusal and not a computation on ``NaN``.

    Parameters
    ----------
    estimator : TMLE
        The estimator.
    data : CausalData
        The prepared data, with a declared missing treatment.
    estimands : tuple of str or None
        The targets the caller named, or ``None`` when the caller asked for the default
        or for ``"all"``, which drop the targets this construction cannot identify.

    Returns
    -------
    CapabilityError or None
        The refusal, or ``None``.
    """
    if not data.has_missing_treatment:
        return None
    name = data.treatment_name
    identification = _IDENTIFICATION.format(A=name)
    method = getattr(estimator, "_assessment_method", None)
    if method == "collaborative_tmle":
        return _message(
            data,
            "C-TMLE selects a treatment mechanism by a collaborative criterion, and no "
            "collaborative score is derived for the composite indicator (roadmap F5).",
            "Fit TMLE or DRTMLE.",
        )
    if method not in ("tmle", "drtmle"):
        return _message(data, "This fit is not one of those compositions.", _MEANS_REMEDY)
    if data.known_treatment is not None:
        return CapabilityError(KNOWN_TREATMENT_DELTA_REFUSAL)
    if bool(getattr(estimator, "randomized", False)):
        return _message(
            data,
            "randomized=True selects the Díaz and van der Laan (2017) construction, which "
            "observes the treatment on every row.",
            "Drop it. The composite construction estimates "
            "P(A = a, Delta_A = 1, Delta = 1 | W) from the data.",
        )
    if getattr(estimator, "cross_fit", False):
        return _message(
            data,
            "The composite construction uses the in-sample (Donsker) theorem. The "
            "cross-fitted missing-data variants are open (roadmap F21).",
            "Pass cross_fit=False (CrossFitting(enabled=False) on a method).",
        )
    if data.has_intermediate:
        return _message(
            data,
            "No derivation of the composite indicator for a controlled direct effect is "
            "written here, and the condition that ties Z to the treatment's recording is not "
            "stated.",
            _DROP_REMEDY,
        )
    for keyword in ("incremental", "policies"):
        if getattr(estimator, keyword, None):
            return _message(data, f"{identification} {keyword}= reads P({name} | W).", _DROP_REMEDY)
    if getattr(estimator, "learned_rule", None) is not None:
        return _message(data, _LEARNED_RULE_REASON, _DROP_REMEDY)
    if method == "drtmle" and (
        getattr(estimator, "interventions", None) or getattr(estimator, "msm", None)
    ):
        # DRTMLE refuses both axes on complete data too; its own message names the reason.
        return None
    named = [target for target in (estimands or ()) if target in _UNIDENTIFIED_TARGETS]
    if named:
        return _message(
            data,
            f"{identification} {', '.join(named)} read the treatment law of every row.",
            _MEANS_REMEDY,
        )
    if (
        method == "drtmle"
        and getattr(estimator, "guard", ())
        and (
            getattr(estimator, "evaluation", None) is not None
            or getattr(estimator, "reduced_crossfit", "pooled") != "pooled"
        )
    ):
        return _message(data, COMPOSITE_NESTED_REFUSAL, "")
    return None


def missing_treatment_design_refusal(
    data: CausalData, *, target: str, axis: str, intermediate: bool
) -> CapabilityError | None:
    """The refusal of a study design with a declared missing treatment, or ``None``.

    :meth:`~cleverly.CausalStudy.identify` raises what this returns, so a design-level
    request meets the reason a fit of the same request meets in
    :func:`missing_treatment_refusal`.

    Parameters
    ----------
    data : CausalData
        The prepared data, with a declared missing treatment.
    target : str
        The registered target the estimand resolves to.
    axis : str
        The target's parameter axis.
    intermediate : bool
        Whether the design declares an intermediate variable.

    Returns
    -------
    CapabilityError or None
        The refusal, or ``None``.
    """
    if not data.has_missing_treatment:
        return None
    if data.known_treatment is not None:
        return CapabilityError(KNOWN_TREATMENT_DELTA_REFUSAL)
    identification = _IDENTIFICATION.format(A=data.treatment_name)
    if intermediate:
        return _message(
            data,
            "No derivation of the composite indicator for a controlled direct effect is "
            "written here, and the condition that ties Z to the treatment's recording is not "
            "stated.",
            _DROP_REMEDY,
        )
    if axis == "learned_rule":
        return _message(data, _LEARNED_RULE_REASON, _DROP_REMEDY)
    if axis not in ("arm", "regime", "msm"):
        return _message(data, f"{identification} A {axis} target reads P(A | W).", _DROP_REMEDY)
    if target in _UNIDENTIFIED_TARGETS:
        return _message(
            data, f"{identification} {target} reads the treatment law of every row.", _MEANS_REMEDY
        )
    return None


def unidentified_targets() -> tuple[str, ...]:
    """The targets a missing treatment leaves unidentified, in a stable order.

    Returns
    -------
    tuple of str
        ``att``, ``atc``, ``ey_obs``, ``par`` and ``paf``.
    """
    return _UNIDENTIFIED_TARGETS
