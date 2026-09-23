r"""Omitted-variable-bias sensitivity analysis.

Positivity diagnostics tell you whether the data can support the estimate.  This
module answers a different question: *how strong would an unmeasured confounder
have to be to overturn the conclusion?*

Following Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2026), the bias from
omitting a confounder is bounded by a product of three interpretable pieces:

.. math::

    |\mathrm{bias}| \le |\rho| \sqrt{\frac{c_D^2}{1 - c_D^2}}\; c_Y\;
                       \underbrace{\sqrt{\sigma^2 \nu^2}}_{\text{max bias}}

where, writing :math:`\alpha` for the Riesz representer of the target parameter,

* :math:`\sigma^2 = E[(Y - \bar Q(A, W))^2]` -- residual outcome variance,
* :math:`\nu^2 = E[\alpha(A, W)^2]` -- how hard the estimand has to work to
  extrapolate (it is large exactly when overlap is poor),
* ``cf_y`` -- the share of *residual* outcome variance the confounder would explain,
* ``cf_d`` -- the corresponding gain in the Riesz representer, i.e. how much the
  confounder would improve prediction of treatment,
* ``rho`` -- how adversarially aligned those two are; ``rho = 1`` is the worst case.

Two things make this useful rather than merely formal.  First, the *robustness
value* :func:`robustness_value` inverts the bound: it reports the single number
``cf_y = cf_d = RV`` at which the conclusion would flip, so there is no need to
guess sensitivity parameters at all.  Second, :func:`benchmark` calibrates
``cf_y``/``cf_d`` against covariates you *did* measure -- "a confounder as strong as
age" is a claim a reader can evaluate, where "cf_y = 0.03" is not.

The parameterisation, including the definition of the benchmark gain statistics,
matches DoubleML's ``sensitivity_analysis`` so numbers are comparable across the two
libraries.

Scope: the bound applies to the linear functionals this library estimates -- the
counterfactual means, their contrasts against the reference arm, and the two conditional
effects.  With more than two arms those are named for the arms they are about, so the
estimand to ask for is ``"ate[medium vs low]"`` rather than ``"ate"``; it is one bound
per contrast, because :math:`\nu^2` is the second moment of *that contrast's* Riesz
representer.  Ratios are not linear functionals of the outcome regression, so use
:mod:`cleverly.sensitivity.evalue` for those.

Scope of the refusals: :data:`_FIT_WIDE_BOUND_RULES` refuses the fits below, for four
reasons.

* A DR-TMLE fit and a collaborative TMLE fit break a premise.  The published ``nu^2``
  estimate and its standard error assume a consistently estimated treatment mechanism.
* A fit with a response mechanism, a fit with an intermediate variable, and a fit indexed
  by a regime, a shift, or an MSM term are well posed.  Theorem 2 of Chernozhukov et al.
  covers each one as a linear functional of the outcome regression, and no implementation
  is registered.
* An ``ipsi`` fit is outside that theorem, because its estimand contains the treatment
  mechanism.
* A longitudinal fit has no registered derivation, and a median-combined repeated fit has
  no influence function for the median bound.

Every entry point in this module reaches those rules through :func:`sensitivity_elements`,
and :class:`~cleverly.assessment.SensitivityFacade` declares the same reason on the
matching capability rows, so a fit the bound refuses is never advertised as available.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy import optimize, stats

from .._inference_status import InferenceStatus, supplies_inference
from .._typing import FloatArray
from ..assessment import SENSITIVITY_ROUTES
from ..estimators.targeting import build_submodel
from ..exceptions import CapabilityError, refuse_inference, repeats_refusal
from ..inference.cluster import influence_variance
from ..inference.influence import spread_name
from ..targets import parameter_stem
from ..targets.population_intervention import is_natural_course_fit
from ..utils.bounds import g_bounds_for
from ..utils.random import resolve_assessment_seed
from ..utils.text import format_table
from ._parameters import ArmParameter, arm_parameters, stratum_refusal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..estimators._nuisance import RepeatFit
    from ..estimators.base import TMLEResult

__all__ = [
    "BenchmarkResult",
    "SensitivityBounds",
    "SensitivityElements",
    "benchmark",
    "omitted_variable_bounds",
    "robustness_value",
    "sensitivity_elements",
]

#: Targets for which the Riesz representer -- and therefore this bound -- is defined.
#:
#: Stems rather than reported names, since a multi-valued treatment names each parameter
#: for the arms it is about: ``ate[medium vs low]`` has a stem of ``ate`` and a
#: representer of its own.  :func:`~cleverly.sensitivity._parameters.arm_parameters` is
#: what turns a fit's arms into the names these stems produce.
LINEAR_ESTIMANDS: frozenset[str] = frozenset({"ate", "ey", "ey1", "ey0", "att", "atc"})

#: Accepted values of ``nu2_estimator``, in one place so the five docstrings that list
#: them and the refusal that rejects the rest cannot drift apart.
#: ``tests/unit/test_omitted_variable_refusals.py`` parses the ``{...}`` literal out of
#: every one of those docstrings and compares it with this tuple.
NU2_ESTIMATORS: tuple[str, ...] = ("auto", "doubly_robust", "plugin")

#: The capability rows :class:`~cleverly.assessment.SensitivityFacade` fills from this
#: module's refusal, named once so the tests that check those rows read the set rather
#: than respell it.  ``benchmark`` shares the refusal on every point fit.  On a
#: longitudinal fit it names its own missing derivation instead.  Read off
#: :data:`~cleverly.assessment.SENSITIVITY_ROUTES` rather than listed here, because that
#: table already says which operations this module answers, and a second list is the
#: registry its comment warns against.
OMITTED_VARIABLE_OPERATIONS: tuple[str, ...] = tuple(
    operation
    for operation, route in SENSITIVITY_ROUTES.items()
    if route.module == "omitted_variable"
)

#: What each non-arm parameter axis reports, named so the refusal says which functional
#: the bound is missing rather than which one the fit lacks.  ``ipsi`` is absent on
#: purpose: an incremental intervention is refused for a different reason, and
#: :func:`_refuse_non_arm_axis` states it separately.
_AXIS_NOUNS: dict[str, str] = {
    "regime": "A regime mean",
    "shift": "A modified-policy mean",
    "msm": "A point-treatment MSM coefficient",
}

#: The longitudinal stop, byte for byte what
#: :class:`~cleverly.assessment.SensitivityFacade` published before the rule table
#: existed.  ``tests/unit/test_assessment_contract.py`` matches the declared reason
#: against the raised one, so the string is the contract rather than the sentence.
_LONGITUDINAL_BOUND_REFUSAL = "no longitudinal sensitivity derivation is registered"

#: The clause both mechanism refusals end on.  Neither estimator assumes a consistent
#: treatment mechanism, and that assumption is what the published bound's ``nu^2`` and
#: its standard error are derived under.
_NO_NU2_DERIVATION = (
    "No derivation registered here gives nu^2, or the bound's standard error, for an "
    "estimator that does not assume a consistent treatment mechanism."
)

_DRTMLE_BOUND_REFUSAL = (
    "the omitted-variable bound has no nu^2 estimate for a 'drtmle' fit. The default "
    "estimator E[2 m(alpha_hat) - alpha_hat^2] equals nu_0^2 minus the squared error of "
    "the fitted representer, by the Riesz identity, so it falls exactly where the fitted "
    "mechanism is wrong -- the case DR-TMLE guards against. The bound would be too "
    "narrow and the robustness value too large. " + _NO_NU2_DERIVATION
)

_CTMLE_BOUND_REFUSAL = (
    "the omitted-variable bound has no nu^2 estimate for a 'collaborative_tmle' fit. The "
    "working mechanism conditions on a function V of W: the selected adjustment set W_S "
    "on the greedy, ordered and discrete paths, and the fitted outcome regression under "
    "'oat'. Where the working mechanism is P(A | V) in the limit, the representer is "
    "E[alpha_W | A, V], whose second moment cannot exceed the second moment of alpha_W, "
    "while sigma^2 still comes from a regression on every "
    "declared covariate. The product sigma^2 nu^2 belongs to no single conditioning set, "
    "and the collaborative robustness value is optimistic by construction. " + _NO_NU2_DERIVATION
)

#: The clause the two well-posed mechanism refusals share: each functional is linear in
#: an outcome regression, which is the hypothesis of the general bound.
_THEOREM_2_COVERS = (
    "so Theorem 2 of Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2026) covers it, "
    "and the bound is well posed here."
)

_RESPONSE_BOUND_REFUSAL = (
    "the omitted-variable bound is not implemented for a fit with a response mechanism. "
    "The identified mean is a linear functional of the regression of Delta Y on "
    "(A, Delta, W), " + _THEOREM_2_COVERS + " Three pieces are missing. The implemented "
    "representer omits the response indicator Delta. sigma^2 averages the respondents, "
    "and the theorem needs E[Delta (Y - Qbar)^2] over every row. cf_d would measure a "
    "joint strength over the treatment and response mechanisms, and this package reports "
    "it as a treatment-side strength."
)

_INTERMEDIATE_BOUND_REFUSAL = (
    "the omitted-variable bound is not implemented for a fit with an intermediate "
    "variable. Each estimand at the fixed level z is a linear functional of the "
    "regression of Y on (A, Z, W), " + _THEOREM_2_COVERS + " The representer carries the "
    "intermediate weight 1{Z = z} / P(Z = z | A, W), so cf_d would measure a joint "
    "strength over the treatment and intermediate mechanisms. This package reports cf_d "
    "and the robustness value as a treatment-side strength."
)

#: Why the median of several fold draws has no bound, in the words
#: :func:`~cleverly.exceptions.repeats_refusal` completes.
_REPEATS_REASON = (
    "A coordinatewise median of the bound's influence terms would not be the influence "
    "function of the median bound. Fit one split for this analysis."
)

#: Appended to the response refusal on a fit that can still run the tilt.  A
#: natural-course fit cannot: both tilt rows are unavailable there
#: (:data:`~cleverly.targets.population_intervention.NATURAL_COURSE_TILT_REFUSAL`), so
#: the pointer would send the reader to a second refusal.
_RESPONSE_TILT_POINTER = (
    " The missingness tilt remains the sensitivity analysis for response: call "
    "sensitivity.missingness() or sensitivity.tipping_gamma()."
)


def _refuse_longitudinal(result: Any) -> str | None:
    """Refuse any assessment family but ``point``.

    First in the table, and not by taste: :class:`~cleverly.longitudinal.LongitudinalData`
    declares no ``has_missing_outcome``, so a later rule would raise ``AttributeError`` on
    a longitudinal result instead of refusing it.
    """
    if getattr(result, "assessment_family", None) != "point":
        return _LONGITUDINAL_BOUND_REFUSAL
    return None


def _refuse_guarded_mechanism(result: Any) -> str | None:
    """Refuse a DR-TMLE fit, whose estimator does not assume a consistent mechanism."""
    if result.fitted_method == "drtmle":
        return _DRTMLE_BOUND_REFUSAL
    return None


def _refuse_selected_mechanism(result: Any) -> str | None:
    """Refuse a collaborative fit, whose representer comes from a selected working g."""
    if result.fitted_method == "collaborative_tmle":
        return _CTMLE_BOUND_REFUSAL
    return None


def _refuse_response_mechanism(result: Any) -> str | None:
    """Refuse a fit whose outcome is unobserved on some rows.

    The predicate is the data flag ``has_missing_outcome``, which the missingness-tilt
    rows read too, so a fit this rule refuses is the fit those rows offer the tilt on.
    It reads the data rather than a fitted missingness nuisance on one repeat, because a
    response mechanism is a property of the identified functional and survives a
    replacement of the stored nuisances.  The sibling surface
    :mod:`~cleverly.sensitivity._simulated_confounding_request` reads the same flag for
    the same boundary.  ``result.data`` has no default here, so a result without
    point-treatment data raises ``AttributeError`` rather than reporting no mechanism.
    """
    if not result.data.has_missing_outcome:
        return None
    if is_natural_course_fit(result):
        return _RESPONSE_BOUND_REFUSAL
    return _RESPONSE_BOUND_REFUSAL + _RESPONSE_TILT_POINTER


def _refuse_intermediate(result: Any) -> str | None:
    """Refuse a fit that declares an intermediate variable.

    The predicate is the one :mod:`~cleverly.sensitivity._simulated_confounding_request`
    uses for the same boundary, so the two surfaces refuse the same fits.  Without this
    rule the bound returns a number, and nothing in it says that ``cf_d`` has changed
    meaning.
    """
    if result.data.has_intermediate or result.intermediate_value is not None:
        return _INTERMEDIATE_BOUND_REFUSAL
    return None


def _refuse_non_arm_axis(result: Any) -> str | None:
    """Refuse a fit whose counterfactuals are indexed by anything but an arm.

    Two reasons, not one.  A regime, a shift, and an MSM coefficient are linear
    functionals of the outcome regression and each has a Riesz representer, so the bound
    is well posed and unimplemented.  An incremental intervention builds its density out
    of the estimated mechanism, so the mechanism is part of the estimand rather than a
    nuisance the bound conditions on, and the regression-only bound does not cover it.
    """
    axis = result.config.parameter_axis
    if axis == "arm":
        return None
    if axis == "ipsi":
        return (
            "the omitted-variable bound does not cover a fit whose parameters are indexed "
            "by 'ipsi'. An incremental intervention tilts the treatment mechanism, so the "
            "mechanism is part of the estimand rather than a nuisance the bound conditions "
            "on. Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2026) bound the bias "
            "of a linear functional of the outcome regression alone."
        )
    noun = _AXIS_NOUNS[axis]
    return (
        f"the omitted-variable bound is not implemented for a fit whose parameters are "
        f"indexed by {axis!r}. {noun} is a linear functional of the outcome regression, "
        f"and it has a Riesz representer, so the bound is well posed here and no "
        f"implementation of it is registered. The implemented bound covers the "
        f"arm-indexed linear estimands {sorted(LINEAR_ESTIMANDS)}."
    )


def _refuse_repeats(result: Any) -> str | None:
    """Refuse a median-combined repeated fit.

    The one rule a refit of the same estimator lifts, which is why it runs last.  Every
    rule above it refuses the fit at any split count, so a reader told to fit one split
    meets no second refusal after the refit.
    """
    return repeats_refusal(
        result.n_repeats, operation="omitted-variable sensitivity", reason=_REPEATS_REASON
    )


#: Every omitted-variable refusal that the requested estimand cannot change, in the one
#: order the five entry points and the capability rows both use.  The table is ordered and
#: each rule assumes its predecessors returned ``None``: ``longitudinal`` establishes that
#: the result carries a point-treatment ``data`` and ``config`` at all, which every rule
#: after it reads.  ``repeats`` is last because it is the only rule a refit with one split
#: lifts.  The names are the introspection contract; a test reads them to pin the order
#: without respelling a message.
_FIT_WIDE_BOUND_RULES: tuple[tuple[str, Callable[[Any], str | None]], ...] = (
    ("longitudinal", _refuse_longitudinal),
    ("drtmle", _refuse_guarded_mechanism),
    ("collaborative_tmle", _refuse_selected_mechanism),
    ("response_mechanism", _refuse_response_mechanism),
    ("intermediate", _refuse_intermediate),
    ("parameter_axis", _refuse_non_arm_axis),
    ("repeats", _refuse_repeats),
)


def fit_wide_bound_refusal(result: Any) -> str | None:
    """Return the first omitted-variable refusal that applies to the whole fit.

    Every boundary in :data:`_FIT_WIDE_BOUND_RULES` is reachable from capability reporting
    and from execution, and the requested estimand cannot change its verdict.  One helper
    answers both callers, so a fit the bound refuses is never advertised as available.
    Estimand-specific checks stay in :func:`resolve_parameter`.

    Parameters
    ----------
    result : Any
        Fitted result inspected by this module or by its assessment facade.

    Returns
    -------
    str or None
        Exact refusal reason, or ``None`` when no fit-wide boundary applies.
    """
    for _name, rule in _FIT_WIDE_BOUND_RULES:
        reason = rule(result)
        if reason is not None:
            return reason
    return None


@dataclass(frozen=True)
class SensitivityElements:
    r"""The ingredients of the bias bound for one estimand.

    Parameters
    ----------
    estimand : str
        Alias of the estimand these elements describe.
    sigma2 : float
        :math:`E[(Y - \bar Q^*(A, W))^2]`, the residual outcome variance.
    nu2 : float
        :math:`E[\alpha(A, W)^2]`, the second moment of the Riesz representer.
    max_bias : float
        :math:`\sqrt{\sigma^2 \nu^2}` -- the largest bias any confounder could produce
        if it explained *all* the residual variation on both sides.
    psi_sigma2 : ndarray
        Influence curve of ``sigma2``.
    psi_nu2 : ndarray
        Influence curve of ``nu2``.
    psi_max_bias : ndarray
        Influence curve of ``max_bias``, so the bias-adjusted bounds get confidence
        intervals rather than being treated as known constants.

    riesz_representer : ndarray
        ``(n,)`` values of :math:`\\alpha(A, W)` for the targeted functional.
    nu2_estimator : str
        Which estimator of ``nu2`` produced these values.
    """

    estimand: str
    sigma2: float
    nu2: float
    max_bias: float
    psi_sigma2: FloatArray
    psi_nu2: FloatArray
    psi_max_bias: FloatArray
    riesz_representer: FloatArray
    nu2_estimator: str


def sensitivity_elements(
    result: TMLEResult,
    estimand: str = "ate",
    *,
    nu2_estimator: str = "auto",
) -> SensitivityElements:
    r"""Compute :math:`\sigma^2`, :math:`\nu^2` and the maximal bias for one estimand.

    This is where every entry point in this module meets
    :data:`_FIT_WIDE_BOUND_RULES`, so :func:`omitted_variable_bounds`,
    :func:`benchmark`, :func:`robustness_value`, and :func:`contour_data` share one
    refusal and none of them pays a learner before hearing it.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.
    estimand : str
        Alias to build the elements for.
    nu2_estimator : {"auto", "doubly_robust", "plugin"}
        ``"doubly_robust"`` uses :math:`E[2 m(O, \alpha) - \alpha^2]`, which is less
        sensitive to error in the estimated propensity than the plug-in
        :math:`E[\alpha^2]` (``"plugin"``).  Both are consistent; ``"auto"`` picks the
        doubly robust form wherever the functional's :math:`m(O, \alpha)` has a closed
        form, which is all of :data:`LINEAR_ESTIMANDS`.

    Returns
    -------
    SensitivityElements
        The residual outcome variance, the Riesz second moment, the maximal bias, and
        the influence curve of each.

    Raises
    ------
    ValueError
        If ``nu2_estimator`` is not one of :data:`NU2_ESTIMATORS`.  The check runs
        before any refusal and before any computation.
    CapabilityError
        If a rule of :data:`_FIT_WIDE_BOUND_RULES` refuses the fit, if the fit reports
        no parameter the bound applies to under ``estimand``, or if the doubly robust
        estimate of :math:`\nu^2` is not positive.
    """
    _resolved_nu2_estimator(nu2_estimator)
    refusal = fit_wide_bound_refusal(result)
    if refusal is not None:
        raise CapabilityError(refusal)
    parameter = resolve_parameter(result, estimand)
    return _elements_for(result, result.repeats[0], parameter, nu2_estimator)


def _resolved_nu2_estimator(nu2_estimator: str) -> str:
    """The estimator ``nu2_estimator`` names, with ``"auto"`` resolved, or a ``ValueError``."""
    if nu2_estimator not in NU2_ESTIMATORS:
        raise ValueError(
            f"nu2_estimator must be one of {list(NU2_ESTIMATORS)}; got {nu2_estimator!r}"
        )
    return "doubly_robust" if nu2_estimator == "auto" else nu2_estimator


def resolve_parameter(result: TMLEResult, estimand: str) -> ArmParameter:
    """The arms a requested estimand is about, or a refusal that says why not.

    One bound is one linear functional, so this is where "which contrast" is decided --
    ``ate`` on a two-armed fit and ``ate[medium vs low]`` on a wider one, each with its
    own Riesz representer.  It reads the request alone.  The caller has already passed
    the fit through :func:`fit_wide_bound_refusal`, whose ``parameter_axis`` rule makes
    the fit arm-indexed, since no estimand name can change that verdict.  The order of
    the checks below matters.  A name this bound could never apply to is refused first,
    so that asking for a risk ratio is not reported as a missing estimand; the
    arm-indexed coverage message comes last, where it describes an arm-indexed fit and
    nothing else.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.
    estimand : str
        Alias the caller asked for.

    Returns
    -------
    ArmParameter
        The reported parameter and the arms it is a functional of.
    """
    stem = parameter_stem(estimand)
    if stem not in LINEAR_ESTIMANDS:
        # The pointer is for the two ratio scales an E-value is defined on. Offered for
        # anything else -- an attributable fraction, say -- it sends the reader from one
        # refusal to a second one.
        pointer = (
            " For a risk ratio or odds ratio use sensitivity.evalue()."
            if stem in {"rr", "or"}
            else ""
        )
        raise CapabilityError(
            f"the omitted-variable bound applies to {sorted(LINEAR_ESTIMANDS)}, not "
            f"{estimand!r}: the bound is on the bias of a linear functional of the "
            f"outcome regression, and {estimand!r} is not one." + pointer
        )
    known = arm_parameters(result)
    available = {name: parameter for name, parameter in known.items() if name in result.estimates}
    if estimand in available:
        return available[estimand]
    # Before the coverage message: this one *was* reported, so "not requested" would be
    # false. It is the derivation that is missing, not the parameter.
    conditional = stratum_refusal(result, estimand, "the omitted-variable bound")
    if conditional is not None:
        raise CapabilityError(conditional)
    if not available:
        # Reached only on an arm-indexed fit, which the fit-wide axis rule has already
        # established. A fit reporting only ``par``, ``paf``, ``rr``, ``or`` or ``ey_obs``
        # is indexed by arms and reports no linear contrast, so a sentence about
        # counterfactuals that are not arms would contradict itself.
        raise CapabilityError(
            "the omitted-variable bound applies to the arm-indexed linear estimands "
            f"{sorted(LINEAR_ESTIMANDS)}, and this arm-indexed fit reported none of them: "
            f"it reported {sorted(result.estimates)}. One bound is the second moment of "
            "one contrast's own Riesz representer, so ask the fit for a counterfactual "
            "mean or a contrast of arms."
        )
    raise CapabilityError(
        f"estimand {estimand!r} was not requested in this fit. The bound is available for "
        f"{sorted(available)} -- one per contrast, since nu^2 is the second moment of "
        "that contrast's own Riesz representer."
    )


def _elements_for(
    result: TMLEResult,
    repeat: RepeatFit,
    parameter: ArmParameter,
    nu2_estimator: str,
) -> SensitivityElements:
    """The bound's pieces under one cross-fitting draw.

    Takes the targeted ``Qbar`` and the mechanism from the same
    :class:`~cleverly.estimators._nuisance.RepeatFit`, which is what makes ``sigma2`` the
    residual variance of the regression whose propensity ``nu2`` was computed from.
    """
    data = result.data
    scaler = repeat.nuisance.scaler
    group = parameter.group
    fluctuation = repeat.fluctuations[group]
    bounds = g_bounds_for(group, result.config.g_bounds, result.config.g_bounds_conditional)
    reference = result.config.reference_arm
    submodel = build_submodel(
        data,
        repeat.nuisance,
        group,
        bounds=bounds,
        nuisance_bound=result.config.missingness_bound,
        intermediate_value=result.intermediate_value,
        # The conditional-effect fluctuations contrast against the arm this fit declared,
        # so the covariate rebuilt here must be the one it was targeted with.
        reference=reference,
    )
    # The weight of the arm the estimand *conditions on*: ``_m_alpha`` multiplies the
    # contrast by 1{A = c} / P(A = c), the observed membership of that arm over its share,
    # and never by a fitted g_c. ``None`` for a mean or an unconditional contrast, which
    # reweight nobody -- and the arm is read off the parameter rather than assumed to be
    # the other one, since with K arms ``att[medium vs low]`` and ``att[high vs low]``
    # condition on different populations.
    conditioning = parameter.conditions_on
    conditioning_indicator: FloatArray | None = None
    conditioning_share: float | None = None
    if conditioning is not None:
        conditioning_indicator = np.asarray(data.treatment == conditioning, dtype=float)
        conditioning_share = float(data.arm_fractions[repeat.nuisance.arms.index(conditioning)])

    # sigma^2: residual variance of the targeted outcome regression, on the original
    # outcome scale so the bound is reported in the units the estimate uses.
    scaled = scaler.scale(data.outcome)
    residual = np.where(data.observed, scaled - fluctuation.targeted.observed, 0.0)
    if not scaler.is_identity:
        residual = residual * scaler.range
    weights = data.weights
    sigma2_element = residual**2
    sigma2 = float(np.average(sigma2_element[data.observed], weights=weights[data.observed]))
    psi_sigma2 = np.where(data.observed, sigma2_element - sigma2, 0.0) * weights

    representer = _riesz_representer(parameter, submodel)

    method = _resolved_nu2_estimator(nu2_estimator)
    if method == "doubly_robust":
        m_alpha = _m_alpha(parameter, submodel, conditioning_indicator, conditioning_share)
        nu2_element = 2.0 * m_alpha - representer**2
        nu2 = float(np.average(nu2_element, weights=weights))
        if nu2 <= 0:
            # A refusal rather than the plug-in value. By the Riesz identity this
            # estimator is nu_0^2 minus the squared error of the fitted representer, so a
            # nonpositive value reports a mechanism fit further from the truth than the
            # representer is large. The plug-in E[alpha_hat^2] is built from that same
            # fitted representer, so substituting it reports a second moment of the wrong
            # function under the name of the right one.
            # ``CapabilityError`` rather than ``ValueError``: ``run_all`` catches only the
            # refusal type, and a bare ``ValueError`` would abort the whole battery.
            raise CapabilityError(
                f"the {method!r} estimator of nu^2 returned {nu2:.6g} for "
                f"{parameter.name!r}, and a second moment cannot be negative. "
                f"E[2 m(alpha_hat) - alpha_hat^2] equals nu_0^2 minus the squared error of "
                f"the fitted representer, so this value reports a treatment mechanism the "
                f"bound's derivation does not cover. The plug-in E[alpha_hat^2] squares "
                f"that same fitted representer, so it is not a substitute. "
                f"(nu2_estimator={nu2_estimator!r} resolved to {method!r}.)"
            )
    else:
        nu2_element = representer**2
        nu2 = float(np.average(nu2_element, weights=weights))
    psi_nu2 = (nu2_element - nu2) * weights

    max_bias = float(np.sqrt(sigma2 * nu2))
    psi_max_bias = (sigma2 * psi_nu2 + nu2 * psi_sigma2) / (2.0 * max_bias)
    return SensitivityElements(
        estimand=parameter.name,
        sigma2=sigma2,
        nu2=nu2,
        max_bias=max_bias,
        psi_sigma2=psi_sigma2,
        psi_nu2=psi_nu2,
        psi_max_bias=psi_max_bias,
        riesz_representer=representer,
        nu2_estimator=method,
    )


def _riesz_representer(parameter: ArmParameter, submodel: Any) -> FloatArray:
    r"""``alpha(A, W)`` for the requested estimand.

    The clever covariate *is* the Riesz representer -- that is why the same object
    both drives the targeting step and controls how much an omitted confounder can
    move the estimate.  Poor overlap inflates both simultaneously.

    Every column is reached by the arm it carries rather than by position, so
    ``ate[high vs low]`` reads the two arms it names out of a ``K``-column covariate
    instead of the two an implementation that counted to two would have found.
    """
    if parameter.group in ("att", "atc"):
        # One column per non-reference arm, and this parameter's is the one carrying the
        # contrast it is named for.
        return submodel.contrast_column_for(parameter.arm)
    if parameter.versus is None:
        return submodel.column_for(parameter.arm)
    return np.asarray(
        submodel.column_for(parameter.arm) - submodel.column_for(parameter.versus), dtype=float
    )


def _m_alpha(
    parameter: ArmParameter,
    submodel: Any,
    conditioning_indicator: FloatArray | None,
    conditioning_share: float | None,
) -> FloatArray:
    r"""The target functional applied to the Riesz representer, ``m(O, alpha)``.

    Used by the doubly robust estimator of :math:`\nu^2`, which relies on the Riesz
    identity :math:`E[m(O, \alpha)] = E[\alpha_0 \alpha]` for every :math:`\alpha`.  Put
    :math:`\hat\alpha` in it and :math:`E[2 m(O, \hat\alpha) - \hat\alpha^2]` equals
    :math:`\nu_0^2 - E[(\hat\alpha - \alpha_0)^2]`, which is what makes the estimator
    first-order insensitive to the fitted mechanism.

    The identity holds only when ``m`` is the functional's own score, with no fitted
    nuisance in it.  An ATT averages the contrast over the units that received the
    conditioning arm ``c``, so its score weights the contrast by the observed
    :math:`1\{A = c\} / P(A = c)`.  That is Example 2 of the Online Appendix of
    Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2026), with
    :math:`\omega = D / P(D = 1)`, and the score of their Theorem 5(2).  DoubleML's
    ``DoubleMLIRM._sensitivity_element_est`` writes the same score, expanded, as
    :math:`D / (p^2 (1 - \hat m))`.  A fitted :math:`\hat g_c(W) / P(A = c)` in that
    place has the right mean only at :math:`\hat g = g_0`.  Anywhere else the identity
    fails, the estimator loses its insensitivity to the mechanism, and it can exceed
    :math:`\nu_0^2`.

    ``conditioning_indicator`` and ``conditioning_share`` belong to the arm the estimand
    conditions on -- the contrast arm for an ATT and the reference for an ATC -- rather
    than to arm 1 and its complement.  With two arms and the default reference those are
    the same two numbers; naming the arm is what keeps them right when a binary fit
    declares the other one, and what gives each of ``K - 1`` contrasts its own
    population.  They are ``None`` for a mean or an unconditional contrast, which
    reweight nobody.
    """
    # ``arms[a][:, c]`` is the covariate at arm ``a`` in the column targeting arm ``c``:
    # the mean submodel has one column per arm, so both indices are arm levels and
    # neither is a positional 0 or 1.
    if parameter.group == "mean":
        columns = submodel.arm_columns
        at = parameter.arm
        if parameter.versus is None:
            return np.asarray(submodel.arms[at][:, columns[at]], dtype=float)
        versus = parameter.versus
        first, second = submodel.arms[at], submodel.arms[versus]
        return np.asarray(
            first[:, columns[at]]
            - first[:, columns[versus]]
            - (second[:, columns[at]] - second[:, columns[versus]]),
            dtype=float,
        )

    # ATT / ATC: the functional carries the observed arm-membership weight, since it
    # averages the contrast over the conditioning arm's subpopulation rather than over
    # everyone. The column is read by the arm whose contrast it carries, not as a literal 0.
    assert parameter.versus is not None
    assert conditioning_indicator is not None and conditioning_share is not None
    column = submodel.contrast_columns[parameter.arm]
    difference = np.asarray(
        submodel.arms[parameter.arm][:, column] - submodel.arms[parameter.versus][:, column],
        dtype=float,
    )
    return np.asarray((conditioning_indicator / conditioning_share) * difference, dtype=float)


@dataclass(frozen=True, init=False)
class SensitivityBounds:
    """Bias-adjusted bounds under an assumed confounder strength.

    Parameters
    ----------
    estimand : str
        Alias of the estimand these bounds describe.
    psi : float
        The unadjusted point estimate.
    cf_y : float
        Share of the residual outcome variation the assumed confounder explains.
    cf_d : float
        Share of the residual treatment variation the assumed confounder explains.
    rho : float
        How adversarially the two are aligned. ``1.0`` is the worst case.
    confounding_strength : float
        The product those three imply.
    max_bias : float
        Largest bias a confounder of that strength could produce.
    lower : float
        Bias-adjusted lower bound on the estimate.
    upper : float
        Bias-adjusted upper bound on the estimate.
    ci_lower : float
        Lower one-sided confidence limit of the adjusted bound. It refuses at a
        non-inferential status; :attr:`plugin_interval_lower` keeps the diagnostic.
    ci_upper : float
        Upper one-sided confidence limit of the adjusted bound. It follows ``ci_lower``.
    level : float
        Coverage level of those limits.
    robustness_value : float
        Confounding strength that would move the point estimate to the null.
    robustness_value_ci : float
        The same strength for the confidence limit. It refuses with ``ci_lower``;
        :attr:`robustness_value_plugin_interval` keeps the diagnostic.
    null_hypothesis : float
        The value the robustness values are measured against.
    inference : str, default="influence_curve"
        The inference status of the estimate the bound adjusts. One of
        :data:`~cleverly.inference.influence.InferenceStatus`, which the
        :doc:`inference reference </technical-reference/inference>` lists.
    """

    estimand: str
    psi: float
    cf_y: float
    cf_d: float
    rho: float
    confounding_strength: float
    max_bias: float
    lower: float
    upper: float
    _ci_lower: float
    _ci_upper: float
    level: float
    robustness_value: float
    _robustness_value_ci: float
    null_hypothesis: float
    #: A plain default, so a bound pickled before the field existed loads as inferential,
    #: which every such bound was: RM11 refused the bound on the only fits that carried
    #: another status.
    inference: InferenceStatus = "influence_curve"

    def __init__(
        self,
        estimand: str,
        psi: float,
        cf_y: float,
        cf_d: float,
        rho: float,
        confounding_strength: float,
        max_bias: float,
        lower: float,
        upper: float,
        ci_lower: float,
        ci_upper: float,
        level: float,
        robustness_value: float,
        robustness_value_ci: float,
        null_hypothesis: float,
        inference: InferenceStatus = "influence_curve",
    ) -> None:
        # Preserve the public constructor while guarding its inferential values.
        for name, value in (
            ("estimand", estimand),
            ("psi", psi),
            ("cf_y", cf_y),
            ("cf_d", cf_d),
            ("rho", rho),
            ("confounding_strength", confounding_strength),
            ("max_bias", max_bias),
            ("lower", lower),
            ("upper", upper),
            ("_ci_lower", ci_lower),
            ("_ci_upper", ci_upper),
            ("level", level),
            ("robustness_value", robustness_value),
            ("_robustness_value_ci", robustness_value_ci),
            ("null_hypothesis", null_hypothesis),
            ("inference", inference),
        ):
            object.__setattr__(self, name, value)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Move the public stored fields of an older pickle behind the guarded accessors."""
        restored = dict(state)
        for name in ("ci_lower", "ci_upper", "robustness_value_ci"):
            if name in restored:
                restored[f"_{name}"] = restored.pop(name)
        self.__dict__.update(restored)

    @property
    def ci_lower(self) -> float:
        """Lower confidence limit, when this bound supplies inference."""
        refuse_inference(self.inference, operation="SensitivityBounds.ci_lower")
        return self._ci_lower

    @property
    def ci_upper(self) -> float:
        """Upper confidence limit, when this bound supplies inference."""
        refuse_inference(self.inference, operation="SensitivityBounds.ci_upper")
        return self._ci_upper

    @property
    def robustness_value_ci(self) -> float:
        """Robustness value for a confidence limit, when inference is supplied."""
        refuse_inference(self.inference, operation="SensitivityBounds.robustness_value_ci")
        return self._robustness_value_ci

    @property
    def plugin_interval_lower(self) -> float:
        """Lower plug-in limit, a diagnostic at a non-inferential status."""
        return self._ci_lower

    @property
    def plugin_interval_upper(self) -> float:
        """Upper plug-in limit, a diagnostic at a non-inferential status."""
        return self._ci_upper

    @property
    def robustness_value_plugin_interval(self) -> float:
        """Robustness value from the plug-in limit, a diagnostic when inference is absent."""
        return self._robustness_value_ci

    @property
    def bias(self) -> float:
        """Magnitude of the bias the assumed confounder could produce."""
        return self.confounding_strength * self.max_bias

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation.

        At a non-inferential :attr:`inference` the mapping carries ``inference``, and
        the three quantities read off the influence curve take the names
        :func:`~cleverly.inference.influence.spread_name` gives them:
        ``plugin_interval_lower``, ``plugin_interval_upper`` and
        ``robustness_value_plugin_interval``.

        Returns
        -------
        dict
            A JSON-compatible mapping of every reported field.
        """
        status = self.inference
        row: dict[str, Any] = {"estimand": self.estimand}
        if not supplies_inference(status):
            row["inference"] = status
        row |= {
            "psi": self.psi,
            "cf_y": self.cf_y,
            "cf_d": self.cf_d,
            "rho": self.rho,
            "max_bias": self.max_bias,
            "bias": self.bias,
            "lower": self.lower,
            "upper": self.upper,
            spread_name("ci_lower", status): self.plugin_interval_lower,
            spread_name("ci_upper", status): self.plugin_interval_upper,
            "robustness_value": self.robustness_value,
            spread_name("robustness_value_ci", status): self.robustness_value_plugin_interval,
        }
        return row

    def summary(self) -> str:
        """Return a printable summary.

        Returns
        -------
        str
            A printable report, one line per reported quantity.
        """
        conclusion = (
            "the sign of the effect survives"
            if (self.lower - self.null_hypothesis) * (self.upper - self.null_hypothesis) > 0
            else "the effect could be explained away"
        )
        status = self.inference
        lower = self.plugin_interval_lower
        upper = self.plugin_interval_upper
        rv_interval = self.robustness_value_plugin_interval
        return "\n".join(
            [
                f"Omitted-variable sensitivity for {self.estimand!r}",
                "-" * 44,
                f"estimate {self.psi:.5g}; maximal bias sqrt(sigma^2 nu^2) = {self.max_bias:.5g}",
                f"assumed confounding: cf_y = {self.cf_y:.3g}, cf_d = {self.cf_d:.3g}, "
                f"rho = {self.rho:.3g}"
                f" -> bias <= {self.bias:.5g}",
                f"bias-adjusted bounds:  [{self.lower:.5g}, {self.upper:.5g}]",
                f"with {self.level:.0%} {spread_name('one-sided CIs', status)}: "
                f"[{lower:.5g}, {upper:.5g}] ({conclusion})",
                "",
                f"robustness value RV   = {self.robustness_value:.4f}: a confounder explaining "
                f"{self.robustness_value:.1%} of the residual variation in BOTH the outcome and "
                f"treatment would move the estimate to {self.null_hypothesis:g}.",
                f"robustness value RVa  = {rv_interval:.4f}: the same, for the "
                f"{self.level:.0%} {spread_name('confidence bound', status)}.",
            ]
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def _confounding_strength(cf_y: float, cf_d: float, rho: float) -> float:
    """``|rho| sqrt(cf_y cf_d / (1 - cf_d))``, the multiplier on the maximal bias."""
    for name, value in (("cf_y", cf_y), ("cf_d", cf_d)):
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must lie in [0, 1); got {value}")
    if not 0.0 <= abs(rho) <= 1.0:
        raise ValueError(f"|rho| must lie in [0, 1]; got {rho}")
    return float(abs(rho) * np.sqrt(cf_y * cf_d / (1.0 - cf_d)))


def omitted_variable_bounds(
    result: TMLEResult,
    estimand: str = "ate",
    *,
    cf_y: float = 0.03,
    cf_d: float = 0.03,
    rho: float = 1.0,
    level: float = 0.95,
    null_hypothesis: float = 0.0,
    nu2_estimator: str = "auto",
) -> SensitivityBounds:
    """Bias-adjusted bounds and robustness values for one estimand.

    Defaults follow the convention of assuming a confounder that explains 3% of the
    residual variation on each side, with worst-case alignment (``rho = 1``).  Prefer
    reading :attr:`SensitivityBounds.robustness_value`, which needs no assumption at
    all.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.
    estimand : str
        Alias to bound.
    cf_y : float
        Assumed share of residual outcome variation the confounder explains.
    cf_d : float
        Assumed share of residual treatment variation it explains.
    rho : float
        How adversarially the two are aligned. ``1.0`` is the worst case.
    level : float
        Coverage level of the reported limits.
    null_hypothesis : float
        Value the robustness values are measured against.
    nu2_estimator : {"auto", "doubly_robust", "plugin"}
        Which estimator of the Riesz second moment to use.

    Returns
    -------
    SensitivityBounds
        Adjusted bounds, their confidence limits, and the robustness values.

    Raises
    ------
    ValueError
        If ``nu2_estimator`` is not one of :data:`NU2_ESTIMATORS`, if ``cf_y`` or
        ``cf_d`` lies outside ``[0, 1)``, or if ``|rho|`` exceeds one.
    CapabilityError
        On every refusal :func:`sensitivity_elements` raises: a fit the rule table
        refuses, an estimand the bound does not cover, or a doubly robust estimate of
        ``nu^2`` that is not positive.
    """
    elements = sensitivity_elements(result, estimand, nu2_estimator=nu2_estimator)
    estimate = result[estimand]
    strength = _confounding_strength(cf_y, cf_d, rho)

    lower = estimate.psi - strength * elements.max_bias
    upper = estimate.psi + strength * elements.max_bias

    # One-sided confidence bounds on each end, accounting for uncertainty in the bias
    # term itself as well as in the estimate.
    quantile = float(stats.norm.ppf(level))
    se_lower = _bound_std_error(estimate.influence_curve, -strength * elements.psi_max_bias, result)
    se_upper = _bound_std_error(estimate.influence_curve, strength * elements.psi_max_bias, result)
    ci_lower = lower - quantile * se_lower
    ci_upper = upper + quantile * se_upper

    rv, rva = _robustness_values(result, elements, estimate.psi, rho, level, null_hypothesis)
    return SensitivityBounds(
        estimand=estimand,
        psi=estimate.psi,
        cf_y=cf_y,
        cf_d=cf_d,
        rho=rho,
        confounding_strength=strength,
        max_bias=elements.max_bias,
        lower=lower,
        upper=upper,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        level=level,
        robustness_value=rv,
        robustness_value_ci=rva,
        null_hypothesis=null_hypothesis,
        inference=estimate.inference,
    )


def _bound_std_error(psi_estimate: FloatArray, psi_bias: FloatArray, result: TMLEResult) -> float:
    """Standard error of a bias-adjusted bound."""
    combined = np.asarray(psi_estimate, dtype=float) + np.asarray(psi_bias, dtype=float)
    return float(np.sqrt(influence_variance(combined, result.data.cluster)))


def _robustness_values(
    result: TMLEResult,
    elements: SensitivityElements,
    psi: float,
    rho: float,
    level: float,
    null_hypothesis: float,
) -> tuple[float, float]:
    """Solve for ``cf_y = cf_d = v`` at which a bound reaches the null."""
    side = 1.0 if null_hypothesis > psi else -1.0
    quantile = float(stats.norm.ppf(level))

    def bound_at(value: float, *, with_ci: bool) -> float:
        strength = _confounding_strength(value, value, rho)
        bias = strength * elements.max_bias
        edge = psi + side * bias
        if not with_ci:
            return edge
        se = _bound_std_error(
            result[elements.estimand].influence_curve,
            side * strength * elements.psi_max_bias,
            result,
        )
        return edge + side * quantile * se

    def objective(value: float, with_ci: bool) -> float:
        return float((bound_at(value, with_ci=with_ci) - null_hypothesis) ** 2)

    rv = float(
        optimize.minimize_scalar(objective, bounds=(0.0, 0.9999), method="bounded", args=(False,)).x
    )
    rva = float(
        optimize.minimize_scalar(objective, bounds=(0.0, 0.9999), method="bounded", args=(True,)).x
    )
    return rv, rva


@dataclass(frozen=True)
class BenchmarkResult:
    """Confounder strengths calibrated against covariates that *were* observed.

    Interpretation: ``cf_y`` and ``cf_d`` are the sensitivity parameters an unobserved
    confounder would need in order to be "as important as" the benchmark covariates,
    measured by how much dropping those covariates degrades the outcome regression and
    the Riesz representer.  ``delta_psi`` is how much the estimate actually moved when
    they were dropped, and ``rho`` is the implied degree of adversity -- a value well
    below 1 says the worst-case ``rho = 1`` is pessimistic for confounders like these.

    Parameters
    ----------
    estimand : str
        Alias of the estimand benchmarked.
    covariates : tuple of str
        The observed covariates the strength is calibrated against.
    cf_y : float
        Share of residual outcome variation those covariates explain.
    cf_d : float
        Share of residual treatment variation those covariates explain.
    rho : float
        The implied alignment of the two.
    delta_psi : float
        How far the estimate moved when they were dropped.
    psi_long : float
        Estimate with the benchmark covariates adjusted for.
    psi_short : float
        Estimate with them dropped.
    sigma2_long : float
        Residual outcome variance with them adjusted for.
    sigma2_short : float
        Residual outcome variance with them dropped.
    nu2_long : float
        Riesz second moment with them adjusted for.
    nu2_short : float
        Riesz second moment with them dropped.
    random_state : int or None
        Seed this benchmark ran under.  Pass it back to :func:`benchmark` to obtain the
        benchmark again.  ``None`` only on a result saved before this field existed.
    """

    estimand: str
    covariates: tuple[str, ...]
    cf_y: float
    cf_d: float
    rho: float
    delta_psi: float
    psi_long: float
    psi_short: float
    sigma2_long: float
    sigma2_short: float
    nu2_long: float
    nu2_short: float
    #: Seed the short refit ran under, resolved rather than requested: an explicit seed,
    #: else the fit's own, else one drawn here.  A benchmark refits, so an estimator
    #: carrying no seed would otherwise give a different answer to the same question and
    #: cache the first one.
    random_state: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation.

        Returns
        -------
        dict
            A JSON-compatible mapping of every reported field.
        """
        return {
            "estimand": self.estimand,
            "covariates": ", ".join(self.covariates),
            "cf_y": self.cf_y,
            "cf_d": self.cf_d,
            "rho": self.rho,
            "delta_psi": self.delta_psi,
            "psi_long": self.psi_long,
            "psi_short": self.psi_short,
        }

    def summary(self) -> str:
        """Return a printable summary.

        Returns
        -------
        str
            A printable report, one line per reported quantity.
        """
        return "\n".join(
            [
                f"Benchmark for {self.estimand!r} against {list(self.covariates)}",
                "-" * 48,
                format_table(
                    ["quantity", "with covariates", "without"],
                    [
                        ["estimate", f"{self.psi_long:.5g}", f"{self.psi_short:.5g}"],
                        ["sigma^2", f"{self.sigma2_long:.5g}", f"{self.sigma2_short:.5g}"],
                        ["nu^2", f"{self.nu2_long:.5g}", f"{self.nu2_short:.5g}"],
                    ],
                ),
                "",
                f"implied cf_y = {self.cf_y:.4f}, cf_d = {self.cf_d:.4f}, rho = {self.rho:.4f}",
                f"the estimate moved by {self.delta_psi:+.5g} when these covariates were dropped",
            ]
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def benchmark(
    result: TMLEResult,
    covariates: Any,
    *,
    estimand: str = "ate",
    nu2_estimator: str = "auto",
    random_state: int | None = None,
) -> BenchmarkResult:
    """Calibrate ``cf_y`` and ``cf_d`` against observed covariates.

    Refits the whole model *without* the named covariates (the "short" model) and
    compares it with the full fit.  The resulting gain statistics say how strong a
    confounder like the dropped ones would be, on the ``cf_y``/``cf_d`` scale, which
    turns an abstract sensitivity parameter into a concrete comparison.

    Note this is a genuine refit, so it costs about as much as the original fit.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.
    covariates : sequence of str
        Observed covariates to calibrate against.
    estimand : str
        Alias to benchmark.
    nu2_estimator : {"auto", "doubly_robust", "plugin"}
        Which estimator of the Riesz second moment to use.
    random_state : int or None
        Seed for the short refit.  ``None`` uses the seed the fit was run with.  A fit
        with no seed draws one.  Either way the result records it under ``random_state``,
        and passing that value back repeats the benchmark.

    Returns
    -------
    BenchmarkResult
        The sensitivity parameters a confounder as important as those covariates
        would need, and how far the estimate moved without them.

    Raises
    ------
    ValueError
        If ``nu2_estimator`` is not one of :data:`NU2_ESTIMATORS`.
    CapabilityError
        If the result carries no fitted estimator, or on every refusal
        :func:`sensitivity_elements` raises for the full fit or the short refit.  The
        full fit is checked before the refit runs.
    """
    estimator = result.estimator
    if estimator is None:
        raise CapabilityError("benchmark needs the fitted estimator that produced the result")
    names = tuple([covariates] if isinstance(covariates, str) else covariates)

    # The short model is a refit, so it carries the same reproducibility question a
    # refutation does: an estimator with no ``random_state`` redraws its folds every time,
    # and the result is cached on the fit and survives ``save``.  Resolve one seed, run the
    # refit under it, and report it.  Same convention as ``cleverly.validation.refute``.
    seed = resolve_assessment_seed(result, random_state)

    long_elements = sensitivity_elements(result, estimand, nu2_estimator=nu2_estimator)
    short_data = result.data.without_covariates(names)
    short_result = estimator.refit(
        short_data, intermediate_value=result.intermediate_value, random_state=seed
    )
    short_elements = sensitivity_elements(short_result, estimand, nu2_estimator=nu2_estimator)

    var_y = float(np.var(result.data.outcome[result.data.observed]))
    r2_long = 1.0 - long_elements.sigma2 / var_y
    r2_short = 1.0 - short_elements.sigma2 / var_y
    r2_riesz = short_elements.nu2 / long_elements.nu2

    cf_y = float(np.clip((r2_long - r2_short) / (1.0 - r2_long), 0.0, 1.0))
    cf_d = float(np.clip((1.0 - r2_riesz) / r2_riesz, 0.0, 1.0)) if r2_riesz > 0 else 1.0

    delta = short_result[estimand].psi - result[estimand].psi
    var_g = short_elements.sigma2 - long_elements.sigma2
    var_riesz = long_elements.nu2 - short_elements.nu2
    if var_g > 0 and var_riesz > 0:
        rho = float(np.clip(abs(delta) / np.sqrt(var_g * var_riesz), 0.0, 1.0)) * float(
            np.sign(delta)
        )
    else:
        rho = float(np.sign(delta))

    return BenchmarkResult(
        estimand=estimand,
        covariates=names,
        cf_y=cf_y,
        cf_d=cf_d,
        rho=rho,
        delta_psi=delta,
        psi_long=result[estimand].psi,
        psi_short=short_result[estimand].psi,
        sigma2_long=long_elements.sigma2,
        sigma2_short=short_elements.sigma2,
        nu2_long=long_elements.nu2,
        nu2_short=short_elements.nu2,
        random_state=seed,
    )


def robustness_value(
    result: TMLEResult,
    estimand: str = "ate",
    *,
    rho: float = 1.0,
    level: float = 0.95,
    null_hypothesis: float = 0.0,
    nu2_estimator: str = "auto",
) -> dict[str, Any]:
    """The confounding strength that would explain the effect away.

    Returns ``{"rv": ..., "rva": ...}``: the value of ``cf_y = cf_d`` at which the point
    estimate reaches ``null_hypothesis``, and the value at which its confidence bound
    does.  This is the single most useful number in this module, because it requires no
    guess about how strong an unmeasured confounder might be -- it reports the
    threshold and lets the reader judge whether it is plausible.

    The confidence bound is read off the estimate's influence curve, so at a
    non-inferential status ``"rva"`` takes the name
    :func:`~cleverly.inference.influence.spread_name` gives it, ``"rv_plugin_interval"``,
    and the mapping opens with ``"inference"`` naming the status, as
    :meth:`SensitivityBounds.to_dict` does.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.
    estimand : str
        Alias to report on.
    rho : float
        How adversarially the two sensitivity parameters are aligned.
    level : float
        Coverage level used for the confidence-limit value.
    null_hypothesis : float
        Value the strength is measured against.
    nu2_estimator : {"auto", "doubly_robust", "plugin"}
        Which estimator of the Riesz second moment to use.

    Returns
    -------
    dict of str to Any
        The strength that moves the point estimate to the null, the one that moves the
        confidence limit there, and ``max_bias``. At a non-inferential status, also the
        status under ``"inference"``.

    Raises
    ------
    ValueError
        If ``nu2_estimator`` is not one of :data:`NU2_ESTIMATORS`, or if ``|rho|``
        exceeds one.
    CapabilityError
        On every refusal :func:`sensitivity_elements` raises: a fit the rule table
        refuses, an estimand the bound does not cover, or a doubly robust estimate of
        ``nu^2`` that is not positive.
    """
    elements = sensitivity_elements(result, estimand, nu2_estimator=nu2_estimator)
    estimate = result[estimand]
    rv, rva = _robustness_values(result, elements, estimate.psi, rho, level, null_hypothesis)
    status = estimate.inference
    values: dict[str, Any] = {} if supplies_inference(status) else {"inference": status}
    values |= {"rv": rv, spread_name("rva", status): rva, "max_bias": elements.max_bias}
    return values


def contour_data(
    result: TMLEResult,
    estimand: str = "ate",
    *,
    rho: float = 1.0,
    grid_size: int = 20,
    grid_bounds: tuple[float, float] = (0.15, 0.15),
    bound: str = "lower",
    nu2_estimator: str = "auto",
) -> Any:
    """A ``cf_d`` x ``cf_y`` grid of bias-adjusted bounds, for a contour plot.

    Returned as a long-format frame (``cf_d``, ``cf_y``, ``value``) rather than a plot, so
    it can be rendered with whatever plotting stack the caller already uses.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.
    estimand : str
        Alias to build the grid for.
    rho : float
        How adversarially the two sensitivity parameters are aligned.
    grid_size : int
        Points along each axis of the grid.
    grid_bounds : tuple of float
        Largest ``cf_d`` and largest ``cf_y`` the grid reaches.
    bound : {"lower", "upper"}
        Which end of the bias-adjusted interval each cell reports.
    nu2_estimator : {"auto", "doubly_robust", "plugin"}
        Which estimator of the Riesz second moment to use.

    Returns
    -------
    DataFrame
        One row per grid cell, in the frame library the fit was given, with columns
        ``cf_d``, ``cf_y``, and ``value``.

    Raises
    ------
    ValueError
        If ``bound`` is not ``"lower"`` or ``"upper"``, if ``nu2_estimator`` is not one
        of :data:`NU2_ESTIMATORS`, if a grid value lies outside ``[0, 1)``, or if
        ``|rho|`` exceeds one.
    CapabilityError
        On every refusal :func:`sensitivity_elements` raises: a fit the rule table
        refuses, an estimand the bound does not cover, or a doubly robust estimate of
        ``nu^2`` that is not positive.
    """
    if bound not in ("lower", "upper"):
        raise ValueError(f"bound must be 'lower' or 'upper'; got {bound!r}")
    elements = sensitivity_elements(result, estimand, nu2_estimator=nu2_estimator)
    psi = result[estimand].psi
    sign = -1.0 if bound == "lower" else 1.0

    cf_d_grid = np.linspace(0.0, grid_bounds[0], grid_size)
    cf_y_grid = np.linspace(0.0, grid_bounds[1], grid_size)
    rows_d, rows_y, values = [], [], []
    for cf_d in cf_d_grid:
        for cf_y in cf_y_grid:
            strength = _confounding_strength(float(cf_y), float(cf_d), rho)
            rows_d.append(float(cf_d))
            rows_y.append(float(cf_y))
            values.append(psi + sign * strength * elements.max_bias)
    return result.data.frame_like({"cf_d": rows_d, "cf_y": rows_y, "value": values})
