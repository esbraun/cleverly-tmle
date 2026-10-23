r"""Omitted-variable-bias sensitivity analysis.

Positivity diagnostics tell you whether the data can support the estimate.  This
module answers a different question: *how strong would an unmeasured confounder
have to be to overturn the conclusion?*

Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2026) bound the bias from omitting a
confounder by a product of three interpretable pieces.  Their Theorem 2 gives the bias of a
linear functional, and Equation (14) in their Section 4 gives the bounds:

.. math::

    |\mathrm{bias}| \le |\rho| \sqrt{\frac{c_D^2}{1 - c_D^2}}\; c_Y\;
                       \underbrace{\sqrt{\sigma^2 \nu^2}}_{\text{bias scale}}

where, writing :math:`\alpha` for the Riesz representer of the target parameter,

* :math:`\sigma^2 = E[(Y - \bar Q(A, W))^2]` -- residual outcome variance,
* :math:`\nu^2 = E[\alpha(A, W)^2]` -- how hard the estimand has to work to
  extrapolate (it is large exactly when overlap is poor),
* ``cf_y`` -- the share of *residual* outcome variance the confounder would explain,
* ``cf_d`` -- the share of the Riesz representer's second moment that only the
  confounder adds; for the ATE, the gain in precision of the treatment model,
* ``rho`` -- how adversarially aligned those two are; ``rho = 1`` is the worst case.

:math:`\sqrt{\sigma^2 \nu^2}` is a scale, not the largest bias: the multiplier
``|rho| sqrt(cf_y cf_d / (1 - cf_d))`` exceeds 1 for strong confounders (about 2.85 at
``cf_y = cf_d = 0.9``).  The field that holds it keeps DoubleML's name, ``max_bias``.

Two things make this useful rather than merely formal.  First, the *robustness
value* :func:`robustness_value` inverts the bound: it reports the single number
``cf_y = cf_d = RV`` at which the conclusion would flip, so there is no need to
guess sensitivity parameters at all.  Second, :func:`benchmark` calibrates
``cf_y``/``cf_d`` against covariates you *did* measure -- "a confounder as strong as
age" is a claim a reader can evaluate, where "cf_y = 0.03" is not.

The parameterisation, including the definition of the benchmark gain statistics,
follows DoubleML's ``sensitivity_analysis``, so a strength means the same thing in both
libraries.  The confidence limits of an ATT or an ATC differ: this module adds the
influence term of the conditioning share to the curve of ``nu^2``, and
``DoubleMLIRM._sensitivity_element_est`` omits it (doubleml-for-py at ``b69f86ef``).  The
estimates differ too, because DoubleML fits a cross-fitted AIPW score and this package a
TMLE.

Inference.  Each bound has one-sided limits, from Theorem 4 in Section 4 of Chernozhukov et
al.  For the ATT and the ATC the curve of the doubly robust ``nu^2`` also carries
``-2 nu^2 (1{A = c} - p) / p``, the influence of the estimated share ``p`` of the
conditioning arm ``c``.  The paper gives this kind of term for the estimate only (Online
Appendix A, "Statistical Inference", Equation (15)).  :func:`_conditioning_share_influence`
holds this package's derivation.
The plug-in estimator of ``nu^2`` squares the fitted representer, so its curve needs the score
of the fitted treatment mechanism.  Its bounds publish limits when the mechanism is declared
known, or is a scikit-learn ``LogisticRegression`` with no penalty or a fixed L2 penalty,
in sample or cross-fitted (:mod:`cleverly.sensitivity._mechanism_stack`).  For such a
mechanism the bound reports the second-order corrected :math:`\hat\nu^2 - \hat B`
(:mod:`cleverly.sensitivity._second_order`) with the influence variance of the uncorrected
curve.  For any other learner it
reports the uncorrected :math:`E_n[\hat\alpha^2]` and publishes no limits
(docs/roadmap.md F26).

Scope: the bound applies to the linear functionals this library estimates -- the
counterfactual means, their contrasts against the reference arm, and the two conditional
effects.  With more than two arms those are named for the arms they are about, so the
estimand to ask for is ``"ate[medium vs low]"`` rather than ``"ate"``; it is one bound
per contrast, because :math:`\nu^2` is the second moment of *that contrast's* Riesz
representer.  Ratios are not linear functionals of the outcome regression, so use
:mod:`cleverly.sensitivity.evalue` for those.

Second mechanisms and strata (X82 part G).  A fit with a response mechanism, a declared
missing treatment, or an intermediate variable at a level ``z`` has a representer that also
divides by that mechanism: :math:`\Delta 1\{A = a\} / (g_a \pi_a)`, the composite
:math:`C_a / g_{c,a}`, or the controlled-direct-effect covariate with :math:`1\{Z = z\}`.
The functional is linear in the regression on the rows the representer reads, so
Theorem 2 of Chernozhukov et al. covers it.  :math:`\sigma^2` is then the mean over every
row of the residual masked to those rows, and ``cf_d`` measures a joint strength over the
treatment mechanism and the second one.  A stratum-conditional parameter restricts the
representer to the stratum and divides by its share, as the ATT divides by the share of its
arm, and the curve of ``nu^2`` carries the influence of that share.  The confidence limits
of these configurations resolve through the ``ovb_limit`` rows of
:mod:`cleverly.validation._interpretation`.  The ``ovb-limit-coverage`` study registers a
row for each single-feature rule on its measured configuration, and every other fit prints
its limits as descriptive.

Repeated cross-fitting (X82 part G).  A fit with ``repeats`` above one has one bound per
draw.  The reported bounds are the medians of the draws' bounds, and each one-sided limit
combines the draws' limits by the median rule of the estimate.  Each draw is retargeted from
its own nuisances for its influence curve.  :func:`sensitivity_elements` describes one draw,
named with ``repeat``.

Scope of the refusals: :data:`_FIT_WIDE_BOUND_RULES` refuses the fits below, for five
reasons.

* A DR-TMLE fit and a collaborative TMLE fit break a premise.  The ``nu^2`` score of
  Lemma 3 of Chernozhukov et al. and the standard error of their Theorem 4 assume a
  consistently estimated treatment mechanism.
* A fit indexed by a regime, a shift, or an MSM term is well posed.  Theorem 2 of
  Chernozhukov et al. covers each one as a linear functional of the outcome regression,
  and no implementation is registered.
* An ``ipsi`` fit is outside that theorem, because its estimand contains the treatment
  mechanism.
* A longitudinal fit has no registered derivation.
* An arm-indexed fit that reports no counterfactual mean and no linear contrast has no
  parameter to bound.  A ratio-only fit, a population attributable risk or fraction, and
  a complete-outcome natural-course mean are such fits.

Every entry point in this module reaches those rules through :func:`sensitivity_elements`,
and :class:`~cleverly.assessment.SensitivityFacade` declares the same reason on the
matching capability rows, so a fit the bound refuses is never advertised as available.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ParamSpec, cast

import numpy as np
from scipy import optimize

from .._inference_status import InferenceStatus, is_labelled
from .._refusal import NO_ALTERNATIVE, Refusal, RefusalReason
from .._support import record_refusal_site
from .._typing import FloatArray
from ..assessment import SENSITIVITY_ROUTES
from ..estimators.composite import composite_state
from ..estimators.ctmle import (
    CTMLEDerivativeFit,
    CTMLESelection,
    declares_no_selection,
    is_per_arm_oat,
)
from ..estimators.direct_effect import declares_intermediate, targeted_rows
from ..estimators.iptw import OPERATION_REFUSALS
from ..estimators.targeting import build_submodel
from ..exceptions import CapabilityError, DataError, refuse_inference
from ..inference.cluster import influence_variance
from ..inference.delta import one_sided_quantile
from ..inference.influence import spread_name
from ..interventions.learned import LEARNED_RULE_BOUND_REFUSAL
from ..targets import parameter_stem
from ..utils.bounds import g_bounds_for
from ..utils.random import resolve_assessment_seed
from ..utils.text import format_table
from ._mechanism_stack import (
    MechanismRefusal,
    configuration_refusal,
    correction_refusal,
    plugin_mechanism_term,
)
from ._parameters import ArmParameter, reported_arm_parameters, stratum_parameter

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..data.causal_data import CausalData
    from ..estimators._nuisance import NuisanceEstimates, RepeatFit
    from ..estimators.base import TMLEResult
    from ..validation._interpretation import Resolution

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

#: The stems of the regime, shift and MSM axes the bound covers (X82 part G).  Each is a
#: linear functional of the outcome regression with a known Riesz representer, read off
#: the fit's own clever covariate.  :func:`axis_parameters` turns a fit's codes into names.
AXIS_ESTIMANDS: frozenset[str] = frozenset(
    {"ey_regime", "ate_regime", "ey_policy", "ate_policy", "msm"}
)

#: The fluctuation group, and the part G feature, of each covered axis.
_AXIS_GROUPS: dict[str, tuple[str, str]] = {
    "regime": ("regime", "regime"),
    "policy": ("mtp", "shift"),
    "msm": ("msm", "msm"),
}

#: The two ratio scales an E-value answers for, and the pointer a refusal appends when a
#: request or a fit names one.  Offered for anything else, such as an attributable
#: fraction, the pointer sends the reader from one refusal to a second one.
_EVALUE_STEMS: frozenset[str] = frozenset({"rr", "or"})
_EVALUE_POINTER = " For a risk ratio or odds ratio use sensitivity.evalue()."

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
    "rr_tilt": "A risk-ratio tilt mean",
}

#: The longitudinal stop. ``tests/unit/test_assessment_contract.py`` matches the declared
#: row against the raised error, and both read this one record. The citation names the
#: research-queue row that holds the missing law.
_LONGITUDINAL_BOUND_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "no longitudinal sensitivity derivation is registered: the bound needs a per-period "
    "latent law that identifies the bias of a sequential g-formula target "
    "(docs/roadmap.md F16 tracks this stop).",
    alternative=NO_ALTERNATIVE,
    roadmap="F16",
)

#: The clause both mechanism refusals end on.  Neither estimator assumes a consistent
#: treatment mechanism, and that assumption is what the published bound's ``nu^2`` and
#: its standard error are derived under.
_NO_NU2_DERIVATION = (
    "No derivation registered here gives nu^2, or the bound's standard error, for an "
    "estimator that does not assume a consistent treatment mechanism."
)

_DRTMLE_BOUND_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "the omitted-variable bound has no nu^2 estimate for a 'drtmle' fit. The default "
    "estimator E[2 m(alpha_hat) - alpha_hat^2] equals nu_0^2 minus the squared error of "
    "the fitted representer, by the Riesz identity, so it falls exactly where the fitted "
    "mechanism is wrong, which is the case DR-TMLE guards against. The bound would be too "
    "narrow and the robustness value too large. " + _NO_NU2_DERIVATION,
    alternative=(
        "Pass assume_consistent_mechanism=True for a point bound that assumes a consistent "
        "treatment mechanism, or fit the ordinary TMLE on the same data for the bound and "
        "its limits."
    ),
    roadmap="X82",
)

_CTMLE_BOUND_REFUSAL = Refusal(
    RefusalReason.IMPLEMENTATION_GAP,
    "the omitted-variable bound has no nu^2 estimate for a 'collaborative_tmle' fit. The "
    "working mechanism conditions on a function V of W: the selected adjustment set W_S "
    "on the greedy, ordered and discrete paths, the selected candidate mechanism on the "
    "lasso and learners paths, and the fitted outcome regression under "
    "'oat'. Where the working mechanism is P(A | V) in the limit, the representer is "
    "E[alpha_W | A, V], whose second moment cannot exceed the second moment of alpha_W, "
    "while sigma^2 still comes from a regression on every "
    "declared covariate. The product sigma^2 nu^2 belongs to no single conditioning set, "
    "and the collaborative robustness value is optimistic by construction. " + _NO_NU2_DERIVATION,
    alternative=(
        "Pass assume_consistent_mechanism=True for a point bound that assumes a consistent "
        "treatment mechanism, or fit the ordinary TMLE on the same data for the bound and "
        "its limits."
    ),
    roadmap="X82",
)

#: The same refusal for the two collaborative fits that select nothing. Their mechanism
#: is fitted on every covariate, so the argument above does not apply to them, and the
#: sentence names the scope rule instead.
_CTMLE_FULL_CANDIDATE_BOUND_REFUSAL = Refusal(
    RefusalReason.API_REDIRECTION,
    "the omitted-variable bound is not offered on a 'collaborative_tmle' fit. This fit "
    "selects nothing: a strategy='discrete' fit that declares one candidate, equal to the "
    "full adjustment set, or a strategy='learners' fit that declares no candidate, equals "
    "the ordinary TMLE fit. The bound reads the fitted method and has no collaborative "
    "route.",
    alternative="Fit the ordinary TMLE (TMLE, or TMLEMethod) on the same data for the bound.",
)

_ZERO_BIAS_LIMITS_REFUSAL = (
    "max_bias is zero: sqrt(sigma^2 nu^2) has no ordinary influence curve at "
    "sigma^2 = 0, so the one-sided limits and confidence-limit robustness value "
    "are not available"
)

_UNREACHABLE_RV = "no equal strength reaches the null"


def _refuse_longitudinal(result: Any) -> Refusal | None:
    """Refuse any assessment family but ``point``.

    First in the table, and not by taste: :class:`~cleverly.longitudinal.LongitudinalData`
    declares no ``has_missing_outcome``, so a later rule would raise ``AttributeError`` on
    a longitudinal result instead of refusing it.
    """
    if getattr(result, "assessment_family", None) != "point":
        return _LONGITUDINAL_BOUND_REFUSAL
    return None


def _refuse_guarded_mechanism(result: Any) -> Refusal | None:
    """Refuse a DR-TMLE fit, whose estimator does not assume a consistent mechanism."""
    if result.fitted_method == "drtmle":
        return _DRTMLE_BOUND_REFUSAL
    return None


def _refuse_adaptive_iptw(result: Any) -> Refusal | None:
    """Refuse an AdaptiveIPTW fit, which has no outcome regression to take a residual from."""
    if result.fitted_method == "adaptive_iptw":
        return OPERATION_REFUSALS["omitted_variable"]
    return None


def _refuse_selected_mechanism(result: Any) -> Refusal | None:
    """Refuse a collaborative fit, whose representer comes from a selected working g.

    The fits that select nothing have their own sentence. The selection records the
    declared ``"discrete"`` list as its path, and a ``"learners"`` path is the declared list
    followed by the terminal candidate, so the path without its last entry stands for the
    declared ``"learners"`` list.

    A ``"derivative"`` fit (C-TMLE0) selects nothing and assumes a consistent mechanism as
    the ordinary TMLE does, so the bound admits it. Its ``benchmark`` refit meets the refit
    refusal of a fit that declared propensity arrays.
    """
    if result.fitted_method != "collaborative_tmle":
        return None
    selection = result.extra.get("ctmle")
    if isinstance(selection, CTMLEDerivativeFit):
        return None
    if isinstance(selection, CTMLESelection) and declares_no_selection(
        selection.strategy,
        selection.path,
        selection.path[:-1] if selection.strategy == "learners" else None,
        selection.covariates,
    ):
        return _CTMLE_FULL_CANDIDATE_BOUND_REFUSAL
    return _CTMLE_BOUND_REFUSAL


def _refuse_non_arm_axis(result: Any) -> Refusal | None:
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
    if axis == "learned_rule":
        # A third reason: the target is data-adaptive, and no bound for it was reviewed.
        # A benchmark refit would also relearn the rules, and so change the target.
        return Refusal(
            RefusalReason.IMPLEMENTATION_GAP,
            "the omitted-variable bound and its benchmark do not cover a learned-rule fit: "
            + LEARNED_RULE_BOUND_REFUSAL
            + ".",
            alternative=NO_ALTERNATIVE,
            roadmap="X11",
        )
    if axis == "ipsi":
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            "the omitted-variable bound does not cover a fit whose parameters are indexed "
            "by 'ipsi'. An incremental intervention tilts the treatment mechanism, so the "
            "mechanism is part of the estimand rather than a nuisance the bound conditions "
            "on. Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2026) bound the bias "
            "of a linear functional of the outcome regression alone.",
        )
    if axis in _AXIS_GROUPS:
        return _refuse_axis_configuration(result, axis)
    noun = _AXIS_NOUNS[axis]
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        f"the omitted-variable bound is not implemented for a fit whose parameters are "
        f"indexed by {axis!r}. {noun} is a linear functional of the outcome regression, "
        f"and it has a Riesz representer, so the bound is well posed here and no "
        f"implementation of it is registered. The implemented bound covers the "
        f"arm-indexed linear estimands {sorted(LINEAR_ESTIMANDS)} and the regime, shift "
        f"and MSM parameters {sorted(AXIS_ESTIMANDS)}.",
        alternative=NO_ALTERNATIVE,
        roadmap="X82",
    )


def _axis_gap(what: str, why: str) -> Refusal:
    """An implementation gap on a covered axis, in one sentence form."""
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        f"the omitted-variable bound covers a {what} only in its basic form, and {why}.",
        alternative=NO_ALTERNATIVE,
        roadmap="X82",
    )


def _refuse_axis_configuration(result: Any, axis: str) -> Refusal | None:
    """Refuse the regime, shift and MSM configurations whose representer is not derived.

    The bound reads the fit's own clever covariate.  A continuous-dose or non-identity MSM,
    and a randomized, fold-tuned or joint policy, change that covariate or the functional
    in a way that no derivation here covers.
    """
    nuisance = result.repeats[0].nuisance
    if axis == "msm":
        msm = nuisance.msm
        if msm is not None and msm.continuous:
            return _axis_gap(
                "working-model coefficient",
                "a continuous-dose MSM reads the treatment density at quadrature doses, "
                "whose representer and Gram matrix the bound does not rebuild",
            )
        if msm is not None and str(msm.link) != "identity":
            return _axis_gap(
                "working-model coefficient",
                f"under link={msm.link!r} the coefficient is not linear in the outcome "
                "regression, so it has no Riesz representer",
            )
    if axis == "policy":
        policies = nuisance.policies
        if policies is not None and any(
            getattr(policies, name) is not None
            for name in ("mixing", "component_rows", "curve_terms", "joint")
        ):
            return _axis_gap(
                "modified-policy mean",
                "a randomized, fold-tuned or joint policy set mixes components whose "
                "representers the bound does not combine",
            )
        if policies is not None and not _atomic_policy_doses(result.data, policies.shifted):
            return _axis_gap(
                "modified-policy mean",
                "a policy on a continuous dose reads a fitted conditional density, and the "
                "bound has no derived Riesz representer or nu^2 guarantee for it; declare "
                "dose_atoms that hold every observed and shifted dose",
            )
    return None


def _atomic_policy_doses(data: Any, shifted: np.ndarray) -> bool:
    """Whether every observed dose and every shifted dose is a declared dose atom.

    The bound's representer for a shift is the density ratio of the induced law over the
    fitted law.  On a dose that the atoms carry, that ratio is a ratio of point masses.
    """
    atoms = np.asarray(getattr(data, "dose_atoms", ()), dtype=float)
    if atoms.size == 0:
        return False
    treatment = np.asarray(data.treatment, dtype=float)
    observed = getattr(data, "treatment_observed", None)
    if observed is not None:
        treatment = treatment[np.asarray(observed, dtype=bool)]
    doses = np.concatenate([treatment.reshape(-1), np.asarray(shifted, dtype=float).reshape(-1)])
    doses = doses[np.isfinite(doses)]
    return bool(np.all(np.isin(doses, atoms)))


def axis_parameters(result: TMLEResult) -> dict[str, ArmParameter]:
    """The reported regime, shift and MSM parameters the bound covers, by name.

    Composed forward through :func:`~cleverly.targets.parameter_name` from the codes the
    fit carries, as :mod:`cleverly.sensitivity.missingness` composes them for the tilt.
    Each parameter keeps its code in ``arm`` and its reference code in ``versus``.

    Parameters
    ----------
    result : TMLEResult
        A point-treatment result.

    Returns
    -------
    dict
        Each reported name and its parameter.  Empty on any other axis.
    """
    from ..targets import parameter_name

    axis = getattr(getattr(result, "config", None), "parameter_axis", None)
    if axis not in _AXIS_GROUPS or getattr(result, "assessment_family", None) != "point":
        return {}
    group = _AXIS_GROUPS[axis][0]
    nuisance = result.repeats[0].nuisance
    estimates = result.estimates
    out: dict[str, ArmParameter] = {}
    if axis == "msm":
        msm = nuisance.msm
        if msm is None:
            return {}
        for code in msm.codes:
            name = parameter_name("msm", arm=msm.labels[code])
            if name in estimates:
                out[name] = ArmParameter(name, "msm", group, code)
        return out
    codes = nuisance.regimes if axis == "regime" else nuisance.policies
    if codes is None:
        return {}
    level, contrast = (
        ("ey_regime", "ate_regime") if axis == "regime" else ("ey_policy", "ate_policy")
    )
    labels = codes.labels
    reference = float(codes.reference)
    for code in codes.codes:
        name = parameter_name(level, arm=labels[code])
        if name in estimates:
            out[name] = ArmParameter(name, level, group, code)
        if code == reference:
            continue
        name = parameter_name(contrast, arm=labels[code], versus=labels[reference])
        if name in estimates:
            out[name] = ArmParameter(name, contrast, group, code, reference)
    return out


def _bound_parameters(result: TMLEResult) -> dict[str, ArmParameter]:
    """Every reported parameter the bound applies to: the arm ones and the axis ones."""
    return {**reported_arm_parameters(result), **axis_parameters(result)}


def _refuse_unbounded_parameters(result: Any) -> Refusal | None:
    """Refuse an arm-indexed fit that reports no parameter the bound applies to.

    The predicate is :func:`~cleverly.sensitivity._parameters.reported_arm_parameters`,
    which :func:`resolve_parameter` reads for the same fit, so no request can lift this
    rule.  A ratio-only fit, a population attributable risk or fraction, and a
    complete-outcome natural-course mean each report none, so every call on such a fit
    refuses, and the five rows read this rule.  The rule follows ``parameter_axis``,
    because its sentence describes an arm-indexed fit.  The E-value pointer is appended
    only on a fit that reports a risk ratio or an odds ratio, the two scales the E-value
    answers for.
    """
    if _bound_parameters(result):
        return None
    ratio = any(parameter_stem(name) in _EVALUE_STEMS for name in result.estimates)
    return Refusal(
        RefusalReason.INVALID_DECLARATION,
        "the omitted-variable bound applies to the arm-indexed linear estimands "
        f"{sorted(LINEAR_ESTIMANDS)}, and this arm-indexed fit reported none of them: "
        f"it reported {sorted(result.estimates)}. One bound is the second moment of "
        "one contrast's own Riesz representer, so ask the fit for a counterfactual "
        "mean or a contrast of arms.",
        alternative=_EVALUE_POINTER.strip() if ratio else None,
    )


#: Every omitted-variable refusal that the requested estimand cannot change, in the one
#: order the five entry points and the capability rows both use.  The table is ordered and
#: each rule assumes its predecessors returned ``None``: ``longitudinal`` establishes that
#: the result carries a point-treatment ``data`` and ``config`` at all, which every rule
#: after it reads.  ``bound_parameters`` follows ``parameter_axis``, because its sentence
#: names an arm-indexed fit.  A fit with ``repeats`` above one passes every rule: each draw
#: has its own bound, and :func:`omitted_variable_bounds` combines them by the median rule.
#: The names are the introspection contract; a test reads them to pin the order without
#: respelling a message.
_FIT_WIDE_BOUND_RULES: tuple[tuple[str, Callable[[Any], Refusal | None]], ...] = (
    ("longitudinal", _refuse_longitudinal),
    ("drtmle", _refuse_guarded_mechanism),
    ("adaptive_iptw", _refuse_adaptive_iptw),
    ("collaborative_tmle", _refuse_selected_mechanism),
    ("parameter_axis", _refuse_non_arm_axis),
    ("bound_parameters", _refuse_unbounded_parameters),
)


def fit_wide_bound_refusal(result: Any) -> Refusal | None:
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
    Refusal or None
        The first fit-wide refusal, or ``None`` when no fit-wide boundary applies.
    """
    for _name, rule in _FIT_WIDE_BOUND_RULES:
        refusal = rule(result)
        if refusal is not None:
            return refusal
    return None


@dataclass(frozen=True)
class SensitivityElements:
    r"""The ingredients of the bias bound for one estimand.

    Parameters
    ----------
    estimand : str
        Alias of the estimand these elements describe.
    sigma2 : float
        :math:`E[R (Y - \bar Q^*(A, W))^2]`, the residual outcome variance, with ``R`` the
        indicator of the rows the representer reads, averaged over every row.  ``R`` is one
        on complete data, :math:`\Delta` with a response mechanism, and also
        :math:`1\{Z = z\}` or :math:`1\{S = s\}` for an intermediate or a stratum.
    nu2 : float
        :math:`E[\alpha(A, W)^2]`, the second moment of the Riesz representer.  Under the
        plug-in estimator with a correction, :math:`E_n[\hat\alpha^2] - \hat B`.
    max_bias : float
        :math:`\sqrt{\sigma^2 \nu^2}`, the bias scale. The bound multiplies it by
        ``|rho| sqrt(cf_y cf_d / (1 - cf_d))``, which can exceed 1, so it is not the
        largest bias. The name follows DoubleML.
    psi_sigma2 : ndarray
        Influence curve of ``sigma2``.
    psi_nu2 : ndarray or None
        Influence curve of ``nu2`` under the doubly robust estimator: the score of Lemma 3
        of Chernozhukov et al. (2026), plus the influence of the conditioning share for the
        ATT and the ATC.  Under the plug-in estimator it adds the term of the fitted mechanism.
        ``None`` under the plug-in estimator when the mechanism is neither declared known nor
        a qualifying ``LogisticRegression`` (docs/roadmap.md F26).
    psi_max_bias : ndarray or None
        Influence curve of ``max_bias``, as in Theorem 4 of the same paper, so the
        bias-adjusted bounds get confidence limits rather than being treated as known
        constants. ``None`` under the plug-in estimator or when ``max_bias`` is zero.

    riesz_representer : ndarray
        ``(n,)`` values of :math:`\\alpha(A, W)` for the targeted functional.
    nu2_estimator : str
        Which estimator of ``nu2`` produced these values.
    limit_refusal : MechanismRefusal or None
        Why the plug-in estimator publishes no limits on this fit, or ``None``.
    plugin_mechanism : str or None
        Under the plug-in estimator with limits, the mechanism whose score the curve
        carries: ``"known"`` or ``"logistic"``. ``None`` otherwise.
    nu2_correction : float or None
        :math:`\hat B`, the second-order bias the plug-in estimate subtracts.  ``None``
        under the doubly robust estimator and wherever the plug-in estimate is uncorrected.
    features : tuple of str, default=()
        The X82 part G configurations the bound reads (:func:`bound_features`).  Empty
        for the shipped bound.
    """

    estimand: str
    sigma2: float
    nu2: float
    max_bias: float
    psi_sigma2: FloatArray
    psi_nu2: FloatArray | None
    psi_max_bias: FloatArray | None
    riesz_representer: FloatArray
    nu2_estimator: str
    limit_refusal: MechanismRefusal | None = None
    plugin_mechanism: str | None = None
    nu2_correction: float | None = None
    features: tuple[str, ...] = ()


def sensitivity_elements(
    result: TMLEResult,
    estimand: str = "ate",
    *,
    nu2_estimator: str = "auto",
    repeat: int | None = None,
) -> SensitivityElements:
    r"""Compute :math:`\sigma^2`, :math:`\nu^2` and the bias scale for one estimand.

    This is where every entry point in this module meets
    :data:`_FIT_WIDE_BOUND_RULES`, so :func:`omitted_variable_bounds`,
    :func:`benchmark`, :func:`robustness_value`, and :func:`contour_data` share one
    refusal and none of them pays a learner before hearing it.

    The elements describe one draw of the cross-fitting split.  A fit with ``repeats``
    above one has one set per draw, so the call names the draw with ``repeat``.
    :func:`omitted_variable_bounds` combines the draws by the median rule.

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
        form, which is all of :data:`LINEAR_ESTIMANDS`.  Only the doubly robust form has
        a derived influence curve for any mechanism learner.  The plug-in form gives the
        bounds confidence limits only for a declared known mechanism or a qualifying
        ``LogisticRegression``, in sample or cross-fitted.
    repeat : int or None
        The draw of the cross-fitting split, from 0 to ``result.n_repeats - 1``.
        ``None`` means the only draw of a fit with one, and a fit with more refuses it.

    Returns
    -------
    SensitivityElements
        The residual outcome variance, the Riesz second moment, the bias scale, and
        the influence curve of each, for the one draw.

    Raises
    ------
    ValueError
        If ``nu2_estimator`` is not one of :data:`NU2_ESTIMATORS`, or if ``repeat`` names
        no draw of the fit.  The checks run before any refusal and before any computation.
    CapabilityError
        If a rule of :data:`_FIT_WIDE_BOUND_RULES` refuses the fit, if the fit reports
        no parameter the bound applies to under ``estimand``, if ``repeat`` is omitted on a
        fit with several draws, or if the doubly robust estimate of :math:`\nu^2` is not
        positive.
    """
    _resolved_nu2_estimator(nu2_estimator)
    draw = _draw_index(result, repeat)
    refusal = fit_wide_bound_refusal(result)
    if refusal is not None:
        raise CapabilityError.from_refusal(refusal)
    if draw is None:
        raise CapabilityError.from_refusal(repeat_required_refusal(result.n_repeats))
    parameter = resolve_parameter(result, estimand)
    return _elements_for(result, result.repeats[draw], parameter, nu2_estimator)


def repeat_required_refusal(n_repeats: int) -> Refusal:
    """The refusal of a bare elements request on a fit with several draws.

    Parameters
    ----------
    n_repeats : int
        How many draws of the cross-fitting split the fit combined.

    Returns
    -------
    Refusal
        An ``invalid_declaration`` refusal that names the ``repeat`` argument.
    """
    return Refusal(
        RefusalReason.INVALID_DECLARATION,
        f"this fit combined {n_repeats} draws of the cross-fitting split, and the elements "
        "of the bound describe one draw.",
        alternative=(
            f"Name the draw with repeat=0 to repeat={n_repeats - 1}. "
            "omitted_variable_bounds() combines the draws by the median rule."
        ),
    )


def _draw_index(result: Any, repeat: int | None) -> int | None:
    """The draw ``repeat`` names, or ``None`` for a bare request on a fit with several."""
    n_repeats = int(getattr(result, "n_repeats", 1))
    if repeat is None:
        return 0 if n_repeats == 1 else None
    if isinstance(repeat, bool) or not isinstance(repeat, (int, np.integer)):
        raise ValueError(f"repeat must be an integer; got {repeat!r}")
    if not 0 <= int(repeat) < n_repeats:
        raise ValueError(f"repeat must lie in [0, {n_repeats - 1}] for this fit; got {repeat}")
    return int(repeat)


@dataclass(frozen=True)
class _Draw:
    """One draw's estimate, its influence curve, and its bound elements."""

    psi: float
    curve: FloatArray | None
    elements: SensitivityElements


#: The fitted methods whose bound refusal ``assume_consistent_mechanism=True`` lifts.
CONSISTENT_MECHANISM_METHODS: tuple[str, ...] = ("drtmle", "collaborative_tmle")


def _consistent_mechanism_limits(method: str) -> MechanismRefusal:
    """Why the point bound of a DR-TMLE or C-TMLE fit has no confidence limits."""
    why = (
        "the DR-TMLE remainder is linear in the error of the mechanism it does not assume "
        "consistent"
        if method == "drtmle"
        else "the selected mechanism has no derived curve for nu^2"
    )
    return MechanismRefusal(
        key="consistent_mechanism",
        text=(
            f"assume_consistent_mechanism=True: the bound of a {method!r} fit is a point "
            f"bound, and no source this package cites gives its standard error, because {why}."
        ),
        alternative="Read the point bounds and the point robustness value, which stay available.",
        reason=RefusalReason.INFERENCE_GAP,
    )


def _lasso_fit(result: TMLEResult) -> bool:
    """Whether a collaborative fit selected its mechanism on a lasso penalty path."""
    selection = result.extra.get("ctmle")
    return isinstance(selection, CTMLESelection) and selection.strategy == "lasso"


def _consistent_mechanism_assumption(result: TMLEResult, mechanism_learner: Any = None) -> str:
    """The sentence a consistent-mechanism point bound prints beside its numbers."""
    if result.fitted_method == "drtmle":
        source = "the fit's primary treatment mechanism"
    elif mechanism_learner is not None:
        source = (
            f"one refit of the declared mechanism learner {mechanism_learner!r} on every "
            "mechanism covariate, not the selected mechanism. The bound holds only if that "
            "learner is consistent for the mechanism on those covariates; the package does "
            "not check it"
        )
    elif _lasso_fit(result):
        source = (
            "one refit of the lasso path's unpenalized member (lambda = 0) on every mechanism "
            "covariate, not the selected mechanism"
        )
    else:
        source = (
            "one refit of the fit's treatment learner on every mechanism covariate, not the "
            "selected mechanism"
        )
    return (
        "assume_consistent_mechanism=True: this point bound assumes no unmeasured confounding "
        f"beyond the stated strengths and a consistent treatment mechanism. Its representer "
        f"comes from {source}."
    )


def _mechanism_fingerprint(result: TMLEResult, mechanism_learner: Any) -> str | None:
    """The provenance name of a declared mechanism learner, or ``None``."""
    if mechanism_learner is None:
        return None
    from ..validation._interpretation import learner_fingerprint

    return learner_fingerprint(mechanism_learner, int(result.data.n))


def _full_mechanism(result: TMLEResult, repeat: RepeatFit, mechanism_learner: Any = None) -> Any:
    """The fit's treatment learner refitted on every mechanism covariate, on the draw's folds.

    A C-TMLE fit conditions its mechanism on a selected function of the covariates.  The
    bound under an assumed consistent mechanism reads the mechanism on every covariate, which
    the fit never estimated, so it is fitted here once per draw.  A lasso fit never uses
    ``treatment_learner`` for its candidates, so its full mechanism is the path's own
    unpenalized member: :class:`~cleverly.learners._l1_logistic.L1Logistic` at
    ``lambda = 0``, on the ``glmnet`` objective the path solves.  A declared
    ``mechanism_learner`` replaces either.
    """
    from ..estimators._nuisance import Propensity, cross_fit_predictions
    from ..learners._l1_logistic import L1Logistic
    from ..learners.super_learner import resolve_learner

    estimator = result.estimator
    data = result.data
    nuisance = repeat.nuisance
    if mechanism_learner is None:
        mechanism_learner = L1Logistic(0.0) if _lasso_fit(result) else estimator.treatment_learner
    learner = resolve_learner(
        mechanism_learner,
        task="classification",
        n_folds=estimator.learner_folds,
        random_state=repeat.seed,
    )
    predictions, _ = cross_fit_predictions(
        learner,
        data.mechanism_covariates,
        data.treatment,
        data.weights,
        nuisance.folds,
        task="classification",
        predict_designs={"g": data.mechanism_covariates},
        groups=data.cluster,
        clip=(0.0, 1.0),
        classes=nuisance.arms,
        n_jobs=estimator.n_jobs,
    )
    return Propensity(predictions["g"], nuisance.arms)


def _consistent_draws(
    result: TMLEResult, estimand: str, nu2_estimator: str, mechanism_learner: Any = None
) -> tuple[_Draw, ...]:
    """The point bound's draws under ``assume_consistent_mechanism=True``.

    The keyword lifts the one rule that refuses a DR-TMLE or a selecting C-TMLE fit.  Every
    other rule of :data:`_FIT_WIDE_BOUND_RULES` still applies.  ``sigma^2`` reads the fit's
    targeted outcome regression, and each bound centres on the fit's own estimate.  No draw
    carries a curve, so the bound has no limits.
    """
    _resolved_nu2_estimator(nu2_estimator)
    method = getattr(result, "fitted_method", None)
    rules = dict(_FIT_WIDE_BOUND_RULES)
    lifted = rules[method](result) if method in CONSISTENT_MECHANISM_METHODS else None
    if lifted is _CTMLE_FULL_CANDIDATE_BOUND_REFUSAL:
        # A fit that selects nothing is the ordinary TMLE fit, so the redirection stands.
        raise CapabilityError.from_refusal(lifted)
    if mechanism_learner is not None and method != "collaborative_tmle":
        if method == "drtmle":
            reason = (
                "A DRTMLE bound reads the fit's own primary mechanism and refits nothing. "
                "Omit mechanism_learner."
            )
        else:
            reason = (
                "This bound already reads the fit's own mechanism and refits nothing, so it "
                "needs neither keyword. Omit mechanism_learner and "
                "assume_consistent_mechanism."
            )
        raise ValueError(
            "mechanism_learner= declares the learner a selecting CTMLE fit refits for the "
            f"consistent-mechanism bound, and this is a {method!r} fit. {reason}"
        )
    if lifted is None:
        raise ValueError(
            "assume_consistent_mechanism=True lifts the bound's refusal of a DRTMLE fit and of "
            "a selecting CTMLE fit, and this fit meets neither. Omit the argument."
        )
    for name, rule in _FIT_WIDE_BOUND_RULES:
        refusal = None if name == method else rule(result)
        if refusal is not None:
            raise CapabilityError.from_refusal(refusal)
    if result.estimator is None:
        raise CapabilityError(
            "assume_consistent_mechanism=True reads the fitted estimator, and this result "
            "carries none",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    if method == "collaborative_tmle" and bool(
        getattr(result.data, "has_missing_treatment", False)
    ):
        raise CapabilityError(
            "assume_consistent_mechanism=True refits the treatment mechanism of a CTMLE fit on "
            "every covariate, and no refit of the composite mechanism of a missing treatment "
            "is implemented.",
            reason=RefusalReason.IMPLEMENTATION_GAP,
            alternative="Fit the ordinary TMLE on the same data for the bound.",
            roadmap="X82",
        )
    if (
        method == "collaborative_tmle"
        and is_per_arm_oat(result.estimator)
        and mechanism_learner is None
    ):
        raise CapabilityError(
            "assume_consistent_mechanism=True refits the fit's treatment learner on every "
            "covariate, and a per-arm outcome-adaptive fit declares a learner that reads one "
            "column, the outcome prediction at one arm. The fit declares no learner for the "
            "mechanism on the full adjustment set, so the representer of the point bound has "
            "no source.",
            reason=RefusalReason.INVALID_DECLARATION,
            alternative=(
                "Declare that learner with mechanism_learner=, for example "
                "mechanism_learner=GLM(), or fit the ordinary TMLE on the same data."
            ),
        )
    parameter = resolve_parameter(result, estimand)
    limits = _consistent_mechanism_limits(str(method))
    draws = []
    for repeat in result.repeats:
        if method == "collaborative_tmle":
            refitted = replace(
                repeat.nuisance,
                propensity=_full_mechanism(result, repeat, mechanism_learner),
            )
            repeat = replace(repeat, nuisance=refitted)
        elements = _elements_for(result, repeat, parameter, nu2_estimator)
        elements = replace(elements, psi_max_bias=None, limit_refusal=limits)
        psi = float(result[estimand].psi) if result.n_repeats == 1 else float(repeat.psi[estimand])
        draws.append(_Draw(psi, None, elements))
    return tuple(draws)


def _bound_draws(
    result: TMLEResult,
    estimand: str,
    nu2_estimator: str,
    *,
    consistent: bool = False,
    mechanism_learner: Any = None,
) -> tuple[_Draw, ...]:
    """Every draw's estimate, curve and elements, after the fit-wide refusals.

    A fit with one draw reads its reported estimate, so every quantity equals the shipped
    one bit for bit.  A fit with several draws keeps each draw's point estimate but not its
    curve, so each draw is retargeted from its own nuisances.  The retarget must reproduce
    the stored point estimate, which checks that it ran the fit's own targeting.
    ``consistent`` is ``assume_consistent_mechanism`` (:func:`_consistent_draws`), and
    ``mechanism_learner`` is the learner it declares.
    """
    if consistent:
        return _consistent_draws(result, estimand, nu2_estimator, mechanism_learner)
    if mechanism_learner is not None:
        raise ValueError(
            "mechanism_learner= declares the mechanism of the consistent-mechanism bound, "
            "so it needs assume_consistent_mechanism=True."
        )
    _resolved_nu2_estimator(nu2_estimator)
    refusal = fit_wide_bound_refusal(result)
    if refusal is not None:
        raise CapabilityError.from_refusal(refusal)
    parameter = resolve_parameter(result, estimand)
    if result.n_repeats == 1:
        estimate = result[estimand]
        elements = _elements_for(result, result.repeats[0], parameter, nu2_estimator)
        return (_Draw(float(estimate.psi), estimate.influence_curve, elements),)
    estimator = result.estimator
    if estimator is None:
        raise CapabilityError(
            "the omitted-variable bound of a fit with several draws retargets each draw, "
            "and this result carries no fitted estimator",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    keys = result.parameter_keys
    key = keys.get(estimand) if keys else None
    target = key.estimand if key is not None else parameter_stem(estimand)
    draws = []
    for index, repeat in enumerate(result.repeats):
        estimates, _ = estimator.retarget(
            result.data,
            repeat.nuisance,
            estimands=(target,),
            intermediate_value=result.intermediate_value,
            g_bounds=result.config.g_bounds,
            g_bounds_conditional=result.config.g_bounds_conditional,
            nuisance_bound=result.config.missingness_bound,
            alpha_sig=result.config.alpha_sig,
            conditions=result.config.support_conditions,
        )
        estimate = estimates.get(estimand)
        stored = repeat.psi.get(estimand)
        if (
            estimate is None
            or stored is None
            or not np.isclose(estimate.psi, stored, rtol=1e-9, atol=1e-12)
        ):
            raise RuntimeError(
                f"retargeting draw {index} for {estimand!r} gave "
                f"{None if estimate is None else estimate.psi!r}, and the fit stored "
                f"{stored!r}. This is a package defect; report it."
            )
        elements = _elements_for(result, repeat, parameter, nu2_estimator)
        draws.append(_Draw(float(stored), estimate.influence_curve, elements))
    return tuple(draws)


def _median_elements(result: TMLEResult, estimand: str, nu2_estimator: str) -> SensitivityElements:
    """The elements of one draw, or the medians of ``sigma2`` and ``nu2`` over several.

    :func:`benchmark` compares these two second moments with and without the named
    covariates.  It reads no curve, so it needs no retarget.  One draw returns its own
    elements unchanged.
    """
    _resolved_nu2_estimator(nu2_estimator)
    refusal = fit_wide_bound_refusal(result)
    if refusal is not None:
        raise CapabilityError.from_refusal(refusal)
    parameter = resolve_parameter(result, estimand)
    per_draw = [
        _elements_for(result, repeat, parameter, nu2_estimator) for repeat in result.repeats
    ]
    if len(per_draw) == 1:
        return per_draw[0]
    sigma2 = float(np.median([elements.sigma2 for elements in per_draw]))
    nu2 = float(np.median([elements.nu2 for elements in per_draw]))
    return replace(per_draw[0], sigma2=sigma2, nu2=nu2, max_bias=float(np.sqrt(sigma2 * nu2)))


def _median_rule(points: FloatArray, std_errors: FloatArray) -> tuple[float, float]:
    r"""The median point and the split-adjusted standard error of several draws.

    The variance is :math:`\operatorname{median}_r\{\hat\sigma_r^2 + (\hat t_r -
    \tilde t)^2\}`, the rule :func:`~cleverly.inference.median_estimates` applies to the
    estimate.  One draw returns its own point and standard error unchanged.
    """
    if len(points) == 1:
        return float(points[0]), float(std_errors[0])
    center = float(np.median(points))
    variance = float(np.median(np.square(std_errors) + np.square(points - center)))
    return center, float(np.sqrt(variance))


def _combined_elements(draws: tuple[_Draw, ...], result: TMLEResult) -> SensitivityElements:
    """The elements a combined bound reports: one draw's, or the medians over draws.

    The curves belong to one draw, so a combined record keeps the first draw's curves and
    only the summary fields of the combined bound read it: the bias scale, the estimator,
    the limit refusal and the features.
    """
    first = draws[0].elements
    if len(draws) == 1:
        return first
    refusal = next(
        (draw.elements.limit_refusal for draw in draws if draw.elements.limit_refusal),
        None,
    )
    corrections = [draw.elements.nu2_correction for draw in draws]
    return replace(
        first,
        sigma2=float(np.median([draw.elements.sigma2 for draw in draws])),
        nu2=float(np.median([draw.elements.nu2 for draw in draws])),
        max_bias=float(np.median([draw.elements.max_bias for draw in draws])),
        psi_max_bias=(
            None
            if any(draw.elements.psi_max_bias is None for draw in draws)
            else first.psi_max_bias
        ),
        limit_refusal=refusal,
        nu2_correction=(
            None
            if any(value is None for value in corrections)
            else float(np.median(cast("list[float]", corrections)))
        ),
    )


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
    the fit arm-indexed and whose ``bound_parameters`` rule makes it report at least one
    parameter the bound applies to, since no estimand name can change either verdict.
    The order of the checks below matters.  A name this bound could never apply to is
    refused first, so that asking for a risk ratio is not reported as a missing
    estimand; the coverage message comes last.  A reported stratum-conditional alias
    resolves to the marginal parameter's arms with the stratum's code.

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
    if stem not in LINEAR_ESTIMANDS | AXIS_ESTIMANDS:
        # The pointer is for the two ratio scales an E-value is defined on. Offered for
        # anything else -- an attributable fraction, say -- it sends the reader from one
        # refusal to a second one.
        pointer = _EVALUE_POINTER.strip() if stem in _EVALUE_STEMS else None
        raise CapabilityError(
            f"the omitted-variable bound applies to {sorted(LINEAR_ESTIMANDS)}, not "
            f"{estimand!r}: the bound is on the bias of a linear functional of the "
            f"outcome regression, and {estimand!r} is not one.",
            reason=RefusalReason.INVALID_DECLARATION,
            alternative=pointer,
        )
    available = _bound_parameters(result)
    if estimand in available:
        return available[estimand]
    # A stratum-conditional parameter has its own representer: the marginal one restricted
    # to the stratum and divided by the stratum's share (``_StratumScale``).
    conditional = stratum_parameter(result, estimand)
    if conditional is not None:
        return conditional
    empty = _refuse_unbounded_parameters(result)
    if empty is not None:
        # Every entry point hears this sentence from the fit-wide table before it gets
        # here. A caller that resolves a parameter directly hears the same sentence.
        raise CapabilityError.from_refusal(empty)
    raise CapabilityError(
        f"estimand {estimand!r} was not requested in this fit. The bound is available for "
        f"{sorted(available)}, one per contrast, since nu^2 is the second moment of "
        "that contrast's own Riesz representer.",
        reason=RefusalReason.INVALID_DECLARATION,
    )


#: The registry instrument the confidence limits of a part G configuration resolve through.
OVB_LIMIT_INSTRUMENT = "ovb_limit"

#: The features of :func:`bound_features`, in the one order their rule joins them, and the
#: noun each second mechanism takes in the joint-strength sentence.  A feature with no noun
#: reads no second mechanism.
_FEATURE_MECHANISMS: dict[str, str | None] = {
    "response": "response",
    "missing_treatment": "recording",
    "intermediate": "intermediate",
    "regime": None,
    "shift": None,
    "msm": None,
    "stratum": None,
    "repeats": None,
}


def bound_features(result: TMLEResult, parameter: ArmParameter) -> tuple[str, ...]:
    """The X82 part G configurations a bound of ``parameter`` on ``result`` reads.

    Each feature changes the representer or the residual mask of the shipped bound.  The
    ``+``-joined tuple is the ``ovb_limit`` rule its confidence limits resolve through, so
    a registered coverage cell licenses exactly the configuration it measured.

    Parameters
    ----------
    result : TMLEResult
        A point-treatment result that the fit-wide rules admit.
    parameter : ArmParameter
        The resolved parameter.

    Returns
    -------
    tuple of str
        The features in :data:`_FEATURE_MECHANISMS` order. Empty for the shipped bound.
    """
    data = result.data
    present = {
        "response": bool(data.has_missing_outcome),
        "missing_treatment": bool(getattr(data, "has_missing_treatment", False)),
        "intermediate": declares_intermediate(result),
        "regime": parameter.group == "regime",
        "shift": parameter.group == "mtp",
        "msm": parameter.group == "msm",
        "stratum": parameter.stratum is not None,
        "repeats": result.n_repeats > 1,
    }
    return tuple(name for name in _FEATURE_MECHANISMS if present[name])


def joint_strength_sentence(features: tuple[str, ...]) -> str | None:
    """The sentence that says which mechanisms ``cf_d`` measures a joint strength over.

    Parameters
    ----------
    features : tuple of str
        The features of :func:`bound_features`.

    Returns
    -------
    str or None
        The sentence, or ``None`` when the representer reads the treatment mechanism alone.
    """
    nouns = [noun for name in features if (noun := _FEATURE_MECHANISMS[name]) is not None]
    if not nouns:
        return None
    named = (
        f"the {nouns[0]} mechanism"
        if len(nouns) == 1
        else f"the {', '.join(nouns[:-1])} and {nouns[-1]} mechanisms"
    )
    return f"cf_d measures a joint strength over the treatment mechanism and {named}."


def _limit_reading(
    result: TMLEResult, estimand: str, features: tuple[str, ...]
) -> Resolution | None:
    """Ask the registry what the confidence limits of a part G configuration may print.

    The shipped configuration has no feature and returns ``None`` without asking, so it
    keeps its output.  The import is local because :mod:`cleverly.validation` imports
    this package.
    """
    if not features:
        return None
    from ..validation._interpretation import fit_configuration, resolve

    try:
        fit = fit_configuration(result, estimand)
    except (AttributeError, KeyError, TypeError, ValueError):
        # An unreadable configuration licenses only through an unrestricted, exact row.
        fit = None
    return resolve(OVB_LIMIT_INSTRUMENT, "+".join(features), fit)


def _bound_inputs(
    result: TMLEResult, repeat: RepeatFit, group: str
) -> tuple[CausalData, NuisanceEstimates, tuple[float, float]]:
    """The data, nuisances, and bounds the representer of ``group`` is built from.

    A composite fit (a declared missing treatment) was targeted along
    :math:`C_a / g_{c,a}` on the composite view, so the representer is read there, as
    :func:`~cleverly.sensitivity.positivity._group_submodel` reads it.  Every other fit
    reads its own data and the group's bounds.
    """
    bounds = g_bounds_for(group, result.config.g_bounds, result.config.g_bounds_conditional)
    data, nuisance = result.data, repeat.nuisance
    if result.extra.get("missing_data") == "composite":
        state = composite_state(
            data, nuisance, g_bounds=bounds, nuisance_bound=result.config.missingness_bound
        )
        return state.data, state.nuisance, state.bounds
    return data, nuisance, bounds


def _representer_rows(data: CausalData) -> FloatArray:
    r"""The observation indicator the submodel applies through the fluctuation's mask.

    :math:`\Delta` on a fit with a response mechanism, :math:`\Delta_A \Delta` on the
    composite view, and one on every row of complete data.  The clever covariate omits it,
    so the representer multiplies it in.
    """
    return np.where(np.asarray(data.observed, dtype=bool), 1.0, 0.0)


def _stratum_rows(result: TMLEResult, parameter: ArmParameter) -> FloatArray | None:
    """``1{S = s}`` for a stratum-conditional parameter, or ``None`` for a marginal one."""
    if parameter.stratum is None:
        return None
    return np.asarray(result.data.strata == parameter.stratum, dtype=float)


def _elements_for(
    result: TMLEResult,
    repeat: RepeatFit,
    parameter: ArmParameter,
    nu2_estimator: str,
) -> SensitivityElements:
    r"""The bound's pieces under one cross-fitting draw.

    Takes the targeted ``Qbar`` and the mechanism from the same
    :class:`~cleverly.estimators._nuisance.RepeatFit`, which is what makes ``sigma2`` the
    residual variance of the regression whose propensity ``nu2`` was computed from.

    The representer is the fit's own clever covariate times the indicator of the rows it
    reads: :math:`\Delta` for a response mechanism, :math:`\Delta 1\{Z = z\}` for an
    intermediate at ``z``, and :math:`\Delta_A \Delta` on the composite view.  The submodel
    applies :math:`\Delta` through the fluctuation's mask rather than in the covariate, so
    the product is taken here.  ``sigma2`` averages the residual masked to the same rows
    over every row.  On complete data the indicator is one everywhere and both are the
    shipped quantities.

    A stratum-conditional parameter restricts both to the stratum.  The representer and
    :math:`m(O, \alpha)` are rescaled from the marginal share to the stratum's, which is
    ``P(S = s)`` for a mean or a difference of means and ``P(A = c, S = s)`` for an
    effect that conditions on arm ``c``.
    """
    data, nuisance, bounds = _bound_inputs(result, repeat, parameter.group)
    scaler = nuisance.scaler
    group = parameter.group
    fluctuation = repeat.fluctuations[group]
    reference = result.config.reference_arm
    submodel = build_submodel(
        data,
        nuisance,
        group,
        bounds=bounds,
        nuisance_bound=result.config.missingness_bound,
        intermediate_value=result.intermediate_value,
        # The conditional-effect fluctuations contrast against the arm this fit declared,
        # so the covariate rebuilt here must be the one it was targeted with.
        reference=reference,
    )
    features = bound_features(result, parameter)
    weights = data.weights
    # The rows the representer reads: the observed outcome (Delta, or Delta_A Delta on the
    # composite view) at the targeted level of the intermediate.
    support = np.asarray(targeted_rows(data, result.intermediate_value), dtype=float)
    stratum = _stratum_rows(result, parameter)
    if stratum is not None:
        support = support * stratum
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
        conditioning_share = float(data.arm_fractions[nuisance.arms.index(conditioning)])

    # sigma^2: residual variance of the targeted outcome regression, on the original
    # outcome scale so the bound is reported in the units the estimate uses.
    scaled = scaler.scale(data.outcome)
    residual = np.where(support > 0.0, scaled - fluctuation.targeted.observed, 0.0)
    if not scaler.is_identity:
        residual = residual * scaler.range
    sigma2_element = residual**2
    sigma2 = float(np.average(sigma2_element, weights=weights))
    psi_sigma2 = (sigma2_element - sigma2) * weights

    axis = _AxisInputs.build(parameter, nuisance, weights)
    representer = _riesz_representer(parameter, submodel, axis) * _representer_rows(data)

    method = _resolved_nu2_estimator(nu2_estimator)
    m_alpha: FloatArray | None = None
    if method == "doubly_robust":
        m_alpha = _m_alpha(parameter, submodel, conditioning_indicator, conditioning_share, axis)
    # The share the estimate divides by, and the indicator of the rows it counts: the
    # conditioning arm's for an ATT or an ATC, and the stratum's when there is one.
    share_indicator, share = conditioning_indicator, conditioning_share
    if stratum is not None:
        representer, m_alpha, share_indicator, share = _restrict_to_stratum(
            representer, m_alpha, stratum, conditioning_indicator, conditioning_share, weights
        )

    if method == "doubly_robust":
        assert m_alpha is not None
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
                f"(nu2_estimator={nu2_estimator!r} resolved to {method!r}.)",
                reason=RefusalReason.INVALID_DECLARATION,
            )
    else:
        nu2_element = representer**2
        nu2 = float(np.average(nu2_element, weights=weights))
    psi_nu2: FloatArray | None = None
    psi_max_bias: FloatArray | None = None
    limit_refusal: MechanismRefusal | None = None
    plugin_mechanism: str | None = None
    mechanism_term: FloatArray | None = None
    stacked: Any = None
    if method == "plugin":
        # E_n[alpha_hat^2] moves at first order with the fitted mechanism, so its curve
        # carries the mechanism score's term. Only a declared known mechanism or a fixed
        # logistic model gives that term (docs/roadmap.md F26); for any other learner an
        # array here would be a curve for a different estimator under this one's name.
        # A part G representer also reads a second mechanism or a stratum share, whose
        # term the stack does not carry.
        if features:
            limit_refusal = configuration_refusal(_configuration_phrase(features))
        else:
            stacked = plugin_mechanism_term(
                result, repeat, parameter, representer, bounds, conditioning_share
            )
            if isinstance(stacked, MechanismRefusal):
                limit_refusal = stacked
            else:
                mechanism_term = stacked.values
    if method == "doubly_robust" or mechanism_term is not None:
        psi_nu2 = (nu2_element - nu2) * weights
        if mechanism_term is not None:
            psi_nu2 = psi_nu2 + mechanism_term
        if share_indicator is not None:
            # The estimate divides by the estimated share of the conditioning arm or of the
            # stratum, so its curve carries that share's influence. See
            # _conditioning_share_influence.
            assert share is not None
            psi_nu2 = psi_nu2 + _conditioning_share_influence(nu2, share_indicator, share, weights)
        if axis is not None and axis.gram_inverse is not None:
            # The representer and m(O, alpha) read the estimated Gram matrix, so the curve
            # carries its influence (_gram_influence).
            psi_nu2 = psi_nu2 + _gram_influence(
                parameter, submodel, axis, representer, nu2, weights
            )
    nu2_correction: float | None = None
    if mechanism_term is not None:
        # The curve is that of E_n[alpha_hat^2]; the bound reports it less its second-order
        # bias and keeps the influence variance of that curve (_second_order).
        assert psi_nu2 is not None
        limit_refusal = stacked.refusal
        if limit_refusal is None:
            correction = float(stacked.correction)
            variance = influence_variance(psi_nu2, data.cluster)
            limit_refusal = correction_refusal(nu2, correction, variance)
            if limit_refusal is None:
                nu2 = nu2 - correction
                nu2_correction = correction
                plugin_mechanism = nuisance.treatment_mechanism
                plugin_mechanism = "known" if plugin_mechanism == "known" else "logistic"
    max_bias = float(np.sqrt(sigma2 * nu2))
    # The square-root delta method requires a positive product. At sigma2=0 the
    # point bound is still defined, but its ordinary influence curve is not.
    if psi_nu2 is not None and limit_refusal is None and max_bias > 0.0:
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
        limit_refusal=limit_refusal,
        plugin_mechanism=plugin_mechanism,
        nu2_correction=nu2_correction,
        features=features,
    )


def _configuration_phrase(features: tuple[str, ...]) -> str:
    """What a part G representer reads beyond the marginal treatment mechanism."""
    nouns = [noun for name in features if (noun := _FEATURE_MECHANISMS[name]) is not None]
    parts = []
    if nouns:
        parts.append(
            "also divides by the "
            + (nouns[0] if len(nouns) == 1 else f"{', '.join(nouns[:-1])} and {nouns[-1]}")
            + (" mechanism" if len(nouns) == 1 else " mechanisms")
        )
    axes = {
        "regime": "is the clever covariate of a declared regime",
        "shift": "is the clever covariate of a modified treatment policy",
        "msm": "is projected through the working model's Gram matrix",
    }
    parts.extend(phrase for name, phrase in axes.items() if name in features)
    if "stratum" in features:
        parts.append("is restricted to a baseline stratum and divides by its share")
    if "repeats" in features:
        parts.append("is combined over draws of the cross-fitting split by the median rule")
    return " and ".join(parts)


def _restrict_to_stratum(
    representer: FloatArray,
    m_alpha: FloatArray | None,
    stratum: FloatArray,
    conditioning_indicator: FloatArray | None,
    conditioning_share: float | None,
    weights: FloatArray,
) -> tuple[FloatArray, FloatArray | None, FloatArray, float]:
    r"""The stratum's representer and :math:`m(O, \alpha)`, from the marginal ones.

    A stratum-conditional mean or difference of means is
    :math:`E[1\{S = s\} f(W) / p_s]` with :math:`p_s = P_w(S = s)`, so its representer is
    :math:`1\{S = s\} \alpha / p_s` and :math:`m(O, \alpha_s) = 1\{S = s\} m / p_s^2`.
    An effect that conditions on arm ``c`` divides by :math:`q = P_w(A = c, S = s)` in place
    of the marginal :math:`p_c`.  Its marginal representer is proportional to
    :math:`1 / p_c` and its marginal :math:`m` to :math:`1 / p_c^2`, so the stratum's
    pieces are the marginal ones times :math:`1\{S = s\} p_c / q` and
    :math:`1\{S = s\} (p_c / q)^2`.  Both forms give :math:`\nu^2 = F / \text{share}^2` with
    ``F`` free of the share, which is what :func:`_conditioning_share_influence` needs.

    Returns
    -------
    tuple
        The representer, :math:`m(O, \alpha)` (``None`` when ``m_alpha`` is ``None``), the
        indicator of the rows the share counts, and the share.
    """
    if conditioning_indicator is None:
        share = float(np.average(stratum, weights=weights))
        scaled_m = None if m_alpha is None else stratum * m_alpha / share**2
        return stratum * representer / share, scaled_m, stratum, share
    assert conditioning_share is not None
    indicator = conditioning_indicator * stratum
    share = float(np.average(indicator, weights=weights))
    ratio = conditioning_share / share
    scaled_m = None if m_alpha is None else stratum * m_alpha * ratio**2
    return stratum * representer * ratio, scaled_m, indicator, share


@dataclass(frozen=True)
class _AxisInputs:
    r"""The known arrays a regime or MSM representer reads beyond the clever covariate.

    Parameters
    ----------
    regimes : ndarray or None
        ``(n, K, R)`` regime densities :math:`g^\star_r(a \mid W)`, for the regime axis.
    design : ndarray or None
        ``(n, K, p)`` working-model design :math:`\varphi(a, V)`, for the MSM axis.
    model_weights : ndarray or None
        ``(n, K)`` working-model weights :math:`h(a, V)`, for the MSM axis.
    gram_inverse : ndarray or None
        ``(p, p)`` inverse of the observation-weighted Gram matrix
        :math:`M = P_n \sum_a h \varphi \varphi^\top`, which the coefficient inverts.
    """

    regimes: FloatArray | None = None
    design: FloatArray | None = None
    model_weights: FloatArray | None = None
    gram_inverse: FloatArray | None = None

    @classmethod
    def build(
        cls, parameter: ArmParameter, nuisance: NuisanceEstimates, weights: FloatArray
    ) -> _AxisInputs | None:
        """The inputs of ``parameter``'s axis, or ``None`` for an arm parameter."""
        if parameter.group == "regime":
            assert nuisance.regimes is not None
            return cls(regimes=np.asarray(nuisance.regimes.values, dtype=float))
        if parameter.group != "msm":
            return None
        msm = nuisance.msm
        assert msm is not None
        design = np.asarray(msm.design, dtype=float)
        model_weights = np.asarray(msm.weights, dtype=float)
        w = np.asarray(weights, dtype=float)
        # The Gram matrix of cleverly.msm.solve_projection under the identity link.
        gram = np.einsum("ijp,ijq,ij,i->pq", design, design, model_weights, w) / float(w.sum())
        return cls(design=design, model_weights=model_weights, gram_inverse=np.linalg.inv(gram))

    def coefficient_rows(self, parameter: ArmParameter) -> FloatArray:
        """``b = M^{-1} e_j``, the row of the inverse Gram matrix coefficient ``j`` reads."""
        assert self.gram_inverse is not None
        return np.asarray(self.gram_inverse[:, int(parameter.arm)], dtype=float)


def _riesz_representer(
    parameter: ArmParameter, submodel: Any, axis: _AxisInputs | None = None
) -> FloatArray:
    r"""``alpha(A, W)`` for the requested estimand.

    The clever covariate *is* the Riesz representer -- that is why the same object
    both drives the targeting step and controls how much an omitted confounder can
    move the estimate.  Poor overlap inflates both simultaneously.

    Every column is reached by the arm it carries rather than by position, so
    ``ate[high vs low]`` reads the two arms it names out of a ``K``-column covariate
    instead of the two an implementation that counted to two would have found.

    A regime column targets one regime, and a contrast of regimes is a difference of two
    columns.  A policy column targets one policy, as an arm column targets one arm.  An MSM
    coefficient ``j`` reads :math:`e_j^\top M^{-1} H(A, W)`, the row of the coefficient's
    residual term in :func:`~cleverly.inference.influence.msm_coefficients`.
    """
    if parameter.group == "msm":
        assert axis is not None
        return np.asarray(submodel.observed @ axis.coefficient_rows(parameter), dtype=float)
    if parameter.group == "regime":
        column = np.asarray(submodel.observed[:, int(parameter.arm)], dtype=float)
        if parameter.versus is None:
            return column
        return np.asarray(column - submodel.observed[:, int(parameter.versus)], dtype=float)
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
    axis: _AxisInputs | None = None,
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
    :math:`1\{A = c\} / P(A = c)`.  That is the ATT of Example 2 in Online Appendix A of
    Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis (2026), with
    :math:`\ell(W_s) = D / P(D = 1)`.  Item (2) of their Theorem 5 writes this :math:`m`
    and the representer.  That theorem establishes the bias formula; the score and the
    limits are Lemma 3 and Theorem 4.  DoubleML's
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
    if parameter.group in ("regime", "msm"):
        assert axis is not None
        return _axis_m_alpha(parameter, submodel, axis)
    # ``arms[a][:, c]`` is the covariate at arm ``a`` in the column targeting arm ``c``:
    # the mean submodel has one column per arm, so both indices are arm levels and
    # neither is a positional 0 or 1.  A policy submodel has the same layout, with the
    # policy in place of the arm: ``arms[s][:, c]`` is the covariate at the dose that
    # policy ``s`` assigns, in the column targeting policy ``c``.
    if parameter.group in ("mean", "mtp"):
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


def _axis_m_alpha(parameter: ArmParameter, submodel: Any, axis: _AxisInputs) -> FloatArray:
    r"""``m(O, alpha)`` for a regime parameter or an MSM coefficient.

    Both functionals are :math:`E[\sum_a c(a, W) \bar Q(a, W)]` with a known weight
    ``c``, so :math:`m(O, \alpha) = \sum_a c(a, W) \alpha(a, W)`.  A regime ``r`` has
    :math:`c = g^\star_r(a \mid W)`, and a contrast the difference of two regimes.  An
    MSM coefficient ``j`` has :math:`c = e_j^\top M^{-1} h(a, V) \varphi(a, V)`.  The
    counterfactual blocks ``submodel.arms[a]`` hold the clever covariate at arm ``a``.
    """
    total = np.zeros(submodel.n, dtype=float)
    if parameter.group == "msm":
        assert axis.design is not None and axis.model_weights is not None
        b = axis.coefficient_rows(parameter)
        for index, level in enumerate(submodel.levels):
            weight = axis.model_weights[:, index] * (axis.design[:, index, :] @ b)
            total += weight * (submodel.arms[level] @ b)
        return total
    assert axis.regimes is not None
    code = int(parameter.arm)
    versus = None if parameter.versus is None else int(parameter.versus)
    for index, level in enumerate(submodel.levels):
        block = submodel.arms[level]
        weight = axis.regimes[:, index, code]
        alpha = block[:, code]
        if versus is not None:
            weight = weight - axis.regimes[:, index, versus]
            alpha = alpha - block[:, versus]
        total += weight * alpha
    return total


def _gram_influence(
    parameter: ArmParameter,
    submodel: Any,
    axis: _AxisInputs,
    representer: FloatArray,
    nu2: float,
    weights: FloatArray,
) -> FloatArray:
    r"""The influence of the estimated Gram matrix on the doubly robust ``nu^2`` of an MSM.

    With :math:`b = M^{-1} e_j`, the estimate is :math:`\hat\nu^2 = b^\top \hat K b`,
    where :math:`\hat K = P_n[2 \sum_a u_a v_a^\top - H H^\top]`, :math:`u_a = h \varphi`
    at arm ``a`` and :math:`v_a` and :math:`H` are the clever covariate at arm ``a`` and
    at the observed arm.  The Gram matrix :math:`\hat M = P_n \sum_a h \varphi
    \varphi^\top` is estimated whenever :math:`h` or :math:`\varphi` reads ``V``.  Since
    :math:`db = -M^{-1} dM\, b`, the delta method adds

    .. math::

        -2 w \Big[\sum_a h(a, V)\, (z^\top \varphi_a)(\varphi_a^\top b) - \nu^2\Big],
        \qquad z = M^{-1} K b ,

    to the curve.  For a marginal MSM the bracket is zero on every row, because each row's
    Gram contribution equals :math:`M`.  ``tests/unit/test_omitted_variable_axes.py``
    compares the curve with its Gateaux derivative on a design that reads ``V``.
    """
    assert axis.design is not None and axis.model_weights is not None
    w = np.asarray(weights, dtype=float)
    mass = float(w.sum())
    b = axis.coefficient_rows(parameter)
    # K b = P_n[2 sum_a u_a alpha(a) - H alpha], with alpha(a) = v_a' b and H' b = alpha.
    # ``representer`` carries the row mask R, and R^2 = R, so H (H' b) = observed * alpha.
    kb = -np.einsum("i,ip,i->p", w, submodel.observed, representer) / mass
    for index, level in enumerate(submodel.levels):
        u = axis.design[:, index, :] * axis.model_weights[:, index][:, None]
        kb += 2.0 * np.einsum("i,ip,i->p", w, u, submodel.arms[level] @ b) / mass
    assert axis.gram_inverse is not None
    z = axis.gram_inverse @ kb
    bracket = np.zeros(submodel.n, dtype=float)
    for index in range(len(submodel.levels)):
        phi = axis.design[:, index, :]
        bracket += axis.model_weights[:, index] * (phi @ z) * (phi @ b)
    return np.asarray(-2.0 * w * (bracket - nu2), dtype=float)


def _conditioning_share_influence(
    nu2: float, indicator: FloatArray, share: float, weights: FloatArray
) -> FloatArray:
    r"""The influence of the estimated conditioning share on the doubly robust ``nu^2``.

    For the ATT and the ATC, the representer and :math:`m(O, \alpha)` both divide by the
    share :math:`p = P_w(A = c)` of the arm :math:`c` that the parameter conditions on.  So
    the estimate is :math:`\hat F / \hat p^2`, where :math:`\hat F` is the weighted mean of
    :math:`\hat p^2 (2 m(O, \hat\alpha) - \hat\alpha^2)`.  At a fixed :math:`p`, the Riesz
    identity makes :math:`\hat F` insensitive at first order to the fitted mechanism, so its
    curve is the score of Lemma 3 of Chernozhukov, Cinelli, Newey, Sharma and Syrgkanis
    (2026).  The weighted share :math:`\hat p` has the curve :math:`w (1\{A = c\} - p)`, and
    the derivative of :math:`F / p^2` in :math:`p` is :math:`-2 \nu^2 / p`.  The delta method
    therefore adds

    .. math::

        -2 \nu^2 w (1\{A = c\} - p) / p

    to the curve.  With ``K`` arms each contrast has its own :math:`c` and :math:`p`.  Under
    clustering the term enters the cluster sum of
    :func:`~cleverly.inference.cluster.influence_variance` like every other row term.

    The paper gives a term of this kind for the estimate only, in Online Appendix A,
    "Statistical Inference", Equation (15), and the estimate's own curve here carries it.
    It gives none for :math:`\nu^2`, so this is the package's derivation.  The R package
    ``dml.sensemakr`` carries the same term since commit ``d5293ecb``, and DoubleML's
    ``DoubleMLIRM._sensitivity_element_est`` omits it.  The weighted rows of the term sum
    to zero, so no check of a mean or of a point can see it.
    ``tests/unit/test_omitted_variable_standard_error.py`` compares the curve with its
    Gateaux derivative row by row.
    """
    return np.asarray(-2.0 * nu2 * (indicator - share) / share * weights, dtype=float)


_P = ParamSpec("_P")

#: The constructor arguments whose values the bound stores under a private field name.
_STORED_LIMITS: tuple[str, ...] = ("ci_lower", "ci_upper", "robustness_value_ci")


def _accepts_stored_limit_names(init: Callable[_P, None]) -> Callable[_P, None]:
    """Let the constructor take the private field names, so that ``replace()`` works.

    ``dataclasses.replace()`` and ``copy.replace()`` pass every field under its stored
    name, and :class:`SensitivityBounds` stores its three limits as ``_ci_lower``,
    ``_ci_upper`` and ``_robustness_value_ci`` so that the public names can refuse.  The
    wrapper renames those keywords to the public arguments, and a public argument passed
    beside its stored name wins, as in ``replace(bound, ci_lower=x)``.  ``functools.wraps``
    keeps the public signature for :func:`inspect.signature`, and the wrapped constructor
    raises its own ``TypeError`` on an unknown or a missing argument.
    """

    @functools.wraps(init)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> None:
        for name in _STORED_LIMITS:
            if f"_{name}" in kwargs:
                stored = kwargs.pop(f"_{name}")
                kwargs.setdefault(name, stored)
        init(*args, **kwargs)

    return wrapper


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
        Share of the Riesz representer's second moment that only the assumed confounder
        adds. For the ATE it is the gain in precision of the treatment model.
    rho : float
        How adversarially the two are aligned. ``1.0`` is the worst case.
    confounding_strength : float
        The product those three imply, ``|rho| sqrt(cf_y cf_d / (1 - cf_d))``.
    max_bias : float
        The bias scale ``sqrt(sigma^2 nu^2)``. :attr:`bias` multiplies it by
        ``confounding_strength``, which can exceed 1. The name follows DoubleML.
    lower : float
        Bias-adjusted lower bound on the estimate.
    upper : float
        Bias-adjusted upper bound on the estimate.
    ci_lower : float
        Lower one-sided confidence limit of the adjusted bound. It refuses at a
        non-inferential status, where :attr:`plugin_interval_lower` keeps the diagnostic.
        It refuses under the plug-in estimator of ``nu^2`` for a mechanism that is neither
        declared known nor a qualifying ``LogisticRegression`` (docs/roadmap.md F26), and
        when the second-order correction of the plug-in ``nu^2`` does not hold
        (``truncation_active``, ``correction_unstable``, ``corrected_nonpositive``).
    ci_upper : float
        Upper one-sided confidence limit of the adjusted bound. It follows ``ci_lower``.
    level : float
        Coverage level of those limits.
    robustness_value : float or None
        Equal confounding strength that moves the point estimate to the null. ``None``
        means no strength below one reaches it.
    robustness_value_ci : float or None
        The same strength for the confidence limit. It refuses with ``ci_lower``;
        :attr:`robustness_value_plugin_interval` keeps the diagnostic. ``None`` means
        the available limit does not reach the null at any equal strength below one.
    null_hypothesis : float
        The value the robustness values are measured against.
    inference : str, default="influence_curve"
        The inference status of the estimate the bound adjusts. One of
        :data:`~cleverly.inference.influence.InferenceStatus`, which the
        :doc:`inference reference </technical-reference/inference>` lists.
    nu2_estimator : str, default="doubly_robust"
        The resolved estimator of ``nu^2`` that built the bound.
    limit_refusal : MechanismRefusal or None, default=None
        Why the six limit accessors refuse under ``"plugin"``, or ``None`` when the fit's
        treatment mechanism is declared known or is a qualifying ``LogisticRegression``.
    plugin_mechanism : str or None, default=None
        ``"known"`` or ``"logistic"`` when a plug-in bound has limits, else ``None``.
    nu2_correction : float or None, default=None
        The second-order bias ``B`` that the plug-in ``nu^2`` subtracts. ``None`` under
        the doubly robust estimator and wherever the plug-in ``nu^2`` is uncorrected.
    features : tuple of str, default=()
        The X82 part G configurations the bound reads (:func:`bound_features`).  Empty
        for the shipped bound.
    limit_reading : Resolution or None, default=None
        What the registry licenses for the confidence limits of a part G configuration.
        ``None`` on the shipped bound, whose limits carry registered coverage evidence.
    assumption : str or None, default=None
        The assumption a ``DRTMLE`` or ``CTMLE`` point bound adds under
        ``assume_consistent_mechanism=True``, as the summary prints it.  ``None`` otherwise.
    mechanism_learner : str or None, default=None
        The fingerprint of the learner declared with ``mechanism_learner=``.  ``None`` when
        none was declared.
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
    #: ``None`` when no derivation in a source this package cites gives the limits, so two
    #: identical plug-in bounds, and a pickle round trip of one, compare equal.  A NaN here
    #: compared unequal to itself.
    _ci_lower: float | None
    _ci_upper: float | None
    level: float
    robustness_value: float | None
    _robustness_value_ci: float | None
    null_hypothesis: float
    inference: InferenceStatus = "influence_curve"
    nu2_estimator: str = "doubly_robust"
    limit_refusal: MechanismRefusal | None = None
    plugin_mechanism: str | None = None
    nu2_correction: float | None = None
    features: tuple[str, ...] = ()
    limit_reading: Resolution | None = None
    assumption: str | None = None
    mechanism_learner: str | None = None

    @_accepts_stored_limit_names
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
        ci_lower: float | None,
        ci_upper: float | None,
        level: float,
        robustness_value: float | None,
        robustness_value_ci: float | None,
        null_hypothesis: float,
        inference: InferenceStatus = "influence_curve",
        nu2_estimator: str = "doubly_robust",
        limit_refusal: MechanismRefusal | None = None,
        plugin_mechanism: str | None = None,
        nu2_correction: float | None = None,
        features: tuple[str, ...] = (),
        limit_reading: Resolution | None = None,
        assumption: str | None = None,
        mechanism_learner: str | None = None,
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
            ("nu2_estimator", nu2_estimator),
            ("limit_refusal", limit_refusal),
            ("plugin_mechanism", plugin_mechanism),
            ("nu2_correction", nu2_correction),
            ("features", tuple(features)),
            ("limit_reading", limit_reading),
            ("assumption", assumption),
            ("mechanism_learner", mechanism_learner),
        ):
            object.__setattr__(self, name, value)

    @property
    def limit_descriptive(self) -> bool:
        """Whether the one-sided limits print without coverage evidence.

        Returns
        -------
        bool
            ``True`` when :attr:`limit_reading` holds a resolution the registry does not
            license.
        """
        return self.limit_reading is not None and not self.limit_reading.licensed

    @property
    def _limits_derived(self) -> bool:
        """Whether a derivation gives the standard error of these limits."""
        return self.limit_refusal is None and self.max_bias > 0.0

    def _limit_reason(self) -> str | None:
        """The estimator or zero-bias reason for withholding limit inference."""
        if self.limit_refusal is not None:
            return self.limit_refusal.text
        if self.max_bias == 0.0:
            return _ZERO_BIAS_LIMITS_REFUSAL + "."
        return None

    def _limit_refusal(self, operation: str, *, diagnostic: bool = False) -> None:
        """Refuse a limit accessor, in the one order every accessor uses.

        The inference status comes first, because it is a property of the estimate the
        bound adjusts.  A diagnostic accessor skips it, since it exists to report the
        spread at a non-inferential status.  The estimator of ``nu^2`` comes second, and no
        name escapes it, because a limit without a derived standard error is not a
        diagnostic of anything.
        """
        if not diagnostic:
            refuse_inference(self.inference, operation=operation)
        reason = self._limit_reason()
        if reason is not None:
            if self.limit_refusal is not None:
                record_refusal_site(
                    "omitted_variable.limit_estimator",
                    self.limit_refusal.reason,
                    self.limit_refusal.roadmap,
                )
                raise CapabilityError(
                    f"{operation} is not defined under {reason}",
                    reason=self.limit_refusal.reason,
                    alternative=self.limit_refusal.alternative,
                    roadmap=self.limit_refusal.roadmap,
                )
            record_refusal_site("omitted_variable.limit_reason", RefusalReason.INFERENCE_GAP, None)
            raise CapabilityError(
                f"{operation} is not defined under {reason}",
                reason=RefusalReason.INFERENCE_GAP,
                alternative="Read the point bounds and the robustness value, which stay available.",
            )

    def _guarded(self, name: str, stored: float | None, *, diagnostic: bool = False) -> float:
        """One stored limit, after :meth:`_limit_refusal` has let it through."""
        operation = f"SensitivityBounds.{name}"
        self._limit_refusal(operation, diagnostic=diagnostic)
        if stored is None:  # pragma: no cover - a derived bound always stores its limits
            raise CapabilityError(
                f"{operation} was not stored on this bound",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        return stored

    @property
    def ci_lower(self) -> float:
        """Lower confidence limit, when this bound supplies inference."""
        return self._guarded("ci_lower", self._ci_lower)

    @property
    def ci_upper(self) -> float:
        """Upper confidence limit, when this bound supplies inference."""
        return self._guarded("ci_upper", self._ci_upper)

    @property
    def robustness_value_ci(self) -> float | None:
        """Robustness value for a confidence limit, when inference is supplied."""
        self._limit_refusal("SensitivityBounds.robustness_value_ci")
        return self._robustness_value_ci

    @property
    def plugin_interval_lower(self) -> float:
        """Lower plug-in limit, a diagnostic at a non-inferential status."""
        return self._guarded("plugin_interval_lower", self._ci_lower, diagnostic=True)

    @property
    def plugin_interval_upper(self) -> float:
        """Upper plug-in limit, a diagnostic at a non-inferential status."""
        return self._guarded("plugin_interval_upper", self._ci_upper, diagnostic=True)

    @property
    def robustness_value_plugin_interval(self) -> float | None:
        """Robustness value from the plug-in limit, a diagnostic when inference is absent."""
        self._limit_refusal("SensitivityBounds.robustness_value_plugin_interval", diagnostic=True)
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
        ``robustness_value_plugin_interval``.  When the plug-in estimator has no curve for the
        fit's mechanism, the mapping carries ``nu2_estimator`` and ``limit_refusal`` and omits
        those three keys under every name.  Under ``"plugin"`` it always carries
        ``nu2_correction``, which is ``None`` where the plug-in ``nu^2`` is uncorrected.

        Returns
        -------
        dict
            A JSON-compatible mapping of every reported field.
        """
        status = self.inference
        row: dict[str, Any] = {"estimand": self.estimand}
        if is_labelled(status):
            row["inference"] = status
        if self.nu2_estimator != "doubly_robust":
            row["nu2_estimator"] = self.nu2_estimator
            row["nu2_correction"] = self.nu2_correction
        reason = self._limit_reason()
        if reason is not None:
            row["limit_refusal"] = reason
        row |= {
            "psi": self.psi,
            "cf_y": self.cf_y,
            "cf_d": self.cf_d,
            "rho": self.rho,
            "max_bias": self.max_bias,
            "bias": self.bias,
            "lower": self.lower,
            "upper": self.upper,
        }
        if self._limits_derived:
            row |= {
                spread_name("ci_lower", status): self.plugin_interval_lower,
                spread_name("ci_upper", status): self.plugin_interval_upper,
            }
        row["robustness_value"] = self.robustness_value
        if self._limits_derived:
            row[spread_name("robustness_value_ci", status)] = self.robustness_value_plugin_interval
        if self.features:
            row["features"] = "+".join(self.features)
        if self.assumption is not None:
            row["assumption"] = self.assumption
        if self.mechanism_learner is not None:
            row["mechanism_learner"] = self.mechanism_learner
        if self.limit_reading is not None and self._limits_derived:
            row |= {
                "limit_rung": self.limit_reading.rung,
                "limit_licensed": self.limit_reading.licensed,
                "limit_sentence": self.limit_reading.sentence,
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
        header = [
            f"Omitted-variable sensitivity for {self.estimand!r}",
            "-" * 44,
            f"estimate {self.psi:.5g}; bias scale sqrt(sigma^2 nu^2) = {self.max_bias:.5g}",
            f"assumed confounding: cf_y = {self.cf_y:.3g}, cf_d = {self.cf_d:.3g}, "
            f"rho = {self.rho:.3g}"
            f" -> bias <= {self.bias:.5g}",
        ]
        joint = joint_strength_sentence(self.features)
        if joint is not None:
            header.append(joint)
        if self.assumption is not None:
            header.append(self.assumption)
        if self.plugin_mechanism is not None and self.nu2_correction is not None:
            source = (
                "the declared known mechanism"
                if self.plugin_mechanism == "known"
                else "the score of the fitted LogisticRegression mechanism, stacked"
            )
            header.append(
                f"plug-in nu^2, second-order corrected by {self.nu2_correction:.5g}; "
                f"its standard error carries {source}"
            )
        elif self.nu2_estimator == "plugin":
            header.append("plug-in nu^2, uncorrected")
        if self.robustness_value is None:
            rv_line = f"robustness value RV   = unavailable: {_UNREACHABLE_RV}."
        else:
            rv_line = (
                f"robustness value RV   = {self.robustness_value:.4f}: a confounder with "
                f"cf_y = cf_d = {self.robustness_value:.1%} (the share of the residual outcome "
                "variation it explains, and the share of the Riesz representer's second moment "
                f"only it adds) would move the estimate to {self.null_hypothesis:g}."
            )
        reason = self._limit_reason()
        if reason is not None:
            return "\n".join(
                [
                    *header,
                    f"bias-adjusted bounds:  [{self.lower:.5g}, {self.upper:.5g}] ({conclusion})",
                    "",
                    rv_line,
                    f"no one-sided limits or RVa under {reason}",
                ]
            )
        status = self.inference
        lower = self.plugin_interval_lower
        upper = self.plugin_interval_upper
        rv_interval = self.robustness_value_plugin_interval
        rva_line = (
            f"robustness value RVa  = unavailable: {_UNREACHABLE_RV}."
            if rv_interval is None
            else f"robustness value RVa  = {rv_interval:.4f}: the same, for the "
            f"{self.level:.0%} {spread_name('confidence bound', status)}."
        )
        return "\n".join(
            [
                *header,
                f"bias-adjusted bounds:  [{self.lower:.5g}, {self.upper:.5g}]",
                f"with {self.level:.0%} {spread_name('one-sided CIs', status)}: "
                f"[{lower:.5g}, {upper:.5g}] ({conclusion})",
                "",
                rv_line,
                rva_line,
                *(
                    []
                    if self.limit_reading is None
                    else ["", f"confidence limits: {self.limit_reading.sentence}"]
                ),
            ]
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def _confounding_strength(cf_y: float, cf_d: float, rho: float) -> float:
    """``|rho| sqrt(cf_y cf_d / (1 - cf_d))``, the multiplier on the bias scale.

    ``cf_y`` is the share ``eta^2`` of residual outcome variation, so it lies in ``[0, 1]``
    and the bound is finite at ``cf_y = 1`` (Theorem 2 of Chernozhukov et al., 2026).
    ``cf_d = 1`` divides by zero, so ``cf_d`` lies in ``[0, 1)``.
    """
    if not 0.0 <= cf_y <= 1.0:
        raise ValueError(f"cf_y must lie in [0, 1]; got {cf_y}")
    if not 0.0 <= cf_d < 1.0:
        raise ValueError(f"cf_d must lie in [0, 1); got {cf_d}. At cf_d = 1 the bound is unbounded")
    if not 0.0 <= abs(rho) <= 1.0:
        raise ValueError(f"|rho| must lie in [0, 1]; got {rho}")
    return float(abs(rho) * np.sqrt(cf_y * cf_d / (1.0 - cf_d)))


def _one_sided_quantile(level: float, df: int | None = None) -> float:
    """Validate the one-sided coverage level before computing its reference quantile.

    ``df`` is the Student t degrees of freedom of the bounded estimate, or ``None`` for the
    normal reference.
    """
    if not 0.5 <= level < 1.0:
        raise ValueError(f"level must lie in [0.5, 1); got {level}")
    return one_sided_quantile(level, df)


def _benchmark_strengths(
    result: TMLEResult,
    estimand: str,
    benchmark: BenchmarkResult,
    cf_y: float | None,
    cf_d: float | None,
) -> tuple[float, float]:
    """The ``(cf_y, cf_d)`` a benchmark of this fit and estimand assigns to the bound.

    The gains enter unchanged.  Remark 8 of Appendix E.5 of Chernozhukov et al.
    (arXiv:2112.13398v6) states that, with ``k_Y = k_D = 1``, the gain metrics serve as
    proxies for the sensitivity parameters themselves.  ``share = gain / (1 + gain)`` would
    be a different, narrower calibration that no cited source gives.
    """
    if not isinstance(benchmark, BenchmarkResult):
        raise TypeError(f"benchmark must be a BenchmarkResult; got {type(benchmark).__name__}")
    supplied = [name for name, value in (("cf_y", cf_y), ("cf_d", cf_d)) if value is not None]
    if supplied:
        raise ValueError(
            f"benchmark= sets cf_y and cf_d, so {' and '.join(supplied)} cannot also be passed"
        )
    if benchmark.estimand != estimand:
        raise ValueError(
            f"the benchmark is of {benchmark.estimand!r}, and this bound is of {estimand!r}. "
            f"Pass estimand={benchmark.estimand!r}, or benchmark {estimand!r}"
        )
    if benchmark.psi_long != result[estimand].psi:
        raise ValueError(
            f"the benchmark was computed on another fit: its full-fit estimate is "
            f"{benchmark.psi_long!r}, and this fit reports {result[estimand].psi!r}"
        )
    if benchmark.cf_d >= 1.0:
        raise ValueError(
            f"a confounder as strong as {list(benchmark.covariates)} has a cf_d gain of "
            f"{benchmark.gain_d:.4g}. The bound is unbounded at cf_d = 1, so this benchmark "
            "calibrates no finite bound"
        )
    return benchmark.cf_y, benchmark.cf_d


def omitted_variable_bounds(
    result: TMLEResult,
    estimand: str = "ate",
    *,
    cf_y: float | None = None,
    cf_d: float | None = None,
    rho: float = 1.0,
    level: float = 0.95,
    null_hypothesis: float = 0.0,
    nu2_estimator: str = "auto",
    benchmark: BenchmarkResult | None = None,
    assume_consistent_mechanism: bool = False,
    mechanism_learner: Any = None,
) -> SensitivityBounds:
    """Bias-adjusted bounds and robustness values for one estimand.

    Defaults follow the convention of assuming a confounder with ``cf_y = cf_d = 0.03``,
    with worst-case alignment (``rho = 1``).  Prefer
    reading :attr:`SensitivityBounds.robustness_value`, which needs no assumption at
    all.  ``benchmark=`` takes the strengths from a :func:`benchmark` of this fit
    instead.  Its gain statistics are the published proxies for the two shares
    (Chernozhukov et al., arXiv:2112.13398v6, Appendix E.5, Remark 8), so they enter the
    bound unchanged.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.
    estimand : str
        Alias to bound.
    cf_y : float or None
        Assumed share of residual outcome variation the confounder explains, in
        ``[0, 1]``.  ``None`` means 0.03, or the benchmark's ``cf_y`` with ``benchmark=``.
    cf_d : float or None
        Assumed share of the Riesz representer's second moment that only the confounder
        adds, in ``[0, 1)``. For the ATE it is the gain in precision of the treatment
        model.  ``None`` means 0.03, or the benchmark's ``cf_d`` with ``benchmark=``.
    rho : float
        How adversarially the two are aligned. ``1.0`` is the worst case.  A benchmark
        does not set it.  Pass ``rho=benchmark.rho`` to use the benchmark's alignment.
    level : float
        Coverage level of the reported limits.
    null_hypothesis : float
        Value the robustness values are measured against.
    nu2_estimator : {"auto", "doubly_robust", "plugin"}
        Which estimator of the Riesz second moment to use.
    benchmark : BenchmarkResult or None
        A benchmark of this fit and this estimand.  Its ``cf_y`` and ``cf_d`` become the
        assumed strengths, so ``cf_y=`` and ``cf_d=`` must be omitted.
    assume_consistent_mechanism : bool
        ``True`` lifts the refusal of a ``DRTMLE`` fit and of a selecting ``CTMLE`` fit, and
        reports a point bound that assumes a consistent treatment mechanism.  The representer
        reads the DR-TMLE primary mechanism, or one refit of the C-TMLE treatment learner on
        every mechanism covariate.  The bound has no confidence limits.  ``False`` keeps the
        refusal.  Any other fit raises ``ValueError`` when it is ``True``.
    mechanism_learner : learner or None
        With ``assume_consistent_mechanism=True`` on a selecting ``CTMLE`` fit, the learner
        refitted on every mechanism covariate in place of the fit's treatment learner.  A
        per-arm outcome-adaptive fit needs it, because its treatment learner reads one
        column.  ``None`` keeps the default refit.  On any other call it raises
        ``ValueError``.

    Returns
    -------
    SensitivityBounds
        Adjusted bounds, their confidence limits, and the robustness values. Under
        ``nu2_estimator="plugin"`` the bounds, ``max_bias`` and the point robustness value
        are reported, and the one-sided limits and the confidence-limit robustness value
        refuse when read, unless the mechanism is declared known or is a qualifying
        ``LogisticRegression`` (docs/roadmap.md F26). A zero bias scale also leaves the point
        bounds available but refuses the limits, whose square-root derivative is undefined.
        An unreachable robustness threshold is ``None``. On a fit with a response mechanism,
        a missing treatment, an intermediate variable, or a stratum parameter, ``features``
        names them and ``limit_reading`` marks the limits descriptive until a registered study
        covers them. With ``repeats`` above one, each bound is the median of the draws'
        bounds, ``max_bias`` the median of their bias scales, and each limit the median rule
        of the draws' limits.

    Raises
    ------
    ValueError
        If ``nu2_estimator`` is not one of :data:`NU2_ESTIMATORS`, if ``cf_y`` lies
        outside ``[0, 1]`` or ``cf_d`` outside ``[0, 1)``, if ``|rho|`` exceeds one, or if
        ``level`` lies outside ``[0.5, 1)``.  With ``benchmark=``, also if ``cf_y=`` or
        ``cf_d=`` is passed, or if the benchmark is of another estimand or another fit.
    CapabilityError
        On every refusal :func:`sensitivity_elements` raises: a fit the rule table
        refuses, an estimand the bound does not cover, or a doubly robust estimate of
        ``nu^2`` that is not positive.

    Notes
    -----
    A fit made with ``treatment_adjustment=`` (``treatment_covariates=`` in
    :meth:`cleverly.TMLE.fit`) lets the treatment mechanism read a subset of the adjustment
    set. The Riesz representer of the bound uses that fitted mechanism. So the bound reads the
    statistical-model restriction that the fit's summary prints, and it is claimed only under
    that restriction. Collaborative fits stay refused. If a later release admits a collaborative
    fit that selects nothing, the same statement covers it.
    """
    _one_sided_quantile(level)
    if benchmark is not None:
        cf_y, cf_d = _benchmark_strengths(result, estimand, benchmark, cf_y, cf_d)
    cf_y = 0.03 if cf_y is None else cf_y
    cf_d = 0.03 if cf_d is None else cf_d
    draws = _bound_draws(
        result,
        estimand,
        nu2_estimator,
        consistent=assume_consistent_mechanism,
        mechanism_learner=mechanism_learner,
    )
    elements = _combined_elements(draws, result)
    estimate = result[estimand]
    quantile = _one_sided_quantile(level, estimate.reference_df)
    strength = _confounding_strength(cf_y, cf_d, rho)
    psis = np.array([draw.psi for draw in draws])
    scales = np.array([draw.elements.max_bias for draw in draws])

    # Each draw's bound, then the median over draws. One draw is the shipped bound.
    lower, _ = _median_rule(psis - strength * scales, np.zeros(len(draws)))
    upper, _ = _median_rule(psis + strength * scales, np.zeros(len(draws)))

    # One-sided limits on each end, Theorem 4 in Section 4 of Chernozhukov et al. (2026).
    # The curve adds the uncertainty of the bias term to that of the estimate. The plug-in
    # has no curve, so its bound stores no limit and its accessors refuse. Several draws
    # combine each draw's limit by the median rule of the estimate.
    ci_lower: float | None = None
    ci_upper: float | None = None
    if elements.psi_max_bias is not None:
        ci_lower = _limit(draws, result, strength, -1.0, quantile)
        ci_upper = _limit(draws, result, strength, 1.0, quantile)

    rv, rva = _robustness_values(
        result, draws, float(estimate.psi), rho, level, null_hypothesis, estimate.reference_df
    )
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
        nu2_estimator=elements.nu2_estimator,
        limit_refusal=elements.limit_refusal,
        plugin_mechanism=elements.plugin_mechanism,
        nu2_correction=elements.nu2_correction,
        features=elements.features,
        limit_reading=_limit_reading(result, estimand, elements.features),
        assumption=(
            _consistent_mechanism_assumption(result, mechanism_learner)
            if assume_consistent_mechanism
            else None
        ),
        mechanism_learner=_mechanism_fingerprint(result, mechanism_learner),
    )


def _bound_std_error(psi_estimate: FloatArray, psi_bias: FloatArray, result: TMLEResult) -> float:
    """Standard error of a bias-adjusted bound."""
    combined = np.asarray(psi_estimate, dtype=float) + np.asarray(psi_bias, dtype=float)
    return float(np.sqrt(influence_variance(combined, result.data.cluster)))


def _edge(draws: tuple[_Draw, ...], signed_strength: float) -> float:
    """The median over draws of ``psi + signed_strength * max_bias``."""
    points = np.array([draw.psi + signed_strength * draw.elements.max_bias for draw in draws])
    return _median_rule(points, np.zeros(len(draws)))[0]


def _limit(
    draws: tuple[_Draw, ...], result: TMLEResult, strength: float, side: float, quantile: float
) -> float:
    """The one-sided limit of the bound at ``strength``, by the median rule.

    ``side`` names the end: ``-1`` for the lower bound, which the limit extends downward,
    and ``1`` for the upper.  It is separate from the strength, which can be zero.
    """
    signed_strength = side * strength
    points, errors = [], []
    for draw in draws:
        psi_bias = draw.elements.psi_max_bias
        assert psi_bias is not None and draw.curve is not None
        points.append(draw.psi + signed_strength * draw.elements.max_bias)
        errors.append(_bound_std_error(draw.curve, signed_strength * psi_bias, result))
    center, error = _median_rule(np.array(points), np.array(errors))
    return center + side * quantile * error


def _robustness_values(
    result: TMLEResult,
    draws: tuple[_Draw, ...],
    psi: float,
    rho: float,
    level: float,
    null_hypothesis: float,
    df: int | None = None,
) -> tuple[float | None, float | None]:
    """Solve for ``cf_y = cf_d = v`` at which a bound reaches the null.

    The point value is ``None`` when no equal strength in ``[0, 1)`` reaches the null.
    The limit value is also ``None`` when its influence curve is unavailable; callers
    omit and refuse that accessor rather than interpreting it as an unreachable limit.
    On several draws both solve for the median bound and its median-rule limit.  The
    median of the draws' bounds reaches the null where the median draw's bound does.
    """
    side = 1.0 if null_hypothesis > psi else -1.0
    elements = _combined_elements(draws, result)
    psi_bias = elements.psi_max_bias
    alignment = abs(rho)
    if not 0.0 <= alignment <= 1.0:
        raise ValueError(f"|rho| must lie in [0, 1]; got {rho}")
    quantile = _one_sided_quantile(level, df)

    def equal_share(strength: float) -> float:
        assert alignment > 0.0
        if strength <= alignment:
            ratio = strength / alignment
            value = float(2.0 * ratio / (np.hypot(ratio, 2.0) + ratio))
        else:
            inverse_ratio = alignment / strength
            value = float(2.0 / (1.0 + np.sqrt(1.0 + 4.0 * inverse_ratio**2)))
        if strength > 0.0 and (not np.isfinite(value) or not 0.0 < value < 1.0):
            raise CapabilityError(
                "the equal-strength robustness threshold exists but is not representable "
                "inside [0, 1) at floating-point precision",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        return value

    def limit_at(strength: float) -> float:
        return _limit(draws, result, strength, side, quantile)

    distance = abs(psi - null_hypothesis)
    if distance == 0.0:
        rv = 0.0
    elif alignment == 0.0 or elements.max_bias == 0.0:
        rv = None
    elif len(draws) == 1:
        rv = equal_share(distance / elements.max_bias)
    else:
        rv = _median_point_root(draws, side, null_hypothesis, equal_share)
    if psi_bias is None:
        return rv, None

    # A baseline one-sided limit already on the null side needs no confounding.
    # Squared-error minimisation misses this case and reports a small positive value.
    def signed_gap(strength: float) -> float:
        return side * (limit_at(strength) - null_hypothesis)

    if signed_gap(0.0) >= 0.0:
        return rv, 0.0
    if alignment == 0.0:
        return rv, None
    high = max(1.0, distance / elements.max_bias)
    if not np.isfinite(high):
        raise CapabilityError(
            "the confidence-limit robustness root exceeds finite precision",
            reason=RefusalReason.INVALID_DECLARATION,
        )
    while signed_gap(high) < 0.0:
        if high > np.finfo(float).max / 2.0:
            raise CapabilityError(
                "the confidence-limit robustness root exceeds finite precision",
                reason=RefusalReason.INVALID_DECLARATION,
            )
        high *= 2.0
    strength = float(optimize.brentq(signed_gap, 0.0, high))
    return rv, equal_share(strength)


def _median_point_root(
    draws: tuple[_Draw, ...],
    side: float,
    null_hypothesis: float,
    equal_share: Callable[[float], float],
) -> float | None:
    """The equal-strength value at which the median bound of several draws meets the null.

    Each draw's bound moves toward the null at the rate of its bias scale, so the median
    bound is monotone in the strength.  It never reaches the null when half the draws or
    more have a zero bias scale and stay on the far side.
    """

    def gap(strength: float) -> float:
        return side * (_edge(draws, side * strength) - null_hypothesis)

    high = 1.0
    while gap(high) < 0.0:
        if high > np.finfo(float).max / 2.0:
            return None
        high *= 2.0
    return equal_share(float(optimize.brentq(gap, 0.0, high)))


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
        Gain in explained outcome variation from adjusting for those covariates,
        ``(R^2_long - R^2_short) / (1 - R^2_long)``, clipped to ``[0, 1]``.  The raw value
        is :attr:`gain_y`.
    cf_d : float
        Gain in the Riesz representer's second moment from adjusting for those
        covariates, ``(nu2_long - nu2_short) / nu2_short``, clipped to ``[0, 1]``.  The raw
        value is :attr:`gain_d`.  Both gains are the benchmark convention of Chernozhukov
        et al. (arXiv:2112.13398v6, Appendix E.5, Remark 8) and of DoubleML.  Remark 8
        states that they serve as proxies for the shares that
        :func:`omitted_variable_bounds` takes, so they enter it unchanged.  Pass the
        whole record as ``benchmark=``.
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
        benchmark again.  ``None`` only on a record built without a seed.
    features : tuple of str
        The second mechanisms and the stratum that the representer carries, as
        :attr:`SensitivityElements.features` names them.  On such a fit ``cf_d`` measures a
        joint strength, and :meth:`summary` says so.

    Attributes
    ----------
    gain_y : float
    gain_d : float
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
    features: tuple[str, ...] = ()

    @property
    def gain_y(self) -> float:
        """The unclipped outcome gain, ``sigma2_short / sigma2_long - 1``.

        A value above 1 says that a confounder as strong as the benchmark covariates would
        explain all of the remaining outcome variation.  ``cf_y`` then holds the largest
        share, 1.
        """
        if self.sigma2_long <= 0.0:
            return float("inf")
        return float(self.sigma2_short / self.sigma2_long - 1.0)

    @property
    def gain_d(self) -> float:
        """The unclipped Riesz gain, ``nu2_long / nu2_short - 1``.

        A value of 1 or more gives ``cf_d = 1``, at which the bound is unbounded.
        """
        if self.nu2_short <= 0.0:
            return float("inf")
        return float(self.nu2_long / self.nu2_short - 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation.

        Returns
        -------
        dict
            A JSON-compatible mapping of every reported field.
        """
        values: dict[str, Any] = {
            "estimand": self.estimand,
            "covariates": ", ".join(self.covariates),
            "cf_y": self.cf_y,
            "cf_d": self.cf_d,
            "gain_y": self.gain_y,
            "gain_d": self.gain_d,
            "rho": self.rho,
            "delta_psi": self.delta_psi,
            "psi_long": self.psi_long,
            "psi_short": self.psi_short,
        }
        if self.features:
            values["features"] = "+".join(self.features)
        return values

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
                *self._joint_lines(),
                *self._clip_lines(),
                f"the estimate moved by {self.delta_psi:+.5g} when these covariates were dropped",
            ]
        )

    def _joint_lines(self) -> list[str]:
        """The joint-strength sentence, on a fit whose representer carries a second mechanism."""
        joint = joint_strength_sentence(self.features)
        return [] if joint is None else [joint]

    def _clip_lines(self) -> list[str]:
        """One line for each gain that the record clips at 1."""
        lines = []
        if self.gain_y > 1.0:
            lines.append(
                f"cf_y is the gain {self.gain_y:.4g} clipped at 1: a confounder this strong "
                "explains all remaining outcome variation"
            )
        if self.gain_d >= 1.0:
            lines.append(
                f"cf_d is the gain {self.gain_d:.4g} clipped at 1: the bound is unbounded "
                "for a confounder this strong"
            )
        return lines

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.summary()


def _benchmark_covariates(data: Any) -> tuple[str, ...]:
    """The covariates a benchmark can name, in fit order, with an encoded covariate once.

    An encoded categorical covariate enters the fit as a block of indicator columns. One
    indicator alone is not a covariate, so the column name stands for the whole block.
    """
    sources = data.indicator_sources
    return tuple(dict.fromkeys(sources.get(name, name) for name in data.covariate_names))


def _benchmark_columns(data: Any, names: tuple[str, ...]) -> tuple[str, ...]:
    """The design columns that the covariate ``names`` stand for, in request order."""
    sources = data.indicator_sources
    return tuple(
        column
        for name in names
        for column in data.covariate_names
        if sources.get(column, column) == name
    )


def _benchmark_names(result: TMLEResult, covariates: Any) -> tuple[str, ...]:
    """The covariate names a ``covariates=`` value requests, one bare name included.

    A malformed request is reported before any refusal: an empty one, an indicator column
    of an encoded covariate, and a name the fit does not adjust for.  :func:`benchmark`
    and the ``benchmark`` method of :class:`~cleverly.assessment.SensitivityFacade` each
    call this first, and :func:`benchmark_refusal` returns ``None`` where it raises.

    Parameters
    ----------
    result : TMLEResult
        The fitted result whose covariates the names must be.
    covariates : str or sequence of str
        The requested covariate names.  An encoded categorical covariate is named by its
        column, not by one of its indicator columns.

    Returns
    -------
    tuple of str
        The requested names, in request order.

    Raises
    ------
    DataError
        If the request names no covariate, names an indicator column, or names a
        covariate the fit does not adjust for.
    """
    names = tuple([covariates] if isinstance(covariates, str) else covariates)
    if not names:
        raise DataError(
            "benchmark needs at least one covariate to drop; covariates= names none, and a "
            "refit that drops nothing measures nothing"
        )
    sources = result.data.indicator_sources
    for name in names:
        if name in sources:
            raise DataError(
                f"{name!r} is an indicator column of the encoded covariate {sources[name]!r}. "
                f"Name {sources[name]!r} to drop its whole encoded block"
            )
    fitted = _benchmark_covariates(result.data)
    unknown = sorted(set(names).difference(fitted))
    if unknown:
        raise DataError(f"unknown covariates {unknown}; this fit adjusts for {list(fitted)}")
    return names


def benchmark_refusal(result: TMLEResult, covariates: Any = None) -> Refusal | None:
    """Return the refusal a :func:`benchmark` request meets for its ``covariates``.

    :func:`benchmark` raises the sentence this function returns, and the ``benchmark``
    capability row of :class:`~cleverly.assessment.SensitivityFacade` quotes it for the
    same request.  The short model is a refit without the named covariates.  A request
    that names every covariate of the fit leaves the refit nothing to adjust for, and
    this package fits no TMLE without a covariate, so the request is refused.  The
    unadjusted comparison is well posed, and no estimator here computes it.

    Parameters
    ----------
    result : TMLEResult
        A fitted point-treatment result.
    covariates : str, sequence of str, or None
        The requested covariate names.  ``None`` stands for the omitted argument, which
        no value can lift on a fit with one covariate.

    Returns
    -------
    Refusal or None
        The refusal, or ``None`` when the covariates leave one to adjust for or
        form a malformed request.  :func:`benchmark` reports a malformed request itself,
        before any refusal.
    """
    if result.data.known_treatment is not None:
        return Refusal(
            RefusalReason.INVALID_DECLARATION,
            "benchmark compares the treatment mechanism with and without the named "
            "covariates, and this fit's data declares the mechanism known, so dropping a "
            "covariate cannot change it. A known design mechanism leaves no omitted "
            "confounder of the treatment to calibrate against.",
        )
    fitted = _benchmark_covariates(result.data)
    try:
        dropped = set(fitted[:1] if covariates is None else _benchmark_names(result, covariates))
    except DataError:
        return None
    if any(name not in dropped for name in fitted):
        return None
    return Refusal(
        RefusalReason.IMPLEMENTATION_GAP,
        f"benchmark cannot drop every covariate of this fit, {list(fitted)}. The short "
        "model would adjust for nothing, and a fit needs at least one covariate.",
        alternative=(
            "Name a proper subset of the covariates, which exists only when the fit "
            "adjusts for two or more."
        ),
        roadmap="X82",
    )


def _axis_benchmark_refusal(axis: str, names: tuple[str, ...]) -> Refusal:
    """Why a benchmark that drops a covariate the declared intervention reads is refused."""
    noun = {"regime": "regime", "policy": "policy", "msm": "working model"}[axis]
    return Refusal(
        RefusalReason.INVALID_DECLARATION,
        f"benchmark refits the model without {list(names)}, and the declared {noun} reads "
        f"them, so the short model would estimate another target. The benchmark compares "
        f"two fits of one parameter.",
        alternative=f"Name covariates that the declared {noun} does not read.",
    )


def _axis_target_moved(result: TMLEResult, short: TMLEResult) -> bool:
    """Whether the short refit evaluated the declared intervention to other values."""
    long, other = result.repeats[0].nuisance, short.repeats[0].nuisance
    axis = result.config.parameter_axis
    if axis == "regime" and long.regimes is not None and other.regimes is not None:
        return not np.array_equal(long.regimes.values, other.regimes.values)
    if axis == "msm" and long.msm is not None and other.msm is not None:
        return not (
            np.array_equal(long.msm.design, other.msm.design)
            and np.array_equal(long.msm.weights, other.msm.weights)
        )
    if axis == "policy" and long.policies is not None and other.policies is not None:
        return not np.array_equal(long.policies.shifted, other.policies.shifted)
    return False


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

    On a fit with ``repeats`` above one, each of ``sigma2`` and ``nu2`` is the median of that
    moment over the draws, taken separately for the full fit and for the refit.  The two
    medians need not come from one draw.

    Parameters
    ----------
    result : TMLEResult
        A fitted result.
    covariates : sequence of str
        Observed covariates to calibrate against.  An encoded categorical covariate is
        named by its column, and the refit drops its whole block of indicator columns.
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
    DataError
        If ``covariates`` is empty, names an indicator column of an encoded covariate, or
        names a covariate the fit does not adjust for.  The argument is checked before any
        refusal.
    ValueError
        If ``nu2_estimator`` is not one of :data:`NU2_ESTIMATORS`.
    CapabilityError
        If the result carries no fitted estimator, on every refusal
        :func:`sensitivity_elements` raises for the full fit or the short refit, and on
        the refusal of ``benchmark_refusal``.  The full fit and the covariates are
        checked before the refit runs.
    """
    names = _benchmark_names(result, covariates)
    estimator = result.estimator
    if estimator is None:
        raise CapabilityError(
            "benchmark needs the fitted estimator that produced the result",
            reason=RefusalReason.INVALID_DECLARATION,
        )

    # The short model is a refit, so it carries the same reproducibility question a
    # refutation does: an estimator with no ``random_state`` redraws its folds every time,
    # and the result is cached on the fit and survives ``save``.  Resolve one seed, run the
    # refit under it, and report it.  Same convention as ``cleverly.validation.refute``.
    seed = resolve_assessment_seed(result, random_state)

    long_elements = _median_elements(result, estimand, nu2_estimator)
    refusal = benchmark_refusal(result, names)
    if refusal is not None:
        raise CapabilityError.from_refusal(refusal)
    short_data = result.data.without_covariates(_benchmark_columns(result.data, names))
    axis = result.config.parameter_axis
    try:
        short_result = estimator.refit(
            short_data, intermediate_value=result.intermediate_value, random_state=seed
        )
    except (KeyError, DataError) as error:
        if axis not in _AXIS_GROUPS:
            raise
        # The refit evaluates the declared regime, policy or working model again on the
        # short data, and the declaration reads a covariate the request drops.
        raise CapabilityError.from_refusal(_axis_benchmark_refusal(axis, names)) from error
    if _axis_target_moved(result, short_result):
        raise CapabilityError.from_refusal(_axis_benchmark_refusal(axis, names))
    short_elements = _median_elements(short_result, estimand, nu2_estimator)

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
        features=long_elements.features,
    )


def robustness_value(
    result: TMLEResult,
    estimand: str = "ate",
    *,
    rho: float = 1.0,
    level: float = 0.95,
    null_hypothesis: float = 0.0,
    nu2_estimator: str = "auto",
    assume_consistent_mechanism: bool = False,
    mechanism_learner: Any = None,
) -> dict[str, Any]:
    """The confounding strength that would explain the effect away.

    Returns ``{"rv": ..., "rva": ...}``: the value of ``cf_y = cf_d`` at which the point
    estimate reaches ``null_hypothesis``, and the value at which its confidence bound
    does. ``None`` means no equal strength below one reaches the null. A confidence
    limit that already crosses the null has ``rva = 0``. This is the single most useful
    number in this module, because it requires no
    guess about how strong an unmeasured confounder might be -- it reports the
    threshold and lets the reader judge whether it is plausible.

    The confidence bound is read off the estimate's influence curve, so at a
    non-inferential status ``"rva"`` takes the name
    :func:`~cleverly.inference.influence.spread_name` gives it, ``"rv_plugin_interval"``,
    and the mapping opens with ``"inference"`` naming the status, as
    :meth:`SensitivityBounds.to_dict` does.  Under ``nu2_estimator="plugin"`` the
    mapping carries ``"nu2_estimator"``.  It carries ``"limit_refusal"`` and no
    confidence-limit value under any name when the mechanism is neither declared known nor a
    qualifying ``LogisticRegression`` (docs/roadmap.md F26).

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
    assume_consistent_mechanism : bool
        ``True`` lifts the refusal of a ``DRTMLE`` fit and of a selecting ``CTMLE`` fit, as
        :func:`omitted_variable_bounds` states.  The point values remain, and no limit does.
    mechanism_learner : learner or None
        The learner of the consistent-mechanism refit, as :func:`omitted_variable_bounds`
        states.

    Returns
    -------
    dict of str to Any
        The strength that moves the point estimate to the null, the one that moves the
        confidence limit there, and ``max_bias``. A threshold is ``None`` when unreachable.
        At a non-inferential status, also the
        status under ``"inference"``. Under the plug-in estimator, ``"nu2_estimator"``, and
        ``"limit_refusal"`` in place of the confidence-limit value when the mechanism does not
        qualify. When :attr:`SensitivityBounds.features` is not empty, such as on a fit
        with a second mechanism, a stratum, a regime, a shift, or an MSM coefficient, also
        ``"limit_licensed"`` and ``"limit_sentence"``, which mark the limit descriptive.

    Raises
    ------
    ValueError
        If ``nu2_estimator`` is not one of :data:`NU2_ESTIMATORS`, or if ``|rho|``
        exceeds one, or if ``level`` lies outside ``[0.5, 1)``.
    CapabilityError
        On every refusal :func:`sensitivity_elements` raises: a fit the rule table
        refuses, an estimand the bound does not cover, or a doubly robust estimate of
        ``nu^2`` that is not positive.
    """
    draws = _bound_draws(
        result,
        estimand,
        nu2_estimator,
        consistent=assume_consistent_mechanism,
        mechanism_learner=mechanism_learner,
    )
    elements = _combined_elements(draws, result)
    estimate = result[estimand]
    rv, rva = _robustness_values(
        result, draws, float(estimate.psi), rho, level, null_hypothesis, estimate.reference_df
    )
    status = estimate.inference
    values: dict[str, Any] = {"inference": status} if is_labelled(status) else {}
    if elements.nu2_estimator != "doubly_robust":
        values["nu2_estimator"] = elements.nu2_estimator
        values["nu2_correction"] = elements.nu2_correction
    if elements.psi_max_bias is None:
        refusal = elements.limit_refusal
        values |= {
            "rv": rv,
            "max_bias": elements.max_bias,
            "limit_refusal": (
                refusal.text if refusal is not None else _ZERO_BIAS_LIMITS_REFUSAL + "."
            ),
        }
        return values
    values |= {"rv": rv, spread_name("rva", status): rva, "max_bias": elements.max_bias}
    reading = _limit_reading(result, estimand, elements.features)
    if reading is not None:
        values |= {"limit_licensed": reading.licensed, "limit_sentence": reading.sentence}
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
    assume_consistent_mechanism: bool = False,
    mechanism_learner: Any = None,
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
    assume_consistent_mechanism : bool
        ``True`` lifts the refusal of a ``DRTMLE`` fit and of a selecting ``CTMLE`` fit, as
        :func:`omitted_variable_bounds` states.  The point values remain, and no limit does.
    mechanism_learner : learner or None
        The learner of the consistent-mechanism refit, as :func:`omitted_variable_bounds`
        states.

    Returns
    -------
    DataFrame
        One row per grid cell, in the frame library the fit was given, with columns
        ``cf_d``, ``cf_y``, and ``value``.

    Raises
    ------
    ValueError
        If ``bound`` is not ``"lower"`` or ``"upper"``, if ``nu2_estimator`` is not one
        of :data:`NU2_ESTIMATORS`, if a ``cf_y`` value lies outside ``[0, 1]`` or a
        ``cf_d`` value outside ``[0, 1)``, or if
        ``|rho|`` exceeds one.
    CapabilityError
        On every refusal :func:`sensitivity_elements` raises: a fit the rule table
        refuses, an estimand the bound does not cover, or a doubly robust estimate of
        ``nu^2`` that is not positive.
    """
    if bound not in ("lower", "upper"):
        raise ValueError(f"bound must be 'lower' or 'upper'; got {bound!r}")
    draws = _bound_draws(
        result,
        estimand,
        nu2_estimator,
        consistent=assume_consistent_mechanism,
        mechanism_learner=mechanism_learner,
    )
    sign = -1.0 if bound == "lower" else 1.0

    cf_d_grid = np.linspace(0.0, grid_bounds[0], grid_size)
    cf_y_grid = np.linspace(0.0, grid_bounds[1], grid_size)
    rows_d, rows_y, values = [], [], []
    for cf_d in cf_d_grid:
        for cf_y in cf_y_grid:
            strength = _confounding_strength(float(cf_y), float(cf_d), rho)
            rows_d.append(float(cf_d))
            rows_y.append(float(cf_y))
            values.append(_edge(draws, sign * strength))
    return result.data.frame_like({"cf_d": rows_d, "cf_y": rows_y, "value": values})
