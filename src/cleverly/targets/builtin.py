"""The estimands a classic point-treatment fit reports.

Each is a thin adapter over the influence-function builders in
:mod:`cleverly.inference.influence`, which are unchanged: this module says *which*
functional and *on what scale*, and the arithmetic stays where its derivation and
its Gateaux tests already are.

Registration order is report order.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .._typing import FloatArray
from ..inference.delta import log_odds_ratio_influence, log_ratio_influence
from ..inference.influence import ParameterEstimate, atc_estimate, att_estimate
from .base import Identification, Target, TargetContext, parameter_name

__all__ = ["BUILTIN_TARGETS"]

_POSITIVITY = (
    "positivity: 0 < P(A = 1 | W) < 1 almost surely, so both counterfactual "
    "means are supported by data at every covariate value",
)
_NO_INTERFERENCE = (
    "no interference: one unit's potential outcome does not depend on other units' "
    "treatment assignments",
)
_POINT_TREATMENT = (
    "consistency: Y = Y^a when A = a",
    *_NO_INTERFERENCE,
    "no unmeasured confounding: Y^a is independent of A given W",
    *_POSITIVITY,
)

_MEAN_ID = Identification(
    assumptions=_POINT_TREATMENT,
    required_nuisances=("outcome_regression", "treatment_mechanism"),
    dr_condition=(
        "consistent if either Qbar(A, W) or g(W) is consistent; with delta= the "
        "mechanism half becomes the product g * P(Delta = 1 | A, W), and with "
        "intermediate= the product g * P(Z = z | A, W) * P(Delta = 1 | A, W)"
    ),
    references=("van der Laan & Rubin (2006)", "Gruber & van der Laan (2010)"),
)

_CONDITIONAL_ID = Identification(
    assumptions=(
        *_POINT_TREATMENT,
        "the conditioning event (having received the arm the effect is reported among) "
        "has positive probability, so the parameter is defined",
        "positivity is only needed *at the reference arm* within that population: the "
        "conditioning arm's own outcomes are already drawn from it and are not reweighted, "
        "which is why g_ref(W) near zero is what these estimands are sensitive to and why "
        "g_bounds='auto' truncates them harder",
    ),
    required_nuisances=("outcome_regression", "treatment_mechanism"),
    dr_condition=(
        "consistent if either Qbar(A, W) or g(W) is consistent; the influence curve "
        "carries an extra term for the randomness of the conditioning event"
    ),
    references=("van der Laan (2010)",),
)


_REGIME_ID = Identification(
    assumptions=(
        "consistency: Y = Y^a when A = a",
        *_NO_INTERFERENCE,
        "no unmeasured confounding: Y^a is independent of A given W",
        "positivity *for the regime*: g(a | W) > 0 wherever the regime assigns arm a "
        "with positive probability -- a weaker requirement than positivity for every "
        "arm when the regime is deterministic, and a different one",
        "the regime is a known function of W: g* does not depend on the observed-data "
        "law, so the influence function carries no term for estimating it. An "
        "intervention defined *through* the estimated mechanism is a different "
        "parameter with a further term, and is ey_ipsi",
    ),
    required_nuisances=("outcome_regression", "treatment_mechanism"),
    dr_condition=(
        "consistent if either Qbar(A, W) or g(W) is consistent; the mechanism half "
        "picks up P(Delta = 1 | A, W) and P(Z = z | A, W) as a product exactly as the "
        "arm-indexed means do"
    ),
    references=("Robins (2004)", "Diaz & van der Laan (2012)", "van der Laan (2013)"),
)


_IPSI_ID = Identification(
    assumptions=(
        "consistency: Y = Y^a when A = a",
        *_NO_INTERFERENCE,
        "no unmeasured confounding: Y^a is independent of A given W",
        "*no positivity assumption*: the clever covariate is delta/D at A=1 and 1/D at "
        "A=0 with D = delta*g + 1 - g, so it lies between min(delta, 1/delta) and "
        "max(delta, 1/delta) however small g is. This is the estimand's reason for "
        "existing, not an oversight in the list",
        "the intervention is a functional of the observed-data law: q_delta is built "
        "out of g, so the influence function carries a term for the pathwise derivative "
        "through it and the estimator fluctuates the mechanism as well as Qbar",
    ),
    required_nuisances=("outcome_regression", "treatment_mechanism"),
    dr_condition=(
        "NOT doubly robust, and the only target here that is not: g appears in the "
        "estimand itself, so every term of the second-order remainder carries "
        "(ghat - g0) as a factor. The remainder vanishes identically when the mechanism "
        "is consistent whatever Qbar does, and no accuracy in Qbar rescues an "
        "inconsistent mechanism -- there the remainder is "
        "(delta - 1) delta E[(g0 - ghat)^2 (Qbar(1,W) - Qbar(0,W)) / (D0 Dhat^2)], "
        "second order but not zero. Read the interval as conditional on g being right. "
        "With delta= even that is not sufficient: the remainder gains a cross term in the "
        "error of the product g*pi, so it needs g right AND one of pi, Qbar right"
    ),
    references=("Kennedy (2019)",),
)


_SHIFT_ID = Identification(
    assumptions=(
        "consistency: Y = Y^a when A = a",
        *_NO_INTERFERENCE,
        "no unmeasured confounding: Y^a is independent of A given W",
        "positivity *for the shifted dose*: g(d(a, w) | w) > 0 wherever g(a | w) > 0, so "
        "the dose the policy assigns is one the data have seen at that covariate value. "
        "This is weaker than positivity at every dose -- which no continuum satisfies -- "
        "and is exactly what the cap is declared to secure",
        "the shift is a known function of (A, W): d does not depend on the observed-data "
        "law, so the influence function carries no term for estimating it. A cap fitted "
        "from the data would break this, which is why cap= is required rather than "
        "defaulted to max(A)",
        "with delta=: missingness at random given (A, W), and positivity for it *at the "
        "assigned dose* -- P(Delta = 1 | A = d(a, w), W = w) > 0, not only at the dose "
        "observed. The fluctuation updates Qbar as a function of the dose, so obtaining "
        "Qbar*(d(A, W), W) reads the mechanism where the policy sends the unit",
        "with intermediate=: the same two statements for P(Z = z | A, W), plus that Delta "
        "is not caused by Z -- the assumption missingness_design() states for an arm",
    ),
    required_nuisances=("outcome_regression", "treatment_density"),
    dr_condition=(
        "consistent if either Qbar(A, W) or the conditional density g(a | W) is "
        "consistent; the mechanism half is a density ratio rather than a propensity, so "
        "its error is the error in g(a - delta | W) / g(a | W) rather than in a "
        "probability. With delta= that half becomes the *product* of the ratio and "
        "P(Delta = 1 | A, W), and with intermediate= the ratio times "
        "P(Z = z | A, W) * P(Delta = 1 | A, W) -- so it is Qbar right OR the whole "
        "product right, exactly as on the arm path, and not either mechanism alone"
    ),
    references=(
        "Diaz & van der Laan (2012)",
        "Haneuse & Rotnitzky (2013)",
        "Diaz, Williams, Hoffman & Schenck (2023)",
    ),
)


_MSM_ID = Identification(
    assumptions=(
        *_POINT_TREATMENT,
        "the working model and its weights are known functions of (a, V): neither phi "
        "nor h depends on the observed-data law, so the influence function carries no "
        "term for estimating them",
        "the weighted Gram matrix is invertible, so the projection is a single "
        "coefficient vector rather than a set of them",
    ),
    required_nuisances=("outcome_regression", "treatment_mechanism"),
    dr_condition=(
        "consistent if either Qbar(A, W) or g(W) is consistent, exactly as for the "
        "arm-indexed means -- beta is a smooth function of them and of nothing else. "
        "Note this says nothing about the working model being correct: beta is defined "
        "as a projection, so it is the same functional either way"
    ),
    references=(
        "Neugebauer & van der Laan (2007)",
        "van der Laan & Rose (2011), chapter 12",
    ),
)

_POPULATION_INTERVENTION_ID = Identification(
    assumptions=(
        *_POINT_TREATMENT,
        "the observed outcome is complete; under missingness at random the natural-course "
        "mean needs an additional outcome/missingness score equation",
    ),
    required_nuisances=("outcome_regression", "treatment_mechanism"),
    dr_condition=(
        "the intervention mean is consistent if either Qbar(A, W) or g(W) is consistent; "
        "the complete-data natural-course mean is empirical and needs neither nuisance"
    ),
    references=("Díaz Muñoz & van der Laan (2012)",),
)


def _ey(ctx: TargetContext) -> list[ParameterEstimate]:
    """``E[Y(a)]`` for every arm, always named with its label.

    The arm-general counterpart of ``ey1`` / ``ey0``, which name the two arms of a binary
    fit and cannot name a third.  Deliberately *not* collapsed to a bare stem on two
    arms: the names have to stay distinct from each other, and ``ey1`` / ``ey0`` are what
    a two-armed report uses anyway.
    """
    return [
        ctx.finish(
            parameter_name("ey", arm=ctx.label(arm)), mean.psi, mean.influence_curve, "level"
        )
        for arm, mean in sorted(ctx.means.items())
    ]


def _ey1(ctx: TargetContext) -> list[ParameterEstimate]:
    mean = ctx.means[1.0]
    return [ctx.finish("ey1", mean.psi, mean.influence_curve, "level")]


def _ey0(ctx: TargetContext) -> list[ParameterEstimate]:
    mean = ctx.means[0.0]
    return [ctx.finish("ey0", mean.psi, mean.influence_curve, "level")]


def _ey_obs(ctx: TargetContext) -> list[ParameterEstimate]:
    r"""The natural-course mean :math:`E[Y]`."""
    mean = ctx.observed_mean
    return [ctx.finish("ey_obs", mean.psi, mean.influence_curve, "level")]


def _par(ctx: TargetContext) -> list[ParameterEstimate]:
    r""":math:`E[Y] - E[Y^{a_0}]`, with ``reference=`` selecting :math:`a_0`."""
    observed = ctx.observed_mean
    intervention = ctx.means[ctx.reference]
    name = ctx.name_for("par", ctx.reference)
    return [
        ctx.finish(
            name,
            observed.psi - intervention.psi,
            observed.influence_curve - intervention.influence_curve,
            "difference",
        )
    ]


def _paf(ctx: TargetContext) -> list[ParameterEstimate]:
    r""":math:`1 - E[Y^{a_0}]/E[Y]`, on its own identity inference scale."""
    observed = ctx.observed_mean
    intervention = ctx.means[ctx.reference]
    if observed.psi <= 0.0:
        raise ValueError("paf is undefined when the observed outcome risk is zero")
    psi = 1.0 - intervention.psi / observed.psi
    curve = (
        -intervention.influence_curve / observed.psi
        + intervention.psi * observed.influence_curve / observed.psi**2
    )
    return [ctx.finish_unscaled(ctx.name_for("paf", ctx.reference), psi, curve, "fraction")]


def _difference_against_reference(ctx: TargetContext, stem: str) -> list[ParameterEstimate]:
    """``E[Y(a)] - E[Y(ref)]``, once per non-reference arm -- or regime.

    Shared by ``ate`` and ``ate_regime``, which are the same functional of whatever
    :attr:`~cleverly.targets.TargetContext.means` is keyed by.  The stem is a parameter
    rather than a constant only because a reported name has to say which of the two it
    came from; see :data:`BUILTIN_TARGETS`.
    """
    reference = ctx.means[ctx.reference]
    return [
        ctx.finish(
            ctx.name_for(stem, arm, versus=ctx.reference),
            ctx.means[arm].psi - reference.psi,
            ctx.means[arm].influence_curve - reference.influence_curve,
            "difference",
        )
        for arm in ctx.contrast_arms
    ]


def _level_per_code(ctx: TargetContext, stem: str) -> list[ParameterEstimate]:
    """``E[Y^{g}]`` for every code the fit's means are keyed by.

    The level-side counterpart of :func:`_difference_against_reference`, and shared by
    ``ey_regime``, ``ey_ipsi`` and ``ey_shift`` for the same reason: they are the same
    functional of whatever :attr:`~cleverly.targets.TargetContext.means` is keyed by, and
    what differs is one level down, in which mean function that property calls.  The stem
    is a parameter only because a reported name has to say which axis it came from.
    """
    return [
        ctx.finish(
            parameter_name(stem, arm=ctx.label(code)),
            mean.psi,
            mean.influence_curve,
            "level",
        )
        for code, mean in sorted(ctx.means.items())
    ]


def _ate(ctx: TargetContext) -> list[ParameterEstimate]:
    """``E[Y(a)] - E[Y(ref)]``, once per non-reference arm."""
    return _difference_against_reference(ctx, "ate")


def _ey_regime(ctx: TargetContext) -> list[ParameterEstimate]:
    """``E[Y^{g*}]`` for every declared regime."""
    return _level_per_code(ctx, "ey_regime")


def _ate_regime(ctx: TargetContext) -> list[ParameterEstimate]:
    """``E[Y^{g*}] - E[Y^{g*_ref}]``, once per non-reference regime."""
    return _difference_against_reference(ctx, "ate_regime")


def _ey_ipsi(ctx: TargetContext) -> list[ParameterEstimate]:
    """``E[Y^{q_delta}]`` for every declared tilt of the treatment mechanism."""
    return _level_per_code(ctx, "ey_ipsi")


def _ate_ipsi(ctx: TargetContext) -> list[ParameterEstimate]:
    """``E[Y^{q_delta}] - E[Y^{q_ref}]``, once per non-reference tilt."""
    return _difference_against_reference(ctx, "ate_ipsi")


def _ey_shift(ctx: TargetContext) -> list[ParameterEstimate]:
    """``E[Y^{d}]`` for every declared shift.

    The same shape as :func:`_ey_regime` because a shift is another thing
    :attr:`~cleverly.targets.TargetContext.means` can be keyed by.  What differs is one
    level down, in which mean function that property calls -- see its docstring.
    """
    return _level_per_code(ctx, "ey_shift")


def _ate_shift(ctx: TargetContext) -> list[ParameterEstimate]:
    """``E[Y^{d}] - E[Y^{d_ref}]``, once per non-reference shift.

    The reference is usually the natural course (``delta=0``), which makes this the
    *effect* of shifting rather than a contrast of two policies -- but it is whichever
    shift the fit declared as reference, exactly as for arms and regimes.
    """
    return _difference_against_reference(ctx, "ate_shift")


def _msm(ctx: TargetContext) -> list[ParameterEstimate]:
    """One coefficient of the declared working model per term.

    :meth:`~cleverly.targets.TargetContext.finish_unscaled` rather than ``finish``,
    because a coefficient vector has no single scale to map back with and the projection
    was therefore solved on the outcome's own scale --
    :func:`~cleverly.inference.influence.msm_coefficients` sets out why.

    The declared ``scale`` is ``"level"`` for every coefficient, which is a statement
    about *inference* and not about the mapping back: a Wald interval on the coefficient
    itself, with no log transform.  That is right for a slope as much as for an intercept.
    """
    return [
        ctx.finish_unscaled(
            parameter_name("msm", arm=ctx.label(code)),
            coefficient.psi,
            coefficient.influence_curve,
            "level",
        )
        for code, coefficient in sorted(ctx.means.items())
    ]


def _ratio_contrasts(
    ctx: TargetContext,
    stem: str,
    influence: Callable[[float, FloatArray, float, FloatArray], tuple[float, FloatArray]],
) -> list[ParameterEstimate]:
    """Build a ratio family from its log-scale influence transformation."""
    reference = ctx.means[ctx.reference]
    out: list[ParameterEstimate] = []
    for arm in ctx.contrast_arms:
        mean = ctx.means[arm]
        log_psi, ic = influence(
            mean.psi, mean.influence_curve, reference.psi, reference.influence_curve
        )
        out.append(
            ctx.finish(
                ctx.name_for(stem, arm, versus=ctx.reference),
                float(np.exp(log_psi)),
                ic,
                "ratio",
                log_psi=log_psi,
            )
        )
    return out


def _rr(ctx: TargetContext) -> list[ParameterEstimate]:
    return _ratio_contrasts(ctx, "rr", log_ratio_influence)


def _or(ctx: TargetContext) -> list[ParameterEstimate]:
    return _ratio_contrasts(ctx, "or", log_odds_ratio_influence)


def _conditional(ctx: TargetContext, stem: str) -> list[ParameterEstimate]:
    """``E[Y^a - Y^r | A = c]``, once per non-reference arm.

    Shared by ``att`` and ``atc``, which differ only in the population ``c`` they
    condition on -- and that is a property of the *group*, so the estimate function is
    chosen by the stem and the arithmetic stays in
    :mod:`cleverly.inference.influence` beside its Gateaux test.

    The names collapse to the bare ``"att"`` / ``"atc"`` on a binary treatment, where
    there is one contrast and the historical short names are unambiguous.
    """
    estimate = att_estimate if stem == "att" else atc_estimate
    effects = estimate(
        ctx.scaled,
        ctx.targeted,
        ctx.submodel,
        ctx.treatment,
        ctx.weights,
        ctx.observed,
        reference=ctx.reference,
    )
    return [
        ctx.finish(
            ctx.name_for(stem, arm, versus=ctx.reference),
            effect.psi,
            effect.influence_curve,
            "difference",
        )
        for arm, effect in sorted(effects.items())
    ]


def _att(ctx: TargetContext) -> list[ParameterEstimate]:
    """``E[Y^a - Y^r | A = a]``: the effect among the units that received ``a``."""
    return _conditional(ctx, "att")


def _atc(ctx: TargetContext) -> list[ParameterEstimate]:
    """``E[Y^a - Y^r | A = r]``: the same contrasts among the reference arm's units."""
    return _conditional(ctx, "atc")


#: In report order.
BUILTIN_TARGETS: tuple[Target, ...] = (
    Target(
        name="ate",
        group="mean",
        scale="difference",
        build=_ate,
        identification=_MEAN_ID,
        in_default_set=True,
        description="average treatment effect, E[Y^1] - E[Y^0]",
    ),
    Target(
        name="att",
        group="att",
        scale="difference",
        build=_att,
        identification=_CONDITIONAL_ID,
        in_default_set=True,
        default_arms="binary",
        undefined_when="the sample contains no units in the conditioning arm",
        description=(
            "average treatment effect on the treated, E[Y^1 - Y^0 | A = 1] -- and with "
            "more arms, E[Y^a - Y^ref | A = a] once per non-reference arm"
        ),
    ),
    Target(
        name="atc",
        group="atc",
        scale="difference",
        build=_atc,
        identification=_CONDITIONAL_ID,
        in_default_set=True,
        default_arms="binary",
        undefined_when="the sample contains no units in the reference arm",
        description=(
            "average treatment effect on the controls, E[Y^1 - Y^0 | A = 0] -- and with "
            "more arms, E[Y^a - Y^ref | A = ref] once per non-reference arm"
        ),
    ),
    Target(
        name="ey",
        group="mean",
        scale="level",
        build=_ey,
        identification=_MEAN_ID,
        in_default_set=True,
        default_arms="multi",
        description="counterfactual mean under each arm, E[Y^a]",
    ),
    Target(
        name="ey1",
        group="mean",
        scale="level",
        build=_ey1,
        identification=_MEAN_ID,
        requires_binary_treatment=True,
        in_default_set=True,
        description="counterfactual mean under treatment, E[Y^1]",
    ),
    Target(
        name="ey0",
        group="mean",
        scale="level",
        build=_ey0,
        identification=_MEAN_ID,
        requires_binary_treatment=True,
        in_default_set=True,
        description="counterfactual mean under control, E[Y^0]",
    ),
    Target(
        name="ey_obs",
        group="mean",
        scale="level",
        build=_ey_obs,
        identification=_POPULATION_INTERVENTION_ID,
        description="natural-course outcome mean, E[Y]",
    ),
    Target(
        name="par",
        group="mean",
        scale="difference",
        build=_par,
        identification=_POPULATION_INTERVENTION_ID,
        description="population attributable risk, E[Y] - E[Y^reference]",
    ),
    Target(
        name="paf",
        group="mean",
        scale="fraction",
        build=_paf,
        identification=_POPULATION_INTERVENTION_ID,
        requires_family="binomial",
        undefined_when="the observed outcome risk is zero, leaving the fraction undefined",
        description="population attributable fraction, 1 - E[Y^reference] / E[Y]",
    ),
    Target(
        name="rr",
        group="mean",
        scale="ratio",
        build=_rr,
        identification=_MEAN_ID,
        requires_family="binomial",
        # Undefined at zero: the log ratio needs both risks strictly positive.
        parameter_bounds=(0.0, float("inf")),
        undefined_when="a counterfactual risk is zero, leaving the log ratio undefined",
        description="risk ratio, E[Y^1] / E[Y^0]",
    ),
    Target(
        name="or",
        group="mean",
        scale="ratio",
        build=_or,
        identification=_MEAN_ID,
        requires_family="binomial",
        # Both risks must lie strictly inside (0, 1) for the odds to be finite.
        parameter_bounds=(0.0, float("inf")),
        undefined_when="a counterfactual risk is 0 or 1, leaving the odds undefined",
        description="odds ratio of the counterfactual risks",
    ),
    Target(
        name="ey_regime",
        group="regime",
        scale="level",
        build=_ey_regime,
        identification=_REGIME_ID,
        parameter_axis="regime",
        in_default_set=True,
        description="counterfactual mean under each declared regime, E[Y^{g*}]",
    ),
    Target(
        name="ate_regime",
        group="regime",
        scale="difference",
        build=_ate_regime,
        identification=_REGIME_ID,
        parameter_axis="regime",
        in_default_set=True,
        description="contrast of each regime against the reference regime",
    ),
    Target(
        name="ey_ipsi",
        group="ipsi",
        scale="level",
        build=_ey_ipsi,
        identification=_IPSI_ID,
        parameter_axis="ipsi",
        requires_binary_treatment=True,
        in_default_set=True,
        description=(
            "counterfactual mean under each declared tilt of the treatment mechanism, "
            "E[Y^{q_delta}] with q_delta the odds of g multiplied by delta"
        ),
    ),
    Target(
        name="ate_ipsi",
        group="ipsi",
        scale="difference",
        build=_ate_ipsi,
        identification=_IPSI_ID,
        parameter_axis="ipsi",
        requires_binary_treatment=True,
        in_default_set=True,
        description="contrast of each tilt against the reference tilt",
    ),
    Target(
        name="ey_shift",
        group="mtp",
        scale="level",
        build=_ey_shift,
        identification=_SHIFT_ID,
        parameter_axis="shift",
        in_default_set=True,
        description="counterfactual mean under each declared shift, E[Y^{d}]",
    ),
    Target(
        name="ate_shift",
        group="mtp",
        scale="difference",
        build=_ate_shift,
        identification=_SHIFT_ID,
        parameter_axis="shift",
        in_default_set=True,
        description="contrast of each shift against the reference shift",
    ),
    Target(
        name="msm",
        group="msm",
        scale="level",
        build=_msm,
        identification=_MSM_ID,
        parameter_axis="msm",
        in_default_set=True,
        description=(
            "coefficients of the declared working model, the h-weighted projection of "
            "the counterfactual means onto m(a, V; beta)"
        ),
    ),
)
